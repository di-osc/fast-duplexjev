use std::{io::Cursor, sync::Arc};

use anyhow::{Context, Result, ensure};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use symphonia::core::{
    formats::{TrackType, probe::Hint},
    io::MediaSourceStream,
};

pub const SAMPLE_RATE: usize = 16_000;
pub(crate) const MEL_BINS: usize = 128;
pub(crate) const MAX_MEL_FRAMES: usize = 3000;
const HOP: usize = 160;
const FFT: usize = 400;

pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: usize,
}

impl Audio {
    pub fn decode(bytes: Vec<u8>) -> Result<Self> {
        let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
        let mut format = symphonia::default::get_probe().probe(
            &Hint::new(),
            source,
            Default::default(),
            Default::default(),
        )?;
        let track = format
            .default_track(TrackType::Audio)
            .context("no audio track")?;
        let track_id = track.id;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
            .context("no audio codec parameters")?;
        let mut decoder =
            symphonia::default::get_codecs().make_audio_decoder(params, &Default::default())?;
        let mut samples = Vec::new();
        let mut interleaved = Vec::<f32>::new();
        let mut sample_rate = 0;
        while let Some(packet) = format.next_packet()? {
            if packet.track_id != track_id {
                continue;
            }
            let decoded = decoder.decode(&packet)?;
            let spec = decoded.spec();
            let rate = spec.rate() as usize;
            ensure!(
                sample_rate == 0 || sample_rate == rate,
                "sample rate changed within audio"
            );
            sample_rate = rate;
            let channels = spec.channels().count();
            ensure!(channels > 0, "audio has no channels");
            interleaved.resize(decoded.samples_interleaved(), 0.0);
            decoded.copy_to_slice_interleaved(&mut interleaved);
            samples.extend(
                interleaved
                    .chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32),
            );
        }
        let audio = Self {
            samples,
            sample_rate,
        };
        audio.validate()?;
        Ok(audio)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            (1..=384_000).contains(&self.sample_rate),
            "invalid audio sample rate"
        );
        ensure!(!self.samples.is_empty(), "audio is empty");
        ensure!(
            self.samples.iter().all(|x| x.is_finite()),
            "audio contains non-finite samples"
        );
        Ok(())
    }
}

pub(crate) struct MelChunk {
    pub values: Vec<f32>,
    pub frames: usize,
}

pub(crate) struct MelExtractor {
    fft: Arc<dyn Fft<f32>>,
    window: [f32; FFT],
    filters: Vec<Vec<(usize, f32)>>,
}

impl MelExtractor {
    pub fn new() -> Self {
        Self {
            fft: FftPlanner::new().plan_fft_forward(FFT),
            window: std::array::from_fn(|i| {
                (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / FFT as f64).cos()) as f32
            }),
            filters: mel_filters(),
        }
    }

    pub fn extract(&self, audio: Audio) -> Result<Vec<MelChunk>> {
        audio.validate()?;
        let mut samples = resample(audio);
        // Match the upstream Decider's right-silence alignment before the STFT.
        const ALIGN: usize = HOP * 8 * 2;
        samples.resize(samples.len().div_ceil(ALIGN) * ALIGN, 0.0);
        let frames = samples.len() / HOP;
        let mut mel = vec![0.0; frames * MEL_BINS];
        let mut bins = [Complex32::default(); FFT];
        let mut scratch = vec![Complex32::default(); self.fft.get_inplace_scratch_len()];
        let mut maximum = f32::NEG_INFINITY;
        for frame in 0..frames {
            for (i, bin) in bins.iter_mut().enumerate() {
                let index = frame as isize * HOP as isize + i as isize - (FFT / 2) as isize;
                let index = index.unsigned_abs();
                let index = if index >= samples.len() {
                    2 * samples.len() - 2 - index
                } else {
                    index
                };
                *bin = Complex32::new(samples[index] * self.window[i], 0.0);
            }
            self.fft.process_with_scratch(&mut bins, &mut scratch);
            let power: [f32; FFT / 2 + 1] = std::array::from_fn(|i| bins[i].norm_sqr());
            for (band, filter) in self.filters.iter().enumerate() {
                let energy: f32 = filter.iter().map(|&(i, w)| power[i] * w).sum();
                let log = energy.max(1e-10).log10();
                mel[band * frames + frame] = log;
                maximum = maximum.max(log);
            }
        }
        mel.iter_mut()
            .for_each(|x| *x = (x.max(maximum - 8.0) + 4.0) / 4.0);
        Ok((0..frames)
            .step_by(MAX_MEL_FRAMES)
            .map(|offset| {
                let count = (frames - offset).min(MAX_MEL_FRAMES);
                let mut values = Vec::with_capacity(count * MEL_BINS);
                for band in 0..MEL_BINS {
                    let start = band * frames + offset;
                    values.extend_from_slice(&mel[start..start + count]);
                }
                MelChunk {
                    values,
                    frames: count,
                }
            })
            .collect())
    }
}

fn mel_filters() -> Vec<Vec<(usize, f32)>> {
    let log_step = 6.4_f64.ln() / 27.0;
    let max_mel = 15.0 + 8.0_f64.ln() / log_step;
    let edges: Vec<_> = (0..MEL_BINS + 2)
        .map(|i| {
            let mel = max_mel * i as f64 / (MEL_BINS + 1) as f64;
            if mel < 15.0 {
                mel * 200.0 / 3.0
            } else {
                1000.0 * (log_step * (mel - 15.0)).exp()
            }
        })
        .collect();
    edges
        .windows(3)
        .map(|edge| {
            let [left, center, right] = [edge[0], edge[1], edge[2]];
            (0..=FFT / 2)
                .filter_map(|bin| {
                    let hz = bin as f64 * SAMPLE_RATE as f64 / FFT as f64;
                    let weight =
                        ((hz - left) / (center - left)).min((right - hz) / (right - center));
                    (weight > 0.0).then_some((bin, (weight * 2.0 / (right - left)) as f32))
                })
                .collect()
        })
        .collect()
}

fn resample(audio: Audio) -> Vec<f32> {
    if audio.sample_rate == SAMPLE_RATE {
        return audio.samples;
    }
    let mut a = audio.sample_rate;
    let mut b = SAMPLE_RATE;
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let up = SAMPLE_RATE / a;
    let down = audio.sample_rate / a;
    // scipy.signal.resample_poly defaults to a Kaiser(5) low-pass FIR.
    let largest = up.max(down);
    let half = 10 * largest;
    let mut kernel: Vec<_> = (0..=2 * half)
        .map(|i| {
            let distance = i as f64 - half as f64;
            let z = std::f64::consts::PI * distance / largest as f64;
            let sinc = if z == 0.0 { 1.0 } else { z.sin() / z };
            let window = bessel_i0(5.0 * (1.0 - (distance / half as f64).powi(2)).sqrt());
            sinc * window
        })
        .collect();
    let scale = up as f64 / kernel.iter().sum::<f64>();
    kernel.iter_mut().for_each(|x| *x *= scale);
    let out_len = (audio.samples.len() * up).div_ceil(down);
    (0..out_len)
        .map(|i| {
            let center = i * down + half;
            let start = center.saturating_sub(2 * half).div_ceil(up);
            let end = (center / up + 1).min(audio.samples.len());
            (start..end)
                .map(|j| audio.samples[j] as f64 * kernel[center - j * up])
                .sum::<f64>() as f32
        })
        .collect()
}

fn bessel_i0(x: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = term;
    for k in 1..=20 {
        term *= (x / (2.0 * f64::from(k))).powi(2);
        sum += term;
    }
    sum
}
