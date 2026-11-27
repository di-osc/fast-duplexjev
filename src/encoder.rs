use anyhow::Result;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, FloatDType, Tensor, TensorData, activation::gelu, module};

use crate::{
    attention::{self, Kv, Segment},
    audio::{MEL_BINS, MelChunk},
    config::AudioConfig,
    layers::{LayerNorm, Linear},
    weights::Weights,
};

pub(crate) fn output_frames(mel_frames: usize) -> usize {
    (mel_frames / 100) * 13 + (mel_frames % 100).div_ceil(8)
}

pub(crate) struct AudioEncoder {
    conv: [(Tensor<4>, Tensor<1>); 3],
    conv_out: Linear,
    position: Tensor<3>,
    layers: Vec<EncoderLayer>,
    norm: LayerNorm,
    channels: usize,
    width: usize,
}

impl AudioEncoder {
    pub fn load(
        weights: &Weights<'_>,
        c: &AudioConfig,
        device: &Device,
        dtype: FloatDType,
    ) -> Result<Self> {
        let channels = c.downsample_hidden_size;
        let conv = |index: usize, input: usize| -> Result<_> {
            let name = format!("audio_tower.conv2d{index}");
            Ok((
                weights.tensor(&format!("{name}.weight"), [channels, input, 3, 3])?,
                weights.tensor(&format!("{name}.bias"), [channels])?,
            ))
        };
        let width = c.d_model;
        let mut position = Vec::with_capacity(13 * width);
        for step in 0..13 {
            for trig in [f32::sin, f32::cos] {
                for i in 0..width / 2 {
                    let exponent = -(i as f32) * (10000.0_f32.ln() / (width / 2 - 1) as f32);
                    position.push(trig(step as f32 * exponent.exp()));
                }
            }
        }
        Ok(Self {
            conv: [conv(1, 1)?, conv(2, channels)?, conv(3, channels)?],
            conv_out: Linear::load(weights, "audio_tower.conv_out", channels * 16, width, false)?,
            position: Tensor::<3>::from_data(TensorData::new(position, [1, 13, width]), device)
                .cast(dtype),
            layers: (0..c.encoder_layers)
                .map(|i| EncoderLayer::load(weights, c, i))
                .collect::<Result<_>>()?,
            norm: LayerNorm::load(weights, "audio_tower.ln_post", width)?,
            channels,
            width,
        })
    }

    pub fn forward(&self, mel: MelChunk, device: &Device, dtype: FloatDType) -> Tensor<3> {
        let chunks = mel.frames.div_ceil(100);
        let mut values = vec![0.0; chunks * MEL_BINS * 100];
        for chunk in 0..chunks {
            let offset = chunk * 100;
            let count = (mel.frames - offset).min(100);
            for band in 0..MEL_BINS {
                let source = band * mel.frames + offset;
                let target = (chunk * MEL_BINS + band) * 100;
                values[target..target + count].copy_from_slice(&mel.values[source..source + count]);
            }
        }
        let mut x =
            Tensor::<4>::from_data(TensorData::new(values, [chunks, 1, MEL_BINS, 100]), device)
                .cast(dtype);
        for (weight, bias) in &self.conv {
            x = gelu(module::conv2d(
                x,
                weight.clone(),
                Some(bias.clone()),
                ConvOptions::new([2, 2], [1, 1], [1, 1], 1),
            ));
        }
        let x = x
            .permute([0, 3, 1, 2])
            .reshape([chunks, 13, self.channels * 16]);
        let x = (self.conv_out.forward(x) + self.position.clone()).reshape([
            1,
            chunks * 13,
            self.width,
        ]);
        let valid = output_frames(mel.frames);
        let mut x = x.narrow(1, 0, valid);
        let segments = (0..valid)
            .step_by(104)
            .map(|start| Segment {
                tokens: start..(start + 104).min(valid),
                prefix: None,
            })
            .collect::<Vec<_>>();
        for layer in &self.layers {
            x = layer.forward(x, &segments);
        }
        self.norm.forward(x)
    }
}

struct EncoderLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    fc1: Linear,
    fc2: Linear,
    pre_attention: LayerNorm,
    pre_mlp: LayerNorm,
    heads: usize,
    width: usize,
}

impl EncoderLayer {
    fn load(w: &Weights<'_>, c: &AudioConfig, index: usize) -> Result<Self> {
        let name = format!("audio_tower.layers.{index}");
        let width = c.d_model;
        let attention = |part: &str| {
            Linear::load(
                w,
                &format!("{name}.self_attn.{part}_proj"),
                width,
                width,
                true,
            )
        };
        Ok(Self {
            q: attention("q")?,
            k: attention("k")?,
            v: attention("v")?,
            output: attention("out")?,
            fc1: Linear::load(w, &format!("{name}.fc1"), width, c.encoder_ffn_dim, true)?,
            fc2: Linear::load(w, &format!("{name}.fc2"), c.encoder_ffn_dim, width, true)?,
            pre_attention: LayerNorm::load(w, &format!("{name}.self_attn_layer_norm"), width)?,
            pre_mlp: LayerNorm::load(w, &format!("{name}.final_layer_norm"), width)?,
            heads: c.encoder_attention_heads,
            width,
        })
    }

    fn forward(&self, x: Tensor<3>, segments: &[Segment]) -> Tensor<3> {
        let [batch, frames, _] = x.dims();
        let norm = self.pre_attention.forward(x.clone());
        let project = |linear: &Linear| {
            linear
                .forward(norm.clone())
                .reshape([batch, frames, self.heads, self.width / self.heads])
                .swap_dims(1, 2)
        };
        let kv = Kv {
            key: project(&self.k),
            value: project(&self.v),
        };
        let context = attention::packed(project(&self.q), &kv, None, segments, false)
            .swap_dims(1, 2)
            .reshape([batch, frames, self.width]);
        let x = x + self.output.forward(context);
        let mlp = self
            .fc2
            .forward(gelu(self.fc1.forward(self.pre_mlp.forward(x.clone()))));
        x + mlp
    }
}
