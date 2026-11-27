use std::{collections::HashSet, fs::File, path::Path, time::Instant};

use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, FloatDType, Tensor, activation::softmax};
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::{
    attention::Segment,
    audio::{Audio, MelChunk, MelExtractor},
    config::Config,
    encoder::{AudioEncoder, output_frames},
    projector::Projector,
    question::{Answer, Answers, Language, Question},
    text::{Decoder, ids},
    weights::Weights,
};

const CHAT_END: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

pub struct Model {
    encoder: AudioEncoder,
    projector: Projector,
    decoder: Decoder,
    tokenizer: Tokenizer,
    extractor: MelExtractor,
    device: Device,
    dtype: FloatDType,
    max_position: usize,
}

pub struct Item {
    pub audio: Audio,
    pub questions: Vec<Question>,
    pub context: Option<String>,
    pub lang: Option<Language>,
}

impl Item {
    pub fn validate(&self) -> Result<()> {
        self.audio.validate()?;
        ensure!(!self.questions.is_empty(), "no questions given");
        let mut ids = HashSet::new();
        for question in &self.questions {
            question.validate()?;
            ensure!(
                ids.insert(&question.id),
                "duplicate question id {:?}",
                question.id
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct BatchLimits {
    pub max_items: usize,
    pub max_tokens: usize,
}

impl Default for BatchLimits {
    fn default() -> Self {
        Self {
            max_items: 4,
            max_tokens: 2048,
        }
    }
}

#[derive(Clone, Default, Serialize)]
pub struct PassStats {
    mode: &'static str,
    clips: usize,
    questions: usize,
    prefix_tokens: usize,
    suffix_tokens: usize,
    prepare_ms: f64,
    forward_ms: f64,
    batches: usize,
    decode_steps: usize,
}

struct Plan {
    mel: Vec<MelChunk>,
    head: Vec<i32>,
    tail: Vec<i32>,
    audio_tokens: usize,
    suffixes: Vec<Suffix>,
}

impl Plan {
    fn prefix_len(&self) -> usize {
        self.head.len() + self.audio_tokens + self.tail.len()
    }

    fn suffix_len(&self) -> usize {
        self.suffixes.iter().map(|suffix| suffix.input.len()).sum()
    }
}

struct Suffix {
    input: Vec<i32>,
    letters: Vec<i32>,
    order: Vec<usize>,
    question: Question,
}

impl Model {
    pub fn load(path: &Path, device: Device, dtype: FloatDType) -> Result<Self> {
        let config: Config = serde_json::from_reader(File::open(path.join("config.json"))?)?;
        config.validate()?;
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!(error))?;
        let audio_dtype = audio_dtype(dtype);
        let weights = Weights::open(path, &device, audio_dtype)?;
        let encoder = AudioEncoder::load(&weights, &config.audio_config, &device, audio_dtype)?;
        let projector = Projector::load(&weights, &config)?;
        let decoder = Decoder::load(&weights.with_dtype(dtype), &config.text_config)?;
        Ok(Self {
            encoder,
            projector,
            decoder,
            tokenizer,
            extractor: MelExtractor::new(),
            device,
            dtype,
            max_position: config.text_config.max_position_embeddings,
        })
    }

    pub fn decide_items(
        &self,
        items: Vec<Item>,
        limits: BatchLimits,
    ) -> Result<(Vec<Answers>, PassStats)> {
        ensure!(
            limits.max_items > 0 && limits.max_tokens > 0,
            "batch limits must be positive"
        );
        let start = Instant::now();
        let plans = items
            .into_iter()
            .map(|item| self.plan(item))
            .collect::<Result<Vec<_>>>()?;
        let mut stats = PassStats {
            mode: "packed",
            clips: plans.len(),
            questions: plans.iter().map(|plan| plan.suffixes.len()).sum(),
            prefix_tokens: plans.iter().map(Plan::prefix_len).sum(),
            suffix_tokens: plans.iter().map(Plan::suffix_len).sum(),
            prepare_ms: ms(start.elapsed()),
            ..Default::default()
        };
        let forward = Instant::now();
        let mut answers = Vec::with_capacity(plans.len());
        let mut plans = plans.into_iter().peekable();
        while let Some(first) = plans.next() {
            let mut tokens = first.prefix_len() + first.suffix_len();
            let mut batch = vec![first];
            while let Some(next) = plans.peek() {
                let next_tokens = next.prefix_len() + next.suffix_len();
                if batch.len() == limits.max_items || tokens + next_tokens > limits.max_tokens {
                    break;
                }
                tokens += next_tokens;
                batch.push(plans.next().expect("peeked plan"));
            }
            answers.extend(self.run(batch)?);
            stats.batches += 1;
        }
        stats.forward_ms = ms(forward.elapsed());
        Ok((answers, stats))
    }

    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!(error))?;
        Ok(encoded.get_ids().iter().map(|&id| id as i32).collect())
    }

    fn plan(&self, item: Item) -> Result<Plan> {
        item.validate()?;
        let lang = item.lang.unwrap_or(item.questions[0].lang);
        let (context_label, audio_label) = match lang {
            Language::En => ("Context:\n", "The user said: "),
            Language::Zh => ("上下文：\n", "用户说："),
        };
        let mut prefix = String::from("<|im_start|>user\n");
        if let Some(context) = item.context.filter(|context| !context.is_empty()) {
            prefix.push_str(context_label);
            prefix.push_str(&context);
            prefix.push_str("\n\n");
        }
        prefix.push_str(audio_label);
        let mel = self.extractor.extract(item.audio)?;
        let mut plan = Plan {
            head: self.encode(&prefix)?,
            tail: self.encode("\n\n")?,
            audio_tokens: mel
                .iter()
                .map(|chunk| output_frames(chunk.frames).div_ceil(2))
                .sum(),
            mel,
            suffixes: Vec::with_capacity(item.questions.len()),
        };
        for question in item.questions {
            let order = question.permutation();
            let input = self.encode(&(question.render(&order) + CHAT_END))?;
            ensure!(
                plan.prefix_len() + input.len() <= self.max_position,
                "question {:?} exceeds the model context of {} tokens",
                question.id,
                self.max_position
            );
            let letters = ('A'..='Z')
                .take(order.len())
                .map(|letter| {
                    let tokens = self.encode(&letter.to_string())?;
                    ensure!(
                        tokens.len() == 1,
                        "candidate {letter} must encode to one token"
                    );
                    Ok(tokens[0])
                })
                .collect::<Result<_>>()?;
            plan.suffixes.push(Suffix {
                input,
                letters,
                order,
                question,
            });
        }
        Ok(plan)
    }

    fn run(&self, plans: Vec<Plan>) -> Result<Vec<Answers>> {
        let mut embeddings = Vec::new();
        let mut prefix_positions = Vec::new();
        let mut prefixes = Vec::new();
        let mut suffix_ids = Vec::new();
        let mut suffix_positions = Vec::new();
        let mut suffixes = Vec::new();
        let mut questions = Vec::new();
        let audio_dtype = audio_dtype(self.dtype);
        for plan in plans {
            let length = plan.prefix_len();
            let prefix = prefix_positions.len()..prefix_positions.len() + length;
            prefix_positions.extend(0..length);
            embeddings.push(self.embed(plan.head));
            for chunk in plan.mel {
                let encoded = self.encoder.forward(chunk, &self.device, audio_dtype);
                embeddings.push(self.projector.forward(encoded).cast(self.dtype));
            }
            embeddings.push(self.embed(plan.tail));
            prefixes.push(Segment {
                tokens: prefix.clone(),
                prefix: None,
            });

            let mut ends = Vec::with_capacity(plan.suffixes.len());
            for question in plan.suffixes {
                let range = suffix_ids.len()..suffix_ids.len() + question.input.len();
                suffix_positions.extend(length..length + range.len());
                suffix_ids.extend(&question.input);
                ends.push((range.end - 1, question));
                suffixes.push(Segment {
                    tokens: range,
                    prefix: Some(prefix.clone()),
                });
            }
            questions.push(ends);
        }
        let cache = self
            .decoder
            .prefix(Tensor::cat(embeddings, 1), &prefix_positions, &prefixes);
        let hidden =
            self.decoder
                .suffix(self.embed(suffix_ids), &suffix_positions, &suffixes, cache);
        questions
            .into_iter()
            .map(|questions| {
                questions
                    .into_iter()
                    .map(|(end, suffix)| {
                        let last = hidden.clone().narrow(1, end, 1);
                        let logits = self
                            .decoder
                            .logits(last, suffix.letters)
                            .cast(FloatDType::F32);
                        let probs = softmax(logits, 2)
                            .into_data()
                            .try_into_vec::<f32>()
                            .map_err(|error| anyhow::anyhow!("{error:?}"))?;
                        ensure!(
                            probs.iter().all(|p| p.is_finite()),
                            "model produced non-finite probabilities"
                        );
                        let best = probs
                            .iter()
                            .enumerate()
                            .max_by(|(ia, a), (ib, b)| {
                                a.total_cmp(b)
                                    .then_with(|| suffix.order[*ib].cmp(&suffix.order[*ia]))
                            })
                            .map(|(i, _)| i)
                            .context("empty candidate logits")?;
                        let round = |x: f32| (x * 1_000_000.0).round() / 1_000_000.0;
                        let options = &suffix.question.options;
                        let answer = Answer {
                            answer: options[suffix.order[best]].clone(),
                            confidence: round(probs[best]),
                            probs: suffix
                                .order
                                .into_iter()
                                .zip(probs)
                                .map(|(index, p)| (options[index].clone(), round(p)))
                                .collect(),
                        };
                        Ok((suffix.question.id, answer))
                    })
                    .collect()
            })
            .collect()
    }

    fn embed(&self, values: Vec<i32>) -> Tensor<3> {
        let length = values.len();
        self.decoder
            .embeddings(ids(values, &self.device).reshape([1, length]))
    }
}

pub(crate) fn ms(duration: std::time::Duration) -> f64 {
    (duration.as_secs_f64() * 100_000.0).round() / 100.0
}

fn audio_dtype(dtype: FloatDType) -> FloatDType {
    // The Metal BF16 audio path can produce different embeddings for identical clips.
    // Keep audio in FP32 and cast once at the boundary with the BF16 decoder.
    if dtype == FloatDType::BF16 {
        FloatDType::F32
    } else {
        dtype
    }
}
