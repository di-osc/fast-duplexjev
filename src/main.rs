mod checkpoint;

use std::{
    collections::BTreeMap,
    fs::{self, File},
    net::{IpAddr, SocketAddr},
    path::PathBuf,
};

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use duplexjev::{
    audio::Audio,
    backend::{Backend, Precision},
    model::{BatchLimits, Item, Model},
    question::{ClipId, Language, Question, Target},
    server,
};
use serde_json::json;

use checkpoint::MODEL_ID;

#[derive(Parser)]
#[command(version, about = "DuplexJev 4B speech decisions with Burn")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the upstream-compatible HTTP decision server.
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        /// Address to bind. Use 0.0.0.0 to listen on all IPv4 interfaces.
        #[arg(long, default_value = "127.0.0.1")]
        host: IpAddr,
        /// TCP port. Use 0 to select a free port.
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// Period in milliseconds for collecting concurrent requests.
        #[arg(long, default_value_t = 160.0)]
        tick_ms: f64,
        /// Maximum clips per neural batch. Extra clips run in later batches in the same tick.
        #[arg(long, default_value_t = 4, value_parser = positive)]
        max_items: usize,
        /// Token budget for grouping clips. A clip larger than this budget runs alone.
        #[arg(long, default_value_t = 2048, value_parser = positive)]
        max_tokens: usize,
        /// Maximum HTTP body size in bytes, including Base64 and JSON.
        #[arg(long, default_value_t = 16 * 1024 * 1024, value_parser = positive)]
        max_body_bytes: usize,
    },
    /// Answer one or more questions about a WAV or FLAC file.
    Decide {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        questions: QuestionArgs,
        /// Optional context, such as the previous assistant turn.
        #[arg(long)]
        context: Option<String>,
        audio: PathBuf,
    },
    /// Answer questions about several WAV or FLAC files.
    Batch {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        questions: QuestionArgs,
        /// Audio clip as id=path. Repeat for each clip.
        #[arg(long = "audio", required = true)]
        audios: Vec<String>,
    },
}

#[derive(Args)]
struct ModelArgs {
    /// Local checkpoint directory or Hugging Face model ID. The default ID is pinned to 4B-Para v1.4.
    #[arg(long, default_value = MODEL_ID)]
    model: String,
    /// Hugging Face revision override. Applies to model IDs.
    #[arg(long)]
    revision: Option<String>,
    /// Directory for downloaded model files. Defaults to the modelhub cache.
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,
    /// Inference backend. Defaults to Metal on macOS builds with Metal enabled, otherwise Flex.
    #[arg(long, visible_alias = "backend", default_value = default_backend())]
    device: Backend,
    /// GPU index for CUDA. Flex and Metal use device 0.
    #[arg(long, default_value_t = 0)]
    device_index: usize,
    /// Qwen3 precision. Auto uses BF16 where supported and FP32 otherwise. BF16 keeps the audio tower in FP32.
    #[arg(long, default_value = "auto")]
    precision: Precision,
}

const fn default_backend() -> &'static str {
    if cfg!(all(feature = "metal", target_vendor = "apple")) {
        "metal"
    } else if cfg!(feature = "flex") {
        "flex"
    } else {
        "cuda"
    }
}

#[derive(Args)]
struct QuestionArgs {
    /// JSON array of {id,text,options,lang[,audio]} question objects.
    #[arg(long)]
    questions: Option<PathBuf>,
    /// Inline question, repeatable. Format id|text|option1,option2 or clip|id|text|option1,option2 for batch.
    #[arg(long = "q")]
    inline: Vec<String>,
    /// Language of inline questions and the audio prompt.
    #[arg(long)]
    lang: Option<Language>,
}

fn positive(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&x| x > 0)
        .ok_or_else(|| "expected a positive integer".into())
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Serve {
            model,
            host,
            port,
            tick_ms,
            max_items,
            max_tokens,
            max_body_bytes,
        } => {
            ensure!(
                tick_ms.is_finite() && tick_ms > 0.0,
                "--tick-ms must be positive and finite"
            );
            server::serve(
                SocketAddr::new(host, port),
                model.load().await?,
                tick_ms,
                BatchLimits {
                    max_items,
                    max_tokens,
                },
                max_body_bytes,
            )
            .await?;
        }
        Command::Decide {
            model,
            questions,
            context,
            audio,
        } => {
            let item = Item {
                audio: Audio::decode(fs::read(&audio)?)?,
                questions: questions.read(false)?,
                context,
                lang: questions.lang,
            };
            item.validate()?;
            let (mut answers, pass) = model
                .load()
                .await?
                .decide_items(vec![item], BatchLimits::default())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "audio": audio, "answers": answers.pop().context("no answer")?, "pass": pass,
                }))?
            );
        }
        Command::Batch {
            model,
            questions,
            audios,
        } => {
            let mut files = BTreeMap::new();
            for entry in audios {
                let (id, path) = entry.split_once('=').context("use --audio id=path")?;
                ensure!(!id.is_empty() && !path.is_empty(), "use --audio id=path");
                ensure!(
                    files.insert(id.to_owned(), PathBuf::from(path)).is_none(),
                    "duplicate clip {id:?}"
                );
            }
            let assigned = server::assign_questions(files.keys(), questions.read(true)?)?;
            let mut clip_ids = Vec::new();
            let mut items = Vec::new();
            for (id, groups) in assigned {
                items.push(Item {
                    audio: Audio::decode(fs::read(&files[&id])?)?,
                    questions: groups,
                    context: None,
                    lang: questions.lang,
                });
                clip_ids.push(id);
            }
            let (answers, pass) = model
                .load()
                .await?
                .decide_items(items, BatchLimits::default())?;
            let answers: BTreeMap<_, _> = clip_ids.into_iter().zip(answers).collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({"answers": answers, "pass": pass}))?
            );
        }
    }
    Ok(())
}

impl QuestionArgs {
    fn read(&self, batch: bool) -> Result<Vec<Question>> {
        let mut questions: Vec<Question> = match &self.questions {
            Some(path) => serde_json::from_reader(File::open(path)?)?,
            None => Vec::new(),
        };
        for spec in &self.inline {
            let mut fields = spec.split('|');
            let audio = if batch {
                Some(Target::One(ClipId::Name(
                    fields.next().context("missing clip id")?.to_owned(),
                )))
            } else {
                None
            };
            let id = fields.next().context("missing question id")?.to_owned();
            let text = fields.next().context("missing question text")?.to_owned();
            let options = fields
                .next()
                .context("missing comma-separated options")?
                .split(',')
                .map(|option| option.trim().to_owned())
                .collect();
            ensure!(fields.next().is_none(), "too many fields in --q");
            questions.push(Question {
                id,
                text,
                options,
                lang: self.lang.unwrap_or_default(),
                audio,
            });
        }
        ensure!(
            !questions.is_empty(),
            "give --questions or at least one --q"
        );
        Ok(questions)
    }
}

impl ModelArgs {
    async fn load(&self) -> Result<Model> {
        let device = self.device.device(self.device_index)?;
        let dtype = self.precision.resolve(self.device, &device)?;
        let path = checkpoint::resolve(
            &self.model,
            self.revision.as_deref(),
            self.cache_dir.as_deref(),
        )
        .await?;
        eprintln!(
            "Loading {} on {:?} ({dtype:?})",
            path.display(),
            self.device
        );
        Model::load(&path, device, dtype)
    }
}
