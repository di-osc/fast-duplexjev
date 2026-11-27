use std::{
    collections::BTreeMap,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{mpsc, oneshot};

use crate::{
    audio::Audio,
    model::{BatchLimits, Item, Model, PassStats, ms},
    question::{Answers, ClipId, Language, Question, Target},
};

#[derive(Clone)]
struct AppState {
    sender: mpsc::Sender<Job>,
    ticks: Arc<AtomicU64>,
    tick_ms: f64,
}

struct Job {
    items: Vec<Item>,
    queued: Instant,
    reply: oneshot::Sender<Result<Envelope<Vec<Answers>>, String>>,
}

#[derive(Serialize)]
struct Envelope<A> {
    answers: A,
    tick: u64,
    requests_in_tick: usize,
    clips_in_pass: usize,
    wait_ms: f64,
    pass_ms: f64,
    pass: PassStats,
}

impl<A> Envelope<A> {
    fn map<B>(self, f: impl FnOnce(A) -> B) -> Envelope<B> {
        Envelope {
            answers: f(self.answers),
            tick: self.tick,
            requests_in_tick: self.requests_in_tick,
            clips_in_pass: self.clips_in_pass,
            wait_ms: self.wait_ms,
            pass_ms: self.pass_ms,
            pass: self.pass,
        }
    }
}

#[derive(Deserialize)]
struct DecideRequest {
    audio_b64: String,
    questions: Vec<Question>,
    context: Option<String>,
    lang: Option<Language>,
}

#[derive(Deserialize)]
struct BatchRequest {
    audios: BTreeMap<String, String>,
    questions: Vec<Question>,
    context: Option<BTreeMap<String, String>>,
    lang: Option<Language>,
}

pub async fn serve(
    address: SocketAddr,
    model: Model,
    tick_ms: f64,
    limits: BatchLimits,
    max_body_bytes: usize,
) -> Result<()> {
    let tick = Duration::try_from_secs_f64(tick_ms / 1000.0).context("invalid tick interval")?;
    ensure!(!tick.is_zero(), "tick interval must be positive");
    let listener = tokio::net::TcpListener::bind(address).await?;
    let app = app(model, tick, tick_ms, limits, max_body_bytes);
    eprintln!("DuplexJev listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

fn app(
    model: Model,
    tick: Duration,
    tick_ms: f64,
    limits: BatchLimits,
    max_body_bytes: usize,
) -> Router {
    let (sender, receiver) = mpsc::channel(128);
    let ticks = Arc::new(AtomicU64::new(0));
    tokio::spawn(worker(
        Arc::new(model),
        receiver,
        ticks.clone(),
        tick,
        limits,
    ));
    Router::new()
        .route("/health", get(health))
        .route("/v1/decide", post(decide))
        .route("/v1/decide_batch", post(decide_batch))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(AppState {
            sender,
            ticks,
            tick_ms,
        })
}

async fn worker(
    model: Arc<Model>,
    mut receiver: mpsc::Receiver<Job>,
    ticks: Arc<AtomicU64>,
    tick: Duration,
    limits: BatchLimits,
) {
    let mut interval = tokio::time::interval_at(tokio::time::Instant::now() + tick, tick);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        let mut replies = Vec::new();
        let mut items = Vec::new();
        for _ in 0..receiver.len() {
            let Ok(mut job) = receiver.try_recv() else {
                break;
            };
            if !job.reply.is_closed() {
                replies.push((job.reply, job.items.len(), job.queued));
                items.append(&mut job.items);
            }
        }
        if replies.is_empty() {
            if receiver.is_closed() {
                return;
            }
            continue;
        }
        let tick = ticks.fetch_add(1, Ordering::Relaxed) + 1;
        let now = Instant::now();
        let requests_in_tick = replies.len();
        let clips_in_pass = items.len();
        let model = model.clone();
        let result = tokio::task::spawn_blocking(move || model.decide_items(items, limits)).await;
        let pass_ms = ms(now.elapsed());
        match result {
            Ok(Ok((answers, pass))) => {
                let mut answers = answers.into_iter();
                for (reply, count, queued) in replies {
                    let _ = reply.send(Ok(Envelope {
                        answers: answers.by_ref().take(count).collect(),
                        tick,
                        requests_in_tick,
                        clips_in_pass,
                        wait_ms: ms(now.saturating_duration_since(queued)),
                        pass_ms,
                        pass: pass.clone(),
                    }));
                }
            }
            failed => {
                let error = match failed {
                    Ok(Err(error)) => error.to_string(),
                    Err(error) => format!("inference worker: {error}"),
                    Ok(Ok(_)) => unreachable!(),
                };
                for (reply, _, _) in replies {
                    let _ = reply.send(Err(error.clone()));
                }
            }
        }
    }
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(
        json!({"ok": true, "tick_ms": state.tick_ms, "ticks": state.ticks.load(Ordering::Relaxed)}),
    )
}

async fn decide(
    State(state): State<AppState>,
    Json(request): Json<DecideRequest>,
) -> Result<Json<Envelope<Answers>>, ApiError> {
    let audio = decode(request.audio_b64).await?;
    let item = Item {
        audio,
        questions: request.questions,
        context: request.context,
        lang: request.lang,
    };
    item.validate().map_err(ApiError::bad_request)?;
    let result = submit(state, vec![item]).await?;
    Ok(Json(result.map(|mut answers| {
        answers.pop().expect("single clip answer")
    })))
}

async fn decide_batch(
    State(state): State<AppState>,
    Json(request): Json<BatchRequest>,
) -> Result<Json<Envelope<BTreeMap<String, Answers>>>, ApiError> {
    let BatchRequest {
        audios,
        questions,
        context,
        lang,
    } = request;
    let assigned = assign_questions(audios.keys(), questions).map_err(ApiError::bad_request)?;
    let (ids, items) = tokio::task::spawn_blocking(move || -> Result<_> {
        let mut ids = Vec::new();
        let mut items = Vec::new();
        for (id, questions) in assigned {
            let audio = Audio::decode(STANDARD.decode(&audios[&id])?)?;
            let context = context
                .as_ref()
                .and_then(|context| context.get(&id))
                .cloned();
            let item = Item {
                audio,
                questions,
                context,
                lang,
            };
            item.validate()?;
            ids.push(id);
            items.push(item);
        }
        Ok((ids, items))
    })
    .await
    .map_err(ApiError::internal)?
    .map_err(ApiError::bad_request)?;
    let result = submit(state, items).await?;
    Ok(Json(
        result.map(|answers| ids.into_iter().zip(answers).collect()),
    ))
}

pub fn assign_questions<'a>(
    ids: impl Iterator<Item = &'a String>,
    questions: Vec<Question>,
) -> Result<BTreeMap<String, Vec<Question>>> {
    let mut clips: BTreeMap<_, Vec<_>> = ids.map(|id| (id.clone(), Vec::new())).collect();
    for question in questions {
        question.validate()?;
        let Some(target) = &question.audio else {
            bail!(
                "question {:?}: set audio to a clip id, a list of ids, or '*'",
                question.id
            );
        };
        let targets = match target {
            Target::One(ClipId::Name(name)) if name == "*" => clips.keys().cloned().collect(),
            Target::One(id) => vec![id.to_string()],
            Target::Many(ids) => ids.iter().map(ToString::to_string).collect(),
        };
        for id in targets {
            let questions = clips.get_mut(&id).with_context(|| {
                format!("question {:?} refers to unknown audio {id:?}", question.id)
            })?;
            ensure!(
                !questions.iter().any(|q: &Question| q.id == question.id),
                "duplicate question id {:?} for audio {id:?}",
                question.id
            );
            questions.push(question.clone());
        }
    }
    clips.retain(|_, questions| !questions.is_empty());
    Ok(clips)
}

async fn decode(encoded: String) -> Result<Audio, ApiError> {
    tokio::task::spawn_blocking(move || -> Result<_> { Audio::decode(STANDARD.decode(encoded)?) })
        .await
        .map_err(ApiError::internal)?
        .map_err(ApiError::bad_request)
}

async fn submit(state: AppState, items: Vec<Item>) -> Result<Envelope<Vec<Answers>>, ApiError> {
    let (reply, result) = oneshot::channel();
    state
        .sender
        .try_send(Job {
            items,
            queued: Instant::now(),
            reply,
        })
        .map_err(|_| {
            ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "inference queue unavailable".into(),
            )
        })?;
    result
        .await
        .map_err(ApiError::internal)?
        .map_err(ApiError::internal)
}

struct ApiError(StatusCode, String);

impl ApiError {
    fn bad_request(error: impl ToString) -> Self {
        Self(StatusCode::BAD_REQUEST, error.to_string())
    }

    fn internal(error: impl ToString) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"detail": self.1}))).into_response()
    }
}
