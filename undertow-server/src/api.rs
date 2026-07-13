//! OpenAI-shape request handlers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use tokio_stream::StreamExt;
use undertow_core::sample::SamplerConfig;
use undertow_tokenizer::ChatMessage;

use crate::state::{
    acquire_slot, chat_prompt_ids, generate_blocking, release_slot, render_metrics, Finish,
    GenerationRequest, ServerState,
};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn request_id(prefix: &str) -> String {
    let n = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{prefix}-{t:x}{n:04x}")
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": { "message": message.into(), "type": "invalid_request_error" }
        })),
    )
        .into_response()
}

fn busy_response() -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        [("retry-after", "5")],
        Json(serde_json::json!({
            "error": { "message": "generation queue is full, retry later", "type": "overloaded_error" }
        })),
    )
        .into_response()
}

/// OpenAI `stop` field: string or array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopField::One(s) => vec![s],
            StopField::Many(v) => v,
        }
    }
}

fn default_max_tokens() -> usize {
    256
}

#[derive(Debug, Deserialize)]
pub(crate) struct SamplingFields {
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
}

impl SamplingFields {
    fn config(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: 0,
            seed: self.seed.unwrap_or(42),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatRequest {
    #[allow(dead_code)]
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessageIn>,
    #[serde(default = "default_max_tokens", alias = "max_completion_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<StopField>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessageIn {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CompletionRequest {
    #[allow(dead_code)]
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<StopField>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn metrics(State(state): State<Arc<ServerState>>) -> Response {
    (
        [("content-type", "text/plain; version=0.0.4")],
        render_metrics(&state),
    )
        .into_response()
}

pub async fn models(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": now_unix(),
            "owned_by": "undertow",
        }]
    }))
}

/// Clamp an over-long prompt to what the context can hold alongside the
/// requested completion, keeping the tail (OpenAI-style behavior).
fn fit_prompt(state: &ServerState, mut ids: Vec<usize>, max_new: usize) -> Vec<usize> {
    let ctx = state.model.max_context();
    let budget = ctx.saturating_sub(max_new.min(ctx / 2)).max(1);
    if ids.len() > budget {
        ids.drain(..ids.len() - budget);
    }
    ids
}

pub async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let prompt_ids = match state.tokenizer.encode(&req.prompt, true) {
        Ok(ids) if !ids.is_empty() => fit_prompt(&state, ids, req.max_tokens),
        Ok(_) => return error_response(StatusCode::BAD_REQUEST, "empty prompt"),
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let gen = GenerationRequest {
        prompt_ids,
        max_new: req.max_tokens,
        sampling: req.sampling.config(),
        stop_strings: req.stop.map(StopField::into_vec).unwrap_or_default(),
    };
    if req.stream {
        stream_response(state, gen, "text_completion", request_id("cmpl")).await
    } else {
        unary_response(state, gen, false, request_id("cmpl")).await
    }
}

pub async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    if req.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage::new(m.role.clone(), m.content.clone()))
        .collect();
    let prompt_ids = match chat_prompt_ids(&state, &messages) {
        Ok(ids) if !ids.is_empty() => fit_prompt(&state, ids, req.max_tokens),
        Ok(_) => return error_response(StatusCode::BAD_REQUEST, "empty rendered prompt"),
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let gen = GenerationRequest {
        prompt_ids,
        max_new: req.max_tokens,
        sampling: req.sampling.config(),
        stop_strings: req.stop.map(StopField::into_vec).unwrap_or_default(),
    };
    if req.stream {
        stream_response(state, gen, "chat.completion.chunk", request_id("chatcmpl")).await
    } else {
        unary_response(state, gen, true, request_id("chatcmpl")).await
    }
}

async fn unary_response(
    state: Arc<ServerState>,
    gen: GenerationRequest,
    chat: bool,
    id: String,
) -> Response {
    let Some(permit) = acquire_slot(&state).await else {
        return busy_response();
    };
    let deadline = Instant::now() + state.config.request_timeout;
    let prompt_tokens = gen.prompt_ids.len();
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut text = String::new();
        let out = generate_blocking(&state2, &gen, deadline, |t| {
            text.push_str(t);
            true
        });
        out.map(|(n, finish)| (text, n, finish))
    })
    .await;
    release_slot(&state);
    drop(permit);
    let (text, completion_tokens, finish) = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    if finish == Finish::TimedOut {
        tracing::warn!(id, completion_tokens, "generation hit the request timeout");
    }
    let usage = serde_json::json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    });
    let body = if chat {
        serde_json::json!({
            "id": id,
            "object": "chat.completion",
            "created": now_unix(),
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": finish.as_openai(),
            }],
            "usage": usage,
        })
    } else {
        serde_json::json!({
            "id": id,
            "object": "text_completion",
            "created": now_unix(),
            "model": state.model_name,
            "choices": [{ "index": 0, "text": text, "finish_reason": finish.as_openai() }],
            "usage": usage,
        })
    };
    Json(body).into_response()
}

async fn stream_response(
    state: Arc<ServerState>,
    gen: GenerationRequest,
    object: &'static str,
    id: String,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let model_name = state.model_name.clone();
    let chat = object == "chat.completion.chunk";
    let id2 = id.clone();
    let state2 = state.clone();

    tokio::spawn(async move {
        let Some(permit) = acquire_slot(&state2).await else {
            // Queue full: a one-chunk stream carrying the error, then done.
            let _ = tx.send(Event::default().data(
                serde_json::json!({
                    "error": { "message": "generation queue is full, retry later", "type": "overloaded_error" }
                })
                .to_string(),
            ));
            let _ = tx.send(Event::default().data("[DONE]"));
            return;
        };
        let deadline = Instant::now() + state2.config.request_timeout;
        let state3 = state2.clone();
        let tx2 = tx.clone();
        let chunk = move |text: &str, finish: Option<&str>| -> Event {
            let delta = if chat {
                serde_json::json!({ "content": text })
            } else {
                serde_json::Value::String(text.to_string())
            };
            let choice = if chat {
                serde_json::json!({ "index": 0, "delta": if finish.is_some() && text.is_empty() {
                    serde_json::json!({})
                } else { delta }, "finish_reason": finish })
            } else {
                serde_json::json!({ "index": 0, "text": text, "finish_reason": finish })
            };
            Event::default().data(
                serde_json::json!({
                    "id": id2,
                    "object": object,
                    "created": now_unix(),
                    "model": model_name,
                    "choices": [choice],
                })
                .to_string(),
            )
        };
        let chunk2 = chunk.clone();
        let result = tokio::task::spawn_blocking(move || {
            // A failed send means the client hung up: stop generating.
            generate_blocking(&state3, &gen, deadline, |t| {
                tx2.send(chunk2(t, None)).is_ok()
            })
        })
        .await;
        release_slot(&state2);
        drop(permit);
        let finish = match result {
            Ok(Ok((_, finish))) => finish,
            _ => Finish::Stop,
        };
        let _ = tx.send(chunk("", Some(finish.as_openai())));
        let _ = tx.send(Event::default().data("[DONE]"));
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(Ok::<_, std::convert::Infallible>);
    Sse::new(stream).into_response()
}
