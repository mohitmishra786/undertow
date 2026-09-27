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
    acquire_slot, chat_prompt_ids, chat_prompt_ids_with_tools, generate_blocking, release_slot,
    render_metrics, Finish, GenerationRequest, ServerState,
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

fn iso8601_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let mut days = secs / 86400;
    let mut year = 1970;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1;
    for &d in &month_days {
        if days < d {
            break;
        }
        days -= d;
        month += 1;
    }
    let day = days + 1;
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

fn value_to_tool_call(v: &serde_json::Value, counter: &mut usize) -> Option<ToolCall> {
    let name = v
        .get("name")
        .and_then(|n| n.as_str())
        .or_else(|| {
            v.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
        })?
        .to_string();
    let arguments = if let Some(args) = v.get("arguments").or_else(|| v.get("parameters")) {
        match args {
            serde_json::Value::Null => "{}".to_string(),
            serde_json::Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    "{}".to_string()
                } else if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    if parsed.is_object() {
                        trimmed.to_string()
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            serde_json::Value::Object(_) => {
                serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
            }
            _ => return None,
        }
    } else {
        "{}".to_string()
    };
    *counter += 1;
    Some(ToolCall {
        id: format!("call_{}{}", request_id("tool"), counter),
        call_type: "function".to_string(),
        function: ToolCallFunction { name, arguments },
    })
}

fn parse_tool_calls_from_str(raw: &str, counter: &mut usize) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) else {
        return out;
    };
    if let Some(arr) = v.as_array() {
        for item in arr {
            if let Some(call) = value_to_tool_call(item, counter) {
                out.push(call);
            }
        }
    } else if let Some(call) = value_to_tool_call(&v, counter) {
        out.push(call);
    }
    out
}

pub(crate) fn extract_tool_calls(text: &str) -> (Option<String>, Option<Vec<ToolCall>>) {
    let mut tool_calls = Vec::new();
    let mut content = text.to_string();
    let mut counter = 0;

    // XML-style <tool_call> ... </tool_call>
    while let Some(start) = content.find("<tool_call>") {
        let after_tag = start + "<tool_call>".len();
        let (call_str, end) = match content[after_tag..].find("</tool_call>") {
            Some(end_rel) => (
                content[after_tag..after_tag + end_rel].to_string(),
                after_tag + end_rel + "</tool_call>".len(),
            ),
            None => (content[after_tag..].to_string(), content.len()),
        };
        let calls = parse_tool_calls_from_str(call_str.trim(), &mut counter);
        tool_calls.extend(calls);
        content.drain(start..end);
    }

    // Markdown ```tool_call ... ``` block
    if tool_calls.is_empty() {
        for tag in ["```tool_call", "```tool-call"] {
            if let Some(start) = content.find(tag) {
                let after_tag = start + tag.len();
                if let Some(end_rel) = content[after_tag..].find("```") {
                    let call_str = content[after_tag..after_tag + end_rel].to_string();
                    let end = after_tag + end_rel + 3;
                    let calls = parse_tool_calls_from_str(call_str.trim(), &mut counter);
                    if !calls.is_empty() {
                        tool_calls.extend(calls);
                        content.drain(start..end);
                        break;
                    }
                }
            }
        }
    }

    // Raw JSON: {"name": "...", ...}
    if tool_calls.is_empty() {
        let trimmed = content.trim();
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            let calls = parse_tool_calls_from_str(trimmed, &mut counter);
            if !calls.is_empty() {
                tool_calls.extend(calls);
                content.clear();
            }
        }
    }

    let clean = content.trim();
    let clean_content = if clean.is_empty() {
        if tool_calls.is_empty() {
            Some(text.to_string())
        } else {
            None
        }
    } else {
        Some(clean.to_string())
    };

    let calls = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };
    (clean_content, calls)
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
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessageIn>,
    #[serde(default = "default_max_tokens", alias = "max_completion_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<StopField>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub(crate) tool_choice: Option<serde_json::Value>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessageIn {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    tool_calls: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessageIn>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    options: Option<OllamaOptions>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct OllamaOptions {
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    num_predict: Option<i64>,
    #[serde(default)]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CompletionRequest {
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
    if let Some(ref req_model) = req.model {
        if req_model != &state.model_name {
            tracing::warn!(
                requested = %req_model,
                loaded = %state.model_name,
                "model mismatch in completion request"
            );
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "model '{req_model}' does not match loaded model '{}'",
                    state.model_name
                ),
            );
        }
    }
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
        stream_response(state, gen, "text_completion", request_id("cmpl"), false).await
    } else {
        unary_response(state, gen, false, request_id("cmpl"), false).await
    }
}

pub async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    if let Some(ref req_model) = req.model {
        if req_model != &state.model_name {
            tracing::warn!(
                requested = %req_model,
                loaded = %state.model_name,
                "model mismatch in chat completion request"
            );
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "model '{req_model}' does not match loaded model '{}'",
                    state.model_name
                ),
            );
        }
    }
    if req.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let messages: Vec<ChatMessage> = req
        .messages
        .into_iter()
        .map(|m| ChatMessage {
            role: m.role,
            content: m.content.unwrap_or_default(),
            name: m.name,
            tool_call_id: m.tool_call_id,
            tool_calls: m.tool_calls,
        })
        .collect();
    let tools_enabled = match &req.tool_choice {
        Some(serde_json::Value::String(s)) if s == "none" => false,
        _ => req.tools.as_ref().is_some_and(|t| !t.is_empty()),
    };
    let tools = if tools_enabled {
        req.tools.as_deref()
    } else {
        None
    };
    let prompt_ids = match match tools {
        Some(t) => chat_prompt_ids_with_tools(&state, &messages, Some(t)),
        None => chat_prompt_ids(&state, &messages),
    } {
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
        stream_response(
            state,
            gen,
            "chat.completion.chunk",
            request_id("chatcmpl"),
            tools_enabled,
        )
        .await
    } else {
        unary_response(state, gen, true, request_id("chatcmpl"), tools_enabled).await
    }
}

async fn unary_response(
    state: Arc<ServerState>,
    gen: GenerationRequest,
    chat: bool,
    id: String,
    tools_enabled: bool,
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
        let (content, tool_calls) = if tools_enabled {
            extract_tool_calls(&text)
        } else {
            (Some(text), None)
        };
        let finish_reason = if tool_calls.is_some() {
            "tool_calls"
        } else {
            finish.as_openai()
        };
        let mut message = serde_json::json!({
            "role": "assistant",
            "content": content,
        });
        if let Some(calls) = tool_calls {
            message["tool_calls"] = serde_json::json!(calls);
        }
        serde_json::json!({
            "id": id,
            "object": "chat.completion",
            "created": now_unix(),
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
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
    tools_enabled: bool,
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
            let mut full_text = String::new();
            let res = generate_blocking(&state3, &gen, deadline, |t| {
                if tools_enabled {
                    full_text.push_str(t);
                    true
                } else {
                    tx2.send(chunk2(t, None)).is_ok()
                }
            });
            res.map(|(n, finish)| (full_text, n, finish))
        })
        .await;
        release_slot(&state2);
        drop(permit);
        let (full_text, finish) = match result {
            Ok(Ok((txt, _, finish))) => (txt, finish),
            _ => (String::new(), Finish::Stop),
        };
        if tools_enabled {
            let (content, tool_calls) = extract_tool_calls(&full_text);
            if let Some(calls) = tool_calls {
                let mut delta = serde_json::json!({ "role": "assistant" });
                if let Some(c) = content {
                    delta["content"] = serde_json::json!(c);
                }
                delta["tool_calls"] = serde_json::json!(calls);
                let choice = serde_json::json!({
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "tool_calls",
                });
                let _ = tx.send(
                    Event::default().data(
                        serde_json::json!({
                            "id": id,
                            "object": object,
                            "created": now_unix(),
                            "model": state.model_name,
                            "choices": [choice],
                        })
                        .to_string(),
                    ),
                );
            } else {
                let choice = serde_json::json!({
                    "index": 0,
                    "delta": { "role": "assistant", "content": full_text },
                    "finish_reason": finish.as_openai(),
                });
                let _ = tx.send(
                    Event::default().data(
                        serde_json::json!({
                            "id": id,
                            "object": object,
                            "created": now_unix(),
                            "model": state.model_name,
                            "choices": [choice],
                        })
                        .to_string(),
                    ),
                );
            }
        } else {
            let _ = tx.send(chunk("", Some(finish.as_openai())));
        }
        let _ = tx.send(Event::default().data("[DONE]"));
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(Ok::<_, std::convert::Infallible>);
    Sse::new(stream).into_response()
}

fn to_ollama_tool_calls(calls: &[ToolCall]) -> Vec<serde_json::Value> {
    calls
        .iter()
        .map(|c| {
            let args_val: serde_json::Value =
                serde_json::from_str(&c.function.arguments).unwrap_or(serde_json::json!({}));
            serde_json::json!({
                "function": {
                    "name": c.function.name,
                    "arguments": args_val,
                }
            })
        })
        .collect()
}

pub async fn ollama_tags(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    let now = iso8601_now();
    let name = &state.model_name;
    let mut names = vec![name.clone()];
    if !name.contains(':') {
        names.insert(0, format!("{name}:latest"));
    }
    let models = names
        .into_iter()
        .map(|n| {
            serde_json::json!({
                "name": n,
                "model": n,
                "modified_at": now,
                "size": 0,
                "digest": "undertow",
                "details": {
                    "parent_model": "",
                    "format": "undertow",
                    "family": "undertow",
                    "families": ["undertow"],
                    "parameter_size": "unknown",
                    "quantization_level": "int8"
                }
            })
        })
        .collect::<Vec<_>>();
    Json(serde_json::json!({ "models": models }))
}

pub async fn ollama_chat(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<OllamaChatRequest>,
) -> Response {
    if let Some(ref req_model) = req.model {
        let req_base = req_model.trim_end_matches(":latest");
        let state_base = state.model_name.trim_end_matches(":latest");
        if req_base != state_base && req_model != &state.model_name {
            tracing::warn!(
                requested = %req_model,
                loaded = %state.model_name,
                "model mismatch in Ollama chat request"
            );
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "model '{req_model}' does not match loaded model '{}'",
                    state.model_name
                ),
            );
        }
    }
    if req.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let messages: Vec<ChatMessage> = req
        .messages
        .into_iter()
        .map(|m| ChatMessage {
            role: m.role,
            content: m.content.unwrap_or_default(),
            name: m.name,
            tool_call_id: m.tool_call_id,
            tool_calls: m.tool_calls,
        })
        .collect();
    let tools_enabled = req.tools.as_ref().is_some_and(|t| !t.is_empty());
    let max_tokens = match req.options.as_ref().and_then(|o| o.num_predict) {
        Some(n) if n > 0 => n as usize,
        _ => default_max_tokens(),
    };
    let sampling = SamplerConfig {
        temperature: req
            .options
            .as_ref()
            .and_then(|o| o.temperature)
            .unwrap_or(1.0),
        top_p: req.options.as_ref().and_then(|o| o.top_p).unwrap_or(1.0),
        top_k: 0,
        seed: req.options.as_ref().and_then(|o| o.seed).unwrap_or(42),
    };
    let stop_strings = req
        .options
        .as_ref()
        .and_then(|o| o.stop.clone())
        .unwrap_or_default();
    let prompt_ids = match chat_prompt_ids_with_tools(&state, &messages, req.tools.as_deref()) {
        Ok(ids) if !ids.is_empty() => fit_prompt(&state, ids, max_tokens),
        Ok(_) => return error_response(StatusCode::BAD_REQUEST, "empty rendered prompt"),
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let gen = GenerationRequest {
        prompt_ids,
        max_new: max_tokens,
        sampling,
        stop_strings,
    };
    if req.stream.unwrap_or(true) {
        ollama_stream_response(state, gen, tools_enabled).await
    } else {
        ollama_unary_response(state, gen, tools_enabled).await
    }
}

async fn ollama_unary_response(
    state: Arc<ServerState>,
    gen: GenerationRequest,
    tools_enabled: bool,
) -> Response {
    let Some(permit) = acquire_slot(&state).await else {
        return busy_response();
    };
    let deadline = Instant::now() + state.config.request_timeout;
    let prompt_tokens = gen.prompt_ids.len();
    let state2 = state.clone();
    let start = Instant::now();
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
    let duration_ns = start.elapsed().as_nanos() as u64;
    let (content, tool_calls) = if tools_enabled {
        extract_tool_calls(&text)
    } else {
        (Some(text), None)
    };
    let done_reason = if tool_calls.is_some() {
        "tool_calls"
    } else {
        finish.as_openai()
    };
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": content.unwrap_or_default(),
    });
    if let Some(calls) = tool_calls {
        message["tool_calls"] = serde_json::json!(to_ollama_tool_calls(&calls));
    }
    let body = serde_json::json!({
        "model": state.model_name,
        "created_at": iso8601_now(),
        "message": message,
        "done": true,
        "done_reason": done_reason,
        "total_duration": duration_ns,
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 0,
        "eval_count": completion_tokens,
        "eval_duration": duration_ns,
    });
    Json(body).into_response()
}

async fn ollama_stream_response(
    state: Arc<ServerState>,
    gen: GenerationRequest,
    tools_enabled: bool,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let model_name = state.model_name.clone();
    let state2 = state.clone();
    let start = Instant::now();
    let prompt_tokens = gen.prompt_ids.len();

    tokio::spawn(async move {
        let Some(permit) = acquire_slot(&state2).await else {
            let err = serde_json::json!({
                "error": "generation queue is full, retry later",
                "done": true,
            });
            let _ = tx.send(format!("{}\n", err));
            return;
        };
        let deadline = Instant::now() + state2.config.request_timeout;
        let state3 = state2.clone();
        let tx2 = tx.clone();
        let model = model_name.clone();
        let result = tokio::task::spawn_blocking(move || {
            let mut full_text = String::new();
            let out = generate_blocking(&state3, &gen, deadline, |t| {
                full_text.push_str(t);
                let chunk = serde_json::json!({
                    "model": model,
                    "created_at": iso8601_now(),
                    "message": { "role": "assistant", "content": t },
                    "done": false,
                });
                tx2.send(format!("{}\n", chunk)).is_ok()
            });
            out.map(|(n, finish)| (full_text, n, finish))
        })
        .await;
        release_slot(&state2);
        drop(permit);
        let duration_ns = start.elapsed().as_nanos() as u64;
        let (full_text, completion_tokens, finish) = match result {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                let err = serde_json::json!({
                    "error": format!("{e}"),
                    "done": true,
                });
                let _ = tx.send(format!("{}\n", err));
                return;
            }
            Err(join_err) => {
                let err = serde_json::json!({
                    "error": format!("task panicked or was canceled: {join_err}"),
                    "done": true,
                });
                let _ = tx.send(format!("{}\n", err));
                return;
            }
        };
        let (content, tool_calls) = if tools_enabled {
            extract_tool_calls(&full_text)
        } else {
            (Some(full_text), None)
        };
        let done_reason = if tool_calls.is_some() {
            "tool_calls"
        } else {
            finish.as_openai()
        };
        let mut final_msg = serde_json::json!({
            "role": "assistant",
            "content": content.unwrap_or_default(),
        });
        if let Some(calls) = tool_calls {
            final_msg["tool_calls"] = serde_json::json!(to_ollama_tool_calls(&calls));
        }
        let final_chunk = serde_json::json!({
            "model": model_name,
            "created_at": iso8601_now(),
            "message": final_msg,
            "done": true,
            "done_reason": done_reason,
            "total_duration": duration_ns,
            "load_duration": 0,
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": 0,
            "eval_count": completion_tokens,
            "eval_duration": duration_ns,
        });
        let _ = tx.send(format!("{}\n", final_chunk));
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map(|s| Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(s)));
    Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_xml_tool_call() {
        let text = "Thinking...<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}</tool_call>Done!";
        let (content, calls) = extract_tool_calls(text);
        assert_eq!(content.as_deref(), Some("Thinking...Done!"));
        let calls = calls.expect("should have extracted 1 tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"London\"}");
    }

    #[test]
    fn extract_markdown_tool_call() {
        let text = "Here is the tool call:\n```tool_call\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2 + 2\"}}\n```\nHope that helps!";
        let (content, calls) = extract_tool_calls(text);
        assert_eq!(
            content.as_deref(),
            Some("Here is the tool call:\n\nHope that helps!")
        );
        let calls = calls.expect("should have extracted 1 tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calculator");
        assert_eq!(calls[0].function.arguments, "{\"expression\":\"2 + 2\"}");
    }

    #[test]
    fn extract_raw_json_tool_call() {
        let text = "{\"name\": \"lookup\", \"arguments\": {\"id\": 42}}";
        let (content, calls) = extract_tool_calls(text);
        assert_eq!(content, None);
        let calls = calls.expect("should have extracted 1 tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(calls[0].function.arguments, "{\"id\":42}");
    }

    #[test]
    fn extract_no_tool_call() {
        let text = "Just normal conversation here.";
        let (content, calls) = extract_tool_calls(text);
        assert_eq!(content.as_deref(), Some("Just normal conversation here."));
        assert!(calls.is_none());
    }
}
