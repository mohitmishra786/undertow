//! OpenAI-compatible HTTP surface over any [`undertow_core::Model`].
//!
//! Endpoints:
//! * `GET /health` — liveness (never requires auth).
//! * `GET /metrics` — Prometheus text (auth-protected when a key is set).
//! * `GET /v1/models` — the one loaded model.
//! * `POST /v1/completions` — raw text completion.
//! * `POST /v1/chat/completions` — chat, with `"stream": true` served as
//!   SSE `chat.completion.chunk` events terminated by `data: [DONE]`.
//!
//! Operational behavior:
//! * Generation is single-flight behind a semaphore; a bounded queue of
//!   waiters gets 429 with `retry-after` beyond `max_queue`.
//! * Every generation carries a deadline, checked per token; a client
//!   disconnect on a stream aborts the forward pass at the next token.
//! * Optional bearer-token auth, optional CORS origin, structured request
//!   logs (method, path, status, latency), graceful shutdown on ctrl-c.

mod api;
mod state;

pub use state::{ServerConfig, ServerState};

use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};

async fn observe(
    State(state): State<Arc<ServerState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let start = std::time::Instant::now();
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

    // Auth: everything except liveness needs the bearer token when set.
    if let Some(key) = &state.config.api_key {
        if path != "/health" {
            let ok = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .is_some_and(|token| token == key);
            if !ok {
                state
                    .metrics
                    .responses_4xx_total
                    .fetch_add(1, Ordering::Relaxed);
                return (
                    StatusCode::UNAUTHORIZED,
                    axum::Json(serde_json::json!({
                        "error": { "message": "missing or invalid bearer token", "type": "authentication_error" }
                    })),
                )
                    .into_response();
            }
        }
    }

    let mut resp = next.run(req).await;

    if let Some(origin) = &state.config.cors_origin {
        if let Ok(v) = origin.parse() {
            resp.headers_mut().insert("access-control-allow-origin", v);
        }
    }

    let status = resp.status();
    if status.is_client_error() {
        state
            .metrics
            .responses_4xx_total
            .fetch_add(1, Ordering::Relaxed);
    } else if status.is_server_error() {
        state
            .metrics
            .responses_5xx_total
            .fetch_add(1, Ordering::Relaxed);
    }
    tracing::info!(
        %method,
        path,
        status = status.as_u16(),
        ms = start.elapsed().as_millis() as u64,
        "request"
    );
    resp
}

/// CORS preflight for the configured origin.
async fn preflight(State(state): State<Arc<ServerState>>) -> Response {
    let mut resp = StatusCode::NO_CONTENT.into_response();
    if let Some(origin) = &state.config.cors_origin {
        let h = resp.headers_mut();
        if let Ok(v) = origin.parse() {
            h.insert("access-control-allow-origin", v);
        }
        h.insert(
            "access-control-allow-methods",
            "GET, POST, OPTIONS".parse().expect("static"),
        );
        h.insert(
            "access-control-allow-headers",
            "authorization, content-type".parse().expect("static"),
        );
        h.insert("access-control-max-age", "86400".parse().expect("static"));
    }
    resp
}

pub fn router(state: Arc<ServerState>) -> axum::Router {
    axum::Router::new()
        .route("/health", get(api::health))
        .route("/metrics", get(api::metrics))
        .route("/v1/models", get(api::models))
        .route("/v1/completions", post(api::completions).options(preflight))
        .route(
            "/v1/chat/completions",
            post(api::chat_completions).options(preflight),
        )
        .layer(axum::middleware::from_fn_with_state(state.clone(), observe))
        .with_state(state)
}

/// Serve until ctrl-c, then stop accepting and let in-flight requests
/// finish (each is already bounded by the request timeout). Builds its
/// own runtime so callers (the CLI) stay synchronous.
pub fn run_blocking(state: ServerState, addr: SocketAddr) -> std::io::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("listening on http://{}", listener.local_addr()?);
        axum::serve(listener, router(Arc::new(state)))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                tracing::info!("shutting down");
            })
            .await
    })
}
