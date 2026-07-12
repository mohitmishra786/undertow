//! OpenAI-compatible HTTP surface over any [`engine_core::Model`].
//!
//! Endpoints:
//! * `GET /health` — liveness.
//! * `GET /v1/models` — the one loaded model.
//! * `POST /v1/completions` — raw text completion.
//! * `POST /v1/chat/completions` — chat, with `"stream": true` served as
//!   SSE `chat.completion.chunk` events terminated by `data: [DONE]`.
//!
//! Concurrency model: requests are accepted concurrently, but generation
//! itself runs single-flight behind a semaphore — one CPU-saturating
//! forward pass at a time beats N passes thrashing each other's expert
//! cache. Queued requests wait their turn; the queue depth is bounded by
//! the HTTP server's own connection limits.
//!
//! Stop strings are honored with holdback: the longest possible stop
//! string is withheld from the stream until it either matches (dropped)
//! or is cleared (flushed), so clients never see a partial stop marker.

mod api;
mod state;

pub use state::ServerState;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::{get, post};

pub fn router(state: Arc<ServerState>) -> axum::Router {
    axum::Router::new()
        .route("/health", get(api::health))
        .route("/v1/models", get(api::models))
        .route("/v1/completions", post(api::completions))
        .route("/v1/chat/completions", post(api::chat_completions))
        .with_state(state)
}

/// Serve until the process dies. Builds its own runtime so callers (the
/// CLI) stay synchronous.
pub fn run_blocking(state: ServerState, addr: SocketAddr) -> std::io::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("listening on http://{}", listener.local_addr()?);
        axum::serve(listener, router(Arc::new(state))).await
    })
}
