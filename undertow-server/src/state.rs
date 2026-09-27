//! Shared server state, limits, metrics, and the blocking generation
//! worker.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Semaphore;
use undertow_core::sample::{Sampler, SamplerConfig};
use undertow_core::{EngineError, Model};
use undertow_tokenizer::{ChatMessage, StreamDecoder, Tokenizer};

/// Operational knobs, all overridable from the CLI.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Require `Authorization: Bearer <key>` when set (health stays open).
    pub api_key: Option<String>,
    /// Wall-clock budget per generation; exceeded requests stop
    /// cooperatively at the next token and report a timeout.
    pub request_timeout: Duration,
    /// Requests allowed to wait for the generation slot beyond the one
    /// running; more than this get 429 immediately.
    pub max_queue: usize,
    /// `Access-Control-Allow-Origin` value; no CORS headers when unset.
    pub cors_origin: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            request_timeout: Duration::from_secs(600),
            max_queue: 32,
            cors_origin: None,
        }
    }
}

/// Cumulative counters exposed at /metrics (Prometheus text format).
#[derive(Debug, Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub responses_5xx_total: AtomicU64,
    pub responses_4xx_total: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    pub generations_cancelled_total: AtomicU64,
    pub generations_timed_out_total: AtomicU64,
    pub queue_rejections_total: AtomicU64,
    pub inflight: AtomicU64,
    pub queued: AtomicU64,
    pub tokens_per_second_bits: AtomicU64,
}

pub struct ServerState {
    pub model: Box<dyn Model>,
    pub tokenizer: Tokenizer,
    /// Name reported by /v1/models and echoed in responses.
    pub model_name: String,
    /// Serializes generations (one forward pass at a time).
    pub gate: Semaphore,
    pub config: ServerConfig,
    pub metrics: Metrics,
}

impl ServerState {
    pub fn new(model: Box<dyn Model>, tokenizer: Tokenizer, model_name: String) -> Self {
        Self::with_config(model, tokenizer, model_name, ServerConfig::default())
    }

    pub fn with_config(
        model: Box<dyn Model>,
        tokenizer: Tokenizer,
        model_name: String,
        config: ServerConfig,
    ) -> Self {
        Self {
            model,
            tokenizer,
            model_name,
            gate: Semaphore::new(1),
            config,
            metrics: Metrics::default(),
        }
    }
}

pub struct GenerationRequest {
    pub prompt_ids: Vec<usize>,
    pub max_new: usize,
    pub sampling: SamplerConfig,
    pub stop_strings: Vec<String>,
}

/// Incremental text emitter with stop-string holdback.
pub struct StopAwareEmitter<'t> {
    decoder: StreamDecoder<'t>,
    stops: Vec<String>,
    holdback: usize,
    pending: String,
    pub stopped: bool,
}

impl<'t> StopAwareEmitter<'t> {
    pub fn new(tokenizer: &'t Tokenizer, stops: Vec<String>) -> Self {
        let holdback = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        Self {
            decoder: StreamDecoder::new(tokenizer),
            stops,
            holdback,
            pending: String::new(),
            stopped: false,
        }
    }

    /// Feed a token; returns text that is safe to emit now (never contains
    /// any prefix of a stop string that could still complete).
    pub fn push(&mut self, id: usize) -> undertow_core::Result<String> {
        let delta = self.decoder.push(id)?;
        self.pending.push_str(&delta);
        self.drain()
    }

    pub fn finish(&mut self) -> undertow_core::Result<String> {
        let rest = self.decoder.finish()?;
        self.pending.push_str(&rest);
        if let Some(idx) = self.match_stop() {
            self.pending.truncate(idx);
            self.stopped = true;
        }
        Ok(std::mem::take(&mut self.pending))
    }

    fn match_stop(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min()
    }

    fn drain(&mut self) -> undertow_core::Result<String> {
        if let Some(idx) = self.match_stop() {
            self.pending.truncate(idx);
            self.stopped = true;
            return Ok(std::mem::take(&mut self.pending));
        }
        if self.pending.len() <= self.holdback {
            return Ok(String::new());
        }
        // Keep the longest possible partial stop; flush at a char boundary.
        let mut cut = self.pending.len() - self.holdback;
        while !self.pending.is_char_boundary(cut) {
            cut -= 1;
        }
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        Ok(out)
    }
}

/// Why a generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// Stop token or stop string.
    Stop,
    /// max_tokens or context window reached.
    Length,
    /// Client went away; whatever was produced stands.
    Cancelled,
    /// Deadline passed; partial output stands.
    TimedOut,
}

impl Finish {
    pub fn as_openai(&self) -> &'static str {
        match self {
            Finish::Stop | Finish::Cancelled => "stop",
            Finish::Length | Finish::TimedOut => "length",
        }
    }
}

/// Blocking generation driving `on_text` with stop-aware deltas.
///
/// `on_text` returns whether to continue: `false` (client disconnected)
/// aborts the forward pass at the next token instead of burning CPU on a
/// dead connection. `deadline` is checked per token for the same reason.
pub fn generate_blocking(
    state: &ServerState,
    req: &GenerationRequest,
    deadline: std::time::Instant,
    mut on_text: impl FnMut(&str) -> bool,
) -> undertow_core::Result<(usize, Finish)> {
    let t0 = std::time::Instant::now();
    let mut sampler = Sampler::new(req.sampling).map_err(EngineError::Other)?;
    let mut emitter = StopAwareEmitter::new(&state.tokenizer, req.stop_strings.clone());
    let stop_ids = state.model.stop_ids().to_vec();
    let mut session = state.model.new_session();
    let mut hit_stop_token = false;
    let mut cancelled = false;
    let mut first_token_time: Option<Instant> = None;
    let mut timed_out = false;
    let produced = undertow_core::generate(
        &mut *session,
        &req.prompt_ids,
        req.max_new,
        &mut sampler,
        &stop_ids,
        |id| {
            if first_token_time.is_none() {
                first_token_time = Some(Instant::now());
            }
            if std::time::Instant::now() >= deadline {
                timed_out = true;
                return false;
            }
            if stop_ids.contains(&id) {
                hit_stop_token = true;
                return true; // counted, not decoded
            }
            match emitter.push(id) {
                Ok(text) => {
                    if !text.is_empty() && !on_text(&text) {
                        cancelled = true;
                        return false;
                    }
                    !emitter.stopped
                }
                Err(_) => false,
            }
        },
    )?;
    if produced > 1 {
        if let Some(t_first) = first_token_time {
            let decode_elapsed = t_first.elapsed().as_secs_f64();
            if decode_elapsed > 0.0 {
                let tps = (produced - 1) as f64 / decode_elapsed;
                state
                    .metrics
                    .tokens_per_second_bits
                    .store(tps.to_bits(), Ordering::Relaxed);
            }
        }
    } else if produced == 1 {
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let tps = 1.0 / elapsed;
            state
                .metrics
                .tokens_per_second_bits
                .store(tps.to_bits(), Ordering::Relaxed);
        }
    }
    state
        .metrics
        .tokens_generated_total
        .fetch_add(produced as u64, Ordering::Relaxed);
    if timed_out {
        state
            .metrics
            .generations_timed_out_total
            .fetch_add(1, Ordering::Relaxed);
        return Ok((produced, Finish::TimedOut));
    }
    if cancelled {
        state
            .metrics
            .generations_cancelled_total
            .fetch_add(1, Ordering::Relaxed);
        return Ok((produced, Finish::Cancelled));
    }
    if !emitter.stopped {
        let rest = emitter.finish()?;
        if !rest.is_empty() {
            on_text(&rest);
        }
    }
    let finish = if emitter.stopped || hit_stop_token {
        Finish::Stop
    } else {
        Finish::Length
    };
    Ok((produced, finish))
}

/// Render a chat into prompt ids.
pub fn chat_prompt_ids(
    state: &ServerState,
    messages: &[ChatMessage],
) -> undertow_core::Result<Vec<usize>> {
    state.tokenizer.encode_chat(messages, true)
}

/// Render a chat with optional tool definitions into prompt ids.
pub fn chat_prompt_ids_with_tools(
    state: &ServerState,
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
) -> undertow_core::Result<Vec<usize>> {
    state
        .tokenizer
        .encode_chat_with_tools(messages, tools, true)
}

/// Acquire the generation slot respecting the queue bound. `None` means
/// the queue is full (429).
pub async fn acquire_slot(state: &Arc<ServerState>) -> Option<tokio::sync::SemaphorePermit<'_>> {
    let m = &state.metrics;
    // Racy check is fine: the bound is operational back-pressure, not an
    // exact admission count.
    if m.queued.load(Ordering::Relaxed) as usize > state.config.max_queue {
        m.queue_rejections_total.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    m.queued.fetch_add(1, Ordering::Relaxed);
    let permit = state.gate.acquire().await.expect("gate never closed");
    m.queued.fetch_sub(1, Ordering::Relaxed);
    m.inflight.fetch_add(1, Ordering::Relaxed);
    Some(permit)
}

pub fn release_slot(state: &ServerState) {
    state.metrics.inflight.fetch_sub(1, Ordering::Relaxed);
}

/// Prometheus text exposition of server + store counters.
pub fn render_metrics(state: &ServerState) -> String {
    let m = &state.metrics;
    let s = state.model.store_stats();
    let mut out = String::new();
    let tps = f64::from_bits(m.tokens_per_second_bits.load(Ordering::Relaxed));
    let gauges: &[(&str, &str, String)] = &[
        (
            "inflight",
            "generations running now",
            m.inflight.load(Ordering::Relaxed).to_string(),
        ),
        (
            "queued",
            "requests waiting for the slot",
            m.queued.load(Ordering::Relaxed).to_string(),
        ),
        (
            "tokens_per_second",
            "instantaneous decode generation rate",
            format!("{tps:.2}"),
        ),
        (
            "expert_cache_hit_ratio",
            "expert cache hit ratio",
            format!("{:.4}", s.hit_rate()),
        ),
        (
            "expert_cache_bytes_used",
            "bytes held in expert cache",
            s.bytes_used.to_string(),
        ),
        (
            "expert_cache_budget_bytes",
            "total allocated expert cache budget bytes",
            s.budget_bytes.to_string(),
        ),
    ];
    for (name, help, v) in gauges {
        out.push_str(&format!(
            "# HELP undertow_{name} {help}\n# TYPE undertow_{name} gauge\nundertow_{name} {v}\n"
        ));
    }
    let counters: &[(&str, &str, u64)] = &[
        (
            "requests_total",
            "http requests received",
            m.requests_total.load(Ordering::Relaxed),
        ),
        (
            "responses_4xx_total",
            "client-error responses",
            m.responses_4xx_total.load(Ordering::Relaxed),
        ),
        (
            "responses_5xx_total",
            "server-error responses",
            m.responses_5xx_total.load(Ordering::Relaxed),
        ),
        (
            "tokens_generated_total",
            "completion tokens produced",
            m.tokens_generated_total.load(Ordering::Relaxed),
        ),
        (
            "generations_cancelled_total",
            "generations aborted by disconnect",
            m.generations_cancelled_total.load(Ordering::Relaxed),
        ),
        (
            "generations_timed_out_total",
            "generations stopped at deadline",
            m.generations_timed_out_total.load(Ordering::Relaxed),
        ),
        (
            "queue_rejections_total",
            "requests rejected with 429",
            m.queue_rejections_total.load(Ordering::Relaxed),
        ),
        ("expert_store_hits_total", "expert cache hits", s.hits),
        ("expert_store_misses_total", "expert cache misses", s.misses),
        (
            "expert_store_bytes_read_total",
            "bytes read from disk",
            s.bytes_read,
        ),
        (
            "prefetch_issued_total",
            "prefetch hints issued",
            s.prefetch_issued,
        ),
        (
            "prefetch_dropped_total",
            "prefetch hints dropped",
            s.prefetch_dropped,
        ),
        (
            "expert_evictions_total",
            "expert weights evicted under memory pressure",
            s.evictions,
        ),
    ];
    for (name, help, v) in counters {
        out.push_str(&format!(
            "# HELP undertow_{name} {help}\n# TYPE undertow_{name} counter\nundertow_{name} {v}\n"
        ));
    }

    out.push_str(
        "# HELP undertow_disk_read_duration_seconds latency of synchronous pread expert fetch calls\n\
         # TYPE undertow_disk_read_duration_seconds histogram\n",
    );
    for (i, &le) in undertow_core::DISK_READ_LATENCY_BUCKETS.iter().enumerate() {
        out.push_str(&format!(
            "undertow_disk_read_duration_seconds_bucket{{le=\"{le}\"}} {}\n",
            s.read_buckets[i]
        ));
    }
    out.push_str(&format!(
        "undertow_disk_read_duration_seconds_bucket{{le=\"+Inf\"}} {}\n",
        s.read_count
    ));
    out.push_str(&format!(
        "undertow_disk_read_duration_seconds_sum {:.6}\n",
        s.read_duration_seconds
    ));
    out.push_str(&format!(
        "undertow_disk_read_duration_seconds_count {}\n",
        s.read_count
    ));

    out
}
