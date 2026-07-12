//! Shared server state and the blocking generation worker.

use engine_core::sample::{Sampler, SamplerConfig};
use engine_core::{EngineError, Model};
use engine_tokenizer::{ChatMessage, StreamDecoder, Tokenizer};
use tokio::sync::Semaphore;

pub struct ServerState {
    pub model: Box<dyn Model>,
    pub tokenizer: Tokenizer,
    /// Name reported by /v1/models and echoed in responses.
    pub model_name: String,
    /// Serializes generations (one forward pass at a time).
    pub gate: Semaphore,
}

impl ServerState {
    pub fn new(model: Box<dyn Model>, tokenizer: Tokenizer, model_name: String) -> Self {
        Self {
            model,
            tokenizer,
            model_name,
            gate: Semaphore::new(1),
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
    pub fn push(&mut self, id: usize) -> engine_core::Result<String> {
        let delta = self.decoder.push(id)?;
        self.pending.push_str(&delta);
        self.drain()
    }

    pub fn finish(&mut self) -> engine_core::Result<String> {
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

    fn drain(&mut self) -> engine_core::Result<String> {
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

/// Blocking generation driving `on_text` with stop-aware deltas. Returns
/// completion token count and finish reason.
pub fn generate_blocking(
    state: &ServerState,
    req: &GenerationRequest,
    mut on_text: impl FnMut(&str),
) -> engine_core::Result<(usize, &'static str)> {
    let mut sampler = Sampler::new(req.sampling).map_err(EngineError::Other)?;
    let mut emitter = StopAwareEmitter::new(&state.tokenizer, req.stop_strings.clone());
    let stop_ids = state.model.stop_ids().to_vec();
    let mut session = state.model.new_session();
    let mut hit_stop_token = false;
    let produced = engine_core::generate(
        &mut *session,
        &req.prompt_ids,
        req.max_new,
        &mut sampler,
        &stop_ids,
        |id| {
            if stop_ids.contains(&id) {
                hit_stop_token = true;
                return true; // counted, not decoded
            }
            match emitter.push(id) {
                Ok(text) => {
                    if !text.is_empty() {
                        on_text(&text);
                    }
                    !emitter.stopped
                }
                Err(_) => false,
            }
        },
    )?;
    if !emitter.stopped {
        let rest = emitter.finish()?;
        if !rest.is_empty() {
            on_text(&rest);
        }
    }
    let finish = if emitter.stopped || hit_stop_token {
        "stop"
    } else if produced >= req.max_new {
        "length"
    } else {
        "stop"
    };
    Ok((produced, finish))
}

/// Render a chat into prompt ids.
pub fn chat_prompt_ids(
    state: &ServerState,
    messages: &[ChatMessage],
) -> engine_core::Result<Vec<usize>> {
    state.tokenizer.encode_chat(messages, true)
}
