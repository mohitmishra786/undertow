//! Family-agnostic model and session interfaces.
//!
//! Frontends (CLI, server) drive inference exclusively through these
//! traits, so adding an architecture family never touches frontend code.
//! Sessions borrow their model; `Box<dyn Session + '_>` keeps that borrow
//! honest without reference counting the weights.

use crate::error::Result;
use crate::sample::Sampler;
use crate::store::StoreStatsSnapshot;
use crate::tensor::Tensor;

/// How a loader should place routed experts.
#[derive(Debug, Clone)]
pub enum StoreChoice {
    /// Stream experts from disk. `cache_budget_bytes: None` auto-sizes
    /// from physical memory minus resident dense weights.
    DiskStreaming {
        cache_budget_bytes: Option<u64>,
        prefetch_workers: usize,
    },
    /// Load every expert into RAM up front.
    Resident,
}

/// Eviction policy for the streaming expert cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CachePolicy {
    #[default]
    Lru,
    /// Importance-weighted (decayed access scores).
    Weighted,
}

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub store: StoreChoice,
    pub cache_policy: CachePolicy,
    /// Hot-expert profile to pin at load (streaming stores only).
    pub pin_profile: Option<crate::profile::ExpertProfile>,
    /// Byte budget for pinned experts; `None` = a quarter of the cache
    /// budget.
    pub pin_budget_bytes: Option<u64>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            store: StoreChoice::DiskStreaming {
                cache_budget_bytes: None,
                prefetch_workers: 2,
            },
            cache_policy: CachePolicy::default(),
            pin_profile: None,
            pin_budget_bytes: None,
        }
    }
}

/// Extract stop-token ids from an `eos_token_id` json value (number or
/// array of numbers; anything else yields none).
pub fn parse_eos_ids(v: &serde_json::Value) -> Vec<usize> {
    match v {
        serde_json::Value::Number(n) => n.as_u64().map(|x| x as usize).into_iter().collect(),
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_u64().map(|u| u as usize))
            .collect(),
        _ => Vec::new(),
    }
}

/// Stop tokens from a checkpoint's `generation_config.json`, when present.
pub fn generation_config_eos(dir: &std::path::Path) -> Vec<usize> {
    let Ok(bytes) = std::fs::read(dir.join("generation_config.json")) else {
        return Vec::new();
    };
    let Ok(root) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Vec::new();
    };
    root.get("eos_token_id")
        .map(parse_eos_ids)
        .unwrap_or_default()
}

/// One conversation/completion in flight (its own KV cache).
pub trait Session {
    /// Run tokens through the model, extending the KV cache. Returns
    /// logits `[seq, vocab]`.
    fn prefill(&mut self, token_ids: &[usize]) -> Result<Tensor>;

    /// Feed one token, get next-token logits.
    fn decode(&mut self, token_id: usize) -> Result<Vec<f32>>;

    /// Roll back to `len` consumed tokens (shared-prefix reuse).
    fn truncate(&mut self, len: usize);

    /// Tokens consumed so far.
    fn position(&self) -> usize;

    /// KV cache bytes currently held.
    fn kv_bytes(&self) -> usize;
}

/// A loaded model of any supported architecture family.
pub trait Model: Send + Sync {
    fn architecture(&self) -> &'static str;

    fn vocab_size(&self) -> usize;

    fn max_context(&self) -> usize;

    /// Stop-token ids declared by the checkpoint.
    fn stop_ids(&self) -> &[usize];

    fn new_session(&self) -> Box<dyn Session + '_>;

    /// Expert-store counters for stats output.
    fn store_stats(&self) -> StoreStatsSnapshot;

    /// Per-expert usage counts (for exporting hot-expert profiles).
    fn expert_usage(&self) -> Vec<(crate::store::ExpertKey, u64)> {
        Vec::new()
    }

    /// Full-sequence teacher-forcing forward with a throwaway session.
    fn forward(&self, token_ids: &[usize]) -> Result<Tensor> {
        self.new_session().prefill(token_ids)
    }
}

/// Drive generation on any session: prefill `prompt`, then sample until
/// `max_new` tokens, a stop token, or `on_token` returns false. Stop
/// tokens are reported to the callback and included in the count.
pub fn generate(
    session: &mut dyn Session,
    prompt: &[usize],
    max_new: usize,
    sampler: &mut Sampler,
    stop_ids: &[usize],
    mut on_token: impl FnMut(usize) -> bool,
) -> Result<usize> {
    let logits = session.prefill(prompt)?;
    let mut last = logits.row(prompt.len() - 1).to_vec();
    let mut produced = 0;
    while produced < max_new {
        let next = sampler.sample(&last);
        produced += 1;
        let keep_going = on_token(next);
        if stop_ids.contains(&next) || !keep_going || produced == max_new {
            break;
        }
        last = session.decode(next)?;
    }
    Ok(produced)
}

/// Greedy decode helper shared by adapters and tests.
pub fn greedy_decode(
    model: &dyn Model,
    prompt: &[usize],
    max_new: usize,
    extra_stop_ids: &[usize],
) -> Result<Vec<usize>> {
    let mut sampler = Sampler::new(crate::sample::SamplerConfig::greedy()).expect("greedy valid");
    let mut stops = model.stop_ids().to_vec();
    stops.extend_from_slice(extra_stop_ids);
    let mut out = prompt.to_vec();
    let mut session = model.new_session();
    generate(&mut *session, prompt, max_new, &mut sampler, &stops, |id| {
        out.push(id);
        true
    })?;
    Ok(out)
}
