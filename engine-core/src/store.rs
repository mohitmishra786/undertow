//! `TieredStore`: the single boundary through which the forward pass
//! reaches routed-expert weights.
//!
//! Implementations today: [`ResidentStore`] (everything in RAM, used for
//! plain f32 checkpoints and tests) and `engine-io`'s `DiskExpertStore`
//! (pread on demand behind an [`crate::cache::ExpertCache`]). The trait
//! deliberately does not assume storage is local to one machine.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use engine_quant::QTensor;

use crate::error::{EngineError, Result};

/// Address of one routed expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: usize,
    pub expert: usize,
}

/// The three FFN matrices of one (SwiGLU) expert, in whatever precision
/// the checkpoint stores them. Kernels dequantize on use; the cache holds
/// the quantized bytes, which is the whole point of the tiered design.
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    /// `[moe_intermediate, hidden]`
    pub gate_proj: QTensor,
    /// `[moe_intermediate, hidden]`
    pub up_proj: QTensor,
    /// `[hidden, moe_intermediate]`
    pub down_proj: QTensor,
}

impl ExpertWeights {
    /// Bytes held in memory; used for cache budgeting.
    pub fn nbytes(&self) -> usize {
        self.gate_proj.nbytes() + self.up_proj.nbytes() + self.down_proj.nbytes()
    }
}

/// Cumulative counters every store implementation exposes. All atomic so
/// stats can be read while inference runs.
#[derive(Debug, Default)]
pub struct StoreStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub bytes_read: AtomicU64,
    pub prefetch_issued: AtomicU64,
    pub prefetch_dropped: AtomicU64,
}

impl StoreStats {
    pub fn snapshot(&self) -> StoreStatsSnapshot {
        StoreStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            prefetch_issued: self.prefetch_issued.load(Ordering::Relaxed),
            prefetch_dropped: self.prefetch_dropped.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub prefetch_issued: u64,
    pub prefetch_dropped: u64,
}

impl StoreStatsSnapshot {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

pub trait TieredStore: Send + Sync {
    /// Fetch one expert's weights, wherever they currently live.
    /// `Arc` so cache tiers and compute can share without copying.
    fn get_expert(&self, key: ExpertKey) -> Result<Arc<ExpertWeights>>;

    /// Non-binding hint that `key` will likely be needed soon
    /// (speculative prefetch path). Default: no-op.
    fn prefetch(&self, _key: ExpertKey) {}

    /// Per-expert access counts recorded so far (for hot-expert
    /// profiles). Default: none recorded.
    fn usage(&self) -> Vec<(ExpertKey, u64)> {
        Vec::new()
    }

    /// Cumulative counters. Implementations without interesting stats may
    /// return zeros.
    fn stats(&self) -> StoreStatsSnapshot {
        StoreStatsSnapshot {
            hits: 0,
            misses: 0,
            bytes_read: 0,
            prefetch_issued: 0,
            prefetch_dropped: 0,
        }
    }
}

/// Everything resident in RAM. Used for plain f32 checkpoints (oracle
/// scale) and as the reference store in tests.
#[derive(Default)]
pub struct ResidentStore {
    experts: HashMap<ExpertKey, Arc<ExpertWeights>>,
    stats: StoreStats,
    usage: std::sync::Mutex<HashMap<ExpertKey, u64>>,
}

impl ResidentStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: ExpertKey, weights: ExpertWeights) {
        self.experts.insert(key, Arc::new(weights));
    }

    pub fn len(&self) -> usize {
        self.experts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }

    /// Total bytes held by all experts.
    pub fn nbytes(&self) -> usize {
        self.experts.values().map(|e| e.nbytes()).sum()
    }
}

impl TieredStore for ResidentStore {
    fn get_expert(&self, key: ExpertKey) -> Result<Arc<ExpertWeights>> {
        match self.experts.get(&key) {
            Some(e) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                *self
                    .usage
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .entry(key)
                    .or_insert(0) += 1;
                Ok(e.clone())
            }
            None => Err(EngineError::ExpertNotFound(key)),
        }
    }

    fn usage(&self) -> Vec<(ExpertKey, u64)> {
        self.usage
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    fn stats(&self) -> StoreStatsSnapshot {
        self.stats.snapshot()
    }
}
