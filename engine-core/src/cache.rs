//! Expert cache policy trait.
//!
//! The disk-backed `TieredStore` (Phase 1) consults an `ExpertCache` for
//! the RAM tier. Policies evolve independently of I/O:
//! LRU baseline → importance/frequency-weighted eviction (MoE-Infinity
//! style) → pinned community hot-profiles.
//!
//! Phase 0 defines the trait only; no implementation ships until the
//! scalar forward pass is proven correct.

use std::sync::Arc;

use crate::store::{ExpertKey, ExpertWeights};

pub trait ExpertCache: Send + Sync {
    /// Look up a cached expert. Implementations update their recency /
    /// frequency bookkeeping on hit.
    fn get(&self, key: ExpertKey) -> Option<Arc<ExpertWeights>>;

    /// Insert after a miss was served from a lower tier. The policy decides
    /// what (if anything) to evict.
    fn insert(&self, key: ExpertKey, weights: Arc<ExpertWeights>);

    /// Pin an expert so it is never evicted (hot-profile support).
    fn pin(&self, key: ExpertKey, weights: Arc<ExpertWeights>);

    /// Record a routing decision even when the weights were not fetched
    /// through this cache (importance signal for weighted policies).
    fn record_use(&self, _key: ExpertKey) {}

    /// Current number of cached (non-pinned) experts.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum experts the cache may hold, if bounded.
    fn capacity(&self) -> Option<usize>;
}
