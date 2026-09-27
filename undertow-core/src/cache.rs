//! Expert cache: eviction policy for the RAM tier, separated from I/O.
//!
//! [`LruExpertCache`] is the default policy: least-recently-used eviction
//! under a byte budget (not an entry count, because int4 and int8 experts
//! differ in size), with a pinned set that never evicts. Importance and
//! frequency weighted policies plug in behind the same trait later; the
//! `record_use` hook already feeds them.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use crate::store::{ExpertKey, ExpertWeights};

pub trait ExpertCache: Send + Sync {
    /// Look up a cached expert. Implementations update their recency /
    /// frequency bookkeeping on hit.
    fn get(&self, key: ExpertKey) -> Option<Arc<ExpertWeights>>;

    /// Insert after a miss was served from a lower tier. The policy decides
    /// what (if anything) to evict.
    fn insert(&self, key: ExpertKey, weights: Arc<ExpertWeights>);

    /// Pin an expert so it is never evicted (hot-profile support). Pinned
    /// bytes count against the budget.
    fn pin(&self, key: ExpertKey, weights: Arc<ExpertWeights>);

    /// Record a routing decision even when the weights were not fetched
    /// through this cache (importance signal for weighted policies).
    fn record_use(&self, _key: ExpertKey) {}

    /// Current number of cached (non-pinned) experts.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes currently held (cached + pinned).
    fn bytes_used(&self) -> usize;

    /// Byte budget, if bounded.
    fn capacity_bytes(&self) -> Option<usize>;

    /// Number of expert evictions performed under memory pressure.
    fn evictions(&self) -> u64 {
        0
    }
}

struct LruInner {
    lru: lru::LruCache<ExpertKey, Arc<ExpertWeights>>,
    pinned: HashMap<ExpertKey, Arc<ExpertWeights>>,
    bytes: usize,
    budget: usize,
    evictions: u64,
}

/// Byte-budgeted LRU with pinning.
///
/// Entries larger than the whole budget are never admitted (they would
/// evict everything and then get evicted themselves on the next insert),
/// but `get_expert` still works for them: the store just returns the
/// freshly loaded copy without caching it.
pub struct LruExpertCache {
    inner: Mutex<LruInner>,
}

impl LruExpertCache {
    /// `budget` is the maximum bytes of expert weights held (cached plus
    /// pinned). A zero budget is valid and caches nothing.
    pub fn new(budget: usize) -> Self {
        Self {
            inner: Mutex::new(LruInner {
                // Unbounded by entry count; we enforce the byte budget.
                lru: lru::LruCache::unbounded(),
                pinned: HashMap::new(),
                bytes: 0,
                budget,
                evictions: 0,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LruInner> {
        // Poisoning only happens if a panic occurred while holding the
        // lock; cache state is a pure performance artifact, so recover.
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl ExpertCache for LruExpertCache {
    fn get(&self, key: ExpertKey) -> Option<Arc<ExpertWeights>> {
        let mut g = self.lock();
        if let Some(w) = g.pinned.get(&key) {
            return Some(w.clone());
        }
        g.lru.get(&key).cloned()
    }

    fn insert(&self, key: ExpertKey, weights: Arc<ExpertWeights>) {
        let size = weights.nbytes();
        let mut g = self.lock();
        if size > g.budget || g.pinned.contains_key(&key) {
            return;
        }
        if let Some(old) = g.lru.put(key, weights) {
            g.bytes -= old.nbytes();
        }
        g.bytes += size;
        while g.bytes > g.budget {
            match g.lru.pop_lru() {
                Some((_, evicted)) => {
                    g.bytes -= evicted.nbytes();
                    g.evictions += 1;
                }
                None => break,
            }
        }
    }

    fn pin(&self, key: ExpertKey, weights: Arc<ExpertWeights>) {
        let size = weights.nbytes();
        let mut g = self.lock();
        if let Some(old) = g.lru.pop(&key) {
            g.bytes -= old.nbytes();
        }
        if let Some(old) = g.pinned.insert(key, weights) {
            g.bytes -= old.nbytes();
        }
        g.bytes += size;
        // Pins are honored even if they blow the budget; evict the
        // unpinned tail to compensate as far as possible.
        while g.bytes > g.budget {
            match g.lru.pop_lru() {
                Some((_, evicted)) => {
                    g.bytes -= evicted.nbytes();
                    g.evictions += 1;
                }
                None => break,
            }
        }
    }

    fn len(&self) -> usize {
        self.lock().lru.len()
    }

    fn bytes_used(&self) -> usize {
        self.lock().bytes
    }

    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.lock().budget)
    }

    fn evictions(&self) -> u64 {
        self.lock().evictions
    }
}

/// Build a cache for the given policy and byte budget.
pub fn build_cache(policy: crate::model::CachePolicy, budget: usize) -> Arc<dyn ExpertCache> {
    match policy {
        crate::model::CachePolicy::Lru => Arc::new(LruExpertCache::new(budget)),
        crate::model::CachePolicy::Weighted => Arc::new(WeightedExpertCache::new(budget)),
    }
}

/// Convenience for sizing an LRU by entry count when expert size is known
/// and uniform (tests, benchmarks).
pub fn lru_for_experts(expert_bytes: usize, count: usize) -> LruExpertCache {
    LruExpertCache::new(expert_bytes.max(1) * count.max(1))
}

struct WeightedEntry {
    weights: Arc<ExpertWeights>,
    score: f64,
    last_tick: u64,
}

struct WeightedInner {
    map: HashMap<ExpertKey, WeightedEntry>,
    pinned: HashMap<ExpertKey, Arc<ExpertWeights>>,
    bytes: usize,
    budget: usize,
    tick: u64,
    evictions: u64,
}

/// Importance-weighted eviction (MoE-Infinity style): each entry carries
/// an exponentially-decayed access score, and the entry with the lowest
/// current score evicts first. A burst of accesses long ago loses to
/// steady recent use; a expert routed every few tokens is effectively
/// unevictable while it stays hot.
///
/// Eviction is an O(n) scan over resident entries. That is deliberate:
/// eviction only happens on a miss, and a miss costs a disk read that
/// dwarfs scanning a few thousand scores.
pub struct WeightedExpertCache {
    inner: Mutex<WeightedInner>,
    /// Accesses over which a score halves.
    half_life: f64,
}

impl WeightedExpertCache {
    pub fn new(budget: usize) -> Self {
        Self::with_half_life(budget, 512.0)
    }

    pub fn with_half_life(budget: usize, half_life: f64) -> Self {
        Self {
            inner: Mutex::new(WeightedInner {
                map: HashMap::new(),
                pinned: HashMap::new(),
                bytes: 0,
                budget,
                tick: 0,
                evictions: 0,
            }),
            half_life: half_life.max(1.0),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, WeightedInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn effective(&self, e: &WeightedEntry, now: u64) -> f64 {
        e.score * 0.5f64.powf((now - e.last_tick) as f64 / self.half_life)
    }

    fn bump(&self, e: &mut WeightedEntry, now: u64) {
        e.score = e.score * 0.5f64.powf((now - e.last_tick) as f64 / self.half_life) + 1.0;
        e.last_tick = now;
    }

    fn evict_to_budget(&self, g: &mut WeightedInner) {
        while g.bytes > g.budget && !g.map.is_empty() {
            let now = g.tick;
            let coldest = g
                .map
                .iter()
                .min_by(|a, b| {
                    self.effective(a.1, now)
                        .partial_cmp(&self.effective(b.1, now))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(k, _)| *k)
                .expect("map not empty");
            if let Some(e) = g.map.remove(&coldest) {
                g.bytes -= e.weights.nbytes();
                g.evictions += 1;
            }
        }
    }
}

impl ExpertCache for WeightedExpertCache {
    fn get(&self, key: ExpertKey) -> Option<Arc<ExpertWeights>> {
        let mut g = self.lock();
        g.tick += 1;
        if let Some(w) = g.pinned.get(&key) {
            return Some(w.clone());
        }
        let now = g.tick;
        let e = g.map.get_mut(&key)?;
        self.bump(e, now);
        Some(e.weights.clone())
    }

    fn insert(&self, key: ExpertKey, weights: Arc<ExpertWeights>) {
        let size = weights.nbytes();
        let mut g = self.lock();
        g.tick += 1;
        if size > g.budget || g.pinned.contains_key(&key) {
            return;
        }
        let now = g.tick;
        match g.map.get_mut(&key) {
            Some(e) => {
                self.bump(e, now);
                // Same key, potentially different bytes (should not
                // happen in practice, but stay budget-correct).
                let old = std::mem::replace(&mut e.weights, weights).nbytes();
                g.bytes = g.bytes - old + size;
            }
            None => {
                g.map.insert(
                    key,
                    WeightedEntry {
                        weights,
                        score: 1.0,
                        last_tick: now,
                    },
                );
                g.bytes += size;
            }
        }
        self.evict_to_budget(&mut g);
    }

    fn pin(&self, key: ExpertKey, weights: Arc<ExpertWeights>) {
        let size = weights.nbytes();
        let mut g = self.lock();
        if let Some(e) = g.map.remove(&key) {
            g.bytes -= e.weights.nbytes();
        }
        if let Some(old) = g.pinned.insert(key, weights) {
            g.bytes -= old.nbytes();
        }
        g.bytes += size;
        self.evict_to_budget(&mut g);
    }

    fn record_use(&self, key: ExpertKey) {
        let mut g = self.lock();
        g.tick += 1;
        let now = g.tick;
        if let Some(e) = g.map.get_mut(&key) {
            self.bump(e, now);
        }
    }

    fn len(&self) -> usize {
        self.lock().map.len()
    }

    fn bytes_used(&self) -> usize {
        self.lock().bytes
    }

    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.lock().budget)
    }

    fn evictions(&self) -> u64 {
        self.lock().evictions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use undertow_quant::{QTensor, QuantFormat};

    fn expert(fill: f32) -> Arc<ExpertWeights> {
        let w: Vec<f32> = (0..32).map(|i| fill + i as f32 * 0.01).collect();
        let q = QTensor::quantize(&w, 4, 8, QuantFormat::Int8).unwrap();
        Arc::new(ExpertWeights {
            gate_proj: q.clone(),
            up_proj: q.clone(),
            down_proj: QTensor::quantize(&w, 8, 4, QuantFormat::Int8).unwrap(),
        })
    }

    fn key(l: usize, e: usize) -> ExpertKey {
        ExpertKey {
            layer: l,
            expert: e,
        }
    }

    #[test]
    fn evicts_least_recently_used_under_byte_budget() {
        let e = expert(1.0);
        let one = e.nbytes();
        let cache = LruExpertCache::new(one * 2);
        cache.insert(key(0, 0), e.clone());
        cache.insert(key(0, 1), e.clone());
        assert_eq!(cache.len(), 2);
        // Touch (0,0) so (0,1) becomes LRU, then overflow.
        assert!(cache.get(key(0, 0)).is_some());
        cache.insert(key(0, 2), e.clone());
        assert!(cache.get(key(0, 0)).is_some());
        assert!(cache.get(key(0, 1)).is_none(), "LRU entry must be evicted");
        assert!(cache.get(key(0, 2)).is_some());
        assert!(cache.bytes_used() <= one * 2);
    }

    #[test]
    fn oversized_entry_not_admitted() {
        let e = expert(1.0);
        let cache = LruExpertCache::new(e.nbytes() - 1);
        cache.insert(key(0, 0), e);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.bytes_used(), 0);
    }

    #[test]
    fn zero_budget_caches_nothing() {
        let cache = LruExpertCache::new(0);
        cache.insert(key(0, 0), expert(1.0));
        assert!(cache.get(key(0, 0)).is_none());
    }

    #[test]
    fn pinned_survives_pressure() {
        let e = expert(1.0);
        let one = e.nbytes();
        let cache = LruExpertCache::new(one * 2);
        cache.pin(key(9, 9), e.clone());
        for i in 0..10 {
            cache.insert(key(0, i), e.clone());
        }
        assert!(cache.get(key(9, 9)).is_some(), "pin must survive churn");
        assert!(cache.bytes_used() <= one * 2);
    }

    #[test]
    fn reinsert_same_key_does_not_leak_bytes() {
        let e = expert(1.0);
        let cache = LruExpertCache::new(e.nbytes() * 4);
        for _ in 0..100 {
            cache.insert(key(0, 0), e.clone());
        }
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes_used(), e.nbytes());
    }

    #[test]
    fn weighted_keeps_hot_expert_over_recent_one_shots() {
        let e = expert(1.0);
        let one = e.nbytes();
        // Room for 3 entries. Expert (0,0) is accessed constantly; a
        // stream of one-shot entries churns the other two slots. Under
        // plain LRU (capacity 3) the hot entry would survive too, so make
        // it adversarial: touch the hot entry, then insert TWO fresh
        // entries; LRU would evict the hot one on the second insert,
        // weighted must not.
        let cache = WeightedExpertCache::with_half_life(one * 3, 1e9);
        cache.insert(key(0, 0), e.clone());
        for round in 0..20 {
            assert!(
                cache.get(key(0, 0)).is_some(),
                "hot expert evicted at round {round}"
            );
            cache.insert(key(1, round), e.clone());
            cache.insert(key(2, round), e.clone());
            assert!(cache.bytes_used() <= one * 3);
        }
    }

    #[test]
    fn weighted_scores_decay() {
        let e = expert(1.0);
        let one = e.nbytes();
        // Half-life of 2 accesses: past glory fades fast. Build up a big
        // score on (0,0), then hammer (0,1); once decay catches up, a new
        // insert must evict (0,0), not the recently busy (0,1).
        let cache = WeightedExpertCache::with_half_life(one * 2, 2.0);
        cache.insert(key(0, 0), e.clone());
        for _ in 0..30 {
            cache.get(key(0, 0));
        }
        cache.insert(key(0, 1), e.clone());
        for _ in 0..200 {
            cache.get(key(0, 1));
        }
        cache.insert(key(0, 2), e.clone());
        assert!(cache.get(key(0, 1)).is_some(), "busy entry must survive");
        assert!(
            cache.get(key(0, 0)).is_none(),
            "decayed entry must be the one evicted"
        );
    }

    #[test]
    fn weighted_pin_and_budget() {
        let e = expert(1.0);
        let one = e.nbytes();
        let cache = WeightedExpertCache::new(one * 2);
        cache.pin(key(9, 9), e.clone());
        for i in 0..10 {
            cache.insert(key(0, i), e.clone());
        }
        assert!(cache.get(key(9, 9)).is_some());
        assert!(cache.bytes_used() <= one * 2);
    }

    #[test]
    fn lru_evictions_counter() {
        let e = expert(1.0);
        let one = e.nbytes();
        let cache = LruExpertCache::new(one * 2);
        assert_eq!(cache.evictions(), 0);
        cache.insert(key(0, 0), e.clone());
        cache.insert(key(0, 1), e.clone());
        assert_eq!(cache.evictions(), 0);
        cache.insert(key(0, 2), e.clone());
        assert_eq!(cache.evictions(), 1);
        cache.insert(key(0, 3), e.clone());
        assert_eq!(cache.evictions(), 2);
    }

    #[test]
    fn weighted_evictions_counter() {
        let e = expert(1.0);
        let one = e.nbytes();
        let cache = WeightedExpertCache::new(one * 2);
        assert_eq!(cache.evictions(), 0);
        cache.insert(key(0, 0), e.clone());
        cache.insert(key(0, 1), e.clone());
        assert_eq!(cache.evictions(), 0);
        cache.insert(key(0, 2), e.clone());
        assert_eq!(cache.evictions(), 1);
    }
}
