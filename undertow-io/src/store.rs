//! `DiskExpertStore`: routed experts streamed from disk on demand.
//!
//! The hot path is `get_expert`: a cache probe, and on miss three `pread`
//! calls (plus scales) that land in buffers we own. Positioned reads are
//! offset-stateless, so any number of threads can fetch different experts
//! from the same shard files without contending on a cursor.
//!
//! Prefetch is a non-binding hint served by a small worker pool. Hints go
//! through a bounded queue; when the queue is full the hint is dropped and
//! counted, never blocked on — a lost hint costs a future cache miss, a
//! blocked decode step costs latency right now.

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use undertow_core::adapter::ExpertNaming;
use undertow_core::store::{ExpertKey, ExpertWeights, StoreStats, StoreStatsSnapshot};
use undertow_core::{ExpertCache, Result, TieredStore};

use crate::reader::{read_qtensor, ShardedModelReader};

/// Logical dims of every routed expert (uniform per model).
#[derive(Debug, Clone, Copy)]
pub struct ExpertDims {
    pub hidden: usize,
    pub moe_intermediate: usize,
}

struct Shared {
    reader: Arc<ShardedModelReader>,
    naming: Arc<dyn ExpertNaming>,
    dims: ExpertDims,
    cache: Arc<dyn ExpertCache>,
    stats: StoreStats,
    /// Keys currently being loaded by a prefetch worker, to avoid
    /// duplicate disk reads across workers.
    in_flight: Mutex<HashSet<ExpertKey>>,
    /// Per-expert access counts (hot-expert profiles).
    usage: Mutex<std::collections::HashMap<ExpertKey, u64>>,
}

impl Shared {
    fn load(&self, key: ExpertKey) -> Result<ExpertWeights> {
        let [gate, up, down] = self.naming.expert_tensor_names(key);
        let (h, m) = (self.dims.hidden, self.dims.moe_intermediate);
        let w = ExpertWeights {
            gate_proj: read_qtensor(&self.reader, &gate, m, h)?,
            up_proj: read_qtensor(&self.reader, &up, m, h)?,
            down_proj: read_qtensor(&self.reader, &down, h, m)?,
        };
        self.stats
            .bytes_read
            .fetch_add(w.nbytes() as u64, Ordering::Relaxed);
        Ok(w)
    }
}

pub struct DiskExpertStore {
    shared: Arc<Shared>,
    prefetch_tx: Option<SyncSender<ExpertKey>>,
    workers: Vec<JoinHandle<()>>,
}

impl DiskExpertStore {
    /// `prefetch_workers = 0` disables background prefetch entirely
    /// (hints become no-ops, still counted as dropped).
    pub fn new(
        reader: Arc<ShardedModelReader>,
        naming: Arc<dyn ExpertNaming>,
        dims: ExpertDims,
        cache: Arc<dyn ExpertCache>,
        prefetch_workers: usize,
    ) -> Self {
        let shared = Arc::new(Shared {
            reader,
            naming,
            dims,
            cache,
            stats: StoreStats::default(),
            in_flight: Mutex::new(HashSet::new()),
            usage: Mutex::new(std::collections::HashMap::new()),
        });
        let mut workers = Vec::new();
        let prefetch_tx = if prefetch_workers > 0 {
            let (tx, rx) = sync_channel::<ExpertKey>(1024);
            let rx = Arc::new(Mutex::new(rx));
            for n in 0..prefetch_workers {
                let shared = shared.clone();
                let rx = rx.clone();
                workers.push(
                    std::thread::Builder::new()
                        .name(format!("undertow-prefetch-{n}"))
                        .spawn(move || prefetch_worker(shared, rx))
                        .expect("spawn prefetch worker"),
                );
            }
            Some(tx)
        } else {
            None
        };
        Self {
            shared,
            prefetch_tx,
            workers,
        }
    }
}

fn prefetch_worker(shared: Arc<Shared>, rx: Arc<Mutex<Receiver<ExpertKey>>>) {
    loop {
        // Hold the receiver lock only while waiting for one item.
        let key = match rx.lock().unwrap_or_else(|p| p.into_inner()).recv() {
            Ok(k) => k,
            Err(_) => return, // store dropped
        };
        if shared.cache.get(key).is_some() {
            continue;
        }
        {
            let mut inflight = shared.in_flight.lock().unwrap_or_else(|p| p.into_inner());
            if !inflight.insert(key) {
                continue; // another worker is on it
            }
        }
        if let Ok(w) = shared.load(key) {
            shared.cache.insert(key, Arc::new(w));
        }
        // Load errors are deliberately swallowed here: a prefetch hint must
        // never crash inference. The synchronous get_expert path will
        // surface the real error if the expert is genuinely unreadable.
        shared
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&key);
    }
}

impl DiskExpertStore {
    /// Load and pin the hottest experts from a profile, stopping once
    /// `budget_bytes` of pinned weights are resident. Unknown experts
    /// (out-of-range indices from a stale profile) are skipped, not fatal.
    pub fn pin_hottest(
        &self,
        keys: impl Iterator<Item = ExpertKey>,
        budget_bytes: u64,
    ) -> Result<u64> {
        let mut pinned = 0u64;
        for key in keys {
            if pinned >= budget_bytes {
                break;
            }
            match self.shared.load(key) {
                Ok(w) => {
                    let bytes = w.nbytes() as u64;
                    self.shared.cache.pin(key, Arc::new(w));
                    pinned += bytes;
                }
                Err(_) => continue,
            }
        }
        Ok(pinned)
    }
}

impl TieredStore for DiskExpertStore {
    fn get_expert(&self, key: ExpertKey) -> Result<Arc<ExpertWeights>> {
        *self
            .shared
            .usage
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(key)
            .or_insert(0) += 1;
        if let Some(w) = self.shared.cache.get(key) {
            self.shared.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(w);
        }
        self.shared.stats.misses.fetch_add(1, Ordering::Relaxed);
        let w = Arc::new(self.shared.load(key)?);
        self.shared.cache.insert(key, w.clone());
        Ok(w)
    }

    fn prefetch(&self, key: ExpertKey) {
        let Some(tx) = &self.prefetch_tx else {
            self.shared
                .stats
                .prefetch_dropped
                .fetch_add(1, Ordering::Relaxed);
            return;
        };
        if self.shared.cache.get(key).is_some() {
            return;
        }
        match tx.try_send(key) {
            Ok(()) => {
                self.shared
                    .stats
                    .prefetch_issued
                    .fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.shared
                    .stats
                    .prefetch_dropped
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn usage(&self) -> Vec<(ExpertKey, u64)> {
        self.shared
            .usage
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    fn stats(&self) -> StoreStatsSnapshot {
        self.shared.stats.snapshot()
    }
}

impl Drop for DiskExpertStore {
    fn drop(&mut self) {
        // Close the queue so workers observe RecvError and exit.
        self.prefetch_tx.take();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{write_safetensors_entries, TensorEntry};
    use undertow_core::{LruExpertCache, QTensor, QuantFormat};

    struct TestNaming;
    impl ExpertNaming for TestNaming {
        fn expert_tensor_names(&self, key: ExpertKey) -> [String; 3] {
            let p = format!("model.layers.{}.mlp.experts.{}", key.layer, key.expert);
            [
                format!("{p}.gate_proj.weight"),
                format!("{p}.up_proj.weight"),
                format!("{p}.down_proj.weight"),
            ]
        }
    }

    const H: usize = 8;
    const M: usize = 4;

    fn expert_value(layer: usize, expert: usize, k: usize) -> f32 {
        ((layer * 1000 + expert * 10 + k) as f32 * 0.001).sin()
    }

    fn write_test_model(dir: &std::path::Path, layers: usize, experts: usize) {
        let mut entries = Vec::new();
        for l in 0..layers {
            for e in 0..experts {
                let names = TestNaming.expert_tensor_names(ExpertKey {
                    layer: l,
                    expert: e,
                });
                let gate: Vec<f32> = (0..M * H).map(|k| expert_value(l, e, k)).collect();
                let up: Vec<f32> = (0..M * H).map(|k| expert_value(l, e, k + 1)).collect();
                let down: Vec<f32> = (0..H * M).map(|k| expert_value(l, e, k + 2)).collect();
                let g = QTensor::quantize(&gate, M, H, QuantFormat::Int4).unwrap();
                let u = QTensor::quantize(&up, M, H, QuantFormat::Int8).unwrap();
                let d = QTensor::quantize(&down, H, M, QuantFormat::Int4).unwrap();
                entries.extend(TensorEntry::from_qtensor(&names[0], &g));
                entries.extend(TensorEntry::from_qtensor(&names[1], &u));
                entries.extend(TensorEntry::from_qtensor(&names[2], &d));
            }
        }
        write_safetensors_entries(dir.join("model.safetensors"), &entries).unwrap();
    }

    fn open_store(dir: &std::path::Path, budget: usize, workers: usize) -> DiskExpertStore {
        let reader = Arc::new(ShardedModelReader::open(dir).unwrap());
        DiskExpertStore::new(
            reader,
            Arc::new(TestNaming),
            ExpertDims {
                hidden: H,
                moe_intermediate: M,
            },
            Arc::new(LruExpertCache::new(budget)),
            workers,
        )
    }

    #[test]
    fn loads_and_caches() {
        let dir = tempfile::tempdir().unwrap();
        write_test_model(dir.path(), 2, 3);
        let store = open_store(dir.path(), usize::MAX / 2, 0);
        let k = ExpertKey {
            layer: 1,
            expert: 2,
        };
        let a = store.get_expert(k).unwrap();
        let b = store.get_expert(k).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "second fetch must be the cached Arc");
        let s = store.stats();
        assert_eq!((s.hits, s.misses), (1, 1));
        assert!(s.bytes_read > 0);
        // Loaded weights are the quantized values we wrote.
        assert_eq!(a.gate_proj.format(), QuantFormat::Int4);
        assert_eq!(a.up_proj.format(), QuantFormat::Int8);
    }

    #[test]
    fn eviction_under_tiny_budget_still_serves_correct_weights() {
        let dir = tempfile::tempdir().unwrap();
        write_test_model(dir.path(), 1, 8);
        // Budget for roughly two experts: constant churn.
        let one = {
            let store = open_store(dir.path(), usize::MAX / 2, 0);
            store
                .get_expert(ExpertKey {
                    layer: 0,
                    expert: 0,
                })
                .unwrap()
                .nbytes()
        };
        let store = open_store(dir.path(), one * 2, 0);
        for round in 0..3 {
            for e in 0..8 {
                let k = ExpertKey {
                    layer: 0,
                    expert: e,
                };
                let w = store.get_expert(k).unwrap();
                // Spot-check content identity through quantization:
                // dequantized gate row 0 must match the source values.
                let deq = w.gate_proj.dequantize();
                let src: Vec<f32> = (0..H).map(|k2| expert_value(0, e, k2)).collect();
                let max = src.iter().fold(0f32, |m, &v| m.max(v.abs()));
                for (a, b) in deq[..H].iter().zip(&src) {
                    assert!((a - b).abs() <= max / 7.0, "round {round} expert {e}");
                }
            }
        }
        let s = store.stats();
        assert!(s.misses > 8, "evictions must force re-reads, stats: {s:?}");
    }

    #[test]
    fn missing_expert_is_an_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        write_test_model(dir.path(), 1, 2);
        let store = open_store(dir.path(), usize::MAX / 2, 0);
        assert!(store
            .get_expert(ExpertKey {
                layer: 5,
                expert: 0
            })
            .is_err());
    }

    #[test]
    fn concurrent_fetches_are_consistent() {
        let dir = tempfile::tempdir().unwrap();
        write_test_model(dir.path(), 2, 8);
        let store = Arc::new(open_store(dir.path(), 1 << 20, 0));
        let mut handles = Vec::new();
        for t in 0..8 {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..50 {
                    let k = ExpertKey {
                        layer: (t + i) % 2,
                        expert: (t * 3 + i) % 8,
                    };
                    let w = store.get_expert(k).unwrap();
                    assert_eq!(w.gate_proj.out_dim(), M);
                    assert_eq!(w.down_proj.out_dim(), H);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn prefetch_warms_cache() {
        let dir = tempfile::tempdir().unwrap();
        write_test_model(dir.path(), 1, 4);
        let store = open_store(dir.path(), usize::MAX / 2, 2);
        for e in 0..4 {
            store.prefetch(ExpertKey {
                layer: 0,
                expert: e,
            });
        }
        // Wait for the pool to drain (bounded, generous).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let done = (0..4).all(|e| {
                store
                    .shared
                    .cache
                    .get(ExpertKey {
                        layer: 0,
                        expert: e,
                    })
                    .is_some()
            });
            if done {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "prefetch timed out");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // All subsequent gets are hits.
        for e in 0..4 {
            store
                .get_expert(ExpertKey {
                    layer: 0,
                    expert: e,
                })
                .unwrap();
        }
        let s = store.stats();
        assert_eq!(s.misses, 0, "prefetched experts must not miss: {s:?}");
        assert_eq!(s.hits, 4);
        assert_eq!(s.prefetch_issued, 4);
    }
}
