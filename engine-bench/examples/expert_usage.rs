//! Print per-layer expert usage for the oracle fixture — a quick check
//! that the synthetic model exercises diverse routing (if every token
//! picked the same experts, the oracle test would prove much less).
//!
//!     cargo run -p engine-bench --example expert_usage

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use engine_core::{ExpertKey, ExpertWeights, StoreStatsSnapshot, TieredStore};

/// Delegates to the real store while counting every fetch.
struct CountingStore {
    inner: Arc<dyn TieredStore>,
    usage: Mutex<HashMap<ExpertKey, usize>>,
}

impl TieredStore for CountingStore {
    fn get_expert(&self, key: ExpertKey) -> engine_core::Result<Arc<ExpertWeights>> {
        *self.usage.lock().unwrap().entry(key).or_insert(0) += 1;
        self.inner.get_expert(key)
    }

    fn stats(&self) -> StoreStatsSnapshot {
        self.inner.stats()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/oracle-tiny");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("reference.json"))?)?;
    let ids: Vec<usize> = reference["full_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let mut model = deepseek_moe::loader::load_model(&dir)?;
    let counting = Arc::new(CountingStore {
        inner: model.store.clone(),
        usage: Mutex::new(HashMap::new()),
    });
    model.store = counting.clone();

    model.forward(&ids)?;

    let usage = counting.usage.lock().unwrap();
    let c = &model.cfg;
    for layer in c.first_k_dense_replace..c.num_hidden_layers {
        let counts: Vec<usize> = (0..c.n_routed_experts)
            .map(|e| *usage.get(&ExpertKey { layer, expert: e }).unwrap_or(&0))
            .collect();
        let used = counts.iter().filter(|&&n| n > 0).count();
        println!(
            "layer {layer}: {used}/{} experts used, assignments: {counts:?}",
            c.n_routed_experts
        );
    }
    Ok(())
}
