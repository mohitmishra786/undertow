//! `TieredStore`: the single boundary through which the forward pass
//! reaches routed-expert weights.
//!
//! Phase 0 implements it with an in-RAM map ([`ResidentStore`]). Phase 1
//! replaces that with a pread-based disk tier + [`crate::cache::ExpertCache`]
//! without changing any caller. The trait deliberately does not assume
//! single-node storage (Phase 4: LAN pooling).

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{EngineError, Result};
use crate::tensor::Tensor;

/// Address of one routed expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: usize,
    pub expert: usize,
}

/// The three FFN matrices of one (SwiGLU) expert.
///
/// Scalar-reference phase: dequantized f32. Quantized in-cache storage
/// arrives with `engine-quant`'s real kernels.
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    /// `[moe_intermediate, hidden]`
    pub gate_proj: Tensor,
    /// `[moe_intermediate, hidden]`
    pub up_proj: Tensor,
    /// `[hidden, moe_intermediate]`
    pub down_proj: Tensor,
}

pub trait TieredStore: Send + Sync {
    /// Fetch one expert's weights, wherever they currently live.
    /// `Arc` so cache tiers and compute can share without copying.
    fn get_expert(&self, key: ExpertKey) -> Result<Arc<ExpertWeights>>;

    /// Non-binding hint that `key` will likely be needed soon
    /// (speculative prefetch path). Default: no-op.
    fn prefetch(&self, _key: ExpertKey) {}
}

/// Phase-0 store: every expert resident in RAM. Exists so the forward pass
/// exercises the `TieredStore` boundary before disk streaming lands.
#[derive(Default)]
pub struct ResidentStore {
    experts: HashMap<ExpertKey, Arc<ExpertWeights>>,
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
}

impl TieredStore for ResidentStore {
    fn get_expert(&self, key: ExpertKey) -> Result<Arc<ExpertWeights>> {
        self.experts
            .get(&key)
            .cloned()
            .ok_or(EngineError::ExpertNotFound(key))
    }
}
