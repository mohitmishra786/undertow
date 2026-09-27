//! Shared machinery for MoE families with grouped-query attention:
//! Mixtral-style and Qwen-MoE-style models differ in tensor naming, router
//! normalization and two attention details (per-head q/k norms, sliding
//! window), so the forward pass, KV cache, session and loader live here
//! once and the family crates stay thin.

pub mod deltanet;
pub mod gqa;
pub mod model;

pub use deltanet::{
    deltanet_forward_cached, deltanet_forward_seq, deltanet_step, DeltaNetDims, DeltaNetState,
    DeltaNetWeights,
};
pub use gqa::{gqa_forward_cached, rope_neox, GqaDims, GqaKvCache, GqaWeights};
pub use model::{load_gqa_model, GqaMoeModel, GqaMoeNaming, GqaMoeSpec, GqaSession};
