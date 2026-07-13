//! Architecture-agnostic core of the engine.
//!
//! This crate owns the trait boundaries that everything else plugs into:
//!
//! * [`adapter::ModelAdapter`] / [`adapter::RouterAdapter`] — describe one
//!   model architecture family (attention kind, router math, expert layout)
//!   without the core knowing anything about specific models.
//! * [`store::TieredStore`] — the disk → RAM weight hierarchy behind a
//!   single `get_expert` call. The disk-backed implementation lives in
//!   `undertow-io`; a RAM-resident one lives here for f32 checkpoints and
//!   tests.
//! * [`cache::ExpertCache`] — eviction policy for the RAM tier.
//!   [`cache::LruExpertCache`] is the byte-budgeted default with pinning.
//! * [`sample`] — deterministic token sampling (greedy, temperature,
//!   top-k, top-p) shared by every frontend.
//! * [`mem`] — physical-memory detection for cache auto-sizing.
//!
//! Design rule: the forward pass may only reach routed-expert weights
//! through `TieredStore`. That keeps the streaming boundary honest.

pub mod adapter;
pub mod cache;
pub mod error;
pub mod mem;
pub mod model;
pub mod profile;
pub mod router;
pub mod sample;
pub mod store;
pub mod tensor;

pub use adapter::{
    AttentionKind, ExpertLayout, ExpertNaming, ModelAdapter, MtpHeadSpec, RopeKind, RouterAdapter,
};
pub use cache::{ExpertCache, LruExpertCache, WeightedExpertCache};
pub use error::{EngineError, Result};
pub use model::{generate, greedy_decode, Model, Session};
pub use profile::ExpertProfile;
pub use router::SoftmaxTopKRouter;
pub use store::{
    ExpertKey, ExpertWeights, ResidentStore, StoreStats, StoreStatsSnapshot, TieredStore,
};
pub use tensor::Tensor;

// The quantized weight type is core vocabulary; re-export so most crates
// only need undertow-core.
pub use undertow_quant::{fast_int8_enabled, set_fast_int8, QTensor, QuantFormat};
