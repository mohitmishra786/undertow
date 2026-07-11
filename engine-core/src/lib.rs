//! Architecture-agnostic core of the engine.
//!
//! This crate owns the trait boundaries that everything else plugs into:
//!
//! * [`adapter::ModelAdapter`] / [`adapter::RouterAdapter`] — describe one
//!   model architecture family (attention kind, router math, expert layout)
//!   without the core knowing anything about specific models.
//! * [`store::TieredStore`] — the disk → RAM → (optional GPU) weight
//!   hierarchy behind a single `get_expert` call. Phase 0 ships a
//!   RAM-resident implementation; disk streaming replaces it without
//!   touching any caller.
//! * [`cache::ExpertCache`] — eviction policy for the RAM tier (LRU first,
//!   importance-weighted later).
//!
//! Design rule: the forward pass may only reach routed-expert weights
//! through `TieredStore`, even while the phase-0 store is a HashMap. That
//! keeps the streaming boundary honest from day one.

pub mod adapter;
pub mod cache;
pub mod error;
pub mod scheduler;
pub mod store;
pub mod tensor;

pub use adapter::{
    AttentionKind, ExpertLayout, ModelAdapter, MtpHeadSpec, RopeKind, RouterAdapter,
};
pub use cache::ExpertCache;
pub use error::{EngineError, Result};
pub use store::{ExpertKey, ExpertWeights, TieredStore};
pub use tensor::Tensor;
