//! DeepSeek-MoE architecture family adapter.
//!
//! One adapter covers GLM-5.2, Kimi K2 and DeepSeek-V3/V4: they share the
//! sigmoid `noaux_tc` router (group-limited top-k over bias-corrected
//! sigmoid scores) and MLA attention (compressed KV latent + partial
//! interleaved RoPE).
//!
//! Everything here is the *scalar reference path*: plain f32, full k/v
//! reconstruction, no KV cache tricks, no weight absorption, no SIMD.
//! Its only job is to be provably correct against a `transformers` oracle;
//! every future optimization is validated against this.

pub mod attention;
pub mod config;
pub mod loader;
pub mod model;
pub mod mtp;
pub mod naming;
pub mod router;

mod adapter;

pub use adapter::DeepseekMoeAdapter;
pub use config::DeepseekConfig;
pub use loader::{LoadOptions, StoreChoice};
pub use model::{DeepseekMoeModel, InferenceSession};
pub use mtp::{
    draft, draft_logits, generate_greedy_mtp, generate_mtp, speculative_loop,
    speculative_loop_with_sampler, MtpHead, MtpState, MtpStats,
};
pub use naming::{classify_tensor, DeepseekExpertNaming};
pub use router::DeepseekSigmoidRouter;
