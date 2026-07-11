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
pub mod router;

mod adapter;

pub use adapter::DeepseekMoeAdapter;
pub use config::DeepseekConfig;
pub use model::DeepseekMoeModel;
pub use router::DeepseekSigmoidRouter;
