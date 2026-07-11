//! The two traits that make the engine architecture-pluggable.
//!
//! A `ModelAdapter` *describes* one architecture family (DeepSeek-MoE,
//! Mixtral, Qwen-MoE, ...): how many layers, which attention variant, how
//! experts are laid out, and which router math applies. The core runtime
//! never branches on a model name — it only consumes these descriptions.
//!
//! A `RouterAdapter` implements the gating function itself. It is
//! deliberately scoped to *pure selection math*: it receives pre-computed
//! gate logits (`hidden · W_gate^T` is an ordinary matmul the runtime owns)
//! and returns the chosen experts with their combination weights. That
//! keeps every I/O and caching concern out of router implementations, and
//! makes routers trivially unit-testable against hand-computed values.

use std::ops::Range;

/// RoPE variant applied inside attention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeKind {
    /// DeepSeek/GLM MLA style: only the last `qk_rope_head_dim` dims of each
    /// query head (and the shared key-rot vector) are rotated, reading
    /// interleaved (even, odd) pairs and writing split halves.
    InterleavedPartial { theta: f32 },
    /// Llama-style rotate-half over the full head dim (future families).
    NeoxFull { theta: f32 },
}

/// Attention family. Dimensions here are per-head unless noted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionKind {
    /// Multi-head Latent Attention (DeepSeek-V3/V4, GLM-5.2, Kimi K2).
    /// KV is compressed to a `kv_lora_rank` latent per token; k/v per head
    /// are reconstructed through `kv_b_proj`.
    Mla {
        num_heads: usize,
        /// `None` = direct `q_proj` (e.g. DeepSeek-V2-Lite); `Some(r)` =
        /// q_a/q_b low-rank path with an RMSNorm between.
        q_lora_rank: Option<usize>,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        rope: RopeKind,
    },
    /// Grouped-query attention (future families).
    Gqa {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: RopeKind,
    },
}

/// How routed + shared experts are laid out in one MoE layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertLayout {
    pub num_routed_experts: usize,
    pub num_experts_per_token: usize,
    pub num_shared_experts: usize,
    /// Intermediate (FFN) width of one routed expert.
    pub moe_intermediate_size: usize,
}

/// Native multi-token-prediction head, when the model ships one
/// (e.g. GLM-5.2's extra MTP layer). Consumed in Phase 2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpHeadSpec {
    /// Number of extra transformer layers dedicated to the MTP head.
    pub num_layers: usize,
    /// Tokens predicted ahead per step.
    pub num_predict: usize,
}

/// One expert chosen by the router for one token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertChoice {
    pub expert: usize,
    /// Final combination weight (already normalized / scaled — the caller
    /// multiplies the expert output by exactly this).
    pub weight: f32,
}

/// Gating function of one MoE architecture family.
///
/// Implementations must be deterministic pure functions of their inputs.
pub trait RouterAdapter: Send + Sync {
    fn num_experts(&self) -> usize;

    fn top_k(&self) -> usize;

    /// Select experts for one token.
    ///
    /// * `gate_logits` — raw router logits, `hidden · W_gate^T`, length
    ///   [`Self::num_experts`]. The matmul is done by the caller.
    /// * `correction_bias` — optional per-expert selection bias
    ///   (DeepSeek `e_score_correction_bias`). Affects *which* experts are
    ///   chosen, never their weights.
    ///
    /// Returns exactly [`Self::top_k`] choices, in selection order.
    fn route(&self, gate_logits: &[f32], correction_bias: Option<&[f32]>) -> Vec<ExpertChoice>;
}

/// Describes one model architecture family to the runtime.
pub trait ModelAdapter: Send + Sync {
    /// Stable architecture id, e.g. `"deepseek_v3"`.
    fn architecture(&self) -> &'static str;

    fn num_layers(&self) -> usize;

    /// Layers whose FFN is a plain dense MLP instead of MoE
    /// (DeepSeek's `first_k_dense_replace` prefix).
    fn dense_layer_range(&self) -> Range<usize>;

    fn attention(&self) -> AttentionKind;

    /// Router for MoE layers. Boxed so one adapter can be shared while each
    /// caller owns its router instance.
    fn router(&self) -> Box<dyn RouterAdapter>;

    fn expert_layout(&self) -> ExpertLayout;

    /// Present only for models that ship a native MTP head.
    fn mtp_head(&self) -> Option<MtpHeadSpec> {
        None
    }
}
