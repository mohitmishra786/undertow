//! HF-style `config.json` for the DeepSeek-MoE family.
//!
//! Field names follow the checkpoint configs (DeepSeek-V3, GLM MoE, Kimi
//! K2 all use this vocabulary). Values are range-validated before any
//! allocation is sized from them — configs arrive from untrusted mirrors.

use engine_core::{EngineError, Result};
use serde::Deserialize;

fn default_one() -> usize {
    1
}
fn default_scale() -> f32 {
    1.0
}
fn default_eps() -> f32 {
    1e-6
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepseekConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Dense-MLP intermediate size (first `first_k_dense_replace` layers).
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    pub num_attention_heads: usize,

    // MoE
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    #[serde(default = "default_one")]
    pub n_group: usize,
    #[serde(default = "default_one")]
    pub topk_group: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "default_scale")]
    pub routed_scaling_factor: f32,

    // MLA
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,

    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    /// DeepSeek-V3 style flat field…
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// …or GLM-5.2 style nested object.
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl DeepseekConfig {
    pub fn from_dir(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let bytes = std::fs::read(&path)?;
        let cfg: Self = serde_json::from_slice(&bytes)
            .map_err(|e| EngineError::InvalidConfig(format!("{}: {e}", path.display())))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
            .or(self.rope_parameters.as_ref().map(|r| r.rope_theta))
            .unwrap_or(10000.0) as f32
    }

    /// Per-head query/key dim: nope part + rope part.
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.qk_head_dim() as f32).sqrt()
    }

    pub fn validate(&self) -> Result<()> {
        macro_rules! ck {
            ($name:literal, $v:expr, $lo:expr, $hi:expr) => {
                if !($lo..=$hi).contains(&$v) {
                    return Err(EngineError::InvalidConfig(format!(
                        "{}={} out of range [{}, {}]",
                        $name, $v, $lo, $hi
                    )));
                }
            };
        }
        ck!("hidden_size", self.hidden_size, 1, 1 << 20);
        ck!("vocab_size", self.vocab_size, 1, 1 << 24);
        ck!("num_hidden_layers", self.num_hidden_layers, 1, 256);
        ck!("num_attention_heads", self.num_attention_heads, 1, 1024);
        ck!("n_routed_experts", self.n_routed_experts, 1, 4096);
        ck!(
            "num_experts_per_tok",
            self.num_experts_per_tok,
            1,
            self.n_routed_experts
        );
        ck!(
            "moe_intermediate_size",
            self.moe_intermediate_size,
            1,
            1 << 20
        );
        ck!("intermediate_size", self.intermediate_size, 1, 1 << 24);
        ck!(
            "first_k_dense_replace",
            self.first_k_dense_replace,
            0,
            self.num_hidden_layers
        );
        ck!("kv_lora_rank", self.kv_lora_rank, 1, 1 << 20);
        ck!("qk_nope_head_dim", self.qk_nope_head_dim, 1, 1 << 16);
        ck!("qk_rope_head_dim", self.qk_rope_head_dim, 2, 1 << 16);
        ck!("v_head_dim", self.v_head_dim, 1, 1 << 16);
        ck!("n_shared_experts", self.n_shared_experts, 0, 64);
        ck!("n_group", self.n_group, 1, self.n_routed_experts);
        ck!("topk_group", self.topk_group, 1, self.n_group);
        if let Some(q) = self.q_lora_rank {
            ck!("q_lora_rank", q, 1, 1 << 20);
        }
        if !self.qk_rope_head_dim.is_multiple_of(2) {
            return Err(EngineError::InvalidConfig(
                "qk_rope_head_dim must be even (interleaved RoPE)".into(),
            ));
        }
        if !self.n_routed_experts.is_multiple_of(self.n_group) {
            return Err(EngineError::InvalidConfig(
                "n_routed_experts must be divisible by n_group".into(),
            ));
        }
        Ok(())
    }
}
