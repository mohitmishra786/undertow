//! Mixtral family adapter.
//!
//! Architecture: GQA attention (optionally sliding-window), softmax top-k
//! router with weights always renormalized over the selected experts,
//! SwiGLU experts named `w1` (gate), `w3` (up), `w2` (down) under
//! `block_sparse_moe`. No shared experts, no dense-layer prefix, no MLA.
//! The forward pass lives in `moe-common`; this crate contributes config
//! parsing, tensor naming and conversion classification.

use std::path::Path;
use std::sync::Arc;

use engine_convert::Disposition;
use engine_core::adapter::ExpertNaming;
use engine_core::model::LoadOptions;
use engine_core::{EngineError, ExpertKey, Result, SoftmaxTopKRouter};
use moe_common::{load_gqa_model, GqaDims, GqaMoeModel, GqaMoeNaming, GqaMoeSpec};
use serde::Deserialize;

fn default_theta() -> f64 {
    1e6
}
fn default_eps() -> f32 {
    1e-5
}
fn default_max_pos() -> usize {
    4096
}

#[derive(Debug, Clone, Deserialize)]
pub struct MixtralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Expert FFN width (Mixtral has no separate moe_intermediate_size).
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

impl MixtralConfig {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let cfg: Self = serde_json::from_slice(&std::fs::read(&path)?)
            .map_err(|e| EngineError::InvalidConfig(format!("{}: {e}", path.display())))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn validate(&self) -> Result<()> {
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads.max(1))
        {
            return Err(EngineError::InvalidConfig(
                "inconsistent head geometry".into(),
            ));
        }
        if self.num_local_experts == 0 || self.num_experts_per_tok > self.num_local_experts {
            return Err(EngineError::InvalidConfig(
                "inconsistent expert counts".into(),
            ));
        }
        if !self.head_dim().is_multiple_of(2) {
            return Err(EngineError::InvalidConfig(
                "head_dim must be even for RoPE".into(),
            ));
        }
        Ok(())
    }

    fn spec(&self) -> GqaMoeSpec {
        GqaMoeSpec {
            architecture: "mixtral",
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            max_position_embeddings: self.max_position_embeddings,
            dims: GqaDims {
                hidden: self.hidden_size,
                num_heads: self.num_attention_heads,
                num_kv_heads: self.num_key_value_heads,
                head_dim: self.head_dim(),
                rope_theta: self.rope_theta as f32,
                rms_eps: self.rms_norm_eps,
                scale: 1.0 / (self.head_dim() as f32).sqrt(),
                sliding_window: self.sliding_window,
            },
            n_experts: self.num_local_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate: self.intermediate_size,
            dense_layers: Vec::new(),
            dense_intermediate: self.intermediate_size,
            tie_word_embeddings: self.tie_word_embeddings,
            eos_ids: self
                .eos_token_id
                .as_ref()
                .map(engine_core::model::parse_eos_ids)
                .unwrap_or_default(),
        }
    }
}

pub struct MixtralNaming;

impl ExpertNaming for MixtralNaming {
    fn expert_tensor_names(&self, key: ExpertKey) -> [String; 3] {
        let p = format!(
            "model.layers.{}.block_sparse_moe.experts.{}",
            key.layer, key.expert
        );
        // w1 = gate, w3 = up, w2 = down (Mixtral's naming).
        [
            format!("{p}.w1.weight"),
            format!("{p}.w3.weight"),
            format!("{p}.w2.weight"),
        ]
    }
}

impl GqaMoeNaming for MixtralNaming {
    fn router_gate(&self, layer: usize) -> String {
        format!("model.layers.{layer}.block_sparse_moe.gate.weight")
    }
}

const DENSE_SUFFIXES: &[&str] = &[
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
];

/// Conversion classification for Mixtral checkpoints.
pub fn classify_tensor(name: &str) -> Disposition {
    if let Some(layer) = expert_layer(name) {
        return Disposition::Expert { layer };
    }
    if name == "model.embed_tokens.weight"
        || name == "lm_head.weight"
        || DENSE_SUFFIXES.iter().any(|s| name.ends_with(s))
    {
        return Disposition::Dense;
    }
    // Norms and the router gate stay f32.
    Disposition::KeepF32
}

fn expert_layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, tail) = rest.split_once('.')?;
    if !tail.starts_with("block_sparse_moe.experts.") || !tail.ends_with(".weight") {
        return None;
    }
    layer_str.parse().ok()
}

pub fn load_model(dir: impl AsRef<Path>) -> Result<GqaMoeModel> {
    load_model_with(dir, &LoadOptions::default())
}

pub fn load_model_with(dir: impl AsRef<Path>, opts: &LoadOptions) -> Result<GqaMoeModel> {
    let cfg = MixtralConfig::from_dir(&dir)?;
    let spec = cfg.spec();
    let router = Box::new(SoftmaxTopKRouter {
        num_experts: cfg.num_local_experts,
        top_k: cfg.num_experts_per_tok,
        // Mixtral always renormalizes the selected weights.
        norm_topk_prob: true,
    });
    load_gqa_model(dir, spec, Arc::new(MixtralNaming), router, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_experts_and_router() {
        assert_eq!(
            classify_tensor("model.layers.3.block_sparse_moe.experts.5.w1.weight"),
            Disposition::Expert { layer: 3 }
        );
        assert_eq!(
            classify_tensor("model.layers.3.block_sparse_moe.gate.weight"),
            Disposition::KeepF32
        );
        assert_eq!(
            classify_tensor("model.layers.3.self_attn.k_proj.weight"),
            Disposition::Dense
        );
        assert_eq!(
            classify_tensor("model.layers.3.input_layernorm.weight"),
            Disposition::KeepF32
        );
    }

    #[test]
    fn naming_matches_classifier() {
        let names = MixtralNaming.expert_tensor_names(ExpertKey {
            layer: 7,
            expert: 2,
        });
        for n in names {
            assert_eq!(classify_tensor(&n), Disposition::Expert { layer: 7 });
        }
    }
}
