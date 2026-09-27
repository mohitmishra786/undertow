//! Qwen3-MoE family adapter.
//!
//! Architecture: GQA attention with per-head RMSNorm on q and k before
//! RoPE, an explicit `head_dim` decoupled from `hidden/heads`, and a
//! softmax top-k router with `norm_topk_prob`. Layers can be forced dense
//! via `mlp_only_layers` or thinned via `decoder_sparse_step`. The forward
//! pass lives in `moe-common`; this crate contributes config parsing,
//! tensor naming and conversion classification.

use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use undertow_convert::Disposition;
use undertow_core::adapter::ExpertNaming;
use undertow_core::model::LoadOptions;
use undertow_core::{EngineError, ExpertKey, Result, SoftmaxTopKRouter};
use undertow_moe_common::{load_gqa_model, GqaDims, GqaMoeModel, GqaMoeNaming, GqaMoeSpec};

fn default_theta() -> f64 {
    10000.0
}
fn default_eps() -> f32 {
    1e-6
}
fn default_max_pos() -> usize {
    4096
}
fn default_one() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
pub struct QwenMoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Dense-MLP width for `mlp_only_layers`.
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit per-head dim (decoupled from hidden/heads in Qwen3; defaults to hidden/heads in Qwen2.5).
    #[serde(default)]
    pub head_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default = "default_one")]
    pub decoder_sparse_step: usize,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

impl QwenMoeConfig {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let bytes = std::fs::read(&path)?;
        Self::from_slice(&bytes)
            .map_err(|e| EngineError::InvalidConfig(format!("{}: {e}", path.display())))
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let mut cfg: Self =
            serde_json::from_slice(bytes).map_err(|e| EngineError::InvalidConfig(e.to_string()))?;
        if cfg.head_dim == 0 && cfg.num_attention_heads > 0 {
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(EngineError::InvalidConfig(
                "inconsistent head geometry".into(),
            ));
        }
        if self.num_experts == 0 || self.num_experts_per_tok > self.num_experts {
            return Err(EngineError::InvalidConfig(
                "inconsistent expert counts".into(),
            ));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(EngineError::InvalidConfig(
                "head_dim must be even for RoPE".into(),
            ));
        }
        if self.decoder_sparse_step == 0 {
            return Err(EngineError::InvalidConfig(
                "decoder_sparse_step must be >= 1".into(),
            ));
        }
        Ok(())
    }

    /// HF semantics: layer `l` is sparse when it is not in
    /// `mlp_only_layers` and `(l + 1) % decoder_sparse_step == 0`.
    fn dense_layers(&self) -> Vec<usize> {
        (0..self.num_hidden_layers)
            .filter(|&l| {
                self.mlp_only_layers.contains(&l)
                    || !(l + 1).is_multiple_of(self.decoder_sparse_step)
            })
            .collect()
    }

    fn spec(&self) -> GqaMoeSpec {
        GqaMoeSpec {
            architecture: "qwen_moe",
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            max_position_embeddings: self.max_position_embeddings,
            dims: GqaDims {
                hidden: self.hidden_size,
                num_heads: self.num_attention_heads,
                num_kv_heads: self.num_key_value_heads,
                head_dim: self.head_dim,
                rope_theta: self.rope_theta as f32,
                rms_eps: self.rms_norm_eps,
                scale: 1.0 / (self.head_dim as f32).sqrt(),
                sliding_window: None,
            },
            n_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate: self.moe_intermediate_size,
            dense_layers: self.dense_layers(),
            dense_intermediate: self.intermediate_size,
            tie_word_embeddings: self.tie_word_embeddings,
            eos_ids: self
                .eos_token_id
                .as_ref()
                .map(undertow_core::model::parse_eos_ids)
                .unwrap_or_default(),
        }
    }
}

pub struct QwenMoeNaming;

impl ExpertNaming for QwenMoeNaming {
    fn expert_tensor_names(&self, key: ExpertKey) -> [String; 3] {
        let p = format!("model.layers.{}.mlp.experts.{}", key.layer, key.expert);
        [
            format!("{p}.gate_proj.weight"),
            format!("{p}.up_proj.weight"),
            format!("{p}.down_proj.weight"),
        ]
    }
}

impl GqaMoeNaming for QwenMoeNaming {
    fn router_gate(&self, layer: usize) -> String {
        format!("model.layers.{layer}.mlp.gate.weight")
    }

    fn qk_norm(&self, layer: usize) -> Option<(String, String)> {
        Some((
            format!("model.layers.{layer}.self_attn.q_norm.weight"),
            format!("model.layers.{layer}.self_attn.k_norm.weight"),
        ))
    }
}

const DENSE_SUFFIXES: &[&str] = &[
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
];

/// Conversion classification for Qwen3-MoE checkpoints.
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
    // Norms (including q_norm/k_norm) and the router gate stay f32.
    Disposition::KeepF32
}

fn expert_layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, tail) = rest.split_once('.')?;
    if !tail.starts_with("mlp.experts.") || !tail.ends_with(".weight") {
        return None;
    }
    layer_str.parse().ok()
}

pub fn load_model(dir: impl AsRef<Path>) -> Result<GqaMoeModel> {
    load_model_with(dir, &LoadOptions::default())
}

pub fn load_model_with(dir: impl AsRef<Path>, opts: &LoadOptions) -> Result<GqaMoeModel> {
    let cfg = QwenMoeConfig::from_dir(&dir)?;
    let spec = cfg.spec();
    let router = Box::new(SoftmaxTopKRouter {
        num_experts: cfg.num_experts,
        top_k: cfg.num_experts_per_tok,
        norm_topk_prob: cfg.norm_topk_prob,
    });
    load_gqa_model(dir, spec, Arc::new(QwenMoeNaming), router, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_covers_the_gate_trap_and_qk_norms() {
        assert_eq!(
            classify_tensor("model.layers.2.mlp.experts.9.up_proj.weight"),
            Disposition::Expert { layer: 2 }
        );
        assert_eq!(
            classify_tensor("model.layers.2.mlp.gate.weight"),
            Disposition::KeepF32
        );
        assert_eq!(
            classify_tensor("model.layers.2.mlp.gate_proj.weight"),
            Disposition::Dense
        );
        assert_eq!(
            classify_tensor("model.layers.2.self_attn.q_norm.weight"),
            Disposition::KeepF32
        );
        assert_eq!(
            classify_tensor("model.layers.2.self_attn.q_proj.weight"),
            Disposition::Dense
        );
    }

    #[test]
    fn dense_layer_selection_follows_hf_semantics() {
        let cfg = QwenMoeConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            moe_intermediate_size: 8,
            num_hidden_layers: 6,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            mlp_only_layers: vec![3],
            decoder_sparse_step: 2,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            tie_word_embeddings: false,
            eos_token_id: None,
        };
        // Sparse layers: (l+1) % 2 == 0 and not in mlp_only_layers:
        // l = 1, 5 sparse; l = 3 would be sparse but is forced dense.
        assert_eq!(cfg.dense_layers(), vec![0, 2, 3, 4]);
    }

    #[test]
    fn naming_matches_classifier() {
        let names = QwenMoeNaming.expert_tensor_names(ExpertKey {
            layer: 11,
            expert: 63,
        });
        for n in names {
            assert_eq!(classify_tensor(&n), Disposition::Expert { layer: 11 });
        }
    }

    #[test]
    fn parse_qwen2_moe_config_without_explicit_head_dim() {
        let json = serde_json::json!({
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "moe_intermediate_size": 1408,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "norm_topk_prob": true,
            "decoder_sparse_step": 1,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 32768,
        });
        let bytes = serde_json::to_vec(&json).unwrap();
        let cfg = QwenMoeConfig::from_slice(&bytes).unwrap();
        // 2048 / 16 = 128
        assert_eq!(cfg.head_dim, 128);
    }
}
