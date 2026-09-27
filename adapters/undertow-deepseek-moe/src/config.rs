//! HF-style `config.json` for the DeepSeek-MoE family.
//!
//! Field names follow the checkpoint configs (DeepSeek-V3, GLM MoE, Kimi
//! K2 all use this vocabulary). Values are range-validated before any
//! allocation is sized from them — configs arrive from untrusted mirrors.

use serde::Deserialize;
use undertow_core::{EngineError, Result};

fn default_one() -> usize {
    1
}
fn default_scale() -> f32 {
    1.0
}
fn default_eps() -> f32 {
    1e-6
}
fn default_max_pos() -> usize {
    4096
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f64>,
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
    #[serde(default)]
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
    /// …or GLM-5.2 style nested object…
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    /// …or DeepSeek-V4/V2 style rope_scaling dict / nested frequency configs.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    /// Extra MTP layer count after the main stack (0 or 1 supported).
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    /// Number or array in checkpoints; parsed via [`Self::eos_ids`].
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

/// Extract stop-token ids from an `eos_token_id` json value (number or
/// array of numbers; anything else yields none).
pub fn parse_eos_ids(v: &serde_json::Value) -> Vec<usize> {
    match v {
        serde_json::Value::Number(n) => n.as_u64().map(|x| x as usize).into_iter().collect(),
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_u64().map(|u| u as usize))
            .collect(),
        _ => Vec::new(),
    }
}

impl DeepseekConfig {
    pub fn from_dir(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let bytes = std::fs::read(&path)?;
        Self::from_slice(&bytes).map_err(|e| EngineError::InvalidConfig(format!("{path:?}: {e}")))
    }

    /// Parse and validate from raw json bytes (also the fuzzing entry).
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let cfg: Self =
            serde_json::from_slice(bytes).map_err(|e| EngineError::InvalidConfig(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn rope_theta(&self) -> f32 {
        if let Some(t) = self.rope_theta {
            return t as f32;
        }
        if let Some(ref p) = self.rope_parameters {
            if let Some(t) = p.get("rope_theta").and_then(|v| v.as_f64()) {
                return t as f32;
            }
            if let Some(t) = p.get("theta").and_then(|v| v.as_f64()) {
                return t as f32;
            }
        }
        if let Some(ref s) = self.rope_scaling {
            if let Some(t) = s.get("rope_theta").and_then(|v| v.as_f64()) {
                return t as f32;
            }
            if let Some(t) = s.get("theta").and_then(|v| v.as_f64()) {
                return t as f32;
            }
        }
        10000.0
    }

    /// Per-head query/key dim: nope part + rope part.
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.qk_head_dim() as f32).sqrt()
    }

    /// Stop tokens declared in config.json.
    pub fn eos_ids(&self) -> Vec<usize> {
        self.eos_token_id
            .as_ref()
            .map(parse_eos_ids)
            .unwrap_or_default()
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
        ck!(
            "max_position_embeddings",
            self.max_position_embeddings,
            1,
            1 << 27
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config_json() -> serde_json::Value {
        serde_json::json!({
            "vocab_size": 1000,
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "n_routed_experts": 16,
            "num_experts_per_tok": 2,
            "kv_lora_rank": 64,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 16,
            "v_head_dim": 32,
        })
    }

    #[test]
    fn parse_with_rope_scaling_yarn_dict() {
        let mut val = base_config_json();
        val["rope_scaling"] = serde_json::json!({
            "type": "yarn",
            "factor": 40.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        });
        let bytes = serde_json::to_vec(&val).unwrap();
        let cfg = DeepseekConfig::from_slice(&bytes).unwrap();
        assert_eq!(cfg.rope_theta(), 10000.0);
    }

    #[test]
    fn parse_with_rope_scaling_explicit_theta() {
        let mut val = base_config_json();
        val["rope_scaling"] = serde_json::json!({
            "type": "linear",
            "rope_theta": 25000.0,
        });
        let bytes = serde_json::to_vec(&val).unwrap();
        let cfg = DeepseekConfig::from_slice(&bytes).unwrap();
        assert_eq!(cfg.rope_theta(), 25000.0);
    }

    #[test]
    fn parse_with_nested_rope_parameters() {
        let mut val = base_config_json();
        val["rope_parameters"] = serde_json::json!({
            "rope_theta": 50000.0,
            "freq_config": {
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
            }
        });
        let bytes = serde_json::to_vec(&val).unwrap();
        let cfg = DeepseekConfig::from_slice(&bytes).unwrap();
        assert_eq!(cfg.rope_theta(), 50000.0);
    }

    #[test]
    fn parse_deepseek_v2_without_q_lora_rank() {
        let val = base_config_json();
        assert!(val.get("q_lora_rank").is_none());
        let bytes = serde_json::to_vec(&val).unwrap();
        let cfg = DeepseekConfig::from_slice(&bytes).unwrap();
        assert_eq!(cfg.q_lora_rank, None);
    }
}
