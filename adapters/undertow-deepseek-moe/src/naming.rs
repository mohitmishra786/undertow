//! Tensor naming and conversion classification for the DeepSeek-MoE
//! family (HF checkpoint conventions, shared by GLM-5.2, Kimi K2 and
//! DeepSeek-V3/V4).

use undertow_convert::Disposition;
use undertow_core::adapter::ExpertNaming;
use undertow_core::ExpertKey;

/// HF tensor names for routed experts.
pub struct DeepseekExpertNaming;

impl ExpertNaming for DeepseekExpertNaming {
    fn expert_tensor_names(&self, key: ExpertKey) -> [String; 3] {
        let p = format!("model.layers.{}.mlp.experts.{}", key.layer, key.expert);
        [
            format!("{p}.gate_proj.weight"),
            format!("{p}.up_proj.weight"),
            format!("{p}.down_proj.weight"),
        ]
    }
}

/// 2-D projections that quantize to the dense format. Everything not
/// matched here or by the expert pattern stays f32 — the safe default for
/// norms, the router gate (`.mlp.gate.weight`, deliberately absent below),
/// biases, and any tensor a future checkpoint adds that we have not
/// audited.
const DENSE_SUFFIXES: &[&str] = &[
    ".self_attn.q_proj.weight",
    ".self_attn.q_a_proj.weight",
    ".self_attn.q_b_proj.weight",
    ".self_attn.kv_a_proj_with_mqa.weight",
    ".self_attn.kv_b_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".mlp.shared_experts.gate_proj.weight",
    ".mlp.shared_experts.up_proj.weight",
    ".mlp.shared_experts.down_proj.weight",
    // MTP layer extras (same suffixes cover any nextn layer index).
    ".eh_proj.weight",
    ".embed_tokens.weight",
    ".shared_head.head.weight",
];

/// Classify one tensor for conversion.
pub fn classify_tensor(name: &str) -> Disposition {
    if let Some(layer) = expert_layer(name) {
        return Disposition::Expert { layer };
    }
    if name == "lm_head.weight" {
        return Disposition::Dense;
    }
    // Suffix match keeps this correct for MTP layers too (their
    // projections share the same suffixes under a different layer index).
    // `.mlp.gate_proj.weight` matching must not swallow the router's
    // `.mlp.gate.weight`; suffixes are distinct strings so it cannot.
    if DENSE_SUFFIXES.iter().any(|s| name.ends_with(s)) {
        return Disposition::Dense;
    }
    Disposition::KeepF32
}

/// `model.layers.{L}.mlp.experts.{E}...` → `L`.
fn expert_layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, tail) = rest.split_once('.')?;
    if !tail.starts_with("mlp.experts.") || !tail.ends_with(".weight") {
        return None;
    }
    layer_str.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_tensors_classified_by_layer() {
        assert_eq!(
            classify_tensor("model.layers.7.mlp.experts.63.gate_proj.weight"),
            Disposition::Expert { layer: 7 }
        );
        assert_eq!(
            classify_tensor("model.layers.61.mlp.experts.0.down_proj.weight"),
            Disposition::Expert { layer: 61 }
        );
    }

    #[test]
    fn router_gate_stays_f32_but_gate_proj_is_dense() {
        assert_eq!(
            classify_tensor("model.layers.3.mlp.gate.weight"),
            Disposition::KeepF32
        );
        assert_eq!(
            classify_tensor("model.layers.0.mlp.gate_proj.weight"),
            Disposition::Dense
        );
        assert_eq!(
            classify_tensor("model.layers.3.mlp.gate.e_score_correction_bias"),
            Disposition::KeepF32
        );
    }

    #[test]
    fn attention_and_embeddings_are_dense() {
        for n in [
            "model.layers.5.self_attn.q_a_proj.weight",
            "model.layers.5.self_attn.kv_b_proj.weight",
            "model.layers.5.self_attn.o_proj.weight",
            "model.layers.5.mlp.shared_experts.up_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert_eq!(classify_tensor(n), Disposition::Dense, "{n}");
        }
    }

    #[test]
    fn norms_and_unknowns_stay_f32() {
        for n in [
            "model.layers.5.input_layernorm.weight",
            "model.layers.5.post_attention_layernorm.weight",
            "model.layers.5.self_attn.q_a_layernorm.weight",
            "model.layers.5.self_attn.kv_a_layernorm.weight",
            "model.norm.weight",
            "model.layers.61.enorm.weight",
            "some.future.tensor",
        ] {
            assert_eq!(classify_tensor(n), Disposition::KeepF32, "{n}");
        }
    }

    #[test]
    fn naming_matches_classifier() {
        // The names the store fetches must classify as experts of the
        // right layer, or a converted model would stream nothing.
        let names = DeepseekExpertNaming.expert_tensor_names(ExpertKey {
            layer: 12,
            expert: 200,
        });
        for n in names {
            assert_eq!(classify_tensor(&n), Disposition::Expert { layer: 12 });
        }
    }
}
