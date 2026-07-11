//! Tiny DeepSeek-V3-architecture oracle model with random weights.
//!
//! The generated directory is loadable by both this engine and
//! `transformers.DeepseekV3ForCausalLM`. Dimensions deliberately exercise
//! the tricky config space: a dense-layer prefix, group-limited routing
//! (`n_group > 1`), shared expert, q-LoRA, `norm_topk_prob` and a
//! non-trivial `routed_scaling_factor`.

use std::path::Path;

use engine_core::{Result, Tensor};
use engine_io::write_safetensors;

use crate::rng::Pcg32;

#[derive(Debug, Clone)]
pub struct OracleSpec {
    pub seed: u64,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub first_k_dense_replace: usize,
    pub num_attention_heads: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Std-dev of the random weight matrices.
    pub init_std: f32,
}

impl Default for OracleSpec {
    fn default() -> Self {
        Self {
            seed: 20260712,
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            moe_intermediate_size: 32,
            num_hidden_layers: 4,
            first_k_dense_replace: 1,
            num_attention_heads: 4,
            n_routed_experts: 16,
            num_experts_per_tok: 4,
            n_shared_experts: 1,
            n_group: 4,
            topk_group: 2,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            q_lora_rank: 32,
            kv_lora_rank: 16,
            qk_nope_head_dim: 16,
            qk_rope_head_dim: 8,
            v_head_dim: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            init_std: 0.05,
        }
    }
}

impl OracleSpec {
    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "first_k_dense_replace": self.first_k_dense_replace,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_attention_heads,
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "n_shared_experts": self.n_shared_experts,
            "n_group": self.n_group,
            "topk_group": self.topk_group,
            "norm_topk_prob": self.norm_topk_prob,
            "routed_scaling_factor": self.routed_scaling_factor,
            "moe_layer_freq": 1,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "head_dim": self.qk_rope_head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "rope_interleave": true,
            "hidden_act": "silu",
            "max_position_embeddings": 4096,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "tie_word_embeddings": false,
            "torch_dtype": "float32",
            "use_cache": true,
            "num_nextn_predict_layers": 0
        })
    }
}

struct Gen {
    rng: Pcg32,
    std: f32,
    tensors: Vec<(String, Tensor)>,
}

impl Gen {
    /// Random weight matrix `[d0, d1]`, N(0, init_std).
    fn mat(&mut self, name: String, d0: usize, d1: usize) {
        let data = (0..d0 * d1)
            .map(|_| self.rng.normal_scaled(self.std))
            .collect();
        self.tensors.push((name, Tensor::new(vec![d0, d1], data)));
    }

    /// Norm weight: near 1 but not exactly 1, so a forward pass that
    /// ignores norm weights cannot silently pass the oracle test.
    fn norm(&mut self, name: String, n: usize) {
        let data = (0..n)
            .map(|_| 1.0 + (self.rng.uniform() - 0.5) * 0.2)
            .collect();
        self.tensors.push((name, Tensor::new(vec![n], data)));
    }

    fn vec1(&mut self, name: String, data: Vec<f32>) {
        let n = data.len();
        self.tensors.push((name, Tensor::new(vec![n], data)));
    }
}

/// Generate an oracle checkpoint (config.json + model.safetensors) in `dir`.
pub fn generate_oracle(dir: impl AsRef<Path>, spec: &OracleSpec) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;

    let s = spec;
    let (hd, heads) = (s.hidden_size, s.num_attention_heads);
    let qh = s.qk_nope_head_dim + s.qk_rope_head_dim;

    let mut g = Gen {
        rng: Pcg32::new(s.seed, 54),
        std: s.init_std,
        tensors: Vec::new(),
    };

    g.mat("model.embed_tokens.weight".into(), s.vocab_size, hd);
    for li in 0..s.num_hidden_layers {
        let p = format!("model.layers.{li}");
        g.norm(format!("{p}.input_layernorm.weight"), hd);
        g.norm(format!("{p}.post_attention_layernorm.weight"), hd);

        g.mat(format!("{p}.self_attn.q_a_proj.weight"), s.q_lora_rank, hd);
        g.norm(format!("{p}.self_attn.q_a_layernorm.weight"), s.q_lora_rank);
        g.mat(
            format!("{p}.self_attn.q_b_proj.weight"),
            heads * qh,
            s.q_lora_rank,
        );
        g.mat(
            format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
            s.kv_lora_rank + s.qk_rope_head_dim,
            hd,
        );
        g.norm(
            format!("{p}.self_attn.kv_a_layernorm.weight"),
            s.kv_lora_rank,
        );
        g.mat(
            format!("{p}.self_attn.kv_b_proj.weight"),
            heads * (s.qk_nope_head_dim + s.v_head_dim),
            s.kv_lora_rank,
        );
        g.mat(
            format!("{p}.self_attn.o_proj.weight"),
            hd,
            heads * s.v_head_dim,
        );

        if li < s.first_k_dense_replace {
            g.mat(format!("{p}.mlp.gate_proj.weight"), s.intermediate_size, hd);
            g.mat(format!("{p}.mlp.up_proj.weight"), s.intermediate_size, hd);
            g.mat(format!("{p}.mlp.down_proj.weight"), hd, s.intermediate_size);
        } else {
            g.mat(format!("{p}.mlp.gate.weight"), s.n_routed_experts, hd);
            // Distinct correction-bias values so bias-vs-weight confusion
            // in the router changes the selected experts and fails loudly.
            let e = s.n_routed_experts;
            let bias: Vec<f32> = (0..e)
                .map(|i| -0.1 + 0.2 * i as f32 / (e - 1).max(1) as f32)
                .collect();
            g.vec1(format!("{p}.mlp.gate.e_score_correction_bias"), bias);
            for ei in 0..e {
                let ep = format!("{p}.mlp.experts.{ei}");
                g.mat(
                    format!("{ep}.gate_proj.weight"),
                    s.moe_intermediate_size,
                    hd,
                );
                g.mat(format!("{ep}.up_proj.weight"), s.moe_intermediate_size, hd);
                g.mat(
                    format!("{ep}.down_proj.weight"),
                    hd,
                    s.moe_intermediate_size,
                );
            }
            if s.n_shared_experts > 0 {
                let si = s.n_shared_experts * s.moe_intermediate_size;
                let sp = format!("{p}.mlp.shared_experts");
                g.mat(format!("{sp}.gate_proj.weight"), si, hd);
                g.mat(format!("{sp}.up_proj.weight"), si, hd);
                g.mat(format!("{sp}.down_proj.weight"), hd, si);
            }
        }
    }
    g.norm("model.norm.weight".into(), hd);
    g.mat("lm_head.weight".into(), s.vocab_size, hd);

    let refs: Vec<(String, &Tensor)> = g.tensors.iter().map(|(n, t)| (n.clone(), t)).collect();
    write_safetensors(dir.join("model.safetensors"), &refs)?;
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&spec.config_json()).expect("static json"),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        generate_oracle(&a, &OracleSpec::default()).unwrap();
        generate_oracle(&b, &OracleSpec::default()).unwrap();
        let fa = std::fs::read(a.join("model.safetensors")).unwrap();
        let fb = std::fs::read(b.join("model.safetensors")).unwrap();
        assert_eq!(fa, fb);
    }
}
