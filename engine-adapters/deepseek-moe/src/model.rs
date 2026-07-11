//! Scalar reference forward pass for the DeepSeek-MoE family.
//!
//! Full-sequence, causal, no KV cache: `forward` recomputes attention over
//! the whole sequence, and `greedy_decode` re-runs `forward` per emitted
//! token. Quadratic and proud of it — this path exists to be *obviously
//! correct* for oracle validation, not fast. Incremental decode with the
//! compressed-KV cache is Phase 1.
//!
//! Routed-expert weights are only reachable through the [`TieredStore`]
//! trait object, even though the phase-0 store is RAM-resident: the MoE
//! block below is already written against the streaming boundary.

use std::sync::Arc;

use engine_core::adapter::RouterAdapter;
use engine_core::{ExpertKey, Tensor, TieredStore};
use engine_quant::{matmul, rmsnorm, silu};

use crate::attention::{mla_forward, MlaDims, MlaWeights};
use crate::config::DeepseekConfig;
use crate::router::DeepseekSigmoidRouter;

/// Plain SwiGLU MLP weights (dense layers, shared experts).
pub struct MlpWeights {
    /// `[inter, hidden]`
    pub gate_proj: Tensor,
    /// `[inter, hidden]`
    pub up_proj: Tensor,
    /// `[hidden, inter]`
    pub down_proj: Tensor,
}

impl MlpWeights {
    /// `out[s] = down( silu(gate(x[s])) * up(x[s]) )`, accumulated into `out`.
    fn forward_add(&self, x: &[f32], seq: usize, out: &mut [f32]) {
        let hidden = self.gate_proj.dim1();
        let inter = self.gate_proj.dim0();
        let mut g = vec![0.0f32; seq * inter];
        let mut u = vec![0.0f32; seq * inter];
        matmul(&mut g, x, &self.gate_proj.data, seq, hidden, inter);
        matmul(&mut u, x, &self.up_proj.data, seq, hidden, inter);
        for (gv, uv) in g.iter_mut().zip(&u) {
            *gv = silu(*gv) * uv;
        }
        let mut d = vec![0.0f32; seq * hidden];
        matmul(&mut d, &g, &self.down_proj.data, seq, inter, hidden);
        for (o, v) in out.iter_mut().zip(&d) {
            *o += v;
        }
    }
}

pub enum FfnBlock {
    Dense(MlpWeights),
    Moe {
        /// Router gate: `[n_routed_experts, hidden]`, kept f32 (small,
        /// numerically sensitive).
        gate: Tensor,
        /// `e_score_correction_bias`, `[n_routed_experts]`.
        correction_bias: Option<Vec<f32>>,
        /// Shared expert(s), fused as one MLP of width
        /// `n_shared * moe_intermediate`.
        shared: Option<MlpWeights>,
    },
}

pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub attn: MlaWeights,
    pub ffn: FfnBlock,
}

pub struct DeepseekMoeModel {
    pub cfg: DeepseekConfig,
    /// `[vocab, hidden]`
    pub embed_tokens: Tensor,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    /// `[vocab, hidden]`
    pub lm_head: Tensor,
    pub router: DeepseekSigmoidRouter,
    pub store: Arc<dyn TieredStore>,
}

impl DeepseekMoeModel {
    fn mla_dims(&self) -> MlaDims {
        let c = &self.cfg;
        MlaDims {
            hidden: c.hidden_size,
            num_heads: c.num_attention_heads,
            qk_nope: c.qk_nope_head_dim,
            qk_rope: c.qk_rope_head_dim,
            v_head: c.v_head_dim,
            kv_lora: c.kv_lora_rank,
            rope_theta: c.rope_theta(),
            rms_eps: c.rms_norm_eps,
            scale: c.attn_scale(),
        }
    }

    /// MoE block: route each position, pull experts through the tiered
    /// store, accumulate weighted outputs + shared expert. `x` is already
    /// post-attention-layernormed; result is accumulated into `out`.
    #[allow(clippy::too_many_arguments)]
    fn moe_forward(
        &self,
        layer_idx: usize,
        gate: &Tensor,
        correction_bias: Option<&[f32]>,
        shared: Option<&MlpWeights>,
        x: &[f32],
        seq: usize,
        out: &mut [f32],
    ) -> engine_core::Result<()> {
        let c = &self.cfg;
        let (hidden, inter) = (c.hidden_size, c.moe_intermediate_size);
        let n_experts = c.n_routed_experts;
        let mut logits = vec![0.0f32; n_experts];
        let mut g = vec![0.0f32; inter];
        let mut u = vec![0.0f32; inter];
        let mut d = vec![0.0f32; hidden];
        for s in 0..seq {
            let xs = &x[s * hidden..(s + 1) * hidden];
            matmul(&mut logits, xs, &gate.data, 1, hidden, n_experts);
            for choice in self.router.route(&logits, correction_bias) {
                let expert = self.store.get_expert(ExpertKey {
                    layer: layer_idx,
                    expert: choice.expert,
                })?;
                matmul(&mut g, xs, &expert.gate_proj.data, 1, hidden, inter);
                matmul(&mut u, xs, &expert.up_proj.data, 1, hidden, inter);
                for (gv, uv) in g.iter_mut().zip(&u) {
                    *gv = silu(*gv) * uv;
                }
                matmul(&mut d, &g, &expert.down_proj.data, 1, inter, hidden);
                let os = &mut out[s * hidden..(s + 1) * hidden];
                for (o, v) in os.iter_mut().zip(&d) {
                    *o += choice.weight * v;
                }
            }
        }
        if let Some(sh) = shared {
            sh.forward_add(x, seq, out);
        }
        Ok(())
    }

    /// Full-sequence teacher-forcing forward. Returns logits `[seq, vocab]`.
    pub fn forward(&self, token_ids: &[usize]) -> engine_core::Result<Tensor> {
        let c = &self.cfg;
        let (seq, hidden) = (token_ids.len(), c.hidden_size);
        let dims = self.mla_dims();

        let mut x = vec![0.0f32; seq * hidden];
        for (s, &id) in token_ids.iter().enumerate() {
            assert!(id < c.vocab_size, "token id {id} out of vocab");
            x[s * hidden..(s + 1) * hidden].copy_from_slice(self.embed_tokens.row(id));
        }

        let mut normed = vec![0.0f32; seq * hidden];
        for (li, layer) in self.layers.iter().enumerate() {
            for s in 0..seq {
                rmsnorm(
                    &mut normed[s * hidden..(s + 1) * hidden],
                    &x[s * hidden..(s + 1) * hidden],
                    &layer.input_norm,
                    c.rms_norm_eps,
                );
            }
            let attn_out = mla_forward(&dims, &layer.attn, &normed, seq);
            for (xv, av) in x.iter_mut().zip(&attn_out) {
                *xv += av;
            }

            for s in 0..seq {
                rmsnorm(
                    &mut normed[s * hidden..(s + 1) * hidden],
                    &x[s * hidden..(s + 1) * hidden],
                    &layer.post_attn_norm,
                    c.rms_norm_eps,
                );
            }
            match &layer.ffn {
                FfnBlock::Dense(mlp) => mlp.forward_add(&normed, seq, &mut x),
                FfnBlock::Moe {
                    gate,
                    correction_bias,
                    shared,
                } => {
                    // MoE writes into a zeroed buffer, then residual-adds:
                    // routed order (weighted) then shared, matching reference.
                    let mut moe_out = vec![0.0f32; seq * hidden];
                    self.moe_forward(
                        li,
                        gate,
                        correction_bias.as_deref(),
                        shared.as_ref(),
                        &normed,
                        seq,
                        &mut moe_out,
                    )?;
                    for (xv, mv) in x.iter_mut().zip(&moe_out) {
                        *xv += mv;
                    }
                }
            }
        }

        let mut logits = Tensor::zeros(vec![seq, c.vocab_size]);
        for s in 0..seq {
            rmsnorm(
                &mut normed[s * hidden..(s + 1) * hidden],
                &x[s * hidden..(s + 1) * hidden],
                &self.final_norm,
                c.rms_norm_eps,
            );
            matmul(
                &mut logits.data[s * c.vocab_size..(s + 1) * c.vocab_size],
                &normed[s * hidden..(s + 1) * hidden],
                &self.lm_head.data,
                1,
                hidden,
                c.vocab_size,
            );
        }
        Ok(logits)
    }

    /// Greedy decode by repeated full forward (oracle-scale only).
    pub fn greedy_decode(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        stop_ids: &[usize],
    ) -> engine_core::Result<Vec<usize>> {
        let mut ids = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let logits = self.forward(&ids)?;
            let last = logits.row(ids.len() - 1);
            let next = argmax(last);
            ids.push(next);
            if stop_ids.contains(&next) {
                break;
            }
        }
        Ok(ids)
    }
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
