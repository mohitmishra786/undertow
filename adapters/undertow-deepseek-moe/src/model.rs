//! DeepSeek-MoE family model: weights, forward pass, inference sessions.
//!
//! Weights are [`QTensor`]s throughout, so the same code runs a plain f32
//! oracle checkpoint and a converted int8/int4 one. Numerically sensitive
//! pieces (norms, router gate, correction bias) are always f32.
//!
//! Routed experts are only reachable through [`TieredStore`]; on a
//! converted checkpoint that is the disk-streaming store, on an f32
//! oracle a RAM-resident one. During single-token decode the router of
//! layer L+1 is speculatively evaluated on layer L's post-attention state
//! and its top-k experts prefetched — a pure I/O hint that never affects
//! output (measured recall of this predictor on GLM-5.2 class models is
//! high enough to hide most of the expert-fetch latency behind compute).

use std::sync::Arc;

use undertow_core::adapter::RouterAdapter;
use undertow_core::{EngineError, ExpertKey, QTensor, Tensor, TieredStore};
use undertow_quant::{matmul, rmsnorm, silu};

use crate::attention::{mla_forward_cached, AttnPath, LayerKvCache, MlaDims, MlaWeights};
use crate::config::DeepseekConfig;
use crate::router::DeepseekSigmoidRouter;

pub use undertow_core::sample::argmax;

/// Plain SwiGLU MLP weights (dense layers, shared experts).
pub struct MlpWeights {
    /// `[inter, hidden]`
    pub gate_proj: QTensor,
    /// `[inter, hidden]`
    pub up_proj: QTensor,
    /// `[hidden, inter]`
    pub down_proj: QTensor,
}

impl MlpWeights {
    pub fn nbytes(&self) -> usize {
        self.gate_proj.nbytes() + self.up_proj.nbytes() + self.down_proj.nbytes()
    }

    /// `out[s] += down( silu(gate(x[s])) * up(x[s]) )`.
    fn forward_add(&self, x: &[f32], seq: usize, out: &mut [f32]) {
        let hidden = self.gate_proj.in_dim();
        let inter = self.gate_proj.out_dim();
        let mut g = vec![0.0f32; seq * inter];
        let mut u = vec![0.0f32; seq * inter];
        self.gate_proj.matmul(&mut g, x, seq);
        self.up_proj.matmul(&mut u, x, seq);
        for (gv, uv) in g.iter_mut().zip(&u) {
            *gv = silu(*gv) * uv;
        }
        let mut d = vec![0.0f32; seq * hidden];
        self.down_proj.matmul(&mut d, &g, seq);
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
    pub embed_tokens: QTensor,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    /// `[vocab, hidden]`
    pub lm_head: QTensor,
    pub router: DeepseekSigmoidRouter,
    pub store: Arc<dyn TieredStore>,
    /// Stop-token ids from config.json / generation_config.json.
    pub stop_ids: Vec<usize>,
    /// Bytes of RAM-resident dense weights (cache budgeting, stats).
    pub dense_bytes: usize,
    /// Native multi-token-prediction head, when the checkpoint ships one.
    pub mtp: Option<crate::mtp::MtpHead>,
    /// Usable context, possibly lowered from the config maximum by a
    /// session KV budget.
    pub effective_max_context: usize,
}

impl DeepseekMoeModel {
    pub fn mla_dims(&self) -> MlaDims {
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

    pub(crate) fn embed_row(&self, id: usize, out: &mut [f32]) {
        out.iter_mut().for_each(|v| *v = 0.0);
        self.embed_tokens.add_scaled_row(id, 1.0, out);
    }

    /// Run one transformer layer in place over `x` `[seq, hidden]`.
    /// `layer_idx` addresses this layer's experts in the store (the MTP
    /// layer lives at index `num_hidden_layers`).
    pub(crate) fn run_layer(
        &self,
        layer: &LayerWeights,
        layer_idx: usize,
        x: &mut [f32],
        seq: usize,
        kv: &mut LayerKvCache,
        path: AttnPath,
    ) -> undertow_core::Result<()> {
        let c = &self.cfg;
        let hidden = c.hidden_size;
        let dims = self.mla_dims();
        let mut normed = vec![0.0f32; seq * hidden];
        for i in 0..seq {
            rmsnorm(
                &mut normed[i * hidden..(i + 1) * hidden],
                &x[i * hidden..(i + 1) * hidden],
                &layer.input_norm,
                c.rms_norm_eps,
            );
        }
        let attn_out = mla_forward_cached(&dims, &layer.attn, &normed, seq, kv, path);
        for (xv, av) in x.iter_mut().zip(&attn_out) {
            *xv += av;
        }
        for i in 0..seq {
            rmsnorm(
                &mut normed[i * hidden..(i + 1) * hidden],
                &x[i * hidden..(i + 1) * hidden],
                &layer.post_attn_norm,
                c.rms_norm_eps,
            );
        }
        match &layer.ffn {
            FfnBlock::Dense(mlp) => mlp.forward_add(&normed, seq, x),
            FfnBlock::Moe {
                gate,
                correction_bias,
                shared,
            } => {
                let mut moe_out = vec![0.0f32; seq * hidden];
                self.moe_forward(
                    layer_idx,
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
        Ok(())
    }

    /// MoE block for `seq` positions of `x` (post-attention-layernormed),
    /// accumulated into `out` (which must be zeroed by the caller).
    ///
    /// Positions are grouped by expert so each expert is fetched once and
    /// multiplied as one batched matmul, and experts are computed on the
    /// rayon pool (which also overlaps their disk fetches). Accumulation
    /// into `out` happens afterwards, per position, in the router's
    /// original choice order, so results are bit-identical to the naive
    /// per-position loop.
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
    ) -> undertow_core::Result<()> {
        use rayon::prelude::*;
        let c = &self.cfg;
        let (hidden, inter) = (c.hidden_size, c.moe_intermediate_size);
        let n_experts = c.n_routed_experts;

        // Route every position first.
        let mut logits = vec![0.0f32; n_experts];
        let mut routes = Vec::with_capacity(seq);
        for s in 0..seq {
            let xs = &x[s * hidden..(s + 1) * hidden];
            matmul(&mut logits, xs, &gate.data, 1, hidden, n_experts);
            routes.push(self.router.route(&logits, correction_bias));
        }

        // Group positions by expert, first-appearance order.
        let mut order: Vec<usize> = Vec::new();
        let mut groups: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (s, choices) in routes.iter().enumerate() {
            for choice in choices {
                groups
                    .entry(choice.expert)
                    .or_insert_with(|| {
                        order.push(choice.expert);
                        Vec::new()
                    })
                    .push(s);
            }
        }

        // Batched expert computation, one task per unique expert.
        let results: Vec<undertow_core::Result<(usize, Vec<f32>)>> = order
            .par_iter()
            .map(|&e| {
                let positions = &groups[&e];
                let n = positions.len();
                let expert = self.store.get_expert(ExpertKey {
                    layer: layer_idx,
                    expert: e,
                })?;
                let mut xs_g = vec![0.0f32; n * hidden];
                for (row, &s) in positions.iter().enumerate() {
                    xs_g[row * hidden..(row + 1) * hidden]
                        .copy_from_slice(&x[s * hidden..(s + 1) * hidden]);
                }
                let mut g = vec![0.0f32; n * inter];
                let mut u = vec![0.0f32; n * inter];
                expert.gate_proj.matmul(&mut g, &xs_g, n);
                expert.up_proj.matmul(&mut u, &xs_g, n);
                for (gv, uv) in g.iter_mut().zip(&u) {
                    *gv = silu(*gv) * uv;
                }
                let mut d = vec![0.0f32; n * hidden];
                expert.down_proj.matmul(&mut d, &g, n);
                Ok((e, d))
            })
            .collect();
        let mut by_expert = std::collections::HashMap::with_capacity(order.len());
        for r in results {
            let (e, d) = r?;
            by_expert.insert(e, d);
        }

        // Accumulate in the original per-position choice order.
        for (s, choices) in routes.iter().enumerate() {
            for choice in choices {
                let d = &by_expert[&choice.expert];
                let row = groups[&choice.expert]
                    .iter()
                    .position(|&p| p == s)
                    .expect("position present in its own group");
                let os = &mut out[s * hidden..(s + 1) * hidden];
                for (o, v) in os.iter_mut().zip(&d[row * hidden..(row + 1) * hidden]) {
                    *o += choice.weight * v;
                }
            }
        }
        if let Some(sh) = shared {
            sh.forward_add(x, seq, out);
        }
        Ok(())
    }

    /// Speculatively route layer `target` on hidden state `x` and hint the
    /// store. Pure I/O optimization: results are never used for compute.
    fn pilot_prefetch(&self, target: usize, x: &[f32]) {
        let layer = &self.layers[target];
        let FfnBlock::Moe {
            gate,
            correction_bias,
            ..
        } = &layer.ffn
        else {
            return;
        };
        let c = &self.cfg;
        let mut nrm = vec![0.0f32; c.hidden_size];
        rmsnorm(&mut nrm, x, &layer.post_attn_norm, c.rms_norm_eps);
        let mut logits = vec![0.0f32; c.n_routed_experts];
        matmul(
            &mut logits,
            &nrm,
            &gate.data,
            1,
            c.hidden_size,
            c.n_routed_experts,
        );
        for choice in self.router.route(&logits, correction_bias.as_deref()) {
            self.store.prefetch(ExpertKey {
                layer: target,
                expert: choice.expert,
            });
        }
    }

    /// Start a fresh inference session (own KV cache).
    pub fn session(&self) -> InferenceSession<'_> {
        InferenceSession {
            model: self,
            kv: (0..self.cfg.num_hidden_layers)
                .map(|_| LayerKvCache::default())
                .collect(),
            pos: 0,
            attn_path: AttnPath::Auto,
            tokens: Vec::new(),
            hidden: Vec::new(),
        }
    }

    /// Full-sequence teacher-forcing forward with a throwaway session.
    /// Returns logits `[seq, vocab]`.
    pub fn forward(&self, token_ids: &[usize]) -> undertow_core::Result<Tensor> {
        let mut s = self.session();
        s.prefill(token_ids)
    }

    /// Greedy decode (temperature 0), stopping on `extra_stop_ids` or the
    /// model's own stop tokens.
    pub fn greedy_decode(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        extra_stop_ids: &[usize],
    ) -> undertow_core::Result<Vec<usize>> {
        undertow_core::greedy_decode(self, prompt, max_new_tokens, extra_stop_ids)
    }
}

impl undertow_core::Model for DeepseekMoeModel {
    fn architecture(&self) -> &'static str {
        "deepseek_moe"
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn max_context(&self) -> usize {
        self.effective_max_context
    }

    fn stop_ids(&self) -> &[usize] {
        &self.stop_ids
    }

    fn new_session(&self) -> Box<dyn undertow_core::Session + '_> {
        Box::new(self.session())
    }

    fn store_stats(&self) -> undertow_core::StoreStatsSnapshot {
        self.store.stats()
    }

    fn expert_usage(&self) -> Vec<(ExpertKey, u64)> {
        self.store.usage()
    }
}

impl undertow_core::Session for InferenceSession<'_> {
    fn prefill(&mut self, token_ids: &[usize]) -> undertow_core::Result<Tensor> {
        InferenceSession::prefill(self, token_ids)
    }

    fn decode(&mut self, token_id: usize) -> undertow_core::Result<Vec<f32>> {
        InferenceSession::decode(self, token_id)
    }

    fn truncate(&mut self, len: usize) {
        InferenceSession::truncate(self, len)
    }

    fn position(&self) -> usize {
        InferenceSession::position(self)
    }

    fn kv_bytes(&self) -> usize {
        InferenceSession::kv_bytes(self)
    }
}

/// One conversation/completion in flight: positions consumed so far plus
/// the per-layer compressed KV cache.
pub struct InferenceSession<'m> {
    pub(crate) model: &'m DeepseekMoeModel,
    kv: Vec<LayerKvCache>,
    pub(crate) pos: usize,
    attn_path: AttnPath,
    /// Consumed token ids (kept only when the model has an MTP head).
    pub(crate) tokens: Vec<usize>,
    /// Last-layer hidden states `[pos, hidden]`, pre-final-norm (kept only
    /// when the model has an MTP head; the draft head consumes them).
    pub(crate) hidden: Vec<f32>,
}

impl<'m> InferenceSession<'m> {
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Force a specific attention path (tests, benchmarks).
    pub fn set_attn_path(&mut self, path: AttnPath) {
        self.attn_path = path;
    }

    /// KV cache bytes currently held.
    pub fn kv_bytes(&self) -> usize {
        self.kv.iter().map(|l| l.nbytes()).sum()
    }

    fn check_capacity(&self, extra: usize) -> undertow_core::Result<()> {
        let max = self.model.effective_max_context;
        if self.pos + extra > max {
            return Err(EngineError::ContextOverflow {
                requested: self.pos + extra,
                max,
            });
        }
        Ok(())
    }

    /// Run `token_ids` through the model, extending the cache. Returns
    /// logits `[seq, vocab]`.
    pub fn prefill(&mut self, token_ids: &[usize]) -> undertow_core::Result<Tensor> {
        if token_ids.is_empty() {
            return Err(EngineError::Other("empty prompt".into()));
        }
        self.check_capacity(token_ids.len())?;
        let m = self.model;
        let c = &m.cfg;
        let (seq, hidden) = (token_ids.len(), c.hidden_size);
        let dims = m.mla_dims();

        let mut x = vec![0.0f32; seq * hidden];
        for (s, &id) in token_ids.iter().enumerate() {
            if id >= c.vocab_size {
                return Err(EngineError::Other(format!(
                    "token id {id} out of vocab ({})",
                    c.vocab_size
                )));
            }
            m.embed_row(id, &mut x[s * hidden..(s + 1) * hidden]);
        }

        let single = seq == 1;
        let mut normed = vec![0.0f32; seq * hidden];
        for (li, layer) in m.layers.iter().enumerate() {
            for s in 0..seq {
                rmsnorm(
                    &mut normed[s * hidden..(s + 1) * hidden],
                    &x[s * hidden..(s + 1) * hidden],
                    &layer.input_norm,
                    c.rms_norm_eps,
                );
            }
            let attn_out = mla_forward_cached(
                &dims,
                &layer.attn,
                &normed,
                seq,
                &mut self.kv[li],
                self.attn_path,
            );
            for (xv, av) in x.iter_mut().zip(&attn_out) {
                *xv += av;
            }
            // Decode-time pilot: hint next layer's experts while this
            // layer's MoE computes.
            if single && li + 1 < m.layers.len() {
                m.pilot_prefetch(li + 1, &x);
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
                    let mut moe_out = vec![0.0f32; seq * hidden];
                    m.moe_forward(
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
        self.pos += seq;
        if m.mtp.is_some() {
            self.tokens.extend_from_slice(token_ids);
            self.hidden.extend_from_slice(&x);
        }

        let mut logits = Tensor::zeros(vec![seq, c.vocab_size]);
        for s in 0..seq {
            rmsnorm(
                &mut normed[s * hidden..(s + 1) * hidden],
                &x[s * hidden..(s + 1) * hidden],
                &m.final_norm,
                c.rms_norm_eps,
            );
            m.lm_head.matvec(
                &mut logits.data[s * c.vocab_size..(s + 1) * c.vocab_size],
                &normed[s * hidden..(s + 1) * hidden],
            );
        }
        Ok(logits)
    }

    /// Feed one token, get next-token logits.
    pub fn decode(&mut self, token_id: usize) -> undertow_core::Result<Vec<f32>> {
        let logits = self.prefill(&[token_id])?;
        Ok(logits.data)
    }

    /// Roll the session back to `len` consumed tokens (shared-prefix reuse).
    pub fn truncate(&mut self, len: usize) {
        let dims = self.model.mla_dims();
        for kv in &mut self.kv {
            kv.truncate(len, &dims);
        }
        self.pos = self.pos.min(len);
        self.tokens.truncate(self.pos);
        self.hidden.truncate(self.pos * self.model.cfg.hidden_size);
    }
}
