//! Generic GQA-MoE model: layer stack, MoE block, sessions, loader.
//!
//! A family crate (mixtral-moe, qwen-moe) supplies three things: a
//! [`GqaMoeSpec`] built from its config.json, a [`GqaMoeNaming`] for its
//! tensor names, and a router. Everything else — forward pass, compressed
//! per-layer KV caches, expert streaming through [`TieredStore`],
//! speculative next-layer prefetch — is shared here and tested once.

use std::path::Path;
use std::sync::Arc;

use engine_core::adapter::{ExpertNaming, RouterAdapter};
use engine_core::model::{generation_config_eos, LoadOptions, StoreChoice};
use engine_core::store::{ExpertKey, ExpertWeights, ResidentStore};
use engine_core::{EngineError, QTensor, Result, StoreStatsSnapshot, Tensor, TieredStore};
use engine_io::{read_qtensor, DiskExpertStore, ExpertDims, ShardedModelReader};
use engine_quant::{matmul, rmsnorm, silu};

use crate::gqa::{gqa_forward_cached, GqaDims, GqaKvCache, GqaWeights};

#[derive(Debug, Clone)]
pub struct GqaMoeSpec {
    pub architecture: &'static str,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub max_position_embeddings: usize,
    pub dims: GqaDims,
    pub n_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate: usize,
    /// Layers whose FFN is a plain dense MLP (Qwen `mlp_only_layers`).
    pub dense_layers: Vec<usize>,
    pub dense_intermediate: usize,
    pub tie_word_embeddings: bool,
    /// eos ids from config.json (generation_config.json is merged in by
    /// the loader).
    pub eos_ids: Vec<usize>,
}

/// Family-specific tensor naming beyond the shared HF conventions.
pub trait GqaMoeNaming: ExpertNaming {
    /// Router gate weight, `[n_experts, hidden]`.
    fn router_gate(&self, layer: usize) -> String;
    /// Prefix for a dense layer's MLP (`{prefix}.gate_proj.weight`, ...).
    fn dense_mlp_prefix(&self, layer: usize) -> String {
        format!("model.layers.{layer}.mlp")
    }
    /// Per-head q/k RMSNorm weight names, if the family has them.
    fn qk_norm(&self, _layer: usize) -> Option<(String, String)> {
        None
    }
}

pub struct MlpWeights {
    pub gate_proj: QTensor,
    pub up_proj: QTensor,
    pub down_proj: QTensor,
}

impl MlpWeights {
    fn nbytes(&self) -> usize {
        self.gate_proj.nbytes() + self.up_proj.nbytes() + self.down_proj.nbytes()
    }

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

pub enum Ffn {
    Dense(MlpWeights),
    Moe {
        /// `[n_experts, hidden]`, f32 (numerically sensitive).
        gate: Tensor,
    },
}

pub struct Layer {
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub attn: GqaWeights,
    pub ffn: Ffn,
}

pub struct GqaMoeModel {
    pub spec: GqaMoeSpec,
    pub embed_tokens: QTensor,
    pub layers: Vec<Layer>,
    pub final_norm: Vec<f32>,
    pub lm_head: QTensor,
    pub router: Box<dyn RouterAdapter>,
    pub store: Arc<dyn TieredStore>,
    pub stop_ids: Vec<usize>,
    pub dense_bytes: usize,
}

impl GqaMoeModel {
    fn embed_row(&self, id: usize, out: &mut [f32]) {
        out.iter_mut().for_each(|v| *v = 0.0);
        self.embed_tokens.add_scaled_row(id, 1.0, out);
    }

    fn moe_forward(
        &self,
        layer_idx: usize,
        gate: &Tensor,
        x: &[f32],
        seq: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let s = &self.spec;
        let (hidden, inter) = (s.hidden_size, s.moe_intermediate);
        let mut logits = vec![0.0f32; s.n_experts];
        let mut g = vec![0.0f32; inter];
        let mut u = vec![0.0f32; inter];
        let mut d = vec![0.0f32; hidden];
        for pos in 0..seq {
            let xs = &x[pos * hidden..(pos + 1) * hidden];
            matmul(&mut logits, xs, &gate.data, 1, hidden, s.n_experts);
            for choice in self.router.route(&logits, None) {
                let expert = self.store.get_expert(ExpertKey {
                    layer: layer_idx,
                    expert: choice.expert,
                })?;
                expert.gate_proj.matvec(&mut g, xs);
                expert.up_proj.matvec(&mut u, xs);
                for (gv, uv) in g.iter_mut().zip(&u) {
                    *gv = silu(*gv) * uv;
                }
                expert.down_proj.matvec(&mut d, &g);
                let os = &mut out[pos * hidden..(pos + 1) * hidden];
                for (o, v) in os.iter_mut().zip(&d) {
                    *o += choice.weight * v;
                }
            }
        }
        Ok(())
    }

    /// Speculative routing of layer `target` on hidden state `x`;
    /// pure I/O hint.
    fn pilot_prefetch(&self, target: usize, x: &[f32]) {
        let layer = &self.layers[target];
        let Ffn::Moe { gate } = &layer.ffn else {
            return;
        };
        let s = &self.spec;
        let mut nrm = vec![0.0f32; s.hidden_size];
        rmsnorm(&mut nrm, x, &layer.post_attn_norm, s.dims.rms_eps);
        let mut logits = vec![0.0f32; s.n_experts];
        matmul(&mut logits, &nrm, &gate.data, 1, s.hidden_size, s.n_experts);
        for choice in self.router.route(&logits, None) {
            self.store.prefetch(ExpertKey {
                layer: target,
                expert: choice.expert,
            });
        }
    }

    pub fn session(&self) -> GqaSession<'_> {
        GqaSession {
            model: self,
            kv: (0..self.spec.num_layers)
                .map(|_| GqaKvCache::default())
                .collect(),
            pos: 0,
        }
    }

    pub fn forward(&self, ids: &[usize]) -> Result<Tensor> {
        self.session().prefill(ids)
    }

    pub fn greedy_decode(
        &self,
        prompt: &[usize],
        max_new: usize,
        extra_stop_ids: &[usize],
    ) -> Result<Vec<usize>> {
        engine_core::greedy_decode(self, prompt, max_new, extra_stop_ids)
    }
}

pub struct GqaSession<'m> {
    model: &'m GqaMoeModel,
    kv: Vec<GqaKvCache>,
    pos: usize,
}

impl GqaSession<'_> {
    pub fn prefill(&mut self, token_ids: &[usize]) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(EngineError::Other("empty prompt".into()));
        }
        let m = self.model;
        let s = &m.spec;
        if self.pos + token_ids.len() > s.max_position_embeddings {
            return Err(EngineError::ContextOverflow {
                requested: self.pos + token_ids.len(),
                max: s.max_position_embeddings,
            });
        }
        let (seq, hidden) = (token_ids.len(), s.hidden_size);
        let mut x = vec![0.0f32; seq * hidden];
        for (i, &id) in token_ids.iter().enumerate() {
            if id >= s.vocab_size {
                return Err(EngineError::Other(format!(
                    "token id {id} out of vocab ({})",
                    s.vocab_size
                )));
            }
            m.embed_row(id, &mut x[i * hidden..(i + 1) * hidden]);
        }

        let single = seq == 1;
        let mut normed = vec![0.0f32; seq * hidden];
        for (li, layer) in m.layers.iter().enumerate() {
            for i in 0..seq {
                rmsnorm(
                    &mut normed[i * hidden..(i + 1) * hidden],
                    &x[i * hidden..(i + 1) * hidden],
                    &layer.input_norm,
                    s.dims.rms_eps,
                );
            }
            let attn = gqa_forward_cached(&s.dims, &layer.attn, &normed, seq, &mut self.kv[li]);
            for (xv, av) in x.iter_mut().zip(&attn) {
                *xv += av;
            }
            if single && li + 1 < m.layers.len() {
                m.pilot_prefetch(li + 1, &x);
            }
            for i in 0..seq {
                rmsnorm(
                    &mut normed[i * hidden..(i + 1) * hidden],
                    &x[i * hidden..(i + 1) * hidden],
                    &layer.post_attn_norm,
                    s.dims.rms_eps,
                );
            }
            match &layer.ffn {
                Ffn::Dense(mlp) => mlp.forward_add(&normed, seq, &mut x),
                Ffn::Moe { gate } => {
                    let mut moe_out = vec![0.0f32; seq * hidden];
                    m.moe_forward(li, gate, &normed, seq, &mut moe_out)?;
                    for (xv, mv) in x.iter_mut().zip(&moe_out) {
                        *xv += mv;
                    }
                }
            }
        }
        self.pos += seq;

        let mut logits = Tensor::zeros(vec![seq, s.vocab_size]);
        for i in 0..seq {
            rmsnorm(
                &mut normed[i * hidden..(i + 1) * hidden],
                &x[i * hidden..(i + 1) * hidden],
                &m.final_norm,
                s.dims.rms_eps,
            );
            m.lm_head.matvec(
                &mut logits.data[i * s.vocab_size..(i + 1) * s.vocab_size],
                &normed[i * hidden..(i + 1) * hidden],
            );
        }
        Ok(logits)
    }

    pub fn decode(&mut self, token_id: usize) -> Result<Vec<f32>> {
        Ok(self.prefill(&[token_id])?.data)
    }

    pub fn truncate(&mut self, len: usize) {
        for kv in &mut self.kv {
            kv.truncate(len, &self.model.spec.dims);
        }
        self.pos = self.pos.min(len);
    }
}

impl engine_core::Model for GqaMoeModel {
    fn architecture(&self) -> &'static str {
        self.spec.architecture
    }

    fn vocab_size(&self) -> usize {
        self.spec.vocab_size
    }

    fn max_context(&self) -> usize {
        self.spec.max_position_embeddings
    }

    fn stop_ids(&self) -> &[usize] {
        &self.stop_ids
    }

    fn new_session(&self) -> Box<dyn engine_core::Session + '_> {
        Box::new(self.session())
    }

    fn store_stats(&self) -> StoreStatsSnapshot {
        self.store.stats()
    }

    fn expert_usage(&self) -> Vec<(ExpertKey, u64)> {
        self.store.usage()
    }
}

impl engine_core::Session for GqaSession<'_> {
    fn prefill(&mut self, token_ids: &[usize]) -> Result<Tensor> {
        GqaSession::prefill(self, token_ids)
    }

    fn decode(&mut self, token_id: usize) -> Result<Vec<f32>> {
        GqaSession::decode(self, token_id)
    }

    fn truncate(&mut self, len: usize) {
        GqaSession::truncate(self, len)
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn kv_bytes(&self) -> usize {
        self.kv.iter().map(|l| l.nbytes()).sum()
    }
}

fn read_1d(r: &ShardedModelReader, name: &str, len: usize) -> Result<Vec<f32>> {
    let t = r.read_f32(name)?;
    if t.shape != [len] {
        return Err(EngineError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![len],
            got: t.shape,
        });
    }
    Ok(t.data)
}

fn read_2d_f32(r: &ShardedModelReader, name: &str, d0: usize, d1: usize) -> Result<Tensor> {
    let t = r.read_f32(name)?;
    if t.shape != [d0, d1] {
        return Err(EngineError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![d0, d1],
            got: t.shape,
        });
    }
    Ok(t)
}

/// Load a GQA-MoE checkpoint (plain or undertow-converted).
pub fn load_gqa_model(
    dir: impl AsRef<Path>,
    spec: GqaMoeSpec,
    naming: Arc<dyn GqaMoeNaming>,
    router: Box<dyn RouterAdapter>,
    opts: &LoadOptions,
) -> Result<GqaMoeModel> {
    let dir = dir.as_ref();
    let reader = Arc::new(ShardedModelReader::open(dir)?);
    let s = &spec;
    let hidden = s.hidden_size;
    let d = &s.dims;

    let embed_tokens = read_qtensor(&reader, "model.embed_tokens.weight", s.vocab_size, hidden)?;
    let lm_head = if s.tie_word_embeddings {
        embed_tokens.clone()
    } else {
        read_qtensor(&reader, "lm_head.weight", s.vocab_size, hidden)?
    };
    let final_norm = read_1d(&reader, "model.norm.weight", hidden)?;

    let mut layers = Vec::with_capacity(s.num_layers);
    let mut dense_bytes = embed_tokens.nbytes() + lm_head.nbytes() + 4 * hidden;
    for li in 0..s.num_layers {
        let p = format!("model.layers.{li}");
        let (q_norm, k_norm) = match naming.qk_norm(li) {
            Some((qn, kn)) => (
                Some(read_1d(&reader, &qn, d.head_dim)?),
                Some(read_1d(&reader, &kn, d.head_dim)?),
            ),
            None => (None, None),
        };
        let attn = GqaWeights {
            q_proj: read_qtensor(
                &reader,
                &format!("{p}.self_attn.q_proj.weight"),
                d.num_heads * d.head_dim,
                hidden,
            )?,
            k_proj: read_qtensor(
                &reader,
                &format!("{p}.self_attn.k_proj.weight"),
                d.num_kv_heads * d.head_dim,
                hidden,
            )?,
            v_proj: read_qtensor(
                &reader,
                &format!("{p}.self_attn.v_proj.weight"),
                d.num_kv_heads * d.head_dim,
                hidden,
            )?,
            o_proj: read_qtensor(
                &reader,
                &format!("{p}.self_attn.o_proj.weight"),
                hidden,
                d.num_heads * d.head_dim,
            )?,
            q_norm,
            k_norm,
        };
        dense_bytes += attn.q_proj.nbytes()
            + attn.k_proj.nbytes()
            + attn.v_proj.nbytes()
            + attn.o_proj.nbytes();

        let ffn = if s.dense_layers.contains(&li) {
            let prefix = naming.dense_mlp_prefix(li);
            let mlp = MlpWeights {
                gate_proj: read_qtensor(
                    &reader,
                    &format!("{prefix}.gate_proj.weight"),
                    s.dense_intermediate,
                    hidden,
                )?,
                up_proj: read_qtensor(
                    &reader,
                    &format!("{prefix}.up_proj.weight"),
                    s.dense_intermediate,
                    hidden,
                )?,
                down_proj: read_qtensor(
                    &reader,
                    &format!("{prefix}.down_proj.weight"),
                    hidden,
                    s.dense_intermediate,
                )?,
            };
            dense_bytes += mlp.nbytes();
            Ffn::Dense(mlp)
        } else {
            let gate = read_2d_f32(&reader, &naming.router_gate(li), s.n_experts, hidden)?;
            dense_bytes += gate.numel() * 4;
            Ffn::Moe { gate }
        };

        let input_norm = read_1d(&reader, &format!("{p}.input_layernorm.weight"), hidden)?;
        let post_attn_norm = read_1d(
            &reader,
            &format!("{p}.post_attention_layernorm.weight"),
            hidden,
        )?;
        dense_bytes += 4 * (input_norm.len() + post_attn_norm.len());
        layers.push(Layer {
            input_norm,
            post_attn_norm,
            attn,
            ffn,
        });
    }

    let naming_experts: Arc<dyn ExpertNaming> = naming;
    let store: Arc<dyn TieredStore> = match &opts.store {
        StoreChoice::Resident => {
            let mut store = ResidentStore::new();
            for li in 0..s.num_layers {
                if s.dense_layers.contains(&li) {
                    continue;
                }
                for e in 0..s.n_experts {
                    let key = ExpertKey {
                        layer: li,
                        expert: e,
                    };
                    let names = naming_experts.expert_tensor_names(key);
                    store.insert(
                        key,
                        ExpertWeights {
                            gate_proj: read_qtensor(
                                &reader,
                                &names[0],
                                s.moe_intermediate,
                                hidden,
                            )?,
                            up_proj: read_qtensor(&reader, &names[1], s.moe_intermediate, hidden)?,
                            down_proj: read_qtensor(
                                &reader,
                                &names[2],
                                hidden,
                                s.moe_intermediate,
                            )?,
                        },
                    );
                }
            }
            Arc::new(store)
        }
        StoreChoice::DiskStreaming {
            cache_budget_bytes,
            prefetch_workers,
        } => {
            let budget = cache_budget_bytes
                .unwrap_or_else(|| engine_core::mem::auto_cache_budget(dense_bytes as u64));
            let cache = engine_core::cache::build_cache(opts.cache_policy, budget as usize);
            let disk = DiskExpertStore::new(
                reader.clone(),
                naming_experts,
                ExpertDims {
                    hidden,
                    moe_intermediate: s.moe_intermediate,
                },
                cache,
                *prefetch_workers,
            );
            if let Some(profile) = &opts.pin_profile {
                if profile.architecture != spec.architecture {
                    return Err(EngineError::InvalidConfig(format!(
                        "profile recorded on {:?}, model is {}",
                        profile.architecture, spec.architecture
                    )));
                }
                let pin_budget = opts.pin_budget_bytes.unwrap_or(budget / 4);
                disk.pin_hottest(profile.hottest(), pin_budget)?;
            }
            Arc::new(disk)
        }
    };

    let mut stop_ids = spec.eos_ids.clone();
    for id in generation_config_eos(dir) {
        if !stop_ids.contains(&id) {
            stop_ids.push(id);
        }
    }

    Ok(GqaMoeModel {
        spec,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        router,
        store,
        stop_ids,
        dense_bytes,
    })
}
