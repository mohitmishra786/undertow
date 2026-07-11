//! Load a DeepSeek-MoE-family checkpoint directory (HF layout) into the
//! scalar reference model.
//!
//! Phase 0: everything — including routed experts — is loaded up front into
//! a [`ResidentStore`]. The point is that `model.rs` already consumes
//! experts via [`TieredStore`], so swapping this for the disk-streaming
//! store later is loader-only surgery.

use std::path::Path;
use std::sync::Arc;

use engine_core::store::{ExpertKey, ExpertWeights, ResidentStore};
use engine_core::{EngineError, Result, Tensor};
use engine_io::ShardedModelReader;

use crate::attention::{MlaWeights, QueryProj};
use crate::config::DeepseekConfig;
use crate::model::{DeepseekMoeModel, FfnBlock, LayerWeights, MlpWeights};
use crate::router::DeepseekSigmoidRouter;

fn expect_2d(t: Tensor, name: &str, d0: usize, d1: usize) -> Result<Tensor> {
    if t.shape != [d0, d1] {
        return Err(EngineError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![d0, d1],
            got: t.shape,
        });
    }
    Ok(t)
}

fn read_2d(r: &ShardedModelReader, name: &str, d0: usize, d1: usize) -> Result<Tensor> {
    expect_2d(r.read_f32(name)?, name, d0, d1)
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

fn read_mlp(
    r: &ShardedModelReader,
    prefix: &str,
    hidden: usize,
    inter: usize,
) -> Result<MlpWeights> {
    Ok(MlpWeights {
        gate_proj: read_2d(r, &format!("{prefix}.gate_proj.weight"), inter, hidden)?,
        up_proj: read_2d(r, &format!("{prefix}.up_proj.weight"), inter, hidden)?,
        down_proj: read_2d(r, &format!("{prefix}.down_proj.weight"), hidden, inter)?,
    })
}

pub fn load_model(dir: impl AsRef<Path>) -> Result<DeepseekMoeModel> {
    let dir = dir.as_ref();
    let cfg = DeepseekConfig::from_dir(dir)?;
    let reader = ShardedModelReader::open(dir)?;
    let c = &cfg;
    let hidden = c.hidden_size;
    let qh = c.qk_head_dim();
    let heads = c.num_attention_heads;

    let embed_tokens = read_2d(&reader, "model.embed_tokens.weight", c.vocab_size, hidden)?;
    let lm_head = if c.tie_word_embeddings {
        embed_tokens.clone()
    } else {
        read_2d(&reader, "lm_head.weight", c.vocab_size, hidden)?
    };
    let final_norm = read_1d(&reader, "model.norm.weight", hidden)?;

    let mut store = ResidentStore::new();
    let mut layers = Vec::with_capacity(c.num_hidden_layers);
    for li in 0..c.num_hidden_layers {
        let p = format!("model.layers.{li}");

        let query = match c.q_lora_rank {
            Some(rank) => QueryProj::Lora {
                q_a: read_2d(
                    &reader,
                    &format!("{p}.self_attn.q_a_proj.weight"),
                    rank,
                    hidden,
                )?,
                q_a_norm: read_1d(
                    &reader,
                    &format!("{p}.self_attn.q_a_layernorm.weight"),
                    rank,
                )?,
                q_b: read_2d(
                    &reader,
                    &format!("{p}.self_attn.q_b_proj.weight"),
                    heads * qh,
                    rank,
                )?,
            },
            None => QueryProj::Direct {
                q_proj: read_2d(
                    &reader,
                    &format!("{p}.self_attn.q_proj.weight"),
                    heads * qh,
                    hidden,
                )?,
            },
        };
        let attn = MlaWeights {
            query,
            kv_a: read_2d(
                &reader,
                &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
                c.kv_lora_rank + c.qk_rope_head_dim,
                hidden,
            )?,
            kv_a_norm: read_1d(
                &reader,
                &format!("{p}.self_attn.kv_a_layernorm.weight"),
                c.kv_lora_rank,
            )?,
            kv_b: read_2d(
                &reader,
                &format!("{p}.self_attn.kv_b_proj.weight"),
                heads * (c.qk_nope_head_dim + c.v_head_dim),
                c.kv_lora_rank,
            )?,
            o_proj: read_2d(
                &reader,
                &format!("{p}.self_attn.o_proj.weight"),
                hidden,
                heads * c.v_head_dim,
            )?,
        };

        let is_dense = li < c.first_k_dense_replace;
        let ffn = if is_dense {
            FfnBlock::Dense(read_mlp(
                &reader,
                &format!("{p}.mlp"),
                hidden,
                c.intermediate_size,
            )?)
        } else {
            for e in 0..c.n_routed_experts {
                let ep = format!("{p}.mlp.experts.{e}");
                let mlp = read_mlp(&reader, &ep, hidden, c.moe_intermediate_size)?;
                store.insert(
                    ExpertKey {
                        layer: li,
                        expert: e,
                    },
                    ExpertWeights {
                        gate_proj: mlp.gate_proj,
                        up_proj: mlp.up_proj,
                        down_proj: mlp.down_proj,
                    },
                );
            }
            let bias_name = format!("{p}.mlp.gate.e_score_correction_bias");
            FfnBlock::Moe {
                gate: read_2d(
                    &reader,
                    &format!("{p}.mlp.gate.weight"),
                    c.n_routed_experts,
                    hidden,
                )?,
                correction_bias: if reader.has(&bias_name) {
                    Some(read_1d(&reader, &bias_name, c.n_routed_experts)?)
                } else {
                    None
                },
                shared: if c.n_shared_experts > 0 {
                    Some(read_mlp(
                        &reader,
                        &format!("{p}.mlp.shared_experts"),
                        hidden,
                        c.n_shared_experts * c.moe_intermediate_size,
                    )?)
                } else {
                    None
                },
            }
        };

        layers.push(LayerWeights {
            input_norm: read_1d(&reader, &format!("{p}.input_layernorm.weight"), hidden)?,
            post_attn_norm: read_1d(
                &reader,
                &format!("{p}.post_attention_layernorm.weight"),
                hidden,
            )?,
            attn,
            ffn,
        });
    }

    let router = DeepseekSigmoidRouter::from_config(c);
    Ok(DeepseekMoeModel {
        cfg,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        router,
        store: Arc::new(store),
    })
}
