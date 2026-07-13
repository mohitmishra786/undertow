//! Load a DeepSeek-MoE-family checkpoint directory (HF layout, plain or
//! undertow-converted) into a model.
//!
//! Dense weights load into RAM as whatever [`undertow_io::read_qtensor`]
//! finds (f32 for plain checkpoints, int8/int4 for converted ones). Routed
//! experts go behind a [`TieredStore`]:
//!
//! * [`StoreChoice::DiskStreaming`] — experts stay on disk, fetched by
//!   `pread` through a byte-budgeted LRU. The production path.
//! * [`StoreChoice::Resident`] — everything loaded up front. For oracle
//!   tests and models that comfortably fit in RAM.

use std::path::Path;
use std::sync::Arc;

use undertow_core::store::{ExpertKey, ExpertWeights, ResidentStore};
use undertow_core::{EngineError, QTensor, Result, Tensor, TieredStore};
use undertow_io::{read_qtensor, DiskExpertStore, ExpertDims, ShardedModelReader};

use crate::attention::{MlaWeights, QueryProj};
use crate::config::DeepseekConfig;
use crate::model::{DeepseekMoeModel, FfnBlock, LayerWeights, MlpWeights};
use crate::naming::DeepseekExpertNaming;
use crate::router::DeepseekSigmoidRouter;

pub use undertow_core::model::{LoadOptions, StoreChoice};

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
        gate_proj: read_qtensor(r, &format!("{prefix}.gate_proj.weight"), inter, hidden)?,
        up_proj: read_qtensor(r, &format!("{prefix}.up_proj.weight"), inter, hidden)?,
        down_proj: read_qtensor(r, &format!("{prefix}.down_proj.weight"), hidden, inter)?,
    })
}

use undertow_core::model::generation_config_eos;

fn read_layer(
    reader: &ShardedModelReader,
    c: &DeepseekConfig,
    li: usize,
    is_dense: bool,
) -> Result<LayerWeights> {
    let hidden = c.hidden_size;
    let qh = c.qk_head_dim();
    let heads = c.num_attention_heads;
    let p = format!("model.layers.{li}");

    let query = match c.q_lora_rank {
        Some(rank) => QueryProj::Lora {
            q_a: read_qtensor(
                reader,
                &format!("{p}.self_attn.q_a_proj.weight"),
                rank,
                hidden,
            )?,
            q_a_norm: read_1d(reader, &format!("{p}.self_attn.q_a_layernorm.weight"), rank)?,
            q_b: read_qtensor(
                reader,
                &format!("{p}.self_attn.q_b_proj.weight"),
                heads * qh,
                rank,
            )?,
        },
        None => QueryProj::Direct {
            q_proj: read_qtensor(
                reader,
                &format!("{p}.self_attn.q_proj.weight"),
                heads * qh,
                hidden,
            )?,
        },
    };
    let attn = MlaWeights {
        query,
        kv_a: read_qtensor(
            reader,
            &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
            c.kv_lora_rank + c.qk_rope_head_dim,
            hidden,
        )?,
        kv_a_norm: read_1d(
            reader,
            &format!("{p}.self_attn.kv_a_layernorm.weight"),
            c.kv_lora_rank,
        )?,
        kv_b: read_qtensor(
            reader,
            &format!("{p}.self_attn.kv_b_proj.weight"),
            heads * (c.qk_nope_head_dim + c.v_head_dim),
            c.kv_lora_rank,
        )?,
        o_proj: read_qtensor(
            reader,
            &format!("{p}.self_attn.o_proj.weight"),
            hidden,
            heads * c.v_head_dim,
        )?,
    };

    let ffn = if is_dense {
        FfnBlock::Dense(read_mlp(
            reader,
            &format!("{p}.mlp"),
            hidden,
            c.intermediate_size,
        )?)
    } else {
        let bias_name = format!("{p}.mlp.gate.e_score_correction_bias");
        FfnBlock::Moe {
            gate: read_2d_f32(
                reader,
                &format!("{p}.mlp.gate.weight"),
                c.n_routed_experts,
                hidden,
            )?,
            correction_bias: if reader.has(&bias_name) {
                Some(read_1d(reader, &bias_name, c.n_routed_experts)?)
            } else {
                None
            },
            shared: if c.n_shared_experts > 0 {
                Some(read_mlp(
                    reader,
                    &format!("{p}.mlp.shared_experts"),
                    hidden,
                    c.n_shared_experts * c.moe_intermediate_size,
                )?)
            } else {
                None
            },
        }
    };

    Ok(LayerWeights {
        input_norm: read_1d(reader, &format!("{p}.input_layernorm.weight"), hidden)?,
        post_attn_norm: read_1d(
            reader,
            &format!("{p}.post_attention_layernorm.weight"),
            hidden,
        )?,
        attn,
        ffn,
    })
}

/// Load with default options (disk streaming, auto budget).
pub fn load_model(dir: impl AsRef<Path>) -> Result<DeepseekMoeModel> {
    load_model_with(dir, &LoadOptions::default())
}

pub fn load_model_with(dir: impl AsRef<Path>, opts: &LoadOptions) -> Result<DeepseekMoeModel> {
    let dir = dir.as_ref();
    let cfg = DeepseekConfig::from_dir(dir)?;
    let reader = Arc::new(ShardedModelReader::open(dir)?);
    let c = &cfg;
    let hidden = c.hidden_size;

    let embed_tokens = read_qtensor(&reader, "model.embed_tokens.weight", c.vocab_size, hidden)?;
    let lm_head = if c.tie_word_embeddings {
        embed_tokens.clone()
    } else {
        read_qtensor(&reader, "lm_head.weight", c.vocab_size, hidden)?
    };
    let final_norm = read_1d(&reader, "model.norm.weight", hidden)?;

    let mut layers = Vec::with_capacity(c.num_hidden_layers);
    for li in 0..c.num_hidden_layers {
        layers.push(read_layer(&reader, c, li, li < c.first_k_dense_replace)?);
    }

    // Native MTP head, when the checkpoint ships one.
    let mtp_prefix = format!("model.layers.{}", c.num_hidden_layers);
    let mtp =
        if c.num_nextn_predict_layers >= 1 && reader.has(&format!("{mtp_prefix}.eh_proj.weight")) {
            Some(crate::mtp::MtpHead {
                enorm: read_1d(&reader, &format!("{mtp_prefix}.enorm.weight"), hidden)?,
                hnorm: read_1d(&reader, &format!("{mtp_prefix}.hnorm.weight"), hidden)?,
                eh_proj: read_qtensor(
                    &reader,
                    &format!("{mtp_prefix}.eh_proj.weight"),
                    hidden,
                    2 * hidden,
                )?,
                layer: read_layer(&reader, c, c.num_hidden_layers, false)?,
                final_norm: read_1d(
                    &reader,
                    &format!("{mtp_prefix}.shared_head.norm.weight"),
                    hidden,
                )?,
                head: read_qtensor(
                    &reader,
                    &format!("{mtp_prefix}.shared_head.head.weight"),
                    c.vocab_size,
                    hidden,
                )?,
            })
        } else {
            None
        };

    let dense_bytes = dense_resident_bytes(&embed_tokens, &lm_head, &layers, c);

    let naming = Arc::new(DeepseekExpertNaming);
    let store: Arc<dyn TieredStore> = match &opts.store {
        StoreChoice::Resident => {
            let mut store = ResidentStore::new();
            let n_moe_layers = c.num_hidden_layers + if mtp.is_some() { 1 } else { 0 };
            for li in c.first_k_dense_replace..n_moe_layers {
                for e in 0..c.n_routed_experts {
                    let key = ExpertKey {
                        layer: li,
                        expert: e,
                    };
                    let names =
                        undertow_core::adapter::ExpertNaming::expert_tensor_names(&*naming, key);
                    store.insert(
                        key,
                        ExpertWeights {
                            gate_proj: read_qtensor(
                                &reader,
                                &names[0],
                                c.moe_intermediate_size,
                                hidden,
                            )?,
                            up_proj: read_qtensor(
                                &reader,
                                &names[1],
                                c.moe_intermediate_size,
                                hidden,
                            )?,
                            down_proj: read_qtensor(
                                &reader,
                                &names[2],
                                hidden,
                                c.moe_intermediate_size,
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
                .unwrap_or_else(|| undertow_core::mem::auto_cache_budget(dense_bytes as u64));
            let cache = undertow_core::cache::build_cache(opts.cache_policy, budget as usize);
            let disk = DiskExpertStore::new(
                reader.clone(),
                naming,
                ExpertDims {
                    hidden,
                    moe_intermediate: c.moe_intermediate_size,
                },
                cache,
                *prefetch_workers,
            );
            if let Some(profile) = &opts.pin_profile {
                if profile.architecture != "deepseek_moe" {
                    return Err(EngineError::InvalidConfig(format!(
                        "profile recorded on {:?}, model is deepseek_moe",
                        profile.architecture
                    )));
                }
                let pin_budget = opts.pin_budget_bytes.unwrap_or(budget / 4);
                disk.pin_hottest(profile.hottest(), pin_budget)?;
            }
            Arc::new(disk)
        }
    };

    let mut stop_ids = cfg.eos_ids();
    for id in generation_config_eos(dir) {
        if !stop_ids.contains(&id) {
            stop_ids.push(id);
        }
    }

    // A session KV budget shrinks the usable context: bytes per position
    // are the compressed latents across layers plus (with MTP) the saved
    // hidden state and token id.
    let per_pos_bytes = cfg.num_hidden_layers * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 4
        + if mtp.is_some() {
            cfg.hidden_size * 4 + 8
        } else {
            0
        };
    let effective_max_context = match opts.kv_budget_bytes {
        Some(budget) => cfg
            .max_position_embeddings
            .min((budget as usize / per_pos_bytes.max(1)).max(1)),
        None => cfg.max_position_embeddings,
    };

    let router = DeepseekSigmoidRouter::from_config(&cfg);
    Ok(DeepseekMoeModel {
        cfg,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        router,
        store,
        stop_ids,
        dense_bytes,
        mtp,
        effective_max_context,
    })
}

fn dense_resident_bytes(
    embed: &QTensor,
    lm_head: &QTensor,
    layers: &[LayerWeights],
    cfg: &DeepseekConfig,
) -> usize {
    let mut total = embed.nbytes() + lm_head.nbytes();
    total += 4 * cfg.hidden_size; // final norm
    for l in layers {
        total += 4 * (l.input_norm.len() + l.post_attn_norm.len());
        let a = &l.attn;
        total += a.kv_a.nbytes() + a.kv_b.nbytes() + a.o_proj.nbytes() + 4 * a.kv_a_norm.len();
        total += match &a.query {
            QueryProj::Lora { q_a, q_a_norm, q_b } => {
                q_a.nbytes() + q_b.nbytes() + 4 * q_a_norm.len()
            }
            QueryProj::Direct { q_proj } => q_proj.nbytes(),
        };
        total += match &l.ffn {
            FfnBlock::Dense(m) => m.nbytes(),
            FfnBlock::Moe {
                gate,
                correction_bias,
                shared,
            } => {
                gate.numel() * 4
                    + correction_bias.as_ref().map_or(0, |b| b.len() * 4)
                    + shared.as_ref().map_or(0, |s| s.nbytes())
            }
        };
    }
    total
}
