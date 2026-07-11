//! Print per-layer expert usage for the oracle fixture — a quick check
//! that the synthetic model exercises diverse routing (if every token
//! picked the same experts, the oracle test would prove much less).
//!
//!     cargo run -p engine-bench --example expert_usage

use engine_core::adapter::RouterAdapter;
use engine_quant::matmul;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/oracle-tiny");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("reference.json"))?)?;
    let ids: Vec<usize> = reference["full_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let model = deepseek_moe::loader::load_model(&dir)?;
    let c = &model.cfg;

    // Re-run the layer stack, but capture routing at each MoE layer by
    // recomputing gate logits on the post-attention-normed hidden states.
    // Cheap trick: run the full forward once per prefix is overkill; instead
    // we reuse the model's own forward but count via the router on hidden
    // states reconstructed layer by layer. Simplest correct option at
    // oracle scale: replicate the residual walk here.
    use engine_quant::rmsnorm;
    let (seq, hidden) = (ids.len(), c.hidden_size);
    let mut x = vec![0.0f32; seq * hidden];
    for (s, &id) in ids.iter().enumerate() {
        x[s * hidden..(s + 1) * hidden].copy_from_slice(model.embed_tokens.row(id));
    }
    let dims = deepseek_moe::attention::MlaDims {
        hidden,
        num_heads: c.num_attention_heads,
        qk_nope: c.qk_nope_head_dim,
        qk_rope: c.qk_rope_head_dim,
        v_head: c.v_head_dim,
        kv_lora: c.kv_lora_rank,
        rope_theta: c.rope_theta(),
        rms_eps: c.rms_norm_eps,
        scale: c.attn_scale(),
    };
    let mut normed = vec![0.0f32; seq * hidden];
    for (li, layer) in model.layers.iter().enumerate() {
        for s in 0..seq {
            rmsnorm(
                &mut normed[s * hidden..(s + 1) * hidden],
                &x[s * hidden..(s + 1) * hidden],
                &layer.input_norm,
                c.rms_norm_eps,
            );
        }
        let attn = deepseek_moe::attention::mla_forward(&dims, &layer.attn, &normed, seq);
        for (xv, av) in x.iter_mut().zip(&attn) {
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
            deepseek_moe::model::FfnBlock::Dense(mlp) => {
                mlp_add(mlp, &normed, seq, &mut x);
            }
            deepseek_moe::model::FfnBlock::Moe {
                gate,
                correction_bias,
                shared,
            } => {
                let mut usage = vec![0usize; c.n_routed_experts];
                let mut logits = vec![0.0f32; c.n_routed_experts];
                for s in 0..seq {
                    matmul(
                        &mut logits,
                        &normed[s * hidden..(s + 1) * hidden],
                        &gate.data,
                        1,
                        hidden,
                        c.n_routed_experts,
                    );
                    for ch in model.router.route(&logits, correction_bias.as_deref()) {
                        usage[ch.expert] += 1;
                    }
                }
                let used = usage.iter().filter(|&&n| n > 0).count();
                println!(
                    "layer {li}: {used}/{} experts used, assignments: {usage:?}",
                    c.n_routed_experts
                );
                // keep the walk faithful: apply the real MoE output
                let mut moe_out = vec![0.0f32; seq * hidden];
                moe_apply(
                    &model,
                    li,
                    gate,
                    correction_bias.as_deref(),
                    shared.as_ref(),
                    &normed,
                    seq,
                    &mut moe_out,
                );
                for (xv, mv) in x.iter_mut().zip(&moe_out) {
                    *xv += mv;
                }
            }
        }
    }
    Ok(())
}

fn mlp_add(mlp: &deepseek_moe::model::MlpWeights, x: &[f32], seq: usize, out: &mut [f32]) {
    use engine_quant::silu;
    let hidden = mlp.gate_proj.dim1();
    let inter = mlp.gate_proj.dim0();
    let (mut g, mut u) = (vec![0.0f32; seq * inter], vec![0.0f32; seq * inter]);
    matmul(&mut g, x, &mlp.gate_proj.data, seq, hidden, inter);
    matmul(&mut u, x, &mlp.up_proj.data, seq, hidden, inter);
    for (gv, uv) in g.iter_mut().zip(&u) {
        *gv = silu(*gv) * uv;
    }
    let mut d = vec![0.0f32; seq * hidden];
    matmul(&mut d, &g, &mlp.down_proj.data, seq, inter, hidden);
    for (o, v) in out.iter_mut().zip(&d) {
        *o += v;
    }
}

#[allow(clippy::too_many_arguments)]
fn moe_apply(
    model: &deepseek_moe::DeepseekMoeModel,
    li: usize,
    gate: &engine_core::Tensor,
    bias: Option<&[f32]>,
    shared: Option<&deepseek_moe::model::MlpWeights>,
    x: &[f32],
    seq: usize,
    out: &mut [f32],
) {
    use engine_core::ExpertKey;
    use engine_quant::silu;
    let c = &model.cfg;
    let (hidden, inter) = (c.hidden_size, c.moe_intermediate_size);
    let mut logits = vec![0.0f32; c.n_routed_experts];
    let (mut g, mut u, mut d) = (
        vec![0.0f32; inter],
        vec![0.0f32; inter],
        vec![0.0f32; hidden],
    );
    for s in 0..seq {
        let xs = &x[s * hidden..(s + 1) * hidden];
        matmul(&mut logits, xs, &gate.data, 1, hidden, c.n_routed_experts);
        for ch in model.router.route(&logits, bias) {
            let e = model
                .store
                .get_expert(ExpertKey {
                    layer: li,
                    expert: ch.expert,
                })
                .unwrap();
            matmul(&mut g, xs, &e.gate_proj.data, 1, hidden, inter);
            matmul(&mut u, xs, &e.up_proj.data, 1, hidden, inter);
            for (gv, uv) in g.iter_mut().zip(&u) {
                *gv = silu(*gv) * uv;
            }
            matmul(&mut d, &g, &e.down_proj.data, 1, inter, hidden);
            for (o, v) in out[s * hidden..(s + 1) * hidden].iter_mut().zip(&d) {
                *o += ch.weight * v;
            }
        }
    }
    if let Some(sh) = shared {
        mlp_add(sh, x, seq, out);
    }
}
