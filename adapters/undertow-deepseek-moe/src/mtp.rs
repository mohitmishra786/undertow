//! Native multi-token prediction (MTP) speculative decoding.
//!
//! DeepSeek-V3-family checkpoints ship an extra transformer layer at index
//! `num_hidden_layers` that predicts token `t+2` from the main model's
//! last hidden state at position `t` and the embedding of token `t+1`:
//!
//! ```text
//! x_t = eh_proj( concat( enorm(embed(t+1)), hnorm(h_t) ) )
//! h'_t = transformer_layer(x_0..x_t)          // own KV cache, causal
//! logits_t = shared_head( shared_head_norm(h'_t) )
//! ```
//!
//! Greedy speculation drafts one token with this head, then verifies both
//! the sampled token and the draft in a single two-token prefill of the
//! main model. Accepted drafts cost one forward for two tokens; rejected
//! drafts truncate one KV position and continue from the verified logits.
//! **Lossless by construction**: the emitted sequence is decided only by
//! the main model's logits, so output is identical with MTP on or off,
//! whatever the draft head predicts. A test pins that invariant.
//!
//! Speculation supports both greedy decoding and stochastic sampling with
//! speculative rejection sampling (Leviathan et al., 2023), guaranteeing
//! exact distribution preservation under any temperature or top_p.

use undertow_core::sample::{argmax, Sampler, SamplerConfig, SpeculativeDecision};
use undertow_core::{EngineError, QTensor};
use undertow_quant::rmsnorm;

use crate::attention::{AttnPath, LayerKvCache};
use crate::model::{DeepseekMoeModel, InferenceSession, LayerWeights};

pub struct MtpHead {
    /// RMSNorm over the next token's embedding.
    pub enorm: Vec<f32>,
    /// RMSNorm over the main model's last hidden state.
    pub hnorm: Vec<f32>,
    /// `[hidden, 2*hidden]`, applied to `concat(enorm(e), hnorm(h))`.
    pub eh_proj: QTensor,
    /// The MTP transformer layer (its experts live in the tiered store at
    /// layer index `num_hidden_layers`).
    pub layer: LayerWeights,
    /// `shared_head.norm.weight`.
    pub final_norm: Vec<f32>,
    /// `shared_head.head.weight`, `[vocab, hidden]`.
    pub head: QTensor,
}

/// Draft-head state for one generation (its own KV cache).
#[derive(Default)]
pub struct MtpState {
    kv: LayerKvCache,
    /// Main-model positions whose MTP input has been fed.
    fed: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MtpStats {
    pub drafted: u64,
    pub accepted: u64,
}

impl MtpStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            return 0.0;
        }
        self.accepted as f64 / self.drafted as f64
    }
}

/// Feed pending MTP inputs up to the session's current position (using
/// `next_token` as the not-yet-prefilled continuation) and return the
/// draft logits for the token after `next_token`.
pub fn draft_logits(
    model: &DeepseekMoeModel,
    mtp: &MtpHead,
    session: &InferenceSession<'_>,
    state: &mut MtpState,
    next_token: usize,
) -> undertow_core::Result<Vec<f32>> {
    let c = &model.cfg;
    let hidden = c.hidden_size;
    let pos = session.pos;
    debug_assert!(state.fed < pos, "draft with nothing pending");
    let pending = pos - state.fed;

    // Build the pending inputs [pending, hidden].
    let mut xs = vec![0.0f32; pending * hidden];
    let mut emb = vec![0.0f32; hidden];
    let mut e_n = vec![0.0f32; hidden];
    let mut h_n = vec![0.0f32; hidden];
    let mut cat = vec![0.0f32; 2 * hidden];
    for (j, i) in (state.fed..pos).enumerate() {
        let t_next = if i + 1 < pos {
            session.tokens[i + 1]
        } else {
            next_token
        };
        model.embed_row(t_next, &mut emb);
        rmsnorm(&mut e_n, &emb, &mtp.enorm, c.rms_norm_eps);
        rmsnorm(
            &mut h_n,
            &session.hidden[i * hidden..(i + 1) * hidden],
            &mtp.hnorm,
            c.rms_norm_eps,
        );
        cat[..hidden].copy_from_slice(&e_n);
        cat[hidden..].copy_from_slice(&h_n);
        mtp.eh_proj
            .matvec(&mut xs[j * hidden..(j + 1) * hidden], &cat);
    }

    model.run_layer(
        &mtp.layer,
        c.num_hidden_layers,
        &mut xs,
        pending,
        &mut state.kv,
        AttnPath::Auto,
    )?;
    state.fed = pos;

    let mut normed = vec![0.0f32; hidden];
    rmsnorm(
        &mut normed,
        &xs[(pending - 1) * hidden..pending * hidden],
        &mtp.final_norm,
        c.rms_norm_eps,
    );
    let mut logits = vec![0.0f32; c.vocab_size];
    mtp.head.matvec(&mut logits, &normed);
    Ok(logits)
}

pub fn draft(
    model: &DeepseekMoeModel,
    mtp: &MtpHead,
    session: &InferenceSession<'_>,
    state: &mut MtpState,
    next_token: usize,
) -> undertow_core::Result<usize> {
    let logits = draft_logits(model, mtp, session, state, next_token)?;
    Ok(argmax(&logits))
}

/// Generation with MTP speculation, supporting both greedy decoding and
/// speculative rejection sampling under temperature > 0.
pub fn generate_mtp(
    model: &DeepseekMoeModel,
    prompt: &[usize],
    max_new: usize,
    sampler_cfg: SamplerConfig,
    extra_stop_ids: &[usize],
    on_token: impl FnMut(usize) -> bool,
) -> undertow_core::Result<(usize, MtpStats)> {
    let Some(mtp) = &model.mtp else {
        return Err(EngineError::Other(
            "model has no MTP head; use plain generation".into(),
        ));
    };
    let mut state = MtpState::default();
    speculative_loop_with_sampler(
        model,
        prompt,
        max_new,
        sampler_cfg,
        extra_stop_ids,
        on_token,
        |sess, next| draft_logits(model, mtp, sess, &mut state, next),
    )
}

/// Greedy generation with MTP speculation. Semantics identical to
/// [`undertow_core::generate`] with a greedy sampler; returns produced count
/// and draft statistics.
pub fn generate_greedy_mtp(
    model: &DeepseekMoeModel,
    prompt: &[usize],
    max_new: usize,
    extra_stop_ids: &[usize],
    on_token: impl FnMut(usize) -> bool,
) -> undertow_core::Result<(usize, MtpStats)> {
    generate_mtp(
        model,
        prompt,
        max_new,
        SamplerConfig::greedy(),
        extra_stop_ids,
        on_token,
    )
}

/// Speculative decoding loop with an arbitrary draft logits source and rejection sampling.
pub fn speculative_loop_with_sampler(
    model: &DeepseekMoeModel,
    prompt: &[usize],
    max_new: usize,
    sampler_cfg: SamplerConfig,
    extra_stop_ids: &[usize],
    mut on_token: impl FnMut(usize) -> bool,
    mut draft_logits_fn: impl FnMut(&InferenceSession<'_>, usize) -> undertow_core::Result<Vec<f32>>,
) -> undertow_core::Result<(usize, MtpStats)> {
    sampler_cfg.validate().map_err(EngineError::Other)?;
    let is_stop = |id: usize| model.stop_ids.contains(&id) || extra_stop_ids.contains(&id);

    let mut session = model.session();
    let mut stats = MtpStats::default();
    let mut sampler = Sampler::new(sampler_cfg).map_err(EngineError::Other)?;

    let logits = session.prefill(prompt)?;
    let mut next = sampler.sample(logits.row(prompt.len() - 1));
    let mut produced = 0usize;
    loop {
        produced += 1;
        let keep_going = on_token(next);
        if is_stop(next) || !keep_going || produced >= max_new {
            break;
        }

        let draft_l = draft_logits_fn(&session, next)?;
        stats.drafted += 1;

        let q_probs = sampler.probs(&draft_l);
        let drafted = sampler.sample_probs(&q_probs);

        let pos_before = session.pos;
        let verify = match session.prefill(&[next, drafted]) {
            Ok(v) => v,
            // Not enough context left for the two-token verify: end the
            // stream cleanly with what was already emitted.
            Err(EngineError::ContextOverflow { .. }) => break,
            Err(e) => return Err(e),
        };

        let p_probs = sampler.probs(verify.row(0));
        let decision = sampler.verify_draft(drafted, &q_probs, &p_probs);

        match decision {
            SpeculativeDecision::Accepted => {
                stats.accepted += 1;
                produced += 1;
                let keep_going = on_token(drafted);
                if is_stop(drafted) || !keep_going || produced >= max_new {
                    break;
                }
                next = sampler.sample(verify.row(1));
            }
            SpeculativeDecision::Rejected(replacement) => {
                // The draft's KV position is wrong; roll it back. `next`'s
                // position (pos_before) is real and stays.
                session.truncate(pos_before + 1);
                next = replacement;
            }
        }
    }
    Ok((produced, stats))
}

/// Greedy speculation with an arbitrary draft source: one verified token
/// per round plus one draft, accepted when the main model agrees. Public
/// so external draft models can plug in; the losslessness tests drive it
/// with both the native MTP head and a perfect oracle draft.
pub fn speculative_loop(
    model: &DeepseekMoeModel,
    prompt: &[usize],
    max_new: usize,
    extra_stop_ids: &[usize],
    mut on_token: impl FnMut(usize) -> bool,
    mut draft_fn: impl FnMut(&InferenceSession<'_>, usize) -> undertow_core::Result<usize>,
) -> undertow_core::Result<(usize, MtpStats)> {
    let is_stop = |id: usize| model.stop_ids.contains(&id) || extra_stop_ids.contains(&id);

    let mut session = model.session();
    let mut stats = MtpStats::default();

    let logits = session.prefill(prompt)?;
    let mut next = argmax(logits.row(prompt.len() - 1));
    let mut produced = 0usize;
    loop {
        produced += 1;
        let keep_going = on_token(next);
        if is_stop(next) || !keep_going || produced >= max_new {
            break;
        }

        let drafted = draft_fn(&session, next)?;
        stats.drafted += 1;

        let pos_before = session.pos;
        let verify = match session.prefill(&[next, drafted]) {
            Ok(v) => v,
            // Not enough context left for the two-token verify: end the
            // stream cleanly with what was already emitted.
            Err(EngineError::ContextOverflow { .. }) => break,
            Err(e) => return Err(e),
        };
        let verified = argmax(verify.row(0));
        if verified == drafted {
            stats.accepted += 1;
            produced += 1;
            let keep_going = on_token(drafted);
            if is_stop(drafted) || !keep_going || produced >= max_new {
                break;
            }
            next = argmax(verify.row(1));
        } else {
            // The draft's KV position is wrong; roll it back. `next`'s
            // position (pos_before) is real and stays.
            session.truncate(pos_before + 1);
            next = verified;
        }
    }
    Ok((produced, stats))
}
