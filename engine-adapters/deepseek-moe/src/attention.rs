//! Multi-head Latent Attention (MLA), scalar reference path.
//!
//! Per token:
//!   q: x → q_a (rank r_q) → RMSNorm → q_b → H heads of [qk_nope | qk_rope],
//!      RoPE on the rope part. (Or a direct q_proj when q_lora_rank is null.)
//!   kv: x → kv_a → [kv latent (rank r_kv) | k_rot], latent is RMSNormed,
//!      k_rot is RoPEd once and shared by all heads.
//!   k/v per head are reconstructed from the latent through kv_b
//!      (`[H*(qk_nope+v_head), r_kv]`).
//!   score(s,t) = (q_nope·k_nope + q_rope·k_rot) / sqrt(qk_nope+qk_rope),
//!      causal softmax, context → o_proj.
//!
//! This is the *non-absorbed* formulation — the mathematically transparent
//! one. Weight absorption (folding kv_b into q/o for O(T·r_kv) decode) is a
//! later optimization that must reproduce this path bit-for-bit-ish.
//!
//! RoPE is DeepSeek/GLM interleaved-partial: input pairs `(v[2j], v[2j+1])`
//! rotate by `pos·theta^(-2j/d)` and land at split-half positions
//! `(v[j], v[half+j])` — identical to `transformers`'
//! `apply_rotary_pos_emb_interleave`, whose de-interleave reshape produces
//! exactly this layout.

use engine_core::Tensor;
use engine_quant::{matmul, rmsnorm, softmax};

/// Query projection: low-rank (q_a/q_b with norm) or direct.
pub enum QueryProj {
    Lora {
        /// `[q_lora_rank, hidden]`
        q_a: Tensor,
        /// `[q_lora_rank]`
        q_a_norm: Vec<f32>,
        /// `[H * (qk_nope + qk_rope), q_lora_rank]`
        q_b: Tensor,
    },
    Direct {
        /// `[H * (qk_nope + qk_rope), hidden]`
        q_proj: Tensor,
    },
}

pub struct MlaWeights {
    pub query: QueryProj,
    /// `[kv_lora_rank + qk_rope, hidden]`
    pub kv_a: Tensor,
    /// `[kv_lora_rank]`
    pub kv_a_norm: Vec<f32>,
    /// `[H * (qk_nope + v_head), kv_lora_rank]`
    pub kv_b: Tensor,
    /// `[hidden, H * v_head]`
    pub o_proj: Tensor,
}

pub struct MlaDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_head: usize,
    pub kv_lora: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// 1/sqrt(qk_nope + qk_rope)
    pub scale: f32,
}

/// Interleaved partial RoPE on `v` (length `dim`, even) at position `pos`.
pub fn rope_interleave(v: &mut [f32], pos: usize, dim: usize, theta: f32) {
    debug_assert!(dim.is_multiple_of(2) && v.len() >= dim);
    let half = dim / 2;
    let input: Vec<f32> = v[..dim].to_vec();
    for j in 0..half {
        let inv = theta.powf(-2.0 * j as f32 / dim as f32);
        let ang = pos as f32 * inv;
        let (sin, cos) = ang.sin_cos();
        let (a, b) = (input[2 * j], input[2 * j + 1]);
        v[j] = a * cos - b * sin;
        v[half + j] = b * cos + a * sin;
    }
}

/// Causal MLA over `x` `[seq, hidden]` (already input-layernormed),
/// positions `0..seq`. Returns `[seq, hidden]` (o_proj applied, no residual).
pub fn mla_forward(d: &MlaDims, w: &MlaWeights, x: &[f32], seq: usize) -> Vec<f32> {
    let (hd, h) = (d.hidden, d.num_heads);
    let qh = d.qk_nope + d.qk_rope;
    let kv_head = d.qk_nope + d.v_head;
    assert_eq!(x.len(), seq * hd);

    // --- queries: [seq, H, qh], rope part rotated per position ---
    let mut q = vec![0.0f32; seq * h * qh];
    match &w.query {
        QueryProj::Lora { q_a, q_a_norm, q_b } => {
            let rank = q_a.dim0();
            let mut resid = vec![0.0f32; rank];
            for s in 0..seq {
                let xs = &x[s * hd..(s + 1) * hd];
                matmul(&mut resid, xs, &q_a.data, 1, hd, rank);
                let normed = {
                    let mut n = vec![0.0f32; rank];
                    rmsnorm(&mut n, &resid, q_a_norm, d.rms_eps);
                    n
                };
                matmul(
                    &mut q[s * h * qh..(s + 1) * h * qh],
                    &normed,
                    &q_b.data,
                    1,
                    rank,
                    h * qh,
                );
            }
        }
        QueryProj::Direct { q_proj } => {
            matmul(&mut q, x, &q_proj.data, seq, hd, h * qh);
        }
    }
    for s in 0..seq {
        for head in 0..h {
            let off = s * h * qh + head * qh + d.qk_nope;
            rope_interleave(&mut q[off..off + d.qk_rope], s, d.qk_rope, d.rope_theta);
        }
    }

    // --- kv latents: latent [seq, kv_lora] (normed), k_rot [seq, qk_rope] (roped) ---
    let mut latent = vec![0.0f32; seq * d.kv_lora];
    let mut k_rot = vec![0.0f32; seq * d.qk_rope];
    let mut comp = vec![0.0f32; d.kv_lora + d.qk_rope];
    for s in 0..seq {
        let xs = &x[s * hd..(s + 1) * hd];
        matmul(&mut comp, xs, &w.kv_a.data, 1, hd, d.kv_lora + d.qk_rope);
        rmsnorm(
            &mut latent[s * d.kv_lora..(s + 1) * d.kv_lora],
            &comp[..d.kv_lora],
            &w.kv_a_norm,
            d.rms_eps,
        );
        let kr = &mut k_rot[s * d.qk_rope..(s + 1) * d.qk_rope];
        kr.copy_from_slice(&comp[d.kv_lora..]);
        rope_interleave(kr, s, d.qk_rope, d.rope_theta);
    }

    // --- reconstruct k_nope|v for every position: [seq, H*(qk_nope+v_head)] ---
    let mut kv = vec![0.0f32; seq * h * kv_head];
    matmul(&mut kv, &latent, &w.kv_b.data, seq, d.kv_lora, h * kv_head);

    // --- causal attention ---
    let mut ctx = vec![0.0f32; seq * h * d.v_head];
    let mut scores = vec![0.0f32; seq];
    for s in 0..seq {
        for head in 0..h {
            let qp = &q[s * h * qh + head * qh..s * h * qh + (head + 1) * qh];
            let (q_nope, q_rope) = qp.split_at(d.qk_nope);
            let n_ctx = s + 1;
            for t in 0..n_ctx {
                let kn = &kv[t * h * kv_head + head * kv_head..][..d.qk_nope];
                let kr = &k_rot[t * d.qk_rope..(t + 1) * d.qk_rope];
                let mut a = 0.0f32;
                for i in 0..d.qk_nope {
                    a += q_nope[i] * kn[i];
                }
                for i in 0..d.qk_rope {
                    a += q_rope[i] * kr[i];
                }
                scores[t] = a * d.scale;
            }
            softmax(&mut scores[..n_ctx]);
            let cx = &mut ctx[(s * h + head) * d.v_head..(s * h + head + 1) * d.v_head];
            for t in 0..n_ctx {
                let vv = &kv[t * h * kv_head + head * kv_head + d.qk_nope..][..d.v_head];
                let a = scores[t];
                for i in 0..d.v_head {
                    cx[i] += a * vv[i];
                }
            }
        }
    }

    let mut out = vec![0.0f32; seq * hd];
    matmul(&mut out, &ctx, &w.o_proj.data, seq, h * d.v_head, hd);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_pos0_is_identity() {
        // At pos 0 all angles are 0: values just de-interleave.
        let mut v = [1.0f32, 2.0, 3.0, 4.0];
        rope_interleave(&mut v, 0, 4, 10000.0);
        assert_eq!(v, [1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn rope_preserves_norm() {
        let mut v = [0.3f32, -1.2, 2.0, 0.5, -0.7, 1.1];
        let before: f32 = v.iter().map(|x| x * x).sum();
        rope_interleave(&mut v, 17, 6, 10000.0);
        let after: f32 = v.iter().map(|x| x * x).sum();
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn rope_relative_position_invariance() {
        // Dot product of a roped q at pos p and roped k at pos p+d must
        // depend only on d: rotate the same two vectors at (0, 5) and
        // (7, 12) and compare scores.
        let dim = 8;
        let qv: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7).sin()).collect();
        let kv: Vec<f32> = (0..dim).map(|i| (i as f32 * 1.3).cos()).collect();
        let dot_at = |pq: usize, pk: usize| {
            let mut q = qv.clone();
            let mut k = kv.clone();
            rope_interleave(&mut q, pq, dim, 10000.0);
            rope_interleave(&mut k, pk, dim, 10000.0);
            q.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>()
        };
        assert!((dot_at(0, 5) - dot_at(7, 12)).abs() < 1e-4);
    }
}
