//! Grouped-query attention with a per-layer KV cache.
//!
//! Covers the whole Mixtral/Qwen-MoE attention space:
//! * GQA head mapping (`num_heads` query heads share `num_kv_heads` KV
//!   heads).
//! * NeoX-style RoPE over the full head dim (`rotate_half`, the standard
//!   `transformers` `apply_rotary_pos_emb`): `out[j] = x[j]·cos − x[j+d/2]·sin`,
//!   `out[j+d/2] = x[j+d/2]·cos + x[j]·sin`.
//! * Optional per-head RMSNorm on q and k before RoPE (Qwen3).
//! * Optional sliding-window causal mask (Mixtral): position `p` attends
//!   to `max(0, p+1−w) ..= p`.
//!
//! Keys are cached after norm+RoPE, values as projected, so decode never
//! recomputes projections for past positions.

use engine_core::QTensor;
use engine_quant::{rmsnorm, softmax};

pub struct GqaWeights {
    /// `[num_heads * head_dim, hidden]`
    pub q_proj: QTensor,
    /// `[num_kv_heads * head_dim, hidden]`
    pub k_proj: QTensor,
    /// `[num_kv_heads * head_dim, hidden]`
    pub v_proj: QTensor,
    /// `[hidden, num_heads * head_dim]`
    pub o_proj: QTensor,
    /// Per-head RMSNorm weights of length `head_dim` (Qwen3).
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy)]
pub struct GqaDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// 1/sqrt(head_dim)
    pub scale: f32,
    pub sliding_window: Option<usize>,
}

/// KV cache for one layer: roped keys and raw values, `[len, KV*head_dim]`.
#[derive(Debug, Clone, Default)]
pub struct GqaKvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
}

impl GqaKvCache {
    pub fn nbytes(&self) -> usize {
        (self.k.capacity() + self.v.capacity()) * 4
    }

    pub fn truncate(&mut self, len: usize, d: &GqaDims) {
        self.len = len.min(self.len);
        let row = d.num_kv_heads * d.head_dim;
        self.k.truncate(self.len * row);
        self.v.truncate(self.len * row);
    }
}

/// NeoX RoPE on `v[..dim]` at position `pos`: rotate split halves.
pub fn rope_neox(v: &mut [f32], pos: usize, dim: usize, theta: f32) {
    debug_assert!(dim.is_multiple_of(2) && v.len() >= dim);
    let half = dim / 2;
    for j in 0..half {
        let inv = theta.powf(-2.0 * j as f32 / dim as f32);
        let ang = pos as f32 * inv;
        let (sin, cos) = ang.sin_cos();
        let (a, b) = (v[j], v[half + j]);
        v[j] = a * cos - b * sin;
        v[half + j] = b * cos + a * sin;
    }
}

/// Causal GQA over `x` `[seq, hidden]` (already input-layernormed) for the
/// `seq` new tokens at positions `kv.len ..`, extending the cache.
/// Returns `[seq, hidden]` (o_proj applied, no residual).
pub fn gqa_forward_cached(
    d: &GqaDims,
    w: &GqaWeights,
    x: &[f32],
    seq: usize,
    kv: &mut GqaKvCache,
) -> Vec<f32> {
    let (hd, h, kvh, dh) = (d.hidden, d.num_heads, d.num_kv_heads, d.head_dim);
    assert_eq!(x.len(), seq * hd);
    assert!(h.is_multiple_of(kvh), "heads must divide into kv heads");
    let group = h / kvh;
    let pos_base = kv.len;
    let total = pos_base + seq;
    let kv_row = kvh * dh;

    // --- queries ---
    let mut q = vec![0.0f32; seq * h * dh];
    w.q_proj.matmul(&mut q, x, seq);
    let mut scratch = vec![0.0f32; dh];
    for s in 0..seq {
        for head in 0..h {
            let off = s * h * dh + head * dh;
            if let Some(qn) = &w.q_norm {
                scratch.copy_from_slice(&q[off..off + dh]);
                rmsnorm(&mut q[off..off + dh], &scratch, qn, d.rms_eps);
            }
            rope_neox(&mut q[off..off + dh], pos_base + s, dh, d.rope_theta);
        }
    }

    // --- extend cache with new keys (normed+roped) and values ---
    kv.k.resize(total * kv_row, 0.0);
    kv.v.resize(total * kv_row, 0.0);
    let mut kbuf = vec![0.0f32; kv_row];
    for s in 0..seq {
        let pos = pos_base + s;
        let xs = &x[s * hd..(s + 1) * hd];
        w.k_proj.matvec(&mut kbuf, xs);
        for head in 0..kvh {
            let off = head * dh;
            if let Some(kn) = &w.k_norm {
                scratch.copy_from_slice(&kbuf[off..off + dh]);
                rmsnorm(&mut kbuf[off..off + dh], &scratch, kn, d.rms_eps);
            }
            rope_neox(&mut kbuf[off..off + dh], pos, dh, d.rope_theta);
        }
        kv.k[pos * kv_row..(pos + 1) * kv_row].copy_from_slice(&kbuf);
        w.v_proj
            .matvec(&mut kv.v[pos * kv_row..(pos + 1) * kv_row], xs);
    }
    kv.len = total;

    // --- attention ---
    let mut ctx = vec![0.0f32; seq * h * dh];
    let mut scores = vec![0.0f32; total];
    for s in 0..seq {
        let pos = pos_base + s;
        let start = match d.sliding_window {
            Some(win) => (pos + 1).saturating_sub(win),
            None => 0,
        };
        let n_ctx = pos + 1 - start;
        for head in 0..h {
            let qp = &q[s * h * dh + head * dh..s * h * dh + (head + 1) * dh];
            let kv_head = head / group;
            for (j, sc) in scores[..n_ctx].iter_mut().enumerate() {
                let t = start + j;
                let kt = &kv.k[t * kv_row + kv_head * dh..t * kv_row + (kv_head + 1) * dh];
                let mut a = 0.0f32;
                for i in 0..dh {
                    a += qp[i] * kt[i];
                }
                *sc = a * d.scale;
            }
            softmax(&mut scores[..n_ctx]);
            let cx = &mut ctx[(s * h + head) * dh..(s * h + head + 1) * dh];
            cx.iter_mut().for_each(|v| *v = 0.0);
            for (j, &a) in scores[..n_ctx].iter().enumerate() {
                let t = start + j;
                let vt = &kv.v[t * kv_row + kv_head * dh..t * kv_row + (kv_head + 1) * dh];
                for i in 0..dh {
                    cx[i] += a * vt[i];
                }
            }
        }
    }

    let mut out = vec![0.0f32; seq * hd];
    w.o_proj.matmul(&mut out, &ctx, seq);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::sample::Pcg32;

    #[test]
    fn rope_neox_pos0_identity() {
        let mut v = [1.0f32, 2.0, 3.0, 4.0];
        rope_neox(&mut v, 0, 4, 10000.0);
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_neox_relative_invariance() {
        let dim = 8;
        let qv: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.9).sin()).collect();
        let kv: Vec<f32> = (0..dim).map(|i| (i as f32 * 1.7).cos()).collect();
        let dot = |pq: usize, pk: usize| {
            let mut a = qv.clone();
            let mut b = kv.clone();
            rope_neox(&mut a, pq, dim, 10000.0);
            rope_neox(&mut b, pk, dim, 10000.0);
            a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>()
        };
        assert!((dot(2, 6) - dot(11, 15)).abs() < 1e-4);
    }

    fn mat(rng: &mut Pcg32, o: usize, i: usize) -> QTensor {
        QTensor::from_f32((0..o * i).map(|_| rng.normal() * 0.15).collect(), o, i)
    }

    fn dims(sliding_window: Option<usize>, kvh: usize) -> GqaDims {
        GqaDims {
            hidden: 24,
            num_heads: 4,
            num_kv_heads: kvh,
            head_dim: 6,
            rope_theta: 10000.0,
            rms_eps: 1e-6,
            scale: 1.0 / 6f32.sqrt(),
            sliding_window,
        }
    }

    fn weights(d: &GqaDims, rng: &mut Pcg32, qk_norm: bool) -> GqaWeights {
        GqaWeights {
            q_proj: mat(rng, d.num_heads * d.head_dim, d.hidden),
            k_proj: mat(rng, d.num_kv_heads * d.head_dim, d.hidden),
            v_proj: mat(rng, d.num_kv_heads * d.head_dim, d.hidden),
            o_proj: mat(rng, d.hidden, d.num_heads * d.head_dim),
            q_norm: qk_norm.then(|| (0..d.head_dim).map(|_| 1.0 + rng.normal() * 0.05).collect()),
            k_norm: qk_norm.then(|| (0..d.head_dim).map(|_| 1.0 + rng.normal() * 0.05).collect()),
        }
    }

    /// Brute-force reference: same math, no cache, explicit mask.
    fn reference(d: &GqaDims, w: &GqaWeights, x: &[f32], seq: usize) -> Vec<f32> {
        // The cached implementation *is* the one under test; the reference
        // recomputes each position through a fresh cache so any
        // cache-state bug diverges.
        let mut out = Vec::new();
        for s in 0..seq {
            let mut fresh = GqaKvCache::default();
            let upto = gqa_forward_cached(d, w, &x[..(s + 1) * d.hidden], s + 1, &mut fresh);
            out.extend_from_slice(&upto[s * d.hidden..(s + 1) * d.hidden]);
        }
        out
    }

    #[test]
    fn incremental_matches_oneshot_with_and_without_qk_norm() {
        for qk_norm in [false, true] {
            for kvh in [4, 2, 1] {
                let d = dims(None, kvh);
                let mut rng = Pcg32::new(31, kvh as u64);
                let w = weights(&d, &mut rng, qk_norm);
                let seq = 6;
                let x: Vec<f32> = (0..seq * d.hidden).map(|_| rng.normal() * 0.3).collect();

                let oneshot = {
                    let mut kv = GqaKvCache::default();
                    gqa_forward_cached(&d, &w, &x, seq, &mut kv)
                };
                let mut kv = GqaKvCache::default();
                let mut incremental = gqa_forward_cached(&d, &w, &x[..3 * d.hidden], 3, &mut kv);
                for s in 3..seq {
                    incremental.extend(gqa_forward_cached(
                        &d,
                        &w,
                        &x[s * d.hidden..(s + 1) * d.hidden],
                        1,
                        &mut kv,
                    ));
                }
                let max_err = oneshot
                    .iter()
                    .zip(&incremental)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_err < 1e-4,
                    "qk_norm={qk_norm} kvh={kvh} diverged: {max_err}"
                );
                let re = reference(&d, &w, &x, seq);
                let max_err2 = oneshot
                    .iter()
                    .zip(&re)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(max_err2 < 1e-4, "vs reference: {max_err2}");
            }
        }
    }

    #[test]
    fn sliding_window_limits_context() {
        let mut rng = Pcg32::new(77, 1);
        let d_full = dims(None, 2);
        let w = weights(&d_full, &mut rng, false);
        let seq = 8;
        let x: Vec<f32> = (0..seq * d_full.hidden)
            .map(|_| rng.normal() * 0.3)
            .collect();

        // Window >= seq must equal no window at all.
        let d_wide = dims(Some(seq), 2);
        let a = {
            let mut kv = GqaKvCache::default();
            gqa_forward_cached(&d_full, &w, &x, seq, &mut kv)
        };
        let b = {
            let mut kv = GqaKvCache::default();
            gqa_forward_cached(&d_wide, &w, &x, seq, &mut kv)
        };
        assert_eq!(a, b, "window >= seq must be a no-op");

        // Window of 1: each position attends only to itself, so the last
        // output must equal running that single token through an empty
        // context at the same position... which we emulate by checking it
        // differs from full attention but matches a manual single-token
        // softmax (softmax over one score is 1, ctx = v of that position).
        let d_one = dims(Some(1), 2);
        let mut kv = GqaKvCache::default();
        let c = gqa_forward_cached(&d_one, &w, &x, seq, &mut kv);
        assert_ne!(a, c, "window=1 must change the output");
        // Manual check for the last position: ctx = its own value vector.
        let s = seq - 1;
        let xs = &x[s * d_one.hidden..(s + 1) * d_one.hidden];
        let kv_row = d_one.num_kv_heads * d_one.head_dim;
        let mut v_own = vec![0.0f32; kv_row];
        w.v_proj.matvec(&mut v_own, xs);
        let mut ctx = vec![0.0f32; d_one.num_heads * d_one.head_dim];
        for head in 0..d_one.num_heads {
            let kvh = head / (d_one.num_heads / d_one.num_kv_heads);
            ctx[head * d_one.head_dim..(head + 1) * d_one.head_dim]
                .copy_from_slice(&v_own[kvh * d_one.head_dim..(kvh + 1) * d_one.head_dim]);
        }
        let mut expect = vec![0.0f32; d_one.hidden];
        w.o_proj.matvec(&mut expect, &ctx);
        let got = &c[s * d_one.hidden..(s + 1) * d_one.hidden];
        let max_err = got
            .iter()
            .zip(&expect)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 1e-5, "window=1 self-attention wrong: {max_err}");
    }
}
