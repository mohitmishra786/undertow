//! Multi-head Latent Attention (MLA) with a compressed KV cache.
//!
//! Per token:
//!   q: x → q_a (rank r_q) → RMSNorm → q_b → H heads of [qk_nope | qk_rope],
//!      RoPE on the rope part. (Or a direct q_proj when q_lora_rank is null.)
//!   kv: x → kv_a → [kv latent (rank r_kv) | k_rot], latent is RMSNormed,
//!      k_rot is RoPEd once and shared by all heads.
//!
//! Only the latent and k_rot are cached: `r_kv + qk_rope` floats per token
//! per layer instead of `H * (qk_nope + qk_rope + v_head)`. Two compute
//! paths consume the cache:
//!
//! * **Reconstruct** (prefill): k/v for every cached position are rebuilt
//!   through `kv_b` in one matmul, then ordinary causal attention. The
//!   mathematically transparent formulation; also the reference the other
//!   path is tested against.
//! * **Absorbed** (decode): for small S, `q_nope` is folded through the K
//!   half of `kv_b` (`qabs = Kᵀ q_nope`), so scores are taken directly
//!   against cached latents, and the context is reconstructed only for the
//!   V half. Cost per step drops from `O(T · H · (nope+vh))` to
//!   `O(T · r_kv)` per head. Same math by linearity, verified in tests.
//!
//! RoPE is DeepSeek/GLM interleaved-partial: input pairs `(v[2j], v[2j+1])`
//! rotate by `pos·theta^(-2j/d)` and land at split-half positions
//! `(v[j], v[half+j])` — identical to `transformers`'
//! `apply_rotary_pos_emb_interleave`.

use undertow_core::QTensor;
use undertow_quant::{rmsnorm, softmax};

/// Query projection: low-rank (q_a/q_b with norm) or direct.
pub enum QueryProj {
    Lora {
        /// `[q_lora_rank, hidden]`
        q_a: QTensor,
        /// `[q_lora_rank]`
        q_a_norm: Vec<f32>,
        /// `[H * (qk_nope + qk_rope), q_lora_rank]`
        q_b: QTensor,
    },
    Direct {
        /// `[H * (qk_nope + qk_rope), hidden]`
        q_proj: QTensor,
    },
}

pub struct MlaWeights {
    pub query: QueryProj,
    /// `[kv_lora_rank + qk_rope, hidden]`
    pub kv_a: QTensor,
    /// `[kv_lora_rank]`
    pub kv_a_norm: Vec<f32>,
    /// `[H * (qk_nope + v_head), kv_lora_rank]`
    pub kv_b: QTensor,
    /// `[hidden, H * v_head]`
    pub o_proj: QTensor,
}

#[derive(Debug, Clone, Copy)]
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

/// Compressed KV cache for one layer: the RMSNormed latent and the RoPEd
/// shared key-rot vector per position.
#[derive(Debug, Clone, Default)]
pub struct LayerKvCache {
    /// `[len, kv_lora]`
    pub latent: Vec<f32>,
    /// `[len, qk_rope]`
    pub k_rot: Vec<f32>,
    pub len: usize,
}

impl LayerKvCache {
    pub fn nbytes(&self) -> usize {
        (self.latent.capacity() + self.k_rot.capacity()) * 4
    }

    pub fn truncate(&mut self, len: usize, dims: &MlaDims) {
        self.len = len.min(self.len);
        self.latent.truncate(self.len * dims.kv_lora);
        self.k_rot.truncate(self.len * dims.qk_rope);
    }
}

/// Which attention path to use for the cached forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnPath {
    /// Absorbed for S <= 4, reconstruct otherwise.
    Auto,
    Absorbed,
    Reconstruct,
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

/// Causal MLA over `x` `[seq, hidden]` (already input-layernormed) for the
/// `seq` new tokens at positions `kv.len .. kv.len + seq`, extending the
/// cache. Returns `[seq, hidden]` (o_proj applied, no residual).
pub fn mla_forward_cached(
    d: &MlaDims,
    w: &MlaWeights,
    x: &[f32],
    seq: usize,
    kv: &mut LayerKvCache,
    path: AttnPath,
) -> Vec<f32> {
    let (hd, h) = (d.hidden, d.num_heads);
    let qh = d.qk_nope + d.qk_rope;
    let kv_head = d.qk_nope + d.v_head;
    assert_eq!(x.len(), seq * hd);
    let pos_base = kv.len;
    let total = pos_base + seq;

    // --- queries: [seq, H, qh], rope part rotated per position ---
    let mut q = vec![0.0f32; seq * h * qh];
    match &w.query {
        QueryProj::Lora { q_a, q_a_norm, q_b } => {
            let rank = q_a.out_dim();
            let mut resid = vec![0.0f32; rank];
            let mut normed = vec![0.0f32; rank];
            for s in 0..seq {
                let xs = &x[s * hd..(s + 1) * hd];
                q_a.matvec(&mut resid, xs);
                rmsnorm(&mut normed, &resid, q_a_norm, d.rms_eps);
                q_b.matvec(&mut q[s * h * qh..(s + 1) * h * qh], &normed);
            }
        }
        QueryProj::Direct { q_proj } => {
            q_proj.matmul(&mut q, x, seq);
        }
    }
    for s in 0..seq {
        for head in 0..h {
            let off = s * h * qh + head * qh + d.qk_nope;
            rope_interleave(
                &mut q[off..off + d.qk_rope],
                pos_base + s,
                d.qk_rope,
                d.rope_theta,
            );
        }
    }

    // --- extend the cache with the new tokens' latent + k_rot ---
    kv.latent.resize(total * d.kv_lora, 0.0);
    kv.k_rot.resize(total * d.qk_rope, 0.0);
    let mut comp = vec![0.0f32; d.kv_lora + d.qk_rope];
    for s in 0..seq {
        let pos = pos_base + s;
        let xs = &x[s * hd..(s + 1) * hd];
        w.kv_a.matvec(&mut comp, xs);
        rmsnorm(
            &mut kv.latent[pos * d.kv_lora..(pos + 1) * d.kv_lora],
            &comp[..d.kv_lora],
            &w.kv_a_norm,
            d.rms_eps,
        );
        let kr = &mut kv.k_rot[pos * d.qk_rope..(pos + 1) * d.qk_rope];
        kr.copy_from_slice(&comp[d.kv_lora..]);
        rope_interleave(kr, pos, d.qk_rope, d.rope_theta);
    }
    kv.len = total;

    let absorbed = match path {
        AttnPath::Auto => seq <= 4,
        AttnPath::Absorbed => true,
        AttnPath::Reconstruct => false,
    };

    let mut ctx = vec![0.0f32; seq * h * d.v_head];
    if absorbed {
        // qabs = K(kv_b)ᵀ q_nope, scores against latents directly. Each
        // (position, head) writes a disjoint ctx slice, so the flattened
        // loop parallelizes with no coordination; per-task scratch is
        // allocated inside (small: kv_lora + context length).
        use rayon::prelude::*;
        let kv_ref = &*kv;
        ctx.par_chunks_mut(d.v_head)
            .enumerate()
            .for_each(|(k, cx)| {
                let (s, head) = (k / h, k % h);
                let n_ctx = pos_base + s + 1;
                let qp = &q[s * h * qh + head * qh..s * h * qh + (head + 1) * qh];
                let (q_nope, q_rope) = qp.split_at(d.qk_nope);
                let rbase = head * kv_head;
                let mut qabs = vec![0.0f32; d.kv_lora];
                for (dd, &qv) in q_nope.iter().enumerate() {
                    w.kv_b.add_scaled_row(rbase + dd, qv, &mut qabs);
                }
                let mut scores = vec![0.0f32; n_ctx];
                for (t, sc) in scores.iter_mut().enumerate() {
                    let lat = &kv_ref.latent[t * d.kv_lora..(t + 1) * d.kv_lora];
                    let kr = &kv_ref.k_rot[t * d.qk_rope..(t + 1) * d.qk_rope];
                    let mut a = 0.0f32;
                    for i in 0..d.kv_lora {
                        a += qabs[i] * lat[i];
                    }
                    for i in 0..d.qk_rope {
                        a += q_rope[i] * kr[i];
                    }
                    *sc = a * d.scale;
                }
                softmax(&mut scores);
                let mut clat = vec![0.0f32; d.kv_lora];
                for (t, &a) in scores.iter().enumerate() {
                    let lat = &kv_ref.latent[t * d.kv_lora..(t + 1) * d.kv_lora];
                    for i in 0..d.kv_lora {
                        clat[i] += a * lat[i];
                    }
                }
                w.kv_b.matvec_rows(rbase + d.qk_nope, d.v_head, &clat, cx);
            });
    } else {
        // Reconstruct k_nope|v for every cached position in one matmul.
        let mut kvr = vec![0.0f32; total * h * kv_head];
        w.kv_b
            .matmul(&mut kvr, &kv.latent[..total * d.kv_lora], total);
        use rayon::prelude::*;
        let kv_ref = &*kv;
        let kvr_ref = &kvr;
        ctx.par_chunks_mut(d.v_head)
            .enumerate()
            .for_each(|(k, cx)| {
                let (s, head) = (k / h, k % h);
                let n_ctx = pos_base + s + 1;
                let qp = &q[s * h * qh + head * qh..s * h * qh + (head + 1) * qh];
                let (q_nope, q_rope) = qp.split_at(d.qk_nope);
                let mut scores = vec![0.0f32; n_ctx];
                for (t, sc) in scores.iter_mut().enumerate() {
                    let kn = &kvr_ref[t * h * kv_head + head * kv_head..][..d.qk_nope];
                    let kr = &kv_ref.k_rot[t * d.qk_rope..(t + 1) * d.qk_rope];
                    let mut a = 0.0f32;
                    for i in 0..d.qk_nope {
                        a += q_nope[i] * kn[i];
                    }
                    for i in 0..d.qk_rope {
                        a += q_rope[i] * kr[i];
                    }
                    *sc = a * d.scale;
                }
                softmax(&mut scores);
                cx.iter_mut().for_each(|v| *v = 0.0);
                for (t, &a) in scores.iter().enumerate() {
                    let vv = &kvr_ref[t * h * kv_head + head * kv_head + d.qk_nope..][..d.v_head];
                    for i in 0..d.v_head {
                        cx[i] += a * vv[i];
                    }
                }
            });
    }

    let mut out = vec![0.0f32; seq * hd];
    w.o_proj.matmul(&mut out, &ctx, seq);
    out
}

/// Stateless full-sequence forward (fresh cache, reconstruct path).
/// The oracle tests validate this; the cached paths are validated against
/// it and against each other.
pub fn mla_forward(d: &MlaDims, w: &MlaWeights, x: &[f32], seq: usize) -> Vec<f32> {
    let mut kv = LayerKvCache::default();
    mla_forward_cached(d, w, x, seq, &mut kv, AttnPath::Reconstruct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use undertow_core::sample::Pcg32;

    #[test]
    fn rope_pos0_is_identity() {
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

    fn tiny_dims() -> MlaDims {
        MlaDims {
            hidden: 16,
            num_heads: 2,
            qk_nope: 6,
            qk_rope: 4,
            v_head: 5,
            kv_lora: 8,
            rope_theta: 10000.0,
            rms_eps: 1e-6,
            scale: 1.0 / (10.0f32).sqrt(),
        }
    }

    fn mat(rng: &mut Pcg32, o: usize, i: usize) -> QTensor {
        let data: Vec<f32> = (0..o * i).map(|_| rng.normal() * 0.15).collect();
        QTensor::from_f32(data, o, i)
    }

    fn tiny_weights(d: &MlaDims, rng: &mut Pcg32) -> MlaWeights {
        let h = d.num_heads;
        let qh = d.qk_nope + d.qk_rope;
        let kvh = d.qk_nope + d.v_head;
        MlaWeights {
            query: QueryProj::Direct {
                q_proj: mat(rng, h * qh, d.hidden),
            },
            kv_a: mat(rng, d.kv_lora + d.qk_rope, d.hidden),
            kv_a_norm: (0..d.kv_lora).map(|_| 1.0 + rng.normal() * 0.05).collect(),
            kv_b: mat(rng, h * kvh, d.kv_lora),
            o_proj: mat(rng, d.hidden, h * d.v_head),
        }
    }

    /// The three ways to run attention over the same tokens must agree:
    /// one-shot reconstruct, incremental reconstruct, incremental absorbed.
    #[test]
    fn cached_paths_match_stateless() {
        let d = tiny_dims();
        let mut rng = Pcg32::new(99, 5);
        let w = tiny_weights(&d, &mut rng);
        let seq = 7;
        let x: Vec<f32> = (0..seq * d.hidden).map(|_| rng.normal() * 0.3).collect();

        let full = mla_forward(&d, &w, &x, seq);

        for path in [AttnPath::Reconstruct, AttnPath::Absorbed] {
            let mut kv = LayerKvCache::default();
            // Prefill the first 4 tokens, then decode 3 one at a time.
            let mut incremental = mla_forward_cached(&d, &w, &x[..4 * d.hidden], 4, &mut kv, path);
            for s in 4..seq {
                let step = mla_forward_cached(
                    &d,
                    &w,
                    &x[s * d.hidden..(s + 1) * d.hidden],
                    1,
                    &mut kv,
                    path,
                );
                incremental.extend(step);
            }
            assert_eq!(kv.len, seq);
            let max_err = full
                .iter()
                .zip(&incremental)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max_err < 1e-4, "{path:?} diverged: {max_err}");
        }
    }

    #[test]
    fn absorbed_equals_reconstruct_with_quantized_kv_b() {
        let d = tiny_dims();
        let mut rng = Pcg32::new(7, 3);
        let mut w = tiny_weights(&d, &mut rng);
        // Quantize kv_b (the matrix both paths traverse differently).
        let deq = w.kv_b.dequantize();
        w.kv_b = QTensor::quantize(
            &deq,
            w.kv_b.out_dim(),
            w.kv_b.in_dim(),
            undertow_core::QuantFormat::Int8,
        )
        .unwrap();
        let seq = 5;
        let x: Vec<f32> = (0..seq * d.hidden).map(|_| rng.normal() * 0.3).collect();
        let mut kv_a = LayerKvCache::default();
        let mut kv_b = LayerKvCache::default();
        let a = mla_forward_cached(&d, &w, &x, seq, &mut kv_a, AttnPath::Absorbed);
        let b = mla_forward_cached(&d, &w, &x, seq, &mut kv_b, AttnPath::Reconstruct);
        let max_err = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 1e-4, "absorbed vs reconstruct: {max_err}");
    }
}
