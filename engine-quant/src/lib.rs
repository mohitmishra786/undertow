//! Matmul kernels and small math primitives.
//!
//! Phase 0 ships the *scalar f32 reference path only*. Every future kernel
//! (int8/int4 dequant-on-use, NEON/AVX2 SIMD) must be validated against
//! these functions before it is trusted. Keep this crate dependency-free
//! and slice-based so the reference path stays trivially auditable.

/// Storage formats the quantized kernels will support. Only `F32` has a
/// compute path today; the variants exist so container/converter code can
/// already speak the right vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    F32,
    /// Per-output-row scale, 8-bit signed weights. (Phase 1)
    Int8,
    /// Per-output-row scale, packed 4-bit. (Phase 1)
    Int4,
    /// Per-output-row scale, packed 2-bit. (Phase 2)
    Int2,
}

/// `out[s, o] = Σ_i x[s, i] * w[o, i]` — row-major everywhere,
/// `w` is `[out_dim, in_dim]` (PyTorch `nn.Linear` layout, i.e. `y = W x`).
///
/// * `x`: `[seq, in_dim]`
/// * `out`: `[seq, out_dim]`
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], seq: usize, in_dim: usize, out_dim: usize) {
    assert_eq!(x.len(), seq * in_dim, "x shape");
    assert_eq!(w.len(), out_dim * in_dim, "w shape");
    assert_eq!(out.len(), seq * out_dim, "out shape");
    for s in 0..seq {
        let xs = &x[s * in_dim..(s + 1) * in_dim];
        let os = &mut out[s * out_dim..(s + 1) * out_dim];
        for (o, oo) in os.iter_mut().enumerate() {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += xs[i] * wr[i];
            }
            *oo = acc;
        }
    }
}

/// RMSNorm: `out[i] = x[i] / rms(x) * weight[i]`, mean computed in f64
/// (matches the f32-sensitive parts of the reference C implementation).
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(weight.len(), n);
    assert_eq!(out.len(), n);
    let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
    let r = 1.0 / ((ms as f32) + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * r * weight[i];
    }
}

/// In-place numerically-stable softmax.
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        // w = I(3), x = [1,2,3] -> out = x
        let w = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let x = [1., 2., 3.];
        let mut out = [0.; 3];
        matmul(&mut out, &x, &w, 1, 3, 3);
        assert_eq!(out, x);
    }

    #[test]
    fn matmul_hand_computed() {
        // w [2,3] = [[1,2,3],[4,5,6]], x [2,3] two rows
        let w = [1., 2., 3., 4., 5., 6.];
        let x = [1., 1., 1., 0., 1., 2.];
        let mut out = [0.; 4];
        matmul(&mut out, &x, &w, 2, 3, 2);
        assert_eq!(out, [6., 15., 8., 17.]);
    }

    #[test]
    fn rmsnorm_unit_weight() {
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        rmsnorm(&mut out, &x, &w, 0.0);
        // rms = sqrt((9+16)/2) = sqrt(12.5)
        let r = 12.5f32.sqrt();
        assert!((out[0] - 3.0 / r).abs() < 1e-6);
        assert!((out[1] - 4.0 / r).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(x[3] > x[2] && x[2] > x[1] && x[1] > x[0]);
    }
}
