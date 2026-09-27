//! Quantized weight storage and matmul kernels.
//!
//! Everything the engine multiplies against a weight matrix goes through
//! [`QTensor`]: f32, int8 or int4 with symmetric per-output-row scales.
//! Kernels are scalar and written for auditability; SIMD variants will be
//! validated against these before they are trusted.
//!
//! Kernel semantics (fixed, tests depend on them):
//!   `out[o] = scale[o] * Σ_i x[i] * q[o][i]`
//! with the sum accumulated in f32 in index order. Dequantization happens
//! per element inside the accumulation, the scale is applied once per row.
//!
//! This crate stays free of workspace dependencies on purpose.

mod kernels;
// Miri interprets rather than executes, so NEON intrinsics are out of its
// reach; under Miri everything routes through the scalar reference path.
#[cfg(all(target_arch = "aarch64", not(miri)))]
mod neon;

#[cfg(all(target_arch = "x86_64", not(miri)))]
pub mod avx2;

#[cfg(all(target_arch = "x86_64", not(miri)))]
pub mod avx512;

mod qtensor;

pub use qtensor::{QTensor, QuantFormat};

/// Opt-in fast int8 path: quantize activations per row and accumulate in
/// integer SDOT lanes. Roughly 2 to 4x on int8 matmuls, at the cost of
/// activation quantization error (~1e-2 relative), so it changes numerics
/// and stays off until enabled explicitly (CLI/server `--fast-int8`).
static FAST_INT8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_fast_int8(enabled: bool) {
    FAST_INT8.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

pub fn fast_int8_enabled() -> bool {
    FAST_INT8.load(std::sync::atomic::Ordering::Relaxed)
}

/// Quantize one activation vector symmetrically to i8 with a single scale.
pub fn quantize_activations(x: &[f32], out: &mut [i8]) -> f32 {
    debug_assert_eq!(x.len(), out.len());
    let max = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
    if max == 0.0 {
        out.iter_mut().for_each(|q| *q = 0);
        return 0.0;
    }
    let scale = max / 127.0;
    let inv = 127.0 / max;
    for (q, &v) in out.iter_mut().zip(x) {
        *q = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

// Only the NEON parity tests use this; on other architectures the module
// would be dead code and fail `-D warnings` in CI.
#[cfg(all(test, any(target_arch = "aarch64", target_arch = "x86_64"), not(miri)))]
pub(crate) mod tests_rng {
    /// Tiny xorshift for kernel property tests (not the oracle RNG).
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            Self(seed.max(1))
        }

        pub fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Uniform-ish in [-1, 1].
        pub fn f32_sym(&mut self) -> f32 {
            ((self.next() >> 40) as f32 / (1u64 << 23) as f32) * 2.0 - 1.0
        }
    }
}

/// RMSNorm: `out[i] = x[i] / rms(x) * weight[i]`, mean computed in f64
/// (norms are numerically sensitive; they stay f32 end to end).
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

/// `out[s, o] = Σ_i x[s, i] * w[o, i]` — plain f32 slices, row-major,
/// `w` is `[out_dim, in_dim]`. Kept public for callers that hold raw f32
/// (router gates, norm-adjacent small matmuls).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        let w = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let x = [1., 2., 3.];
        let mut out = [0.; 3];
        matmul(&mut out, &x, &w, 1, 3, 3);
        assert_eq!(out, x);
    }

    #[test]
    fn matmul_hand_computed() {
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
