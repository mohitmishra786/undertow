//! NEON (aarch64) kernels.
//!
//! Same semantics as the scalar kernels in `kernels.rs`, which remain the
//! source of truth: every function here is property-tested against its
//! scalar counterpart on random inputs, including odd dimensions and the
//! packed-nibble tail. Accumulation happens in four independent f32 lanes
//! that are summed at the end, so results can differ from scalar by normal
//! f32 reassociation noise (~1e-6 relative), never more.
//!
//! NEON is baseline on aarch64; no runtime feature detection is needed.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

/// Dot product of `x` and `w` (f32), 16 elements per iteration.
#[inline]
unsafe fn dot_f32(x: &[f32], w: &[f32], n: usize) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let chunks = n / 16;
    for c in 0..chunks {
        let i = c * 16;
        acc0 = vfmaq_f32(
            acc0,
            vld1q_f32(x.as_ptr().add(i)),
            vld1q_f32(w.as_ptr().add(i)),
        );
        acc1 = vfmaq_f32(
            acc1,
            vld1q_f32(x.as_ptr().add(i + 4)),
            vld1q_f32(w.as_ptr().add(i + 4)),
        );
        acc2 = vfmaq_f32(
            acc2,
            vld1q_f32(x.as_ptr().add(i + 8)),
            vld1q_f32(w.as_ptr().add(i + 8)),
        );
        acc3 = vfmaq_f32(
            acc3,
            vld1q_f32(x.as_ptr().add(i + 12)),
            vld1q_f32(w.as_ptr().add(i + 12)),
        );
    }
    let mut acc = vaddvq_f32(acc0) + vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3);
    for i in chunks * 16..n {
        acc += x[i] * w[i];
    }
    acc
}

/// Dot product of f32 `x` against an i8 row, 16 elements per iteration.
#[inline]
unsafe fn dot_i8(x: &[f32], q: &[i8], n: usize) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let chunks = n / 16;
    for c in 0..chunks {
        let i = c * 16;
        let qv = vld1q_s8(q.as_ptr().add(i));
        let lo16 = vmovl_s8(vget_low_s8(qv));
        let hi16 = vmovl_s8(vget_high_s8(qv));
        let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
        let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
        let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
        let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
        acc0 = vfmaq_f32(acc0, vld1q_f32(x.as_ptr().add(i)), f0);
        acc1 = vfmaq_f32(acc1, vld1q_f32(x.as_ptr().add(i + 4)), f1);
        acc2 = vfmaq_f32(acc2, vld1q_f32(x.as_ptr().add(i + 8)), f2);
        acc3 = vfmaq_f32(acc3, vld1q_f32(x.as_ptr().add(i + 12)), f3);
    }
    let mut acc = vaddvq_f32(acc0) + vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3);
    for i in chunks * 16..n {
        acc += x[i] * q[i] as f32;
    }
    acc
}

/// Dot product of f32 `x` against a packed-int4 row (low nibble first),
/// 16 logical elements (8 bytes) per iteration.
#[inline]
unsafe fn dot_i4(x: &[f32], packed: &[u8], n: usize) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let chunks = n / 16;
    let low_mask = vdup_n_u8(0x0F);
    for c in 0..chunks {
        let b = vld1_u8(packed.as_ptr().add(c * 8));
        // Even logical indices live in low nibbles, odd in high; zip to
        // restore interleaved order, then sign-extend 4-bit two's
        // complement by shifting through the top of the i8 lane.
        let even = vand_u8(b, low_mask);
        let odd = vshr_n_u8(b, 4);
        let inter = vzip1_u8(even, odd); // only 8 lanes needed per half
        let inter2 = vzip2_u8(even, odd);
        let q16 = vcombine_u8(inter, inter2);
        let signed = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(q16), 4), 4);
        let lo16 = vmovl_s8(vget_low_s8(signed));
        let hi16 = vmovl_s8(vget_high_s8(signed));
        let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
        let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
        let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
        let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
        let i = c * 16;
        acc0 = vfmaq_f32(acc0, vld1q_f32(x.as_ptr().add(i)), f0);
        acc1 = vfmaq_f32(acc1, vld1q_f32(x.as_ptr().add(i + 4)), f1);
        acc2 = vfmaq_f32(acc2, vld1q_f32(x.as_ptr().add(i + 8)), f2);
        acc3 = vfmaq_f32(acc3, vld1q_f32(x.as_ptr().add(i + 12)), f3);
    }
    let mut acc = vaddvq_f32(acc0) + vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3);
    for (i, &xv) in x.iter().enumerate().take(n).skip(chunks * 16) {
        acc += xv * crate::kernels::unpack_i4(packed, i) as f32;
    }
    acc
}

/// Integer path availability. The widening-multiply idiom below is
/// baseline NEON, so this is always true on aarch64; kept as a function
/// so the dispatch reads the same once SDOT intrinsics stabilize and this
/// becomes a real feature check.
pub fn dotprod_available() -> bool {
    true
}

/// i8 x i8 -> i32 dot product, 16 lanes per iteration via widening
/// multiply (`vmull_s8`) and pairwise accumulate (`vpadalq_s16`). Safe
/// against overflow: 8 products of at most 127*127 sum to < 2^17 in i16
/// pairs, accumulated into i32 lanes. Upgrade to SDOT (`vdotq_s32`) when
/// std::arch stabilizes it.
///
/// # Safety
/// Both slices must be at least `n` long.
unsafe fn dot_i8_sdot(a: &[i8], b: &[i8], n: usize) -> i32 {
    let mut acc = vdupq_n_s32(0);
    let chunks = n / 16;
    for c in 0..chunks {
        let av = vld1q_s8(a.as_ptr().add(c * 16));
        let bv = vld1q_s8(b.as_ptr().add(c * 16));
        let lo = vmull_s8(vget_low_s8(av), vget_low_s8(bv));
        let hi = vmull_s8(vget_high_s8(av), vget_high_s8(bv));
        acc = vpadalq_s16(acc, lo);
        acc = vpadalq_s16(acc, hi);
    }
    let mut total = vaddvq_s32(acc);
    for i in chunks * 16..n {
        total += a[i] as i32 * b[i] as i32;
    }
    total
}

/// Fast int8 matmul: activations quantized per row (one scale each),
/// integer accumulation, result rescaled by `a_scale * w_scale`. Only
/// called when the fast-int8 switch is on and SDOT exists; numerics carry
/// activation quantization error, which is why this is opt-in.
pub fn matmul_i8_sdot(
    out: &mut [f32],
    x: &[f32],
    q: &[i8],
    scales: &[f32],
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert!(dotprod_available());
    let mut qa = vec![0i8; in_dim];
    for s in 0..seq {
        let xs = &x[s * in_dim..(s + 1) * in_dim];
        let a_scale = crate::quantize_activations(xs, &mut qa);
        let os = &mut out[s * out_dim..(s + 1) * out_dim];
        for (o, oo) in os.iter_mut().enumerate() {
            let row = &q[o * in_dim..(o + 1) * in_dim];
            // SAFETY: dotprod availability asserted above; both slices are
            // exactly in_dim long.
            let acc = unsafe { dot_i8_sdot(&qa, row, in_dim) };
            *oo = acc as f32 * a_scale * scales[o];
        }
    }
}

pub fn matmul_f32(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) {
    for s in 0..seq {
        let xs = &x[s * in_dim..(s + 1) * in_dim];
        let os = &mut out[s * out_dim..(s + 1) * out_dim];
        for (o, oo) in os.iter_mut().enumerate() {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            // SAFETY: xs and wr are both exactly in_dim long; dot_f32 only
            // reads full 16-lane chunks inside that bound plus a scalar
            // tail.
            *oo = unsafe { dot_f32(xs, wr, in_dim) };
        }
    }
}

pub fn matmul_i8(
    out: &mut [f32],
    x: &[f32],
    q: &[i8],
    scales: &[f32],
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) {
    for s in 0..seq {
        let xs = &x[s * in_dim..(s + 1) * in_dim];
        let os = &mut out[s * out_dim..(s + 1) * out_dim];
        for (o, oo) in os.iter_mut().enumerate() {
            let row = &q[o * in_dim..(o + 1) * in_dim];
            // SAFETY: bounds as in matmul_f32.
            *oo = unsafe { dot_i8(xs, row, in_dim) } * scales[o];
        }
    }
}

pub fn matmul_i4(
    out: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scales: &[f32],
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) {
    let row_bytes = in_dim.div_ceil(2);
    for s in 0..seq {
        let xs = &x[s * in_dim..(s + 1) * in_dim];
        let os = &mut out[s * out_dim..(s + 1) * out_dim];
        for (o, oo) in os.iter_mut().enumerate() {
            let row = &packed[o * row_bytes..(o + 1) * row_bytes];
            // SAFETY: 16 logical elements consume 8 bytes; chunks stay
            // within row_bytes, tail is scalar via unpack_i4.
            *oo = unsafe { dot_i4(xs, row, in_dim) } * scales[o];
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{QTensor, QuantFormat};

    /// The integer-accumulation path differs from scalar only by
    /// activation quantization: error stays within a few parts in a
    /// hundred of the row magnitude.
    #[test]
    fn fast_int8_kernel_close_to_scalar() {
        let mut rng = crate::tests_rng::Rng::new(0xFA57);
        for &(o, i) in &[(4usize, 64usize), (7, 130), (3, 16)] {
            let w: Vec<f32> = (0..o * i).map(|_| rng.f32_sym()).collect();
            let x: Vec<f32> = (0..2 * i).map(|_| rng.f32_sym()).collect();
            let t = QTensor::quantize(&w, o, i, QuantFormat::Int8).unwrap();
            let QTensor::Int8 { q, scales, .. } = &t else {
                unreachable!()
            };
            let mut fast = vec![0f32; 2 * o];
            super::matmul_i8_sdot(&mut fast, &x, q, scales, 2, i, o);
            let mut exact = vec![0f32; 2 * o];
            t.matmul_scalar(&mut exact, &x, 2);
            let mag = exact.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            for (f, e) in fast.iter().zip(&exact) {
                assert!(
                    (f - e).abs() <= 0.03 * mag,
                    "[{o},{i}] fast {f} vs exact {e} (mag {mag})"
                );
            }
        }
    }

    /// NEON matmul must agree with scalar matmul on random inputs for
    /// every format and awkward dimension.
    #[test]
    fn neon_matches_scalar_for_all_formats() {
        let mut rng = crate::tests_rng::Rng::new(0xDEC0DE);
        for &(o, i) in &[
            (1usize, 1usize),
            (3, 16),
            (5, 17),
            (8, 64),
            (7, 130),
            (2, 31),
        ] {
            let w: Vec<f32> = (0..o * i).map(|_| rng.f32_sym()).collect();
            let x: Vec<f32> = (0..3 * i).map(|_| rng.f32_sym()).collect();
            for fmt in [QuantFormat::F32, QuantFormat::Int8, QuantFormat::Int4] {
                let t = QTensor::quantize(&w, o, i, fmt).unwrap();
                let mut simd = vec![0f32; 3 * o];
                t.matmul(&mut simd, &x, 3); // dispatches to NEON on aarch64
                let mut scalar = vec![0f32; 3 * o];
                t.matmul_scalar(&mut scalar, &x, 3);
                for (a, b) in simd.iter().zip(&scalar) {
                    assert!(
                        (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                        "{fmt:?} [{o},{i}]: {a} vs {b}"
                    );
                }
            }
        }
    }
}
