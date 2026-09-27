//! AVX2+FMA (x86_64) kernels.
//!
//! Same contract as the NEON module: identical semantics to the scalar
//! kernels up to f32 lane reassociation, property-tested against them.
//! Availability is checked once at runtime; without AVX2+FMA the dispatch
//! falls back to scalar.

#![cfg(all(target_arch = "x86_64", not(miri)))]

use std::arch::x86_64::*;

pub fn avx2_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32(x: &[f32], w: &[f32], n: usize) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let chunks = n / 16;
    for c in 0..chunks {
        let i = c * 16;
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr().add(i)),
            _mm256_loadu_ps(w.as_ptr().add(i)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr().add(i + 8)),
            _mm256_loadu_ps(w.as_ptr().add(i + 8)),
            acc1,
        );
    }
    let mut acc = hsum256(acc0) + hsum256(acc1);
    for (i, &xv) in x.iter().enumerate().take(n).skip(chunks * 16) {
        acc += xv * w[i];
    }
    acc
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i8(x: &[f32], q: &[i8], n: usize) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let chunks = n / 16;
    for c in 0..chunks {
        let i = c * 16;
        // 16 i8 -> two 8-lane f32 vectors.
        let qv = _mm_loadu_si128(q.as_ptr().add(i) as *const __m128i);
        let lo16 = _mm256_cvtepi8_epi32(qv);
        let hi16 = _mm256_cvtepi8_epi32(_mm_srli_si128(qv, 8));
        let f0 = _mm256_cvtepi32_ps(lo16);
        let f1 = _mm256_cvtepi32_ps(hi16);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i)), f0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), f1, acc1);
    }
    let mut acc = hsum256(acc0) + hsum256(acc1);
    for (i, &xv) in x.iter().enumerate().take(n).skip(chunks * 16) {
        acc += xv * q[i] as f32;
    }
    acc
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i4(x: &[f32], packed: &[u8], n: usize) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let low_mask = _mm_set1_epi8(0x0F);
    let chunks = n / 16;
    for c in 0..chunks {
        // 8 bytes hold 16 logical nibbles (low nibble = even index).
        let bytes = _mm_loadl_epi64(packed.as_ptr().add(c * 8) as *const __m128i);
        let even = _mm_and_si128(bytes, low_mask);
        let odd = _mm_and_si128(_mm_srli_epi16(bytes, 4), low_mask);
        // Interleave to restore logical order, then sign-extend 4-bit
        // two's complement through the top of each i8 lane.
        let inter = _mm_unpacklo_epi8(even, odd);
        // Sign-extend the 4-bit two's-complement values inside 16-bit
        // lanes: the nibble must travel to the top of the lane and back,
        // so the shift is 12, not the 4 the 8-bit NEON idiom uses. The
        // 4-shift variant leaves negative nibbles positive, which is
        // exactly the bug the parity test on real AVX2 hardware caught.
        let signed = _mm_srai_epi16(_mm_slli_epi16(_mm_cvtepi8_epi16(inter), 12), 12);
        let signed_hi = _mm_srai_epi16(
            _mm_slli_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(inter, 8)), 12),
            12,
        );
        let f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(signed));
        let f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(signed_hi));
        let i = c * 16;
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i)), f0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), f1, acc1);
    }
    let mut acc = hsum256(acc0) + hsum256(acc1);
    for (i, &xv) in x.iter().enumerate().take(n).skip(chunks * 16) {
        acc += xv * crate::kernels::unpack_i4(packed, i) as f32;
    }
    acc
}

macro_rules! rowwise {
    ($dot:ident, $out:ident, $x:ident, $rows:expr, $scales:expr, $seq:ident, $id:ident, $od:ident, $row_len:expr) => {
        for s in 0..$seq {
            let xs = &$x[s * $id..(s + 1) * $id];
            let os = &mut $out[s * $od..(s + 1) * $od];
            for (o, oo) in os.iter_mut().enumerate() {
                let row = &$rows[o * $row_len..(o + 1) * $row_len];
                // SAFETY: AVX2+FMA availability is checked by the caller;
                // slices are exactly one row / one activation vector.
                let acc = unsafe { $dot(xs, row, $id) };
                *oo = acc * $scales(o);
            }
        }
    };
}

pub fn matmul_f32(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) {
    assert!(avx2_available(), "AVX2+FMA required");
    rowwise!(
        dot_f32,
        out,
        x,
        w,
        |_o| 1.0f32,
        seq,
        in_dim,
        out_dim,
        in_dim
    );
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
    assert!(avx2_available(), "AVX2+FMA required");
    rowwise!(
        dot_i8,
        out,
        x,
        q,
        |o: usize| scales[o],
        seq,
        in_dim,
        out_dim,
        in_dim
    );
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
    assert!(avx2_available(), "AVX2+FMA required");
    let row_bytes = in_dim.div_ceil(2);
    rowwise!(
        dot_i4,
        out,
        x,
        packed,
        |o: usize| scales[o],
        seq,
        in_dim,
        out_dim,
        row_bytes
    );
}

#[cfg(test)]
mod tests {
    use crate::{QTensor, QuantFormat};

    #[test]
    fn avx2_matches_scalar_for_all_formats() {
        if !super::avx2_available() {
            assert!(
                std::env::var_os("CI").is_none(),
                "CI x86 runners have AVX2+FMA; refusing to skip the parity test there"
            );
            eprintln!("AVX2 not available on this CPU; skipping");
            return;
        }
        let mut rng = crate::tests_rng::Rng::new(0xA5A5);
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
                t.matmul(&mut simd, &x, 3);
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
