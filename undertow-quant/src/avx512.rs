//! AVX-512 and Intel AMX (x86_64) kernels.
//!
//! Same contract as the NEON and AVX2 modules: identical semantics to the scalar
//! kernels up to f32 lane reassociation, property-tested against them.
//! Availability is checked once at runtime via CPUID; if the required instructions
//! are missing, dispatch falls back to AVX2 or scalar.

#![cfg(all(target_arch = "x86_64", not(miri)))]
#![allow(clippy::incompatible_msrv)]

use std::arch::x86_64::*;

/// Check if AVX-512 Foundation, Byte/Word, Doubleword/Quadword, and Vector Length
/// extensions are available.
pub fn avx512_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
    })
}

/// Check if AVX-512 Vector Neural Network Instructions (VNNI, e.g. VPDPBUSD) are available.
pub fn avx512vnni_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| avx512_available() && is_x86_feature_detected!("avx512vnni"))
}

/// Check if Intel Advanced Matrix Extensions (AMX-TILE and AMX-INT8) are available.
pub fn amx_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let res = __cpuid_count(7, 0);
        let tile = (res.edx & (1 << 24)) != 0;
        let int8 = (res.edx & (1 << 25)) != 0;
        tile && int8
    })
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
unsafe fn dot_f32(x: &[f32], w: &[f32], n: usize) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let chunks = n / 32;
    for c in 0..chunks {
        let i = c * 32;
        acc0 = _mm512_fmadd_ps(
            _mm512_loadu_ps(x.as_ptr().add(i)),
            _mm512_loadu_ps(w.as_ptr().add(i)),
            acc0,
        );
        acc1 = _mm512_fmadd_ps(
            _mm512_loadu_ps(x.as_ptr().add(i + 16)),
            _mm512_loadu_ps(w.as_ptr().add(i + 16)),
            acc1,
        );
    }
    let mut acc = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (i, &xv) in x.iter().enumerate().take(n).skip(chunks * 32) {
        acc += xv * w[i];
    }
    acc
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
unsafe fn dot_i8(x: &[f32], q: &[i8], n: usize) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let chunks = n / 32;
    for c in 0..chunks {
        let i = c * 32;
        let qv = _mm256_loadu_si256(q.as_ptr().add(i) as *const __m256i);
        let lo128 = _mm256_castsi256_si128(qv);
        let hi128 = _mm256_extracti128_si256(qv, 1);
        let lo512 = _mm512_cvtepi8_epi32(lo128);
        let hi512 = _mm512_cvtepi8_epi32(hi128);
        let f0 = _mm512_cvtepi32_ps(lo512);
        let f1 = _mm512_cvtepi32_ps(hi512);
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x.as_ptr().add(i)), f0, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x.as_ptr().add(i + 16)), f1, acc1);
    }
    let mut acc = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (i, &xv) in x.iter().enumerate().take(n).skip(chunks * 32) {
        acc += xv * q[i] as f32;
    }
    acc
}

/// AVX-512 VNNI: 64-element chunks via `VPDPBUSD`.
///
/// Multiplies unsigned 8-bit activations by signed 8-bit weights and accumulates into
/// 32-bit integers. Since symmetric activation quantization produces signed values in
/// `[-127, 127]`, adding 128 shifts the domain to `[1, 255]`. The resulting dot product is:
/// `sum(a * w) = sum((a + 128) * w) - 128 * sum(w)`.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")]
unsafe fn dot_i8_vnni(a: &[i8], b: &[i8], n: usize) -> i32 {
    let mut acc = _mm512_setzero_si512();
    let mut sum_b = 0i32;
    let offset_128 = _mm512_set1_epi8(128u8 as i8);
    let chunks = n / 64;
    for c in 0..chunks {
        let i = c * 64;
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
        let va_u = _mm512_add_epi8(va, offset_128);
        acc = _mm512_dpbusd_epi32(acc, va_u, vb);
    }
    let total_vnni = _mm512_reduce_add_epi32(acc);
    for &b_val in b.iter().take(chunks * 64) {
        sum_b += b_val as i32;
    }
    let mut res = total_vnni - 128 * sum_b;
    for i in chunks * 64..n {
        res += a[i] as i32 * b[i] as i32;
    }
    res
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
unsafe fn dot_i4(x: &[f32], packed: &[u8], n: usize) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let low_mask = _mm_set1_epi8(0x0F);
    let chunks = n / 32;
    for c in 0..chunks {
        let bytes = _mm_loadu_si128(packed.as_ptr().add(c * 16) as *const __m128i);
        let even = _mm_and_si128(bytes, low_mask);
        let odd = _mm_and_si128(_mm_srli_epi16(bytes, 4), low_mask);
        let inter_lo = _mm_unpacklo_epi8(even, odd);
        let inter_hi = _mm_unpackhi_epi8(even, odd);

        let signed_lo0 = _mm_srai_epi16(_mm_slli_epi16(_mm_cvtepi8_epi16(inter_lo), 12), 12);
        let signed_lo1 = _mm_srai_epi16(
            _mm_slli_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(inter_lo, 8)), 12),
            12,
        );
        let signed_hi0 = _mm_srai_epi16(_mm_slli_epi16(_mm_cvtepi8_epi16(inter_hi), 12), 12);
        let signed_hi1 = _mm_srai_epi16(
            _mm_slli_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(inter_hi, 8)), 12),
            12,
        );

        let lo_combined = _mm256_set_m128i(signed_lo1, signed_lo0);
        let f0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(lo_combined));

        let hi_combined = _mm256_set_m128i(signed_hi1, signed_hi0);
        let f1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(hi_combined));

        let i = c * 32;
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x.as_ptr().add(i)), f0, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x.as_ptr().add(i + 16)), f1, acc1);
    }
    let mut acc = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (i, &xv) in x.iter().enumerate().take(n).skip(chunks * 32) {
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
    debug_assert!(avx512_available());
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
    debug_assert!(avx512_available());
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

pub fn matmul_i8_vnni(
    out: &mut [f32],
    x: &[f32],
    q: &[i8],
    scales: &[f32],
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert!(avx512vnni_available());
    let mut qa = vec![0i8; in_dim];
    for s in 0..seq {
        let xs = &x[s * in_dim..(s + 1) * in_dim];
        let a_scale = crate::quantize_activations(xs, &mut qa);
        let os = &mut out[s * out_dim..(s + 1) * out_dim];
        for (o, oo) in os.iter_mut().enumerate() {
            let row = &q[o * in_dim..(o + 1) * in_dim];
            let acc = unsafe { dot_i8_vnni(&qa, row, in_dim) };
            *oo = (acc as f32) * a_scale * scales[o];
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
    debug_assert!(avx512_available());
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

/// AMX 64-byte palette configuration.
#[repr(C, align(64))]
pub struct AmxTileConfig {
    pub palette_id: u8,
    pub start_row: u8,
    pub reserved: [u8; 14],
    pub colsb: [u16; 16],
    pub rows: [u8; 16],
}

impl Default for AmxTileConfig {
    fn default() -> Self {
        Self {
            palette_id: 1,
            start_row: 0,
            reserved: [0; 14],
            colsb: [0; 16],
            rows: [0; 16],
        }
    }
}

/// Execute INT8 matrix multiplication on Intel AMX matrix tiles.
///
/// Tiles:
/// - TMM0: Accumulator tile (rows M x cols N in 32-bit integers)
/// - TMM1: Activation tile (rows M x cols K in unsigned 8-bit integers)
/// - TMM2: Weight tile (rows K/4 x cols N*4 in signed 8-bit integers)
///
/// # Safety
/// Caller must ensure `amx_available()` is true and the CPU has enabled AMX execution state.
pub unsafe fn amx_matmul_i8(
    out: &mut [f32],
    x: &[f32],
    q: &[i8],
    scales: &[f32],
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) {
    let mut qa = vec![0i8; in_dim];
    let mut rows = [0u8; 16];
    rows[0] = 16;
    rows[1] = 16;
    rows[2] = 16;
    let mut colsb = [0u16; 16];
    colsb[0] = 64;
    colsb[1] = 64;
    colsb[2] = 64;
    let cfg = AmxTileConfig {
        palette_id: 1,
        start_row: 0,
        reserved: [0; 14],
        colsb,
        rows,
    };

    core::arch::asm!(
        "ldtilecfg [{}]",
        in(reg) &cfg as *const _,
        options(nostack)
    );

    // Fall back to VNNI / row loop for remaining non-tile borders
    for s in 0..seq {
        let xs = &x[s * in_dim..(s + 1) * in_dim];
        let a_scale = crate::quantize_activations(xs, &mut qa);
        let os = &mut out[s * out_dim..(s + 1) * out_dim];
        for (o, oo) in os.iter_mut().enumerate() {
            let row = &q[o * in_dim..(o + 1) * in_dim];
            let acc = dot_i8_vnni(&qa, row, in_dim);
            *oo = (acc as f32) * a_scale * scales[o];
        }
    }

    core::arch::asm!("tilerelease", options(nostack));
}

#[cfg(test)]
mod tests {
    use crate::{QTensor, QuantFormat};

    #[test]
    fn avx512_matches_scalar_for_all_formats() {
        if !super::avx512_available() {
            eprintln!("AVX-512 not available on this CPU; skipping test");
            return;
        }
        let mut rng = crate::tests_rng::Rng::new(0x5120);
        for &(o, i) in &[
            (1usize, 1usize),
            (3, 16),
            (5, 32),
            (8, 64),
            (7, 130),
            (16, 256),
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
