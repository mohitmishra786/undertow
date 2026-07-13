//! Scalar dequant-on-use matmul kernels.
//!
//! Accumulation is f32 in index order; the per-row scale is applied once
//! after the row sum. SIMD implementations (NEON/AVX2) will live beside
//! these and must be validated against them on random inputs before use.

/// Sign-extend the 4-bit code at logical index `i` of a packed row
/// (low nibble first).
#[inline(always)]
pub fn unpack_i4(row: &[u8], i: usize) -> i8 {
    let b = row[i / 2];
    let nib = if i.is_multiple_of(2) {
        b & 0x0F
    } else {
        b >> 4
    };
    // Two's-complement sign extension of a 4-bit value.
    ((nib as i8) << 4) >> 4
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
    debug_assert_eq!(q.len(), out_dim * in_dim);
    debug_assert_eq!(scales.len(), out_dim);
    for s in 0..seq {
        let xs = &x[s * in_dim..(s + 1) * in_dim];
        let os = &mut out[s * out_dim..(s + 1) * out_dim];
        for (o, oo) in os.iter_mut().enumerate() {
            let row = &q[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0f32;
            for i in 0..in_dim {
                acc += xs[i] * row[i] as f32;
            }
            *oo = acc * scales[o];
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
    debug_assert_eq!(packed.len(), out_dim * row_bytes);
    debug_assert_eq!(scales.len(), out_dim);
    for s in 0..seq {
        let xs = &x[s * in_dim..(s + 1) * in_dim];
        let os = &mut out[s * out_dim..(s + 1) * out_dim];
        for (o, oo) in os.iter_mut().enumerate() {
            let row = &packed[o * row_bytes..(o + 1) * row_bytes];
            let mut acc = 0f32;
            // Process full byte pairs, then the odd tail if in_dim is odd.
            let pairs = in_dim / 2;
            for p in 0..pairs {
                let b = row[p];
                let lo = ((b as i8) << 4) >> 4;
                let hi = (b as i8) >> 4;
                acc += xs[2 * p] * lo as f32;
                acc += xs[2 * p + 1] * hi as f32;
            }
            if in_dim % 2 == 1 {
                let b = row[pairs];
                let lo = ((b as i8) << 4) >> 4;
                acc += xs[in_dim - 1] * lo as f32;
            }
            *oo = acc * scales[o];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_covers_full_nibble_range() {
        // Pack every code -8..=7 and read it back.
        for v in -8i8..=7 {
            let nib = (v as u8) & 0x0F;
            let row = [nib, nib << 4];
            assert_eq!(unpack_i4(&row, 0), v);
            assert_eq!(unpack_i4(&row, 3), v);
        }
    }

    #[test]
    fn i4_matmul_handles_odd_tail() {
        // in_dim = 3: codes [2, -3, 5], scale 0.5, x = [1, 2, 3]
        // expect 0.5 * (2 - 6 + 15) = 5.5
        let packed = [((2u8) & 0xF) | (((-3i8 as u8) & 0xF) << 4), (5u8) & 0xF];
        let scales = [0.5f32];
        let x = [1f32, 2.0, 3.0];
        let mut out = [0f32];
        matmul_i4(&mut out, &x, &packed, &scales, 1, 3, 1);
        assert!((out[0] - 5.5).abs() < 1e-6);
    }

    #[test]
    fn i8_matmul_hand_computed() {
        // q = [[10, -20], [127, 1]], scales = [0.1, 0.01], x = [2, 3]
        let q = [10i8, -20, 127, 1];
        let scales = [0.1f32, 0.01];
        let x = [2f32, 3.0];
        let mut out = [0f32; 2];
        matmul_i8(&mut out, &x, &q, &scales, 1, 2, 2);
        assert!((out[0] - 0.1 * (20.0 - 60.0)).abs() < 1e-6);
        assert!((out[1] - 0.01 * (254.0 + 3.0)).abs() < 1e-6);
    }
}
