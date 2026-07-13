//! Quantized 2-D weight tensor with symmetric per-output-row scales.
//!
//! Layouts:
//! * `F32`: plain row-major `[out_dim, in_dim]`.
//! * `Int8`: `q[o][i]` in `[-127, 127]`, `w ≈ q * scale[o]`.
//! * `Int4`: two values per byte, low nibble first (even index in the low
//!   nibble), two's-complement nibbles clamped to `[-7, 7]`; each row is
//!   padded to a whole byte, so a row occupies `ceil(in_dim / 2)` bytes.
//!
//! Quantization is symmetric: `scale = max|w| / qmax`, `q = round(w/scale)`.
//! An all-zero row gets `scale = 0` and zero codes, which dequantizes to
//! exact zeros without dividing by zero.

use crate::kernels;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    F32,
    Int8,
    Int4,
}

impl QuantFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" | "F32" => Some(Self::F32),
            "int8" | "Int8" | "i8" => Some(Self::Int8),
            "int4" | "Int4" | "i4" => Some(Self::Int4),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }

    /// Bytes of packed payload for a `[out_dim, in_dim]` matrix
    /// (excluding scales).
    pub fn payload_bytes(self, out_dim: usize, in_dim: usize) -> usize {
        match self {
            Self::F32 => out_dim * in_dim * 4,
            Self::Int8 => out_dim * in_dim,
            Self::Int4 => out_dim * in_dim.div_ceil(2),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QTensor {
    F32 {
        out_dim: usize,
        in_dim: usize,
        data: Vec<f32>,
    },
    Int8 {
        out_dim: usize,
        in_dim: usize,
        q: Vec<i8>,
        scales: Vec<f32>,
    },
    Int4 {
        out_dim: usize,
        in_dim: usize,
        packed: Vec<u8>,
        scales: Vec<f32>,
    },
}

impl QTensor {
    pub fn from_f32(data: Vec<f32>, out_dim: usize, in_dim: usize) -> Self {
        assert_eq!(data.len(), out_dim * in_dim);
        Self::F32 {
            out_dim,
            in_dim,
            data,
        }
    }

    /// Build from packed payload + scales as read from a converted
    /// checkpoint. Validates sizes.
    pub fn from_quantized(
        fmt: QuantFormat,
        payload: Vec<u8>,
        scales: Vec<f32>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self, String> {
        if payload.len() != fmt.payload_bytes(out_dim, in_dim) {
            return Err(format!(
                "payload size {} does not match {}x{} {}",
                payload.len(),
                out_dim,
                in_dim,
                fmt.name()
            ));
        }
        match fmt {
            QuantFormat::F32 => {
                let data = payload
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Ok(Self::F32 {
                    out_dim,
                    in_dim,
                    data,
                })
            }
            QuantFormat::Int8 | QuantFormat::Int4 => {
                if scales.len() != out_dim {
                    return Err(format!(
                        "scales len {} != out_dim {}",
                        scales.len(),
                        out_dim
                    ));
                }
                if fmt == QuantFormat::Int8 {
                    let q = payload.into_iter().map(|b| b as i8).collect();
                    Ok(Self::Int8 {
                        out_dim,
                        in_dim,
                        q,
                        scales,
                    })
                } else {
                    Ok(Self::Int4 {
                        out_dim,
                        in_dim,
                        packed: payload,
                        scales,
                    })
                }
            }
        }
    }

    /// Quantize an f32 matrix. Returns an error on non-finite input, which
    /// a converter must treat as a corrupt source tensor.
    pub fn quantize(
        w: &[f32],
        out_dim: usize,
        in_dim: usize,
        fmt: QuantFormat,
    ) -> Result<Self, String> {
        assert_eq!(w.len(), out_dim * in_dim);
        if let Some(bad) = w.iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "non-finite weight at flat index {bad} (row {}, col {})",
                bad / in_dim,
                bad % in_dim
            ));
        }
        match fmt {
            QuantFormat::F32 => Ok(Self::F32 {
                out_dim,
                in_dim,
                data: w.to_vec(),
            }),
            QuantFormat::Int8 => {
                let mut q = vec![0i8; out_dim * in_dim];
                let mut scales = vec![0f32; out_dim];
                for o in 0..out_dim {
                    let row = &w[o * in_dim..(o + 1) * in_dim];
                    let max = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
                    if max > 0.0 {
                        let s = max / 127.0;
                        scales[o] = s;
                        for i in 0..in_dim {
                            q[o * in_dim + i] = (row[i] / s).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                }
                Ok(Self::Int8 {
                    out_dim,
                    in_dim,
                    q,
                    scales,
                })
            }
            QuantFormat::Int4 => {
                let row_bytes = in_dim.div_ceil(2);
                let mut packed = vec![0u8; out_dim * row_bytes];
                let mut scales = vec![0f32; out_dim];
                for o in 0..out_dim {
                    let row = &w[o * in_dim..(o + 1) * in_dim];
                    let max = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
                    if max > 0.0 {
                        let s = max / 7.0;
                        scales[o] = s;
                        for i in 0..in_dim {
                            let qv = (row[i] / s).round().clamp(-7.0, 7.0) as i8;
                            let nib = (qv as u8) & 0x0F;
                            let byte = &mut packed[o * row_bytes + i / 2];
                            if i % 2 == 0 {
                                *byte |= nib;
                            } else {
                                *byte |= nib << 4;
                            }
                        }
                    }
                }
                Ok(Self::Int4 {
                    out_dim,
                    in_dim,
                    packed,
                    scales,
                })
            }
        }
    }

    pub fn format(&self) -> QuantFormat {
        match self {
            Self::F32 { .. } => QuantFormat::F32,
            Self::Int8 { .. } => QuantFormat::Int8,
            Self::Int4 { .. } => QuantFormat::Int4,
        }
    }

    pub fn out_dim(&self) -> usize {
        match self {
            Self::F32 { out_dim, .. } | Self::Int8 { out_dim, .. } | Self::Int4 { out_dim, .. } => {
                *out_dim
            }
        }
    }

    pub fn in_dim(&self) -> usize {
        match self {
            Self::F32 { in_dim, .. } | Self::Int8 { in_dim, .. } | Self::Int4 { in_dim, .. } => {
                *in_dim
            }
        }
    }

    /// Bytes held in memory (payload + scales), used for cache budgeting.
    pub fn nbytes(&self) -> usize {
        match self {
            Self::F32 { data, .. } => data.len() * 4,
            Self::Int8 { q, scales, .. } => q.len() + scales.len() * 4,
            Self::Int4 { packed, scales, .. } => packed.len() + scales.len() * 4,
        }
    }

    /// Packed payload + scales, for writing to a converted checkpoint.
    /// Scales are empty for F32.
    pub fn to_parts(&self) -> (Vec<u8>, &[f32]) {
        match self {
            Self::F32 { data, .. } => {
                let mut b = Vec::with_capacity(data.len() * 4);
                for v in data {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                (b, &[])
            }
            Self::Int8 { q, scales, .. } => (q.iter().map(|&v| v as u8).collect(), scales),
            Self::Int4 { packed, scales, .. } => (packed.clone(), scales),
        }
    }

    /// Dequantize the full matrix to f32 (reference/debug path).
    pub fn dequantize(&self) -> Vec<f32> {
        let (o, i) = (self.out_dim(), self.in_dim());
        let mut out = vec![0f32; o * i];
        for r in 0..o {
            self.add_scaled_row(r, 1.0, &mut out[r * i..(r + 1) * i]);
        }
        out
    }

    /// `out[s, o] = scale[o] * Σ_i x[s, i] * q[o, i]`.
    ///
    /// Dispatches to NEON on aarch64 and parallelizes across rayon workers
    /// when the work is large enough to pay for it. Every output element's
    /// accumulation order is unchanged by either, so results are identical
    /// to [`Self::matmul_scalar`] up to SIMD lane reassociation only —
    /// threading never changes numerics.
    pub fn matmul(&self, out: &mut [f32], x: &[f32], seq: usize) {
        let (od, id) = (self.out_dim(), self.in_dim());
        assert_eq!(x.len(), seq * id, "x shape");
        assert_eq!(out.len(), seq * od, "out shape");

        // Below this many multiply-accumulates, thread coordination costs
        // more than it saves (measured on M4; conservative for smaller
        // machines).
        const PAR_THRESHOLD: usize = 1 << 18;
        let work = seq * od * id;
        if work >= PAR_THRESHOLD && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            if seq > 1 {
                // One task per position; each computes a full output row.
                out.par_chunks_mut(od)
                    .zip(x.par_chunks(id))
                    .for_each(|(o_row, x_row)| self.matmul_rows(o_row, x_row, 0, od));
            } else {
                // Single position: split the output rows into contiguous
                // blocks, one task per block.
                let blocks = (rayon::current_num_threads() * 4).min(od).max(1);
                let rows_per = od.div_ceil(blocks);
                out.par_chunks_mut(rows_per)
                    .enumerate()
                    .for_each(|(b, o_rows)| {
                        self.matmul_rows(o_rows, x, b * rows_per, o_rows.len())
                    });
            }
            return;
        }
        for s in 0..seq {
            self.matmul_rows(
                &mut out[s * od..(s + 1) * od],
                &x[s * id..(s + 1) * id],
                0,
                od,
            );
        }
    }

    /// Compute output rows `[row_start, row_start + nrows)` for a single
    /// activation vector into `out` (length `nrows`). Rows are contiguous
    /// in the payload, so this is the existing kernels over sub-slices.
    fn matmul_rows(&self, out: &mut [f32], x: &[f32], row_start: usize, nrows: usize) {
        let id = self.in_dim();
        debug_assert_eq!(out.len(), nrows);
        debug_assert!(row_start + nrows <= self.out_dim());
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            match self {
                Self::F32 { data, .. } => crate::neon::matmul_f32(
                    out,
                    x,
                    &data[row_start * id..(row_start + nrows) * id],
                    1,
                    id,
                    nrows,
                ),
                Self::Int8 { q, scales, .. }
                    if crate::fast_int8_enabled() && crate::neon::dotprod_available() =>
                {
                    crate::neon::matmul_i8_sdot(
                        out,
                        x,
                        &q[row_start * id..(row_start + nrows) * id],
                        &scales[row_start..row_start + nrows],
                        1,
                        id,
                        nrows,
                    )
                }
                Self::Int8 { q, scales, .. } => crate::neon::matmul_i8(
                    out,
                    x,
                    &q[row_start * id..(row_start + nrows) * id],
                    &scales[row_start..row_start + nrows],
                    1,
                    id,
                    nrows,
                ),
                Self::Int4 { packed, scales, .. } => {
                    let rb = id.div_ceil(2);
                    crate::neon::matmul_i4(
                        out,
                        x,
                        &packed[row_start * rb..(row_start + nrows) * rb],
                        &scales[row_start..row_start + nrows],
                        1,
                        id,
                        nrows,
                    )
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            if crate::avx2::avx2_available() {
                match self {
                    Self::F32 { data, .. } => crate::avx2::matmul_f32(
                        out,
                        x,
                        &data[row_start * id..(row_start + nrows) * id],
                        1,
                        id,
                        nrows,
                    ),
                    Self::Int8 { q, scales, .. } => crate::avx2::matmul_i8(
                        out,
                        x,
                        &q[row_start * id..(row_start + nrows) * id],
                        &scales[row_start..row_start + nrows],
                        1,
                        id,
                        nrows,
                    ),
                    Self::Int4 { packed, scales, .. } => {
                        let rb = id.div_ceil(2);
                        crate::avx2::matmul_i4(
                            out,
                            x,
                            &packed[row_start * rb..(row_start + nrows) * rb],
                            &scales[row_start..row_start + nrows],
                            1,
                            id,
                            nrows,
                        )
                    }
                }
                return;
            }
        }
        #[cfg(any(not(target_arch = "aarch64"), miri))]
        {
            match self {
                Self::F32 { data, .. } => crate::matmul(
                    out,
                    x,
                    &data[row_start * id..(row_start + nrows) * id],
                    1,
                    id,
                    nrows,
                ),
                Self::Int8 { q, scales, .. } => kernels::matmul_i8(
                    out,
                    x,
                    &q[row_start * id..(row_start + nrows) * id],
                    &scales[row_start..row_start + nrows],
                    1,
                    id,
                    nrows,
                ),
                Self::Int4 { packed, scales, .. } => {
                    let rb = id.div_ceil(2);
                    kernels::matmul_i4(
                        out,
                        x,
                        &packed[row_start * rb..(row_start + nrows) * rb],
                        &scales[row_start..row_start + nrows],
                        1,
                        id,
                        nrows,
                    )
                }
            }
        }
    }

    /// Scalar reference matmul, identical semantics on every platform.
    pub fn matmul_scalar(&self, out: &mut [f32], x: &[f32], seq: usize) {
        let (od, id) = (self.out_dim(), self.in_dim());
        assert_eq!(x.len(), seq * id, "x shape");
        assert_eq!(out.len(), seq * od, "out shape");
        match self {
            Self::F32 { data, .. } => crate::matmul(out, x, data, seq, id, od),
            Self::Int8 { q, scales, .. } => kernels::matmul_i8(out, x, q, scales, seq, id, od),
            Self::Int4 { packed, scales, .. } => {
                kernels::matmul_i4(out, x, packed, scales, seq, id, od)
            }
        }
    }

    /// Single-row convenience: `out = W x` for one activation vector.
    pub fn matvec(&self, out: &mut [f32], x: &[f32]) {
        self.matmul(out, x, 1);
    }

    /// `acc[i] += factor * w[row, i]` — dequantizes one row on the fly.
    /// Needed by MLA weight absorption (`q_nope` folded through `kv_b`).
    pub fn add_scaled_row(&self, row: usize, factor: f32, acc: &mut [f32]) {
        let id = self.in_dim();
        assert!(row < self.out_dim());
        assert_eq!(acc.len(), id);
        match self {
            Self::F32 { data, .. } => {
                let r = &data[row * id..(row + 1) * id];
                for i in 0..id {
                    acc[i] += factor * r[i];
                }
            }
            Self::Int8 { q, scales, .. } => {
                let s = factor * scales[row];
                let r = &q[row * id..(row + 1) * id];
                for i in 0..id {
                    acc[i] += s * r[i] as f32;
                }
            }
            Self::Int4 { packed, scales, .. } => {
                let s = factor * scales[row];
                let row_bytes = id.div_ceil(2);
                let r = &packed[row * row_bytes..(row + 1) * row_bytes];
                for (i, a) in acc.iter_mut().enumerate() {
                    *a += s * kernels::unpack_i4(r, i) as f32;
                }
            }
        }
    }

    /// `out[j] = scale * Σ_i x[i] * w[row0 + j, i]` for `j in 0..nrows` —
    /// matvec over a contiguous row block. Needed by MLA weight absorption
    /// (the V block of `kv_b` for one head).
    pub fn matvec_rows(&self, row0: usize, nrows: usize, x: &[f32], out: &mut [f32]) {
        let id = self.in_dim();
        assert!(row0 + nrows <= self.out_dim());
        assert_eq!(x.len(), id);
        assert_eq!(out.len(), nrows);
        match self {
            Self::F32 { data, .. } => {
                for j in 0..nrows {
                    let r = &data[(row0 + j) * id..(row0 + j + 1) * id];
                    let mut acc = 0f32;
                    for i in 0..id {
                        acc += x[i] * r[i];
                    }
                    out[j] = acc;
                }
            }
            Self::Int8 { q, scales, .. } => {
                for j in 0..nrows {
                    let r = &q[(row0 + j) * id..(row0 + j + 1) * id];
                    let mut acc = 0f32;
                    for i in 0..id {
                        acc += x[i] * r[i] as f32;
                    }
                    out[j] = acc * scales[row0 + j];
                }
            }
            Self::Int4 { packed, scales, .. } => {
                let row_bytes = id.div_ceil(2);
                for j in 0..nrows {
                    let r = &packed[(row0 + j) * row_bytes..(row0 + j + 1) * row_bytes];
                    let mut acc = 0f32;
                    for (i, &xv) in x.iter().enumerate() {
                        acc += xv * kernels::unpack_i4(r, i) as f32;
                    }
                    out[j] = acc * scales[row0 + j];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_matrix(o: usize, i: usize) -> Vec<f32> {
        // Deterministic, mixed-sign, mixed-magnitude values.
        (0..o * i)
            .map(|k| ((k as f32 * 0.7).sin() * 0.9) + ((k % 7) as f32 - 3.0) * 0.01)
            .collect()
    }

    #[test]
    fn int8_roundtrip_error_bound() {
        let (o, i) = (8, 33);
        let w = test_matrix(o, i);
        let t = QTensor::quantize(&w, o, i, QuantFormat::Int8).unwrap();
        let back = t.dequantize();
        for r in 0..o {
            let max = w[r * i..(r + 1) * i]
                .iter()
                .fold(0f32, |m, &v| m.max(v.abs()));
            let half_step = max / 127.0 / 2.0 + 1e-9;
            for c in 0..i {
                let err = (w[r * i + c] - back[r * i + c]).abs();
                assert!(err <= half_step * 1.01, "int8 err {err} > {half_step}");
            }
        }
    }

    #[test]
    fn int4_roundtrip_error_bound() {
        let (o, i) = (5, 17); // odd in_dim exercises nibble padding
        let w = test_matrix(o, i);
        let t = QTensor::quantize(&w, o, i, QuantFormat::Int4).unwrap();
        let back = t.dequantize();
        for r in 0..o {
            let max = w[r * i..(r + 1) * i]
                .iter()
                .fold(0f32, |m, &v| m.max(v.abs()));
            let half_step = max / 7.0 / 2.0 + 1e-9;
            for c in 0..i {
                let err = (w[r * i + c] - back[r * i + c]).abs();
                assert!(err <= half_step * 1.01, "int4 err {err} > {half_step}");
            }
        }
    }

    #[test]
    fn quantized_matmul_matches_dequant_reference() {
        let (o, i, s) = (6, 21, 3);
        let w = test_matrix(o, i);
        let x: Vec<f32> = (0..s * i).map(|k| ((k as f32) * 0.31).cos()).collect();
        for fmt in [QuantFormat::Int8, QuantFormat::Int4] {
            let t = QTensor::quantize(&w, o, i, fmt).unwrap();
            let deq = t.dequantize();
            let mut expect = vec![0f32; s * o];
            crate::matmul(&mut expect, &x, &deq, s, i, o);
            let mut got = vec![0f32; s * o];
            t.matmul(&mut got, &x, s);
            for (g, e) in got.iter().zip(&expect) {
                // Same math up to scale-factor association; tiny fp slack.
                assert!(
                    (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                    "{fmt:?}: {g} vs {e}"
                );
            }
        }
    }

    #[test]
    fn f32_matmul_is_exact_passthrough() {
        let (o, i) = (4, 9);
        let w = test_matrix(o, i);
        let x: Vec<f32> = (0..i).map(|k| k as f32 * 0.1 - 0.4).collect();
        let t = QTensor::from_f32(w.clone(), o, i);
        let mut a = vec![0f32; o];
        let mut b = vec![0f32; o];
        t.matvec(&mut a, &x);
        crate::matmul(&mut b, &x, &w, 1, i, o);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_rows_are_safe() {
        let (o, i) = (3, 8);
        let mut w = test_matrix(o, i);
        for v in &mut w[i..2 * i] {
            *v = 0.0; // middle row all zero
        }
        for fmt in [QuantFormat::Int8, QuantFormat::Int4] {
            let t = QTensor::quantize(&w, o, i, fmt).unwrap();
            let back = t.dequantize();
            assert!(back[i..2 * i].iter().all(|&v| v == 0.0));
            let x = vec![1.0f32; i];
            let mut out = vec![0f32; o];
            t.matvec(&mut out, &x);
            assert_eq!(out[1], 0.0);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn non_finite_weights_rejected() {
        let mut w = test_matrix(2, 4);
        w[5] = f32::NAN;
        assert!(QTensor::quantize(&w, 2, 4, QuantFormat::Int8).is_err());
        w[5] = f32::INFINITY;
        assert!(QTensor::quantize(&w, 2, 4, QuantFormat::Int4).is_err());
    }

    #[test]
    fn parts_roundtrip() {
        let (o, i) = (4, 11);
        let w = test_matrix(o, i);
        for fmt in [QuantFormat::F32, QuantFormat::Int8, QuantFormat::Int4] {
            let t = QTensor::quantize(&w, o, i, fmt).unwrap();
            let (payload, scales) = t.to_parts();
            let rebuilt = QTensor::from_quantized(fmt, payload, scales.to_vec(), o, i).unwrap();
            assert_eq!(t, rebuilt);
        }
    }

    #[test]
    fn row_ops_match_dequant() {
        let (o, i) = (7, 13);
        let w = test_matrix(o, i);
        let x: Vec<f32> = (0..i).map(|k| ((k * 3) as f32 * 0.17).sin()).collect();
        for fmt in [QuantFormat::F32, QuantFormat::Int8, QuantFormat::Int4] {
            let t = QTensor::quantize(&w, o, i, fmt).unwrap();
            let deq = t.dequantize();
            // add_scaled_row
            let mut acc = vec![0f32; i];
            t.add_scaled_row(3, 2.5, &mut acc);
            for c in 0..i {
                assert!((acc[c] - 2.5 * deq[3 * i + c]).abs() < 1e-5);
            }
            // matvec_rows over a block
            let (r0, n) = (2, 4);
            let mut out = vec![0f32; n];
            t.matvec_rows(r0, n, &x, &mut out);
            for j in 0..n {
                let mut e = 0f32;
                for c in 0..i {
                    e += x[c] * deq[(r0 + j) * i + c];
                }
                assert!(
                    (out[j] - e).abs() < 1e-4,
                    "{fmt:?} row {j}: {} vs {e}",
                    out[j]
                );
            }
        }
    }
}
