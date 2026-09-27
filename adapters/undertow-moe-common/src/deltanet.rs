//! Gated DeltaNet / KDA (Kimi Delta Attention) linear recurrent attention.
//!
//! Implements causal linear recurrence with $O(1)$ memory complexity per token:
//! $$S_t = \alpha_t S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^T$$
//! $$y_t = (S_t q_t) \odot \text{silu}(g_t)$$
//!
//! State per head is fixed at $[d_v \times d_k]$ floats, independent of context
//! length, perfectly suited for frontier 2026 hybrid architectures (Qwen3.5/3.8,
//! Kimi K3, GLM-5.3). Incremental single-token decode and chunked multi-token
//! prefill maintain identical state trajectories and numeric outputs.

use undertow_core::QTensor;
use undertow_quant::{sigmoid, silu};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaNetDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

pub struct DeltaNetWeights {
    /// `[num_heads * key_dim, hidden]`
    pub q_proj: QTensor,
    /// `[num_heads * key_dim, hidden]`
    pub k_proj: QTensor,
    /// `[num_heads * value_dim, hidden]`
    pub v_proj: QTensor,
    /// Decay rate gate `[num_heads, hidden]`
    pub alpha_proj: QTensor,
    /// Write rate gate `[num_heads, hidden]`
    pub beta_proj: QTensor,
    /// Output gate `[num_heads * value_dim, hidden]`
    pub g_proj: QTensor,
    /// Output projection `[hidden, num_heads * value_dim]`
    pub o_proj: QTensor,
    /// Optional bias for alpha gate
    pub alpha_bias: Option<Vec<f32>>,
}

/// Linear attention recurrent state for one layer: $[H \times d_v \times d_k]$ floats.
#[derive(Debug, Clone, Default)]
pub struct DeltaNetState {
    pub s: Vec<f32>,
    pub len: usize,
}

impl DeltaNetState {
    pub fn new(d: &DeltaNetDims) -> Self {
        Self {
            s: vec![0.0f32; d.num_heads * d.value_dim * d.key_dim],
            len: 0,
        }
    }

    pub fn nbytes(&self) -> usize {
        self.s.capacity() * std::mem::size_of::<f32>()
    }

    pub fn reset(&mut self) {
        self.s.fill(0.0);
        self.len = 0;
    }
}

/// Single-token causal linear recurrence step for incremental decode.
pub fn deltanet_step(
    x: &[f32],
    state: &mut DeltaNetState,
    w: &DeltaNetWeights,
    d: &DeltaNetDims,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), d.hidden);
    debug_assert_eq!(out.len(), d.hidden);

    let required_state = d.num_heads * d.value_dim * d.key_dim;
    if state.s.len() < required_state {
        state.s.resize(required_state, 0.0);
    }

    let h = d.num_heads;
    let dk = d.key_dim;
    let dv = d.value_dim;

    let mut q = vec![0.0f32; h * dk];
    let mut k = vec![0.0f32; h * dk];
    let mut v = vec![0.0f32; h * dv];
    let mut alpha_raw = vec![0.0f32; h];
    let mut beta_raw = vec![0.0f32; h];
    let mut g_raw = vec![0.0f32; h * dv];

    w.q_proj.matmul(&mut q, x, 1);
    w.k_proj.matmul(&mut k, x, 1);
    w.v_proj.matmul(&mut v, x, 1);
    w.alpha_proj.matmul(&mut alpha_raw, x, 1);
    w.beta_proj.matmul(&mut beta_raw, x, 1);
    w.g_proj.matmul(&mut g_raw, x, 1);

    if let Some(ref bias) = w.alpha_bias {
        for (a, &b) in alpha_raw.iter_mut().zip(bias) {
            *a += b;
        }
    }

    let mut y = vec![0.0f32; h * dv];
    let mut v_hat = vec![0.0f32; dv];
    let mut e = vec![0.0f32; dv];
    let mut u = vec![0.0f32; dv];
    let mut k_norm = vec![0.0f32; dk];

    for head in 0..h {
        let q_head = &q[head * dk..(head + 1) * dk];
        let k_head = &k[head * dk..(head + 1) * dk];
        let v_head = &v[head * dv..(head + 1) * dv];
        let g_head = &g_raw[head * dv..(head + 1) * dv];
        let y_head = &mut y[head * dv..(head + 1) * dv];

        // 1. L2 normalize key vector
        let norm_sq: f32 = k_head.iter().map(|&val| val * val).sum();
        let inv_norm = 1.0 / norm_sq.sqrt().max(1e-6);
        for i in 0..dk {
            k_norm[i] = k_head[i] * inv_norm;
        }

        let alpha = sigmoid(alpha_raw[head]);
        let beta = sigmoid(beta_raw[head]);

        let state_offset = head * dv * dk;
        let s_head = &mut state.s[state_offset..state_offset + dv * dk];

        // 2. Retrieve predicted value v_hat = S * k_norm
        for r in 0..dv {
            let row_offset = r * dk;
            let mut acc = 0.0f32;
            for c in 0..dk {
                acc += s_head[row_offset + c] * k_norm[c];
            }
            v_hat[r] = acc;
            e[r] = v_head[r] - acc;
        }

        // 3. Update memory state S = alpha * S + beta * (e outer k_norm)
        for (r, &e_val) in e.iter().enumerate() {
            let row_offset = r * dk;
            let beta_e = beta * e_val;
            for c in 0..dk {
                s_head[row_offset + c] = alpha * s_head[row_offset + c] + beta_e * k_norm[c];
            }
        }

        // 4. Retrieve query output u = S * q
        for r in 0..dv {
            let row_offset = r * dk;
            let mut acc = 0.0f32;
            for c in 0..dk {
                acc += s_head[row_offset + c] * q_head[c];
            }
            u[r] = acc;
            y_head[r] = acc * silu(g_head[r]);
        }
    }

    w.o_proj.matmul(out, &y, 1);
    state.len += 1;
}

/// Multi-token causal linear recurrence for prefill.
pub fn deltanet_forward_seq(
    x: &[f32],
    seq: usize,
    state: &mut DeltaNetState,
    w: &DeltaNetWeights,
    d: &DeltaNetDims,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), seq * d.hidden);
    debug_assert_eq!(out.len(), seq * d.hidden);

    for t in 0..seq {
        let xt = &x[t * d.hidden..(t + 1) * d.hidden];
        let ot = &mut out[t * d.hidden..(t + 1) * d.hidden];
        deltanet_step(xt, state, w, d, ot);
    }
}

/// Cached linear recurrence forward pass: single-token decode or multi-token prefill.
pub fn deltanet_forward_cached(
    x: &[f32],
    seq: usize,
    state: &mut DeltaNetState,
    w: &DeltaNetWeights,
    d: &DeltaNetDims,
    out: &mut [f32],
) {
    if seq == 1 {
        deltanet_step(x, state, w, d, out);
    } else {
        deltanet_forward_seq(x, seq, state, w, d, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use undertow_core::QuantFormat;

    fn make_test_weights(d: &DeltaNetDims) -> DeltaNetWeights {
        let h = d.num_heads;
        let dk = d.key_dim;
        let dv = d.value_dim;
        let hidden = d.hidden;

        let q_data: Vec<f32> = (0..(h * dk) * hidden)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let k_data: Vec<f32> = (0..(h * dk) * hidden)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..(h * dv) * hidden)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let a_data: Vec<f32> = (0..h * hidden)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let b_data: Vec<f32> = (0..h * hidden)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();
        let g_data: Vec<f32> = (0..(h * dv) * hidden)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();
        let o_data: Vec<f32> = (0..hidden * (h * dv))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.02)
            .collect();

        DeltaNetWeights {
            q_proj: QTensor::quantize(&q_data, h * dk, hidden, QuantFormat::F32).unwrap(),
            k_proj: QTensor::quantize(&k_data, h * dk, hidden, QuantFormat::F32).unwrap(),
            v_proj: QTensor::quantize(&v_data, h * dv, hidden, QuantFormat::F32).unwrap(),
            alpha_proj: QTensor::quantize(&a_data, h, hidden, QuantFormat::F32).unwrap(),
            beta_proj: QTensor::quantize(&b_data, h, hidden, QuantFormat::F32).unwrap(),
            g_proj: QTensor::quantize(&g_data, h * dv, hidden, QuantFormat::F32).unwrap(),
            o_proj: QTensor::quantize(&o_data, hidden, h * dv, QuantFormat::F32).unwrap(),
            alpha_bias: Some(vec![0.5; h]),
        }
    }

    #[test]
    fn step_equals_forward_seq_equivalence() {
        let dims = DeltaNetDims {
            hidden: 32,
            num_heads: 4,
            key_dim: 8,
            value_dim: 8,
        };
        let weights = make_test_weights(&dims);

        let seq = 5;
        let x: Vec<f32> = (0..seq * dims.hidden)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.1)
            .collect();

        // 1. One-shot sequence pass
        let mut state_seq = DeltaNetState::new(&dims);
        let mut out_seq = vec![0.0f32; seq * dims.hidden];
        deltanet_forward_seq(&x, seq, &mut state_seq, &weights, &dims, &mut out_seq);

        // 2. Incremental step-by-step pass
        let mut state_step = DeltaNetState::new(&dims);
        let mut out_step = vec![0.0f32; seq * dims.hidden];
        for t in 0..seq {
            let xt = &x[t * dims.hidden..(t + 1) * dims.hidden];
            let ot = &mut out_step[t * dims.hidden..(t + 1) * dims.hidden];
            deltanet_step(xt, &mut state_step, &weights, &dims, ot);
        }

        // Outputs must match to floating-point precision
        for (idx, (a, b)) in out_seq.iter().zip(&out_step).enumerate() {
            assert!((a - b).abs() < 1e-6, "output mismatch at {idx}: {a} vs {b}");
        }

        // Final state matrices must be bit-identical
        assert_eq!(state_seq.len, state_step.len);
        assert_eq!(state_seq.s, state_step.s);
    }

    #[test]
    fn deltanet_decay_and_delta_update_rule() {
        let dims = DeltaNetDims {
            hidden: 2,
            num_heads: 1,
            key_dim: 2,
            value_dim: 2,
        };

        // Identity projections for simplicity of hand-verification
        let q_proj = QTensor::quantize(&[1.0, 0.0, 0.0, 1.0], 2, 2, QuantFormat::F32).unwrap();
        let k_proj = QTensor::quantize(&[1.0, 0.0, 0.0, 1.0], 2, 2, QuantFormat::F32).unwrap();
        let v_proj = QTensor::quantize(&[1.0, 0.0, 0.0, 1.0], 2, 2, QuantFormat::F32).unwrap();
        let alpha_proj = QTensor::quantize(&[0.0, 0.0], 1, 2, QuantFormat::F32).unwrap();
        let beta_proj = QTensor::quantize(&[0.0, 0.0], 1, 2, QuantFormat::F32).unwrap();
        let g_proj = QTensor::quantize(&[0.0, 0.0, 0.0, 0.0], 2, 2, QuantFormat::F32).unwrap();
        let o_proj = QTensor::quantize(&[1.0, 0.0, 0.0, 1.0], 2, 2, QuantFormat::F32).unwrap();

        let weights = DeltaNetWeights {
            q_proj,
            k_proj,
            v_proj,
            alpha_proj,
            beta_proj,
            g_proj,
            o_proj,
            alpha_bias: None,
        };

        let mut state = DeltaNetState::new(&dims);
        let x = vec![1.0, 0.0];
        let mut out = vec![0.0; 2];

        // Step 1:
        // q = [1, 0], k = [1, 0] (norm=1), v = [1, 0]
        // alpha = sigmoid(0) = 0.5, beta = sigmoid(0) = 0.5
        // v_hat = S_0 * k = [0, 0]
        // e = [1, 0] - [0, 0] = [1, 0]
        // S_1 = 0.5 * 0 + 0.5 * [1, 0]^T [1, 0] = [[0.5, 0], [0, 0]]
        // u = S_1 * q = [0.5, 0]
        // silu(0) = 0.0 -> gated out is 0
        deltanet_step(&x, &mut state, &weights, &dims, &mut out);

        assert_eq!(state.len, 1);
        assert!((state.s[0] - 0.5).abs() < 1e-6);
        assert!((state.s[1] - 0.0).abs() < 1e-6);
        assert!((state.s[2] - 0.0).abs() < 1e-6);
        assert!((state.s[3] - 0.0).abs() < 1e-6);
    }
}
