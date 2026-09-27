//! Generic router implementations shared by architecture families.

use crate::adapter::{ExpertChoice, RouterAdapter};

/// Softmax-then-top-k routing: Mixtral (weights always renormalized over
/// the selected experts) and Qwen-MoE (renormalization behind
/// `norm_topk_prob`). Selection ties break toward the lowest expert index,
/// matching `torch.topk`.
#[derive(Debug, Clone)]
pub struct SoftmaxTopKRouter {
    pub num_experts: usize,
    pub top_k: usize,
    pub norm_topk_prob: bool,
}

impl RouterAdapter for SoftmaxTopKRouter {
    fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn top_k(&self) -> usize {
        self.top_k
    }

    fn route(&self, gate_logits: &[f32], correction_bias: Option<&[f32]>) -> Vec<ExpertChoice> {
        assert_eq!(gate_logits.len(), self.num_experts, "gate logits length");
        debug_assert!(
            correction_bias.is_none(),
            "softmax router has no correction bias"
        );
        let mut probs = gate_logits.to_vec();
        undertow_quant::softmax(&mut probs);

        let mut idx = Vec::with_capacity(self.top_k);
        topk_indices(&probs, self.top_k, &mut idx);

        let mut weights: Vec<f32> = idx.iter().map(|&i| probs[i]).collect();
        if self.norm_topk_prob {
            let sum: f32 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
        }
        idx.into_iter()
            .zip(weights)
            .map(|(expert, weight)| ExpertChoice { expert, weight })
            .collect()
    }
}

#[derive(Copy, Clone, Default, PartialEq)]
struct MinNode {
    val: f32,
    idx: usize,
}

#[inline(always)]
fn is_worse(a: &MinNode, b: &MinNode) -> bool {
    if a.val < b.val {
        true
    } else if a.val > b.val {
        false
    } else {
        a.idx > b.idx
    }
}

#[inline]
fn sift_up(heap: &mut [MinNode], mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if is_worse(&heap[i], &heap[parent]) {
            heap.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

#[inline]
fn sift_down(heap: &mut [MinNode], mut i: usize, len: usize) {
    loop {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut worst = i;
        if left < len && is_worse(&heap[left], &heap[worst]) {
            worst = left;
        }
        if right < len && is_worse(&heap[right], &heap[worst]) {
            worst = right;
        }
        if worst != i {
            heap.swap(i, worst);
            i = worst;
        } else {
            break;
        }
    }
}

fn topk_indices_with_heap(
    values: &[f32],
    k_eff: usize,
    heap: &mut [MinNode],
    out: &mut Vec<usize>,
) {
    for (i, &val) in values.iter().enumerate().take(k_eff) {
        heap[i] = MinNode { val, idx: i };
        sift_up(heap, i);
    }

    for (i, &val) in values.iter().enumerate().skip(k_eff) {
        let cand = MinNode { val, idx: i };
        if is_worse(&heap[0], &cand) {
            heap[0] = cand;
            sift_down(heap, 0, k_eff);
        }
    }

    for step in (1..k_eff).rev() {
        heap.swap(0, step);
        sift_down(heap, 0, step);
    }

    out.reserve(k_eff);
    for item in &heap[..k_eff] {
        out.push(item.idx);
    }
}

/// Indices of the `k` largest values, lowest index first on ties
/// (matches `torch.topk` ordering for distinct ranks).
///
/// Runs in O(N log K) time with zero heap allocations for K <= 64.
pub fn topk_indices(values: &[f32], k: usize, out: &mut Vec<usize>) {
    out.clear();
    let n = values.len();
    let k_eff = k.min(n);
    if k_eff == 0 {
        return;
    }

    if k_eff == 1 {
        let mut best = 0;
        for (i, &val) in values.iter().enumerate().skip(1) {
            if val > values[best] {
                best = i;
            }
        }
        out.push(best);
        return;
    }

    const STACK_LIMIT: usize = 64;
    if k_eff <= STACK_LIMIT {
        let mut stack_heap = [MinNode::default(); STACK_LIMIT];
        topk_indices_with_heap(values, k_eff, &mut stack_heap[..k_eff], out);
    } else {
        let mut heap_buf = vec![MinNode::default(); k_eff];
        topk_indices_with_heap(values, k_eff, &mut heap_buf, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_top_probabilities_and_normalizes() {
        let r = SoftmaxTopKRouter {
            num_experts: 4,
            top_k: 2,
            norm_topk_prob: true,
        };
        let out = r.route(&[1.0, 3.0, 2.0, -1.0], None);
        assert_eq!(out[0].expert, 1);
        assert_eq!(out[1].expert, 2);
        // Renormalized over the two winners: e^3/(e^3+e^2), e^2/(e^3+e^2).
        let e3 = 3f32.exp();
        let e2 = 2f32.exp();
        assert!((out[0].weight - e3 / (e3 + e2)).abs() < 1e-6);
        assert!((out[1].weight - e2 / (e3 + e2)).abs() < 1e-6);
        assert!((out[0].weight + out[1].weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn without_norm_weights_are_raw_softmax() {
        let r = SoftmaxTopKRouter {
            num_experts: 3,
            top_k: 2,
            norm_topk_prob: false,
        };
        let logits = [0.5f32, 1.5, -0.5];
        let out = r.route(&logits, None);
        let mut probs = logits.to_vec();
        undertow_quant::softmax(&mut probs);
        assert_eq!(out[0].expert, 1);
        assert!((out[0].weight - probs[1]).abs() < 1e-7);
        assert!((out[1].weight - probs[0]).abs() < 1e-7);
    }

    #[test]
    fn exact_ties_pick_lowest_index() {
        let r = SoftmaxTopKRouter {
            num_experts: 4,
            top_k: 2,
            norm_topk_prob: true,
        };
        let out = r.route(&[1.0, 1.0, 1.0, 1.0], None);
        assert_eq!(out[0].expert, 0);
        assert_eq!(out[1].expert, 1);
    }

    #[test]
    fn topk_indices_edge_cases() {
        let mut out = Vec::new();

        // Empty input
        topk_indices(&[], 5, &mut out);
        assert!(out.is_empty());

        // k = 0
        topk_indices(&[1.0, 2.0, 3.0], 0, &mut out);
        assert!(out.is_empty());

        // k = 1
        topk_indices(&[1.0, 5.0, 2.0, 5.0], 1, &mut out);
        assert_eq!(out, vec![1]); // first 5.0 wins

        // k > len
        topk_indices(&[2.0, 3.0, 1.0], 10, &mut out);
        assert_eq!(out, vec![1, 0, 2]);

        // Negative values and ties
        topk_indices(&[-5.0, -1.0, -1.0, -3.0], 2, &mut out);
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn topk_indices_parity_with_reference() {
        // Brute-force reference implementation
        fn ref_topk(values: &[f32], k: usize) -> Vec<usize> {
            let mut out = Vec::new();
            for _ in 0..k.min(values.len()) {
                let mut best: Option<usize> = None;
                for (i, &v) in values.iter().enumerate() {
                    if out.contains(&i) {
                        continue;
                    }
                    match best {
                        Some(b) if values[b] >= v => {}
                        _ => best = Some(i),
                    }
                }
                out.push(best.expect("k <= len"));
            }
            out
        }

        // Test with large N=512, K=16 and synthetic scores with intentional ties
        let mut values = Vec::with_capacity(512);
        for i in 0..512 {
            // Periodic values to generate plenty of exact numerical ties
            values.push(((i * 37) % 50) as f32 * 0.1);
        }

        let mut actual = Vec::new();
        topk_indices(&values, 16, &mut actual);
        let expected = ref_topk(&values, 16);
        assert_eq!(actual, expected);

        // Also test K > 64 (beyond STACK_LIMIT)
        let mut actual_large_k = Vec::new();
        topk_indices(&values, 80, &mut actual_large_k);
        let expected_large_k = ref_topk(&values, 80);
        assert_eq!(actual_large_k, expected_large_k);
    }
}
