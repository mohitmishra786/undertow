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

        let mut idx: Vec<usize> = Vec::with_capacity(self.top_k);
        for _ in 0..self.top_k.min(self.num_experts) {
            let mut best: Option<usize> = None;
            for (i, &p) in probs.iter().enumerate() {
                if idx.contains(&i) {
                    continue;
                }
                match best {
                    Some(b) if probs[b] >= p => {}
                    _ => best = Some(i),
                }
            }
            idx.push(best.expect("top_k <= num_experts"));
        }

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
}
