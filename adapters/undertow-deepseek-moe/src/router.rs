//! DeepSeek sigmoid `noaux_tc` router.
//!
//! Reference semantics (`transformers` `DeepseekV3TopkRouter`, also used by
//! GLM-5.2 and Kimi K2):
//!
//! 1. `scores = sigmoid(gate_logits)`
//! 2. Selection ranks `scores + e_score_correction_bias` — the bias is the
//!    aux-loss-free load-balancing correction ("noaux_tc") and must *never*
//!    leak into the combination weights.
//! 3. Group limitation: experts are split into `n_group` groups; each group
//!    is scored by the sum of its top-2 corrected scores, only the best
//!    `topk_group` groups stay eligible.
//! 4. Top-k over eligible corrected scores picks the experts; the weights
//!    are the *uncorrected* sigmoid scores of those experts.
//! 5. Optional weight normalization (`norm_topk_prob`), then
//!    `routed_scaling_factor`.
//!
//! Tie-breaking follows `torch.topk`: lowest index wins.

use undertow_core::adapter::{ExpertChoice, RouterAdapter};
use undertow_core::router::topk_indices;
use undertow_quant::sigmoid;

use crate::config::DeepseekConfig;

#[derive(Debug, Clone)]
pub struct DeepseekSigmoidRouter {
    pub num_experts: usize,
    pub top_k: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
}

impl DeepseekSigmoidRouter {
    pub fn from_config(cfg: &DeepseekConfig) -> Self {
        Self {
            num_experts: cfg.n_routed_experts,
            top_k: cfg.num_experts_per_tok,
            n_group: cfg.n_group,
            topk_group: cfg.topk_group,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
        }
    }
}

impl RouterAdapter for DeepseekSigmoidRouter {
    fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn top_k(&self) -> usize {
        self.top_k
    }

    fn route(&self, gate_logits: &[f32], correction_bias: Option<&[f32]>) -> Vec<ExpertChoice> {
        let e = self.num_experts;
        assert_eq!(gate_logits.len(), e, "gate logits length");

        let scores: Vec<f32> = gate_logits.iter().map(|&l| sigmoid(l)).collect();
        let mut corrected = scores.clone();
        if let Some(bias) = correction_bias {
            assert_eq!(bias.len(), e, "correction bias length");
            for (c, b) in corrected.iter_mut().zip(bias) {
                *c += b;
            }
        }

        // Group limitation: keep only experts in the topk_group best groups.
        let mut eligible = vec![true; e];
        if self.n_group > 1 && self.topk_group < self.n_group {
            let group_size = e / self.n_group;
            let mut scratch = Vec::with_capacity(2);
            let group_scores: Vec<f32> = (0..self.n_group)
                .map(|g| {
                    let grp = &corrected[g * group_size..(g + 1) * group_size];
                    topk_indices(grp, 2, &mut scratch);
                    scratch.iter().map(|&i| grp[i]).sum()
                })
                .collect();
            let mut top_groups = Vec::with_capacity(self.topk_group);
            topk_indices(&group_scores, self.topk_group, &mut top_groups);
            eligible.fill(false);
            for &g in &top_groups {
                eligible[g * group_size..(g + 1) * group_size].fill(true);
            }
        }

        let masked: Vec<f32> = corrected
            .iter()
            .zip(&eligible)
            .map(|(&c, &ok)| if ok { c } else { f32::NEG_INFINITY })
            .collect();
        let mut idx = Vec::with_capacity(self.top_k);
        topk_indices(&masked, self.top_k, &mut idx);

        let mut weights: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
        if self.norm_topk_prob {
            let denom: f32 = weights.iter().sum::<f32>() + 1e-20;
            for w in &mut weights {
                *w /= denom;
            }
        }
        idx.iter()
            .zip(&weights)
            .map(|(&expert, &w)| ExpertChoice {
                expert,
                weight: w * self.routed_scaling_factor,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router(e: usize, k: usize) -> DeepseekSigmoidRouter {
        DeepseekSigmoidRouter {
            num_experts: e,
            top_k: k,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: false,
            routed_scaling_factor: 1.0,
        }
    }

    #[test]
    fn picks_largest_logits_without_bias() {
        let r = router(4, 2);
        let out = r.route(&[0.1, 2.0, -1.0, 1.0], None);
        assert_eq!(out[0].expert, 1);
        assert_eq!(out[1].expert, 3);
        assert!((out[0].weight - sigmoid(2.0)).abs() < 1e-7);
    }

    #[test]
    fn bias_changes_selection_but_not_weights() {
        let r = router(3, 1);
        // Without bias expert 0 wins; bias pushes expert 2 ahead.
        let logits = [1.0, 0.0, 0.9];
        let bias = [0.0, 0.0, 0.5];
        let out = r.route(&logits, Some(&bias));
        assert_eq!(out[0].expert, 2);
        // Weight must be the raw sigmoid score, not the biased one.
        assert!((out[0].weight - sigmoid(0.9)).abs() < 1e-7);
    }

    #[test]
    fn norm_and_scale_applied() {
        let mut r = router(4, 2);
        r.norm_topk_prob = true;
        r.routed_scaling_factor = 2.5;
        let out = r.route(&[3.0, 2.0, -3.0, -3.0], None);
        let s0 = sigmoid(3.0);
        let s1 = sigmoid(2.0);
        let denom = s0 + s1 + 1e-20;
        assert!((out[0].weight - 2.5 * s0 / denom).abs() < 1e-6);
        assert!((out[1].weight - 2.5 * s1 / denom).abs() < 1e-6);
    }

    #[test]
    fn group_limitation_masks_losing_groups() {
        // 8 experts, 4 groups of 2, keep best 2 groups.
        // Group scores (sum of top-2 corrected): g0 strong, g3 strong.
        let r = DeepseekSigmoidRouter {
            num_experts: 8,
            top_k: 4,
            n_group: 4,
            topk_group: 2,
            norm_topk_prob: false,
            routed_scaling_factor: 1.0,
        };
        // g1 holds the single largest logit (expert 2) but a weak partner,
        // so its top-2 sum loses to g0 and g3; expert 2 must NOT be picked.
        let logits = [2.0, 1.9, 3.0, -5.0, -1.0, -1.0, 2.0, 1.8];
        let out = r.route(&logits, None);
        let picked: Vec<usize> = out.iter().map(|c| c.expert).collect();
        // Eligible: {0,1,6,7}. Experts 0 and 6 tie exactly (same logit) —
        // lowest index first, then 1, then 7.
        assert_eq!(picked, vec![0, 6, 1, 7]);
        assert!(!picked.contains(&2));
    }
}
