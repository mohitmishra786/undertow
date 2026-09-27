//! Token sampling, deterministic under a seed.
//!
//! One implementation shared by every frontend so `run`, `chat` and the
//! future server sample identically. Greedy is the `temperature == 0`
//! case, not a separate code path callers can get subtly wrong.

/// PCG32: small, fast, and stable across platforms and releases. Sampling
/// reproducibility with a fixed seed is part of this crate's contract.
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (stream << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1) with 24 bits of mantissa (exact in f32).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Approximately N(0, 1): Irwin–Hall with n = 12. Used by the oracle
    /// generator; kept here so all deterministic randomness lives in one
    /// audited place.
    pub fn normal(&mut self) -> f32 {
        let mut acc = 0.0f32;
        for _ in 0..12 {
            acc += self.uniform();
        }
        acc - 6.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    /// 0 means greedy (argmax). Must be finite and >= 0.
    pub temperature: f32,
    /// Nucleus threshold in (0, 1]; 1 disables.
    pub top_p: f32,
    /// Keep only the k most likely tokens; 0 disables.
    pub top_k: usize,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            seed: 42,
        }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(format!(
                "temperature must be finite and >= 0, got {}",
                self.temperature
            ));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(format!("top_p must be in (0, 1], got {}", self.top_p));
        }
        Ok(())
    }
}

pub struct Sampler {
    cfg: SamplerConfig,
    rng: Pcg32,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            rng: Pcg32::new(cfg.seed, 0x5eed),
            cfg,
        })
    }

    /// Sample a token id from raw logits. Never panics on well-formed
    /// input of length >= 1; NaN logits are treated as -inf.
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        assert!(!logits.is_empty(), "empty logits");
        if self.cfg.temperature == 0.0 {
            return argmax(logits);
        }

        // Collect candidate (id, logit), dropping NaN.
        let mut cand: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, l)| !l.is_nan())
            .collect();
        if cand.is_empty() {
            return 0;
        }
        // Sort descending by logit; stable order for exact ties.
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if self.cfg.top_k > 0 && self.cfg.top_k < cand.len() {
            cand.truncate(self.cfg.top_k);
        }

        // Softmax over the survivors at the given temperature.
        let t = self.cfg.temperature;
        let max = cand[0].1;
        let mut probs: Vec<f32> = cand.iter().map(|(_, l)| ((l - max) / t).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Nucleus: keep the smallest prefix whose mass reaches top_p.
        let mut keep = probs.len();
        if self.cfg.top_p < 1.0 {
            let mut acc = 0.0f32;
            for (i, p) in probs.iter().enumerate() {
                acc += p;
                if acc >= self.cfg.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        let mass: f32 = probs[..keep].iter().sum();

        let mut r = self.rng.uniform() * mass;
        for i in 0..keep {
            r -= probs[i];
            if r <= 0.0 {
                return cand[i].0;
            }
        }
        cand[keep - 1].0
    }

    pub fn config(&self) -> &SamplerConfig {
        &self.cfg
    }

    pub fn rng_mut(&mut self) -> &mut Pcg32 {
        &mut self.rng
    }

    /// Compute full normalized probability distribution over vocabulary
    /// matching the sampler's temperature, top_k, and top_p settings.
    pub fn probs(&self, logits: &[f32]) -> Vec<f32> {
        compute_probs(logits, &self.cfg)
    }

    /// Sample a token id from an explicit normalized probability distribution.
    pub fn sample_probs(&mut self, probs: &[f32]) -> usize {
        sample_from_probs(probs, &mut self.rng)
    }

    /// Verify a draft token using speculative rejection sampling (Leviathan et al., 2023).
    pub fn verify_draft(
        &mut self,
        draft_token: usize,
        q_probs: &[f32],
        p_probs: &[f32],
    ) -> SpeculativeDecision {
        speculative_rejection_sample(draft_token, q_probs, p_probs, &mut self.rng)
    }
}

/// Decision of speculative rejection sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeDecision {
    /// The draft token is accepted.
    Accepted,
    /// The draft token is rejected; includes the replacement token sampled
    /// from the normalized positive difference distribution:
    /// `p'(x) = max(0, p(x) - q(x)) / sum_{x'} max(0, p(x') - q(x'))`.
    Rejected(usize),
}

/// Compute full normalized probability distribution over vocabulary from logits.
pub fn compute_probs(logits: &[f32], cfg: &SamplerConfig) -> Vec<f32> {
    assert!(!logits.is_empty(), "empty logits");
    let mut out = vec![0.0f32; logits.len()];
    if cfg.temperature == 0.0 {
        out[argmax(logits)] = 1.0;
        return out;
    }

    let mut cand: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, l)| !l.is_nan())
        .collect();
    if cand.is_empty() {
        out[0] = 1.0;
        return out;
    }

    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if cfg.top_k > 0 && cfg.top_k < cand.len() {
        cand.truncate(cfg.top_k);
    }

    let t = cfg.temperature;
    let max = cand[0].1;
    let mut probs: Vec<f32> = cand.iter().map(|(_, l)| ((l - max) / t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    let mut keep = probs.len();
    if cfg.top_p < 1.0 {
        let mut acc = 0.0f32;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if acc >= cfg.top_p {
                keep = i + 1;
                break;
            }
        }
    }

    let mass: f32 = probs[..keep].iter().sum();
    if mass > 0.0 {
        for i in 0..keep {
            out[cand[i].0] = probs[i] / mass;
        }
    } else {
        out[cand[0].0] = 1.0;
    }
    out
}

/// Sample a token index from an explicit probability distribution.
pub fn sample_from_probs(probs: &[f32], rng: &mut Pcg32) -> usize {
    if probs.is_empty() {
        return 0;
    }
    let mut cand: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| *p > 0.0 && !p.is_nan())
        .collect();
    if cand.is_empty() {
        return argmax(probs);
    }
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mass: f32 = cand.iter().map(|(_, p)| *p).sum();
    if mass <= 0.0 {
        return cand[0].0;
    }
    let mut r = rng.uniform() * mass;
    for &(id, p) in &cand {
        r -= p;
        if r <= 0.0 {
            return id;
        }
    }
    cand.last().map(|&(id, _)| id).unwrap_or(0)
}

/// Perform speculative rejection sampling (Leviathan et al., 2023).
///
/// `draft_token`: the candidate token proposed by the draft model.
/// `q_probs`: draft model probability distribution over vocabulary.
/// `p_probs`: base model probability distribution over vocabulary.
/// `rng`: PCG32 random number generator.
pub fn speculative_rejection_sample(
    draft_token: usize,
    q_probs: &[f32],
    p_probs: &[f32],
    rng: &mut Pcg32,
) -> SpeculativeDecision {
    let p_draft = p_probs.get(draft_token).copied().unwrap_or(0.0);
    let q_draft = q_probs.get(draft_token).copied().unwrap_or(0.0);

    let alpha = if q_draft <= 0.0 {
        if p_draft > 0.0 {
            0.0
        } else {
            1.0
        }
    } else {
        (p_draft / q_draft).min(1.0)
    };

    let u = rng.uniform();
    if u < alpha {
        return SpeculativeDecision::Accepted;
    }

    // Rejected: sample from normalized positive difference distribution (p(x) - q(x))+
    let max_len = p_probs.len().max(q_probs.len());
    let mut diff = vec![0.0f32; max_len];
    let mut sum_diff = 0.0f32;
    for (i, item) in diff.iter_mut().enumerate() {
        let p = p_probs.get(i).copied().unwrap_or(0.0);
        let q = q_probs.get(i).copied().unwrap_or(0.0);
        let d = (p - q).max(0.0);
        *item = d;
        sum_diff += d;
    }

    let replacement = if sum_diff > 1e-12 {
        for d in &mut diff {
            *d /= sum_diff;
        }
        sample_from_probs(&diff, rng)
    } else {
        sample_from_probs(p_probs, rng)
    };

    SpeculativeDecision::Rejected(replacement)
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_zero_is_argmax() {
        let mut s = Sampler::new(SamplerConfig::greedy()).unwrap();
        assert_eq!(s.sample(&[0.1, 3.0, -1.0, 2.9]), 1);
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            seed: 7,
        };
        let logits = [1.0f32, 0.5, 0.2, -0.4, 2.0];
        let a: Vec<usize> = {
            let mut s = Sampler::new(cfg).unwrap();
            (0..20).map(|_| s.sample(&logits)).collect()
        };
        let b: Vec<usize> = {
            let mut s = Sampler::new(cfg).unwrap();
            (0..20).map(|_| s.sample(&logits)).collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn top_k_one_is_argmax_even_with_temperature() {
        let cfg = SamplerConfig {
            temperature: 5.0,
            top_p: 1.0,
            top_k: 1,
            seed: 1,
        };
        let mut s = Sampler::new(cfg).unwrap();
        for _ in 0..10 {
            assert_eq!(s.sample(&[0.0, 9.0, 1.0]), 1);
        }
    }

    #[test]
    fn nucleus_excludes_tail() {
        // Token 0 has ~all the mass; top_p = 0.5 must always pick it.
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_p: 0.5,
            top_k: 0,
            seed: 3,
        };
        let mut s = Sampler::new(cfg).unwrap();
        for _ in 0..50 {
            assert_eq!(s.sample(&[10.0, 0.0, 0.0, 0.0]), 0);
        }
    }

    #[test]
    fn nan_logits_ignored() {
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 3,
        };
        let mut s = Sampler::new(cfg).unwrap();
        let pick = s.sample(&[f32::NAN, 1.0, f32::NAN]);
        assert_eq!(pick, 1);
    }

    #[test]
    fn invalid_config_rejected() {
        assert!(Sampler::new(SamplerConfig {
            temperature: -1.0,
            ..Default::default()
        })
        .is_err());
        assert!(Sampler::new(SamplerConfig {
            top_p: 0.0,
            ..Default::default()
        })
        .is_err());
    }

    #[test]
    fn samples_follow_distribution_roughly() {
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 11,
        };
        let mut s = Sampler::new(cfg).unwrap();
        // p(1)/p(0) = e^2 ≈ 7.39
        let logits = [0.0f32, 2.0];
        let n = 5000;
        let ones = (0..n).filter(|_| s.sample(&logits) == 1).count();
        let frac = ones as f64 / n as f64;
        assert!((frac - 0.8808).abs() < 0.03, "frac {frac}");
    }

    #[test]
    fn greedy_rejection_sampling_matches_argmax() {
        let mut rng = Pcg32::new(42, 1);
        let q = [0.0, 1.0, 0.0];
        let p_match = [0.0, 1.0, 0.0];
        let p_diff = [0.0, 0.0, 1.0];

        // Matching draft is accepted
        let d1 = speculative_rejection_sample(1, &q, &p_match, &mut rng);
        assert_eq!(d1, SpeculativeDecision::Accepted);

        // Mismatched draft is rejected with target's argmax
        let d2 = speculative_rejection_sample(1, &q, &p_diff, &mut rng);
        assert_eq!(d2, SpeculativeDecision::Rejected(2));
    }

    #[test]
    fn speculative_rejection_sampling_preserves_target_distribution() {
        let mut rng = Pcg32::new(12345, 6789);
        let p = [0.10f32, 0.40, 0.30, 0.20];
        let q = [0.40f32, 0.10, 0.20, 0.30];

        let n = 25000;
        let mut counts = [0usize; 4];
        let mut accepted_count = 0usize;

        for _ in 0..n {
            let draft = sample_from_probs(&q, &mut rng);
            let final_token = match speculative_rejection_sample(draft, &q, &p, &mut rng) {
                SpeculativeDecision::Accepted => {
                    accepted_count += 1;
                    draft
                }
                SpeculativeDecision::Rejected(repl) => repl,
            };
            counts[final_token] += 1;
        }

        // Acceptance rate should roughly equal sum_x min(p(x), q(x)) = 0.1 + 0.1 + 0.2 + 0.2 = 0.60
        let acceptance_rate = accepted_count as f64 / n as f64;
        assert!(
            (acceptance_rate - 0.60).abs() < 0.03,
            "acceptance rate {acceptance_rate} should be near 0.60"
        );

        // Emitted token frequencies must match target distribution p
        for i in 0..4 {
            let freq = counts[i] as f64 / n as f64;
            let target = p[i] as f64;
            assert!(
                (freq - target).abs() < 0.02,
                "token {i} frequency {freq} deviates from target {target}"
            );
        }
    }

    #[test]
    fn compute_probs_sums_to_one() {
        let logits = [1.0, 2.5, -0.5, 0.0, 3.2];
        let cfg = SamplerConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 3,
            seed: 42,
        };
        let probs = compute_probs(&logits, &cfg);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Non-survivors are 0
        let non_zeros = probs.iter().filter(|&&p| p > 0.0).count();
        assert!(non_zeros <= 3);
    }
}
