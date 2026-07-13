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
}
