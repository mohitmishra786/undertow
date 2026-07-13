//! Deterministic RNG for oracle-model generation.
//!
//! Requirements: identical output on every platform and toolchain, forever
//! — checked-in golden references depend on the exact weights. So: PCG32
//! (pure integer math) and an Irwin–Hall approximate normal (12 uniforms,
//! add/sub only). No libm calls anywhere in the generation path, because
//! `ln`/`cos` results may differ by an ulp across platforms.

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

    /// Approximately N(0, 1): Irwin–Hall with n=12 (sum of 12 uniforms has
    /// variance exactly 1 around mean 6). Plenty for random test weights.
    pub fn normal(&mut self) -> f32 {
        let mut acc = 0.0f32;
        for _ in 0..12 {
            acc += self.uniform();
        }
        acc - 6.0
    }

    /// N(0, std).
    pub fn normal_scaled(&mut self, std: f32) -> f32 {
        self.normal() * std
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Pcg32::new(42, 7);
        let mut b = Pcg32::new(42, 7);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn normal_moments_roughly_right() {
        let mut rng = Pcg32::new(1, 1);
        let n = 20_000;
        let samples: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let var: f32 = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }
}
