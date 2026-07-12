//! Shareable hot-expert profiles.
//!
//! Expert activation is heavily skewed and largely stable across users of
//! the same model on similar workloads, so a usage histogram recorded on
//! one machine is worth pinning on another. A profile is a plain JSON
//! file: `{version, architecture, counts: [[layer, expert, count], ...]}`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};
use crate::store::ExpertKey;

pub const PROFILE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertProfile {
    pub version: u32,
    /// Architecture id the profile was recorded on; pinning refuses a
    /// mismatch rather than pinning meaningless expert indices.
    pub architecture: String,
    /// `[layer, expert, count]`, sorted descending by count.
    pub counts: Vec<(usize, usize, u64)>,
}

impl ExpertProfile {
    pub fn new(architecture: &str, mut counts: Vec<(usize, usize, u64)>) -> Self {
        counts.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        Self {
            version: PROFILE_VERSION,
            architecture: architecture.to_string(),
            counts,
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let profile: Self = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| EngineError::Other(format!("{}: bad profile: {e}", path.display())))?;
        if profile.version != PROFILE_VERSION {
            return Err(EngineError::Other(format!(
                "{}: profile version {} unsupported (expected {PROFILE_VERSION})",
                path.display(),
                profile.version
            )));
        }
        Ok(profile)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(self).expect("profile serializes"),
        )?;
        Ok(())
    }

    /// Fold another profile's counts into this one (for accumulating
    /// across runs, or merging community submissions).
    pub fn merge(&mut self, other: &ExpertProfile) {
        let mut map: std::collections::HashMap<(usize, usize), u64> =
            self.counts.iter().map(|&(l, e, c)| ((l, e), c)).collect();
        for &(l, e, c) in &other.counts {
            *map.entry((l, e)).or_insert(0) += c;
        }
        let counts: Vec<(usize, usize, u64)> =
            map.into_iter().map(|((l, e), c)| (l, e, c)).collect();
        *self = Self::new(&self.architecture, counts);
    }

    /// Hottest experts first.
    pub fn hottest(&self) -> impl Iterator<Item = ExpertKey> + '_ {
        self.counts
            .iter()
            .map(|&(layer, expert, _)| ExpertKey { layer, expert })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_ordering() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.json");
        let p = ExpertProfile::new("deepseek_moe", vec![(0, 1, 5), (2, 3, 50), (1, 0, 20)]);
        assert_eq!(p.counts[0], (2, 3, 50));
        p.save(&path).unwrap();
        let q = ExpertProfile::load(&path).unwrap();
        assert_eq!(q.counts, p.counts);
        assert_eq!(
            q.hottest().next(),
            Some(ExpertKey {
                layer: 2,
                expert: 3
            })
        );
    }

    #[test]
    fn merge_accumulates() {
        let mut a = ExpertProfile::new("x", vec![(0, 0, 10), (0, 1, 5)]);
        let b = ExpertProfile::new("x", vec![(0, 1, 20), (1, 0, 1)]);
        a.merge(&b);
        assert_eq!(a.counts[0], (0, 1, 25));
        assert_eq!(a.counts.len(), 3);
    }

    #[test]
    fn version_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.json");
        std::fs::write(
            &path,
            serde_json::json!({"version": 99, "architecture": "x", "counts": []}).to_string(),
        )
        .unwrap();
        assert!(ExpertProfile::load(&path).is_err());
    }
}
