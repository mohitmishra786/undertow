//! Shared model-loading flags and stats output.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use engine_core::model::{CachePolicy, LoadOptions, StoreChoice};
use engine_core::{ExpertProfile, Model, Session};

#[derive(Args, Debug, Clone)]
pub struct ModelArgs {
    /// Model directory (config.json + safetensors, plain or converted).
    #[arg(long)]
    pub model: PathBuf,
    /// Expert-cache budget in MiB. Omit for auto-sizing from physical RAM.
    #[arg(long)]
    pub cache_budget_mb: Option<u64>,
    /// Background prefetch worker threads (0 disables prefetch).
    #[arg(long, default_value_t = 2)]
    pub prefetch_workers: usize,
    /// Load every expert into RAM up front instead of streaming.
    #[arg(long)]
    pub resident: bool,
    /// Cache eviction policy: lru or weighted.
    #[arg(long, default_value = "lru")]
    pub cache_policy: String,
    /// Hot-expert profile to pin at load.
    #[arg(long)]
    pub profile: Option<PathBuf>,
    /// Byte budget for pinned experts in MiB (default: quarter of the
    /// cache budget).
    #[arg(long)]
    pub pin_budget_mb: Option<u64>,
    /// Write an expert-usage profile here when the command finishes
    /// (merged into the file if it already exists).
    #[arg(long)]
    pub profile_out: Option<PathBuf>,
}

impl ModelArgs {
    pub fn load_options(&self) -> Result<LoadOptions> {
        let store = if self.resident {
            StoreChoice::Resident
        } else {
            StoreChoice::DiskStreaming {
                cache_budget_bytes: self.cache_budget_mb.map(|mb| mb * 1024 * 1024),
                prefetch_workers: self.prefetch_workers,
            }
        };
        let cache_policy = match self.cache_policy.as_str() {
            "lru" => CachePolicy::Lru,
            "weighted" => CachePolicy::Weighted,
            other => anyhow::bail!("unknown cache policy {other:?} (lru|weighted)"),
        };
        let pin_profile = match &self.profile {
            Some(path) => Some(ExpertProfile::load(path)?),
            None => None,
        };
        Ok(LoadOptions {
            store,
            cache_policy,
            pin_profile,
            pin_budget_bytes: self.pin_budget_mb.map(|mb| mb * 1024 * 1024),
        })
    }

    /// Export (and merge) the usage profile if --profile-out was given.
    pub fn export_profile(&self, model: &dyn Model) -> Result<()> {
        let Some(path) = &self.profile_out else {
            return Ok(());
        };
        let counts: Vec<(usize, usize, u64)> = model
            .expert_usage()
            .into_iter()
            .map(|(k, c)| (k.layer, k.expert, c))
            .collect();
        let mut profile = ExpertProfile::new(model.architecture(), counts);
        if path.exists() {
            let existing = ExpertProfile::load(path)?;
            if existing.architecture == profile.architecture {
                profile.merge(&existing);
            }
        }
        profile.save(path)?;
        tracing::info!("wrote expert profile to {}", path.display());
        Ok(())
    }
}

pub fn load_model(args: &ModelArgs) -> Result<Box<dyn Model>> {
    let t0 = std::time::Instant::now();
    let model = crate::registry::load_any(&args.model, &args.load_options()?)
        .with_context(|| format!("loading model from {}", args.model.display()))?;
    tracing::info!(
        "loaded {} model: vocab {}, context {}, in {:.1}s",
        model.architecture(),
        model.vocab_size(),
        model.max_context(),
        t0.elapsed().as_secs_f64()
    );
    Ok(model)
}

pub fn print_stats(model: &dyn Model, session: &dyn Session) {
    let s = model.store_stats();
    eprintln!(
        "expert store: {} hits / {} misses ({:.1}% hit rate), {:.2} GB read from disk",
        s.hits,
        s.misses,
        s.hit_rate() * 100.0,
        s.bytes_read as f64 / 1e9
    );
    eprintln!(
        "prefetch: {} issued, {} dropped",
        s.prefetch_issued, s.prefetch_dropped
    );
    eprintln!(
        "kv cache: {:.2} MB across {} positions",
        session.kv_bytes() as f64 / 1e6,
        session.position()
    );
}
