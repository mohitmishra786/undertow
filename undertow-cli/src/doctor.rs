//! `undertow doctor`: is this machine ready for that model?
//!
//! Reports physical memory, measures sequential disk read throughput the
//! way the engine actually reads (large positioned reads, cold-ish), and,
//! given a model directory, estimates the resident footprint, the expert
//! pool size, and a tok/s ceiling implied by disk bandwidth alone.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

pub fn run(model: Option<PathBuf>, disk_probe_mb: u64) -> Result<()> {
    println!("undertow doctor");
    println!("===============");

    // Memory.
    match undertow_core::mem::total_memory_bytes() {
        Some(total) => {
            println!("memory: {:.1} GB physical", total as f64 / 1e9);
            match undertow_core::mem::available_memory_bytes() {
                Some(avail) => println!("        {:.1} GB reported available", avail as f64 / 1e9),
                None => println!(
                    "        (no availability metric on this platform; budgets use totals)"
                ),
            }
        }
        None => {
            println!("memory: could not detect physical RAM; pass --cache-budget-mb explicitly")
        }
    }
    println!(
        "cores:  {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    );

    // Disk throughput, measured where the model lives (or cwd).
    let probe_dir = model
        .as_deref()
        .filter(|p| p.is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let bw =
        measure_read_bandwidth(&probe_dir, disk_probe_mb).context("measuring disk throughput")?;
    println!(
        "disk:   {:.2} GB/s sequential read ({} MB probe in {})",
        bw / 1e9,
        disk_probe_mb,
        probe_dir.display()
    );

    let Some(dir) = model else {
        println!();
        println!("pass --model <dir> for a fit estimate against a specific checkpoint");
        return Ok(());
    };

    // Model fit: resident dense weights vs streamed expert pool.
    let ty = crate::registry::model_type(&dir)?;
    println!();
    println!("model:  {} ({})", dir.display(), ty);
    let (dense, experts) = shard_sizes(&dir)?;
    println!(
        "        dense/resident shards: {:.2} GB",
        dense as f64 / 1e9
    );
    println!(
        "        expert pool on disk:   {:.2} GB",
        experts as f64 / 1e9
    );

    if let Some(total) = undertow_core::mem::total_memory_bytes() {
        let budget = undertow_core::mem::auto_cache_budget(dense);
        println!(
            "        auto cache budget:     {:.2} GB ({:.0}% of the expert pool)",
            budget as f64 / 1e9,
            100.0 * budget as f64 / experts.max(1) as f64
        );
        let verdict = if dense + 512 * 1024 * 1024 > total {
            "does not fit: resident weights alone exceed RAM"
        } else if budget >= experts {
            "comfortable: the whole expert pool can stay cached"
        } else {
            "streams: expect disk reads on cold or shifting expert sets"
        };
        println!("        verdict: {verdict}");
    }

    // Disk-implied ceiling: worst case reads every routed expert per token.
    if experts > 0 {
        if let Ok(cfg_bytes) = std::fs::read(dir.join("config.json")) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&cfg_bytes) {
                let layers = v["num_hidden_layers"].as_u64().unwrap_or(0);
                let per_tok = v["num_experts_per_tok"].as_u64().unwrap_or(0);
                let n_experts = v["n_routed_experts"]
                    .as_u64()
                    .or_else(|| v["num_local_experts"].as_u64())
                    .or_else(|| v["num_experts"].as_u64())
                    .unwrap_or(0);
                if layers > 0 && per_tok > 0 && n_experts > 0 {
                    let expert_bytes = experts / (layers * n_experts).max(1);
                    let cold_bytes_per_token = expert_bytes * layers * per_tok;
                    println!(
                        "        cold-cache disk floor: {:.1} tok/s ({} experts x {} layers, {:.1} MB/token)",
                        bw / cold_bytes_per_token.max(1) as f64,
                        per_tok,
                        layers,
                        cold_bytes_per_token as f64 / 1e6
                    );
                    println!("        (warm-cache throughput is compute-bound and far higher)");
                }
            }
        }
    }
    Ok(())
}

/// Write then read back a probe file with the same positioned reads the
/// engine uses. Honest caveat: the page cache can flatter this number;
/// the write is fsynced and the file removed afterwards.
fn measure_read_bandwidth(dir: &Path, probe_mb: u64) -> Result<f64> {
    let path = dir.join(".undertow-doctor-probe");
    let chunk = vec![0x5Au8; 1 << 20];
    {
        let mut f = std::fs::File::create(&path)?;
        for _ in 0..probe_mb {
            f.write_all(&chunk)?;
        }
        f.sync_all()?;
    }
    let file = std::fs::File::open(&path)?;
    let mut buf = vec![0u8; 1 << 20];
    let start = Instant::now();
    let mut total = 0u64;
    for i in 0..probe_mb {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            file.read_exact_at(&mut buf, i << 20)?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = &file;
            f.seek(SeekFrom::Start(i << 20))?;
            f.read_exact(&mut buf)?;
        }
        total += buf.len() as u64;
    }
    let secs = start.elapsed().as_secs_f64();
    drop(file);
    let _ = std::fs::remove_file(&path);
    Ok(total as f64 / secs.max(1e-9))
}

/// Split a checkpoint's safetensors bytes into resident (dense) and
/// streamed (expert) shards by the converter's layout, falling back to
/// counting expert tensors for unconverted checkpoints.
fn shard_sizes(dir: &Path) -> Result<(u64, u64)> {
    let mut dense = 0u64;
    let mut experts = 0u64;
    let mut saw_expert_shards = false;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".safetensors") {
            continue;
        }
        let len = entry.metadata()?.len();
        if name.starts_with("experts-") {
            experts += len;
            saw_expert_shards = true;
        } else {
            dense += len;
        }
    }
    if !saw_expert_shards {
        // Unconverted checkpoint: apportion by tensor names.
        let reader = undertow_io::ShardedModelReader::open(dir)?;
        let (mut d, mut e) = (0u64, 0u64);
        for name in reader.tensor_names() {
            let info = reader.info(name)?;
            if name.contains(".experts.") {
                e += info.nbytes;
            } else {
                d += info.nbytes;
            }
        }
        return Ok((d, e));
    }
    Ok((dense, experts))
}
