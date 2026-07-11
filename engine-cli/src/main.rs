//! `undertow` CLI.
//!
//! Phase 0 surface: `run` executes the scalar reference forward pass on an
//! HF-layout checkpoint directory (oracle-scale models); `bench gen-oracle`
//! produces a synthetic test model. `convert` and `chat` land with the
//! quantized converter and tokenizer work (Phase 1).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use deepseek_moe::model::argmax;

#[derive(Parser)]
#[command(
    name = "undertow",
    version,
    about = "Tiered-memory MoE inference engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the scalar reference forward pass (greedy decode over token ids).
    Run {
        /// Model directory (config.json + safetensors).
        #[arg(long)]
        model: PathBuf,
        /// Comma-separated prompt token ids, e.g. "3,14,159,26".
        #[arg(long)]
        prompt_ids: String,
        /// Tokens to generate.
        #[arg(long, default_value_t = 16)]
        max_new: usize,
    },
    /// Convert a checkpoint to quantized expert shards. (Phase 1)
    Convert,
    /// Interactive chat. (Phase 1, needs tokenizer)
    Chat,
    /// Benchmarks and test-model utilities.
    Bench {
        #[command(subcommand)]
        command: BenchCommand,
    },
}

#[derive(Subcommand)]
enum BenchCommand {
    /// Generate a tiny synthetic oracle model (real architecture, random
    /// weights) for correctness testing.
    GenOracle {
        /// Output directory.
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 20260712)]
        seed: u64,
    },
    /// Time the reference forward pass on a model directory.
    Forward {
        #[arg(long)]
        model: PathBuf,
        /// Sequence length of the synthetic prompt.
        #[arg(long, default_value_t = 16)]
        seq: usize,
        #[arg(long, default_value_t = 8)]
        iters: usize,
    },
}

fn parse_ids(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|t| t.trim().parse::<usize>().context("bad token id"))
        .collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run {
            model,
            prompt_ids,
            max_new,
        } => {
            let prompt = parse_ids(&prompt_ids)?;
            let m = deepseek_moe::loader::load_model(&model)?;
            eprintln!(
                "loaded {} layers, {} routed experts/layer",
                m.cfg.num_hidden_layers, m.cfg.n_routed_experts
            );
            let out = m.greedy_decode(&prompt, max_new, &[])?;
            println!(
                "{}",
                out.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        Command::Convert => bail!("convert: not implemented in phase 0 (scalar reference only)"),
        Command::Chat => bail!("chat: not implemented in phase 0 (needs tokenizer, phase 1)"),
        Command::Bench { command } => match command {
            BenchCommand::GenOracle { out, seed } => {
                let spec = engine_bench::OracleSpec {
                    seed,
                    ..Default::default()
                };
                engine_bench::generate_oracle(&out, &spec)?;
                println!("oracle model written to {}", out.display());
            }
            BenchCommand::Forward { model, seq, iters } => {
                let m = deepseek_moe::loader::load_model(&model)?;
                let prompt: Vec<usize> = (0..seq).map(|i| (i * 7 + 3) % m.cfg.vocab_size).collect();
                let t0 = Instant::now();
                let mut last = 0;
                for _ in 0..iters {
                    let logits = m.forward(&prompt)?;
                    last = argmax(logits.row(seq - 1));
                }
                let dt = t0.elapsed().as_secs_f64() / iters as f64;
                println!("seq={seq}: {:.3} ms/forward (last argmax {last})", dt * 1e3);
            }
        },
    }
    Ok(())
}
