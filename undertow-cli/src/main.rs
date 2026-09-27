//! `undertow` CLI.
//!
//! * `convert` — quantize an HF checkpoint into a streaming layout.
//! * `run` — one completion (text prompt via the model tokenizer, or raw
//!   token ids), with sampling and cache controls.
//! * `chat` — interactive session with streaming output and KV-prefix
//!   reuse across turns.
//! * `bench` — forward timing and oracle-model utilities.

mod chat;
mod common;
mod doctor;
mod registry;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use undertow_core::sample::{Sampler, SamplerConfig};
use undertow_core::QuantFormat;

use common::{load_model, print_stats, ModelArgs};

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

#[derive(Args, Debug, Clone)]
struct SamplingArgs {
    /// 0 = greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    /// 0 disables top-k.
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

impl SamplingArgs {
    fn sampler(&self, default_temperature: f32) -> Result<Sampler> {
        let temperature = if self.temperature < 0.0 {
            default_temperature
        } else {
            self.temperature
        };
        Sampler::new(SamplerConfig {
            temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
        })
        .map_err(|e| anyhow::anyhow!(e))
    }
}

#[derive(Subcommand)]
enum Command {
    /// Quantize an HF checkpoint into a streaming undertow checkpoint.
    Convert {
        /// Source model directory (config.json + safetensors).
        #[arg(long)]
        src: PathBuf,
        /// Output directory.
        #[arg(long)]
        out: PathBuf,
        /// Routed-expert format: int4, int8 or f32.
        #[arg(long, default_value = "int4")]
        experts: String,
        /// Dense/attention format: int8, int4 or f32.
        #[arg(long, default_value = "int8")]
        dense: String,
        /// Rows per streamed quantization chunk.
        #[arg(long, default_value_t = 1024)]
        row_chunk: usize,
        /// Rewrite shards that already exist.
        #[arg(long)]
        force: bool,
    },
    /// Run one completion.
    Run {
        #[command(flatten)]
        model: ModelArgs,
        /// Prompt text (needs tokenizer.json next to the model).
        #[arg(long, conflicts_with = "prompt_ids")]
        prompt: Option<String>,
        /// Comma-separated raw token ids, e.g. "3,14,159".
        #[arg(long)]
        prompt_ids: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_new: usize,
        #[command(flatten)]
        sampling: SamplingArgs,
        /// Print store/cache statistics at the end.
        #[arg(long)]
        stats: bool,
        /// Native MTP speculative decoding (greedy only; DeepSeek-family
        /// checkpoints that ship an MTP head).
        #[arg(long)]
        mtp: bool,
    },
    /// Interactive chat (streaming, KV-prefix reuse between turns).
    Chat {
        #[command(flatten)]
        model: ModelArgs,
        /// System prompt for the conversation.
        #[arg(long)]
        system: Option<String>,
        #[arg(long, default_value_t = 512)]
        max_new: usize,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Check whether this machine can run a given model: RAM, disk
    /// throughput, resident-vs-streamed footprint, and the disk-implied
    /// throughput floor.
    Doctor {
        /// Model directory to size up (optional; hardware-only otherwise).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Size of the disk throughput probe in MB.
        #[arg(long, default_value_t = 256)]
        disk_probe_mb: u64,
    },
    /// OpenAI-compatible HTTP server.
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Require this bearer token on every endpoint except /health
        /// (env: UNDERTOW_API_KEY).
        #[arg(long, env = "UNDERTOW_API_KEY")]
        api_key: Option<String>,
        /// Per-request generation deadline in seconds.
        #[arg(long, default_value_t = 600)]
        timeout_secs: u64,
        /// Requests allowed to wait for the generation slot; beyond this,
        /// 429.
        #[arg(long, default_value_t = 32)]
        max_queue: usize,
        /// Access-Control-Allow-Origin value; omit to send no CORS headers.
        #[arg(long)]
        cors_origin: Option<String>,
    },
    /// Benchmarks and test-model utilities.
    Bench {
        #[command(subcommand)]
        command: BenchCommand,
    },
    /// Manage expert usage profiles.
    Profile {
        #[command(subcommand)]
        command: ProfileCommand,
    },
}

#[derive(Subcommand)]
enum ProfileCommand {
    /// Merge multiple expert usage profiles into a single profile.
    Merge {
        /// Input profile JSON files to merge.
        #[arg(long = "inputs", required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,
        /// Destination path for the merged profile JSON.
        #[arg(long = "out")]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum BenchCommand {
    /// Generate a synthetic oracle model (real architecture, random
    /// weights) for correctness testing and benchmarking. Defaults match
    /// the checked-in tiny fixture; larger dims give meaningful perf
    /// numbers without real weights.
    GenOracle {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 20260712)]
        seed: u64,
        #[arg(long, default_value_t = 64)]
        hidden: usize,
        #[arg(long, default_value_t = 4)]
        layers: usize,
        #[arg(long, default_value_t = 16)]
        experts: usize,
        #[arg(long, default_value_t = 4)]
        experts_per_tok: usize,
        #[arg(long, default_value_t = 32)]
        moe_inter: usize,
        #[arg(long, default_value_t = 256)]
        vocab: usize,
    },
    /// Time prefill and decode on a model directory.
    Forward {
        #[command(flatten)]
        model: ModelArgs,
        /// Prefill length of the synthetic prompt.
        #[arg(long, default_value_t = 16)]
        seq: usize,
        /// Decode steps to time after prefill.
        #[arg(long, default_value_t = 32)]
        decode: usize,
    },
}

fn parse_fmt(s: &str) -> Result<QuantFormat> {
    QuantFormat::parse(s).with_context(|| format!("unknown quant format {s:?} (int4|int8|f32)"))
}

fn parse_ids(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|t| t.trim().parse::<usize>().context("bad token id"))
        .collect()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Convert {
            src,
            out,
            experts,
            dense,
            row_chunk,
            force,
        } => {
            let opts = undertow_convert::ConvertOptions {
                expert_format: parse_fmt(&experts)?,
                dense_format: parse_fmt(&dense)?,
                row_chunk,
                force,
            };
            let t0 = Instant::now();
            let classify = registry::classifier_for(&src)?;
            let report = undertow_convert::convert(&src, &out, &classify, &opts)
                .with_context(|| format!("converting {}", src.display()))?;
            println!(
                "converted {} tensors into {} shards ({} skipped as already complete)",
                report.tensors, report.shards_written, report.shards_skipped
            );
            println!(
                "{:.2} GB in -> {:.2} GB out in {:.1}s",
                report.bytes_in as f64 / 1e9,
                report.bytes_out as f64 / 1e9,
                t0.elapsed().as_secs_f64()
            );
        }
        Command::Run {
            model: margs,
            prompt,
            prompt_ids,
            max_new,
            sampling,
            stats,
            mtp,
        } => {
            if mtp && sampling.temperature != 0.0 {
                bail!("--mtp requires greedy decoding (temperature 0)");
            }
            let model = load_model(&margs)?;
            let (ids, tokenizer) = match (&prompt, &prompt_ids) {
                (Some(text), None) => {
                    let tok = undertow_tokenizer::Tokenizer::from_dir(&margs.model)
                        .context("text prompts need tokenizer.json; use --prompt-ids otherwise")?;
                    let ids = tok.encode(text, true).map_err(anyhow::Error::from)?;
                    (ids, Some(tok))
                }
                (None, Some(s)) => (parse_ids(s)?, None),
                _ => bail!("exactly one of --prompt / --prompt-ids is required"),
            };
            let mut generated = Vec::new();
            let t0 = Instant::now();
            let mut session = model.new_session();
            let n = if mtp {
                let ds = registry::as_deepseek(&margs.model, &margs.load_options()?)?;
                let (n, mtp_stats) =
                    undertow_deepseek_moe::generate_greedy_mtp(&ds, &ids, max_new, &[], |id| {
                        generated.push(id);
                        true
                    })?;
                eprintln!(
                    "mtp: {}/{} drafts accepted ({:.0}%)",
                    mtp_stats.accepted,
                    mtp_stats.drafted,
                    mtp_stats.acceptance_rate() * 100.0
                );
                n
            } else {
                let mut sampler = sampling.sampler(0.0)?;
                let stop_ids = model.stop_ids().to_vec();
                undertow_core::generate(
                    &mut *session,
                    &ids,
                    max_new,
                    &mut sampler,
                    &stop_ids,
                    |id| {
                        generated.push(id);
                        true
                    },
                )?
            };
            let dt = t0.elapsed().as_secs_f64();
            match &tokenizer {
                Some(tok) => println!("{}", tok.decode(&generated).map_err(anyhow::Error::from)?),
                None => println!(
                    "{}",
                    generated
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            }
            eprintln!(
                "{n} tokens in {dt:.2}s ({:.2} tok/s, prompt {} tokens)",
                n as f64 / dt,
                ids.len()
            );
            if stats {
                print_stats(&*model, &*session);
            }
            margs.export_profile(&*model)?;
        }
        Command::Chat {
            model: margs,
            system,
            max_new,
            sampling,
        } => {
            let model = load_model(&margs)?;
            let tok = undertow_tokenizer::Tokenizer::from_dir(&margs.model)
                .context("chat needs tokenizer.json in the model directory")?;
            let sampler = sampling.sampler(0.7)?;
            chat::run_chat(&*model, &tok, system, max_new, sampler)?;
            margs.export_profile(&*model)?;
        }
        Command::Doctor {
            model,
            disk_probe_mb,
        } => doctor::run(model, disk_probe_mb.clamp(16, 4096))?,
        Command::Serve {
            model: margs,
            host,
            port,
            api_key,
            timeout_secs,
            max_queue,
            cors_origin,
        } => {
            let model = load_model(&margs)?;
            let tok = undertow_tokenizer::Tokenizer::from_dir(&margs.model)
                .context("serving needs tokenizer.json in the model directory")?;
            let name = margs
                .model
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "undertow".into());
            let addr: std::net::SocketAddr = format!("{host}:{port}")
                .parse()
                .with_context(|| format!("bad listen address {host}:{port}"))?;
            let config = undertow_server::ServerConfig {
                api_key,
                request_timeout: std::time::Duration::from_secs(timeout_secs.max(1)),
                max_queue,
                cors_origin,
            };
            let state = undertow_server::ServerState::with_config(model, tok, name, config);
            undertow_server::run_blocking(state, addr)?;
        }
        Command::Bench { command } => match command {
            BenchCommand::GenOracle {
                out,
                seed,
                hidden,
                layers,
                experts,
                experts_per_tok,
                moe_inter,
                vocab,
            } => {
                let spec = undertow_bench::OracleSpec {
                    seed,
                    hidden_size: hidden,
                    num_hidden_layers: layers,
                    n_routed_experts: experts,
                    num_experts_per_tok: experts_per_tok,
                    moe_intermediate_size: moe_inter,
                    intermediate_size: 4 * moe_inter,
                    vocab_size: vocab,
                    q_lora_rank: (hidden / 2).max(8),
                    kv_lora_rank: (hidden / 4).max(8),
                    ..Default::default()
                };
                undertow_bench::generate_oracle(&out, &spec)?;
                println!("oracle model written to {}", out.display());
            }
            BenchCommand::Forward {
                model: margs,
                seq,
                decode,
            } => {
                let model = load_model(&margs)?;
                let prompt: Vec<usize> =
                    (0..seq).map(|i| (i * 7 + 3) % model.vocab_size()).collect();
                let mut session = model.new_session();
                let t0 = Instant::now();
                let logits = session.prefill(&prompt)?;
                let prefill_s = t0.elapsed().as_secs_f64();
                let mut next = undertow_core::sample::argmax(logits.row(seq - 1));
                let t1 = Instant::now();
                for _ in 0..decode {
                    let l = session.decode(next)?;
                    next = undertow_core::sample::argmax(&l);
                }
                let decode_s = t1.elapsed().as_secs_f64();
                println!(
                    "prefill {seq} tokens: {:.1} ms ({:.1} tok/s)",
                    prefill_s * 1e3,
                    seq as f64 / prefill_s
                );
                println!(
                    "decode {decode} tokens: {:.1} ms ({:.2} tok/s)",
                    decode_s * 1e3,
                    decode as f64 / decode_s
                );
                print_stats(&*model, &*session);
                margs.export_profile(&*model)?;
            }
        },
        Command::Profile { command } => match command {
            ProfileCommand::Merge { inputs, out } => {
                if inputs.is_empty() {
                    bail!("at least one input profile is required");
                }
                let mut profiles = Vec::with_capacity(inputs.len());
                for path in &inputs {
                    let p = undertow_core::ExpertProfile::load(path)
                        .with_context(|| format!("loading profile from {}", path.display()))?;
                    profiles.push(p);
                }
                let merged = undertow_core::ExpertProfile::merge_all(&profiles)?;
                merged
                    .save(&out)
                    .with_context(|| format!("saving merged profile to {}", out.display()))?;
                println!(
                    "Merged {} profiles (arch: {}) -> {} ({} active experts)",
                    inputs.len(),
                    merged.architecture,
                    out.display(),
                    merged.counts.len()
                );
            }
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_profile_merge_cli() {
        let cli = Cli::try_parse_from([
            "undertow",
            "profile",
            "merge",
            "--inputs",
            "p1.json",
            "p2.json",
            "--out",
            "merged.json",
        ])
        .unwrap();

        match cli.command {
            Command::Profile {
                command: ProfileCommand::Merge { inputs, out },
            } => {
                assert_eq!(
                    inputs,
                    vec![PathBuf::from("p1.json"), PathBuf::from("p2.json")]
                );
                assert_eq!(out, PathBuf::from("merged.json"));
            }
            _ => panic!("expected Profile::Merge command"),
        }
    }
}
