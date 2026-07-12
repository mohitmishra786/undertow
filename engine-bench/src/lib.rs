//! Synthetic "oracle" model generation.
//!
//! Real architecture, random weights, tiny dimensions: enough to validate
//! every line of the forward pass on a laptop in milliseconds, without a
//! single real model byte. The generated directory is a normal HF-layout
//! checkpoint (config.json + model.safetensors) so both this engine *and*
//! `transformers` can load it — the `transformers` output is the golden
//! reference (see `tools/make_reference.py`).

pub mod oracle;
pub mod rng;

pub use oracle::{generate_mixtral_oracle, generate_oracle, generate_qwen_oracle, OracleSpec};
