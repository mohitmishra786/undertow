# Changelog

Format follows keepachangelog.com; versions follow semver once published.

## Unreleased

### Added
- Rayon parallelism across matmul rows, attention heads and expert
  batches; results stay bit-identical to the serial path.
- Prefill expert batching: each routed expert is fetched once per forward
  and multiplied as one batch.
- Opt-in fast int8 path (`--fast-int8`): quantized activations with
  integer accumulation on NEON.
- AVX2 kernels for x86_64 with runtime detection, parity-tested against
  the scalar reference.
- Criterion micro-benchmarks and a scheduled benchmark workflow.
- Server hardening: per-request deadlines, bounded queue with 429,
  client-disconnect cancellation, graceful shutdown, optional bearer
  auth, CORS, Prometheus /metrics, structured request logs.
- Session KV budget (`--kv-budget-mb`) and OpenAI-style prompt clamping.
- Corrupted-checkpoint tests and three libFuzzer targets (safetensors
  header, model config, chat template) with a weekly fuzz workflow.
- Manual tag-triggered release workflow producing a draft release with
  binaries, checksums, an SBOM and build provenance.
- Declared MSRV 1.88, checked in CI, and inherited by every workspace
  member so the floor is enforced by cargo rather than by convention.

### Fixed
- safetensors headers with malformed `data_offsets` arrays or overflowing
  shape products are rejected instead of panicking.
- Chat template rendering is fuel-limited so untrusted templates cannot
  stall the process.
- Byte-to-word decoding uses `slice::as_chunks`, clearing the
  `chunks_exact` clippy lint on current stable.
- Test tokenizer setup matches the `tokenizers` 0.23 owned-`AddedToken`
  API, so the workspace builds against the bumped dependency.
- Transitive `h2` and `rustls` updated past RUSTSEC-2026-0258 and
  RUSTSEC-2026-0285, keeping `cargo audit` and `cargo deny` clean.

## 0.1.0 (2026-07-12)

Initial engine: three adapter families (DeepSeek-style, Mixtral,
Qwen3-MoE) validated token-exactly against transformers oracles; disk
streaming with LRU and weighted caches; int8/int4 quantization with NEON;
streaming resumable converter; tokenizer with chat templates; sampling;
MTP speculative decoding; OpenAI-compatible server; CLI.
