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
- Integration test in `undertow-convert` asserting bit-identical output
  shards across varying row chunk sizes (1, 13, and 1024).
- CLI utility `undertow profile merge` to combine and rank multiple
  expert usage profiles into a unified hot-expert profile.
- Extended Prometheus `/metrics` telemetry with cache hit ratio, RAM bytes used,
  cache budget, expert evictions total, synchronous pread latency histogram,
  and instantaneous decode generation rate.
- Structured tool / function calling support (`tools`, `tool_choice`) with
  automatic extraction of XML, markdown, and raw JSON tool calls in `undertow-server`.
- Ollama API compatibility shim in `undertow-server` exposing `/api/tags` and
  streaming/unary `/api/chat` (application/x-ndjson).
- OS buffer cache bypass in `undertow-io`: `F_NOCACHE` enabled by default on macOS
  and `posix_fadvise` page cache invalidation on Linux, preventing RAM page cache
  inflation during multi-thousand token generation runs.

### Changed
- Router top-k expert selection in `undertow-core` and `undertow-deepseek-moe`
  optimized from $O(K^2 \cdot N)$ to $O(N \log K)$ via fixed-capacity binary min-heap
  with zero heap allocations for $K \le 64$, preserving bit-identical `torch.topk`
  descending rank and lowest-index tie-breaking parity.

### Fixed
- Fuzz target `model_config` expanded to cover `MixtralConfig` and `QwenConfig`
  alongside DeepSeek; hardened division and alignment checks in Mixtral and
  Qwen config parsers against zero counts.
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
- Architecture documentation (`docs/ARCHITECTURE.md`) reconciled with
  the shipped AVX2 kernel implementation and future AVX-512/AMX roadmap.
- Model name validation in `undertow-server` validates incoming `ChatRequest`
  and `CompletionRequest` `model` parameter against the loaded model name,
  eliminating `#[allow(dead_code)]` annotations and rejecting mismatches.
- Model registry recognizes `deepseek_v4`, `deepseek_v2`, and `qwen2_moe`
  architectures; DeepSeek config parses `rope_scaling` dictionaries and
  nested frequency configurations gracefully, and Qwen adapter derives head
  dimensions for Qwen2.5-MoE checkpoints lacking explicit `head_dim`.

## 0.1.0 (2026-07-12)

Initial engine: three adapter families (DeepSeek-style, Mixtral,
Qwen3-MoE) validated token-exactly against transformers oracles; disk
streaming with LRU and weighted caches; int8/int4 quantization with NEON;
streaming resumable converter; tokenizer with chat templates; sampling;
MTP speculative decoding; OpenAI-compatible server; CLI.
