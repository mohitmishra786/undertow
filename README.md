<p align="center">
  <img src="assets/undertow.svg" width="500" alt="undertow, the current under the surface">
</p>

Undertow is a Rust inference engine for the largest open-weight MoE models, the GLM-5.2, Kimi K2, DeepSeek-V3/V4 class, built on one observation: a trillion-parameter sparse model touches maybe 1 or 2 percent of its weights per token, so RAM should be a cache over the model, not a container for it. Routed experts live on disk and stream in on demand. Dense and shared weights stay resident. The thing you need to buy becomes disk, which is cheap, instead of RAM, which lately is not.

The engine is architecture-pluggable: one adapter per model family behind small trait boundaries, starting with the DeepSeek-style family because a single adapter covers three major model lines. The reasoning and roadmap are in [docs/project-proposal.md](docs/project-proposal.md).

## Where it stands

Phase 0 is done. That means the trait boundaries (`ModelAdapter`, `RouterAdapter`, `TieredStore`, `ExpertCache`), the DeepSeek family adapter with the sigmoid noaux_tc router and MLA attention, a pread-based safetensors reader, and a scalar f32 forward pass validated against a `transformers` oracle: teacher-forcing argmax and greedy decode match token for token, with max logit error around 1e-6.

There is no disk streaming yet, no cache, no quantized kernels. That ordering is the point. Hoare said there are two ways to build software: so simple there are obviously no deficiencies, or so complicated there are no obvious deficiencies. The scalar path is the first kind, and every optimization that follows has to reproduce its output before we trust it.

## Try it

```sh
cargo test --workspace

cargo run -p engine-cli -- run \
    --model engine-bench/fixtures/oracle-tiny \
    --prompt-ids "3,14,159,26,53,58,200,11,77,240,5,99" --max-new 16
```

Prompts are raw token ids for now. Austere, but honest: the tokenizer arrives in Phase 1 along with the converter and incremental decode. Crate layout and design notes are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

Apache-2.0
