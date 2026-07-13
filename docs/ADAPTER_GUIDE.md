# Adding a model family

This is the document that matters most if you want to contribute: the adapter pattern is the whole extension surface, and a new family never touches the runtime. The Mixtral and Qwen3-MoE adapters were added this way and each is a few hundred lines, most of them config parsing and tests.

## What an adapter is

An adapter crate under `adapters/` supplies four things:

1. **A config struct** parsed from the family's `config.json`, with range validation. Configs arrive from untrusted mirrors; validate before any allocation is sized from a field. Look at `undertow-qwen-moe/src/lib.rs` for the shape: serde struct, `from_dir`, `validate`, and a `spec()` that maps config vocabulary onto the shared model spec.
2. **Tensor naming**: an `ExpertNaming` impl mapping `(layer, expert)` to the three FFN tensor names, plus whatever family-specific names exist (router gate, q/k norms). This is what lets the disk store stream experts without knowing your family exists.
3. **A router**: either reuse `SoftmaxTopKRouter` from undertow-core (Mixtral, Qwen) or implement `RouterAdapter` if the gating math is new (the DeepSeek sigmoid noaux_tc router is the worked example). Routers are pure functions of gate logits; keep them that way, it is what makes them testable by hand.
4. **A conversion classifier**: `classify_tensor(name) -> Disposition` deciding expert versus dense versus keep-f32. The recurring trap is a router gate name one substring away from a quantizable projection (`.mlp.gate.weight` versus `.mlp.gate_proj.weight`); write the test for your family's version of it.

If your family uses grouped-query attention, `undertow-moe-common` already has the forward pass, KV cache, sessions and loader; your crate stays thin like `undertow-mixtral-moe`. A genuinely new attention scheme means a forward-pass module like `undertow-deepseek-moe/src/attention.rs`, with the incremental-versus-oneshot equivalence tests that go with it.

## The validation contract

No adapter merges without an oracle. The recipe, all of it mechanical:

1. Add a generator to `undertow-bench/src/oracle.rs` producing a tiny random-weight checkpoint with the real architecture. Use the crate's deterministic RNG only (no libm), so the fixture is byte-stable across platforms. Make the dims exercise the family's tricky config space: expert groups, decoupled head dims, dense-layer patterns, whatever your family has.
2. Run it through the upstream implementation once: `uv run --with torch --with transformers python undertow-bench/tools/make_reference.py <fixture-dir>`. Check in the fixture and the reference.
3. Register the fixture in `undertow-bench/tests/oracle.rs`. Your forward pass must match teacher-forcing argmax at every position and the greedy continuation token for token, with max logit error in the 1e-6 range.
4. Add one arm to the CLI registry (`undertow-cli/src/registry.rs`) mapping the `model_type` string to your loader and classifier.

If the oracle test passes, conversion, streaming, sampling, the server and the CLI all work for your family with no further wiring; that is the point of the boundaries.

## Numerics rules worth knowing before you start

- Norm weights, router gates and biases stay f32 everywhere. Do not quantize them, and classify them `KeepF32`.
- Threading and batching must never change results. Parallel loops write disjoint slices; batched accumulation replays the router's original choice order. If your change needs a tolerance bump in an existing test, it is wrong.
- The scalar kernels in `undertow-quant` are the reference; SIMD or quantized fast paths are validated against them, and anything that changes numerics (like activation quantization) ships opt-in.
