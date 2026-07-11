# Architecture

This is a description of what is actually built and why it has the shape it has. For the vision, the market argument and the phase plan, read [project-proposal.md](project-proposal.md). If something here disagrees with the code, the code wins and this file has a bug.

## Crate map

<p align="center">
  <img src="../assets/crates.svg" width="640" alt="Crate dependency graph">
</p>

`engine-core` sits at the bottom on purpose. It owns the trait vocabulary and depends on nothing else in the workspace, so an adapter author never pulls in I/O or kernel code just to implement an interface. `engine-quant` is slice-based and dependency-free, and I want to keep it that way: those scalar kernels are the reference that every future quantized or SIMD kernel gets validated against, so they have to stay small enough to audit by reading.

Adapters live one crate per architecture family. The core never branches on a model name. When Mixtral or Qwen support lands, it lands as a sibling of `deepseek-moe`, not as an `if` in the runtime.

## The two traits that matter

**`RouterAdapter` is pure selection math.** It receives gate logits that the caller already computed (the gating matmul is just a matmul, the runtime owns it) and returns chosen experts with their combination weights:

```rust
fn route(&self, gate_logits: &[f32], correction_bias: Option<&[f32]>) -> Vec<ExpertChoice>;
```

Keeping I/O, batching and caching out of this signature means a router implementation is a deterministic function you can unit test against values computed by hand, which is exactly what the tests do. The DeepSeek implementation carries the full noaux_tc semantics: sigmoid scores, a correction bias that influences which experts get picked but never their weights, group-limited top-k where each group is ranked by the sum of its two best corrected scores, then `norm_topk_prob` and `routed_scaling_factor`. One implementation, three model families.

**`TieredStore` is the streaming boundary.** The MoE block reaches routed-expert weights only through `get_expert(layer, id)`. In Phase 0 the store behind that call is a HashMap holding everything in RAM, which sounds silly until you notice what it buys: the compute code is already written against the streaming interface, so swapping in the disk-backed store later touches the loader and nothing else.

<p align="center">
  <img src="../assets/tiers.svg" width="700" alt="Tiered storage design">
</p>

The diagram shows the target design. Today only the RAM tier exists. `ExpertCache` is a trait without an implementation for the same reason there is no streaming yet: I am not writing cache code until the forward pass it serves is proven correct, and now that it is, that work can start. The store trait also avoids assuming storage is local to one machine, because Phase 4 wants to pool disks over a LAN and I would rather not redesign the boundary then.

## MLA, the transparent formulation

`deepseek-moe/src/attention.rs` implements multi-head latent attention the readable way: reconstruct every context position's per-head keys and values from the compressed KV latent through `kv_b_proj`, then do ordinary causal attention. Quadratic, allocation-happy, correct. The two well-known optimizations, weight absorption for decode and the compressed KV cache, are Phase 1 work and both have to reproduce this path's output.

RoPE here is the interleaved partial variant the whole family uses: only the last `qk_rope_head_dim` dims of each head rotate, input pairs `(2j, 2j+1)` land at split-half positions `(j, half+j)`. This matches what `transformers` calls `apply_rotary_pos_emb_interleave`, and the oracle test confirms the match numerically rather than by reading the two implementations side by side and hoping.

## How correctness is established

The method is simple, and it deserves to be standard practice:

1. `engine-bench` generates a tiny checkpoint with the real architecture (a dense-layer prefix, q-LoRA MLA, expert groups, a shared expert, the correction bias) and random weights. The generator's RNG is PCG32 plus an Irwin-Hall normal, no libm anywhere, so it produces bit-identical files on every platform. That is what lets the fixture and its golden reference live in git.
2. `tools/make_reference.py` runs that checkpoint through `transformers.DeepseekV3ForCausalLM` once and records teacher-forcing logits and a greedy continuation.
3. The Rust tests replay the same inputs and must match token-exactly at all 28 positions, plus the full greedy decode, with max logit error observed around 1e-6. A separate test regenerates the oracle and asserts byte-identity with the fixture, so the generator cannot drift away from the golden file without someone noticing.

One check I did not want to skip: whether the tiny model routes diversely enough to mean anything. It does, 12 to 13 of 16 experts get used per MoE layer (`cargo run -p engine-bench --example expert_usage`), so group limitation and the bias-versus-weight separation are genuinely exercised, not just present in the config.

## Why pread and not mmap

With a 350 to 600 GB expert pool behind an 8 to 64 GB RAM budget, mmap hands residency decisions to the page cache, and RSS becomes something that happens to you rather than something you chose. That failure mode is worst on exactly the unified-memory machines this engine targets. Positioned reads into buffers we own keep RSS flat and leave residency to the tier scheduler. They are also offset-stateless, so concurrent expert fetches never fight over a shared file cursor. The fast-path I/O work in Phase 1 keeps this constraint.
