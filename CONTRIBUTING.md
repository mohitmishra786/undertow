# Contributing

Thanks for looking. The short version: correctness is anchored to
`transformers` oracles and everything else is tested relative to that
anchor, so the bar for a change is "the suite stays green and any new
behavior arrives with the test that proves it."

## Getting productive

```sh
cargo test --workspace        # full correctness story, seconds on a laptop
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

No model downloads are needed; synthetic oracle checkpoints ship in
`undertow-bench/fixtures/`. CI runs the suite on Linux x64 and arm64, macOS
and Windows, plus Miri on the kernel crate, so cfg-gated code gets linted
on every platform it exists for.

## What makes a change mergeable

- New behavior comes with tests. Bug fixes come with the test that would
  have caught the bug.
- Numerics changes are opt-in, never silent. Threading, batching and
  storage-tier changes must keep results bit-identical; the pipeline
  tests enforce this and are not to be loosened.
- Docs are written in plain prose (see the existing files for tone), and
  rustdoc builds clean with warnings as errors.
- New dependencies get a hard look: cargo-deny gates licenses and
  advisories, and small trees are preferred.

## Adding a model family

That path is fully documented in
[docs/ADAPTER_GUIDE.md](docs/ADAPTER_GUIDE.md); an adapter never touches
the runtime, so it is the friendliest large contribution to make.

## Security issues

Privately, please: see [SECURITY.md](SECURITY.md).
