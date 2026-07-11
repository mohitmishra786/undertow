"""Produce the golden reference for an oracle model using `transformers`.

Loads the checkpoint written by `undertow bench gen-oracle` into
DeepseekV3ForCausalLM (the upstream-blessed implementation of the
DeepSeek-MoE family) and records:

  * tf_logits  — full teacher-forcing logits over `full_ids` [seq, vocab]
  * tf_pred    — per-position argmax of those logits
  * full_ids   — prompt + greedy continuation (use_cache decode)

The Rust correctness test (engine-bench/tests/oracle.rs) replays the same
inputs and must match tf_pred / full_ids token-exactly and tf_logits within
f32 accumulation tolerance.

Run (regenerates engine-bench/fixtures/oracle-tiny/reference.json):

    uv run --with torch --with transformers \
        python engine-bench/tools/make_reference.py engine-bench/fixtures/oracle-tiny
"""

import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

PROMPT_IDS = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
MAX_NEW_TOKENS = 16


def main() -> None:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "engine-bench/fixtures/oracle-tiny")
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)  # nothing random should run, but be safe
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=cfg, torch_dtype=torch.float32
    ).eval()

    # The engine assumes the interleaved-RoPE MLA formulation; fail loudly if
    # this transformers version resolved the config differently.
    interleave = getattr(model.config, "rope_interleave", None)
    assert interleave is not False, f"expected interleaved rope, got {interleave!r}"

    ids = torch.tensor([PROMPT_IDS])
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=0,
        )
    full_ids = out[0].tolist()

    with torch.no_grad():
        tf_logits = model(torch.tensor([full_ids]), use_cache=False).logits[0]
    tf_pred = tf_logits.argmax(-1).tolist()

    # Greedy consistency: teacher-forcing argmax at position i must equal the
    # generated token i+1 for i >= len(prompt)-1.
    for i in range(len(PROMPT_IDS) - 1, len(full_ids) - 1):
        assert tf_pred[i] == full_ids[i + 1], f"tf/generate mismatch at {i}"

    ref = {
        "prompt_ids": PROMPT_IDS,
        "full_ids": full_ids,
        "tf_pred": tf_pred,
        "tf_logits": [[round(v, 6) for v in row] for row in tf_logits.tolist()],
        "generator": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "model_class": model.__class__.__name__,
        },
    }
    out_path = model_dir / "reference.json"
    out_path.write_text(json.dumps(ref))
    print(f"wrote {out_path}")
    print("full_ids:", full_ids)


if __name__ == "__main__":
    main()
