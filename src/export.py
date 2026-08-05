"""Export a trained model to a flat binary the C runtime reads, plus a golden
logits reference so the C port can be proven correct before it touches hardware.

**fp32, not quantized.** esp32-tinyllm quantizes because a 28.9M-parameter model
does not otherwise fit; this one is 312,128 parameters -- 1219 KB fp32, which an
ESP32-S3 holds in flash without noticing. Quantization would buy ~900 KB nobody
needs and cost the thing that matters most right now: an exact host check.
`verify.c` can demand near-bit-level agreement instead of a tolerance, so a
C-vs-PyTorch mismatch means a porting bug and never "probably rounding".
Revisit only if the model grows.

Format is deliberately dead-simple (the C reader is ~40 lines):

    [magic "CMD1"][vocab dim n_layers n_heads ffn seq_len : int32][rope_theta : f32]
    then tensors as raw little-endian fp32, in the fixed order below.

There are no names in the file, only offsets. The order here and `llm_load()` in
firmware/common/llm.h are one contract: change one, change the other.

    uv run python src/export.py [run-tag]
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Config, TinyLM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "firmware" / "model"
MAGIC = 0x434D4431          # "CMD1"

# The prompt the golden is taken over. A pin-numbered command with a two-digit
# pin on purpose: it exercises the digit path, which is where a tokenizer or
# emission bug would do real damage.
GOLDEN_PROMPT = "blink pin 38 every 300ms"


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1-s0"
    run = ROOT / "runs" / f"cmd-{tag}.pt"
    if not run.exists():
        raise SystemExit(f"missing {run}")

    ck = torch.load(run, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = TinyLM(cfg).eval()
    model.load_state_dict(ck["state"])
    sd = model.state_dict()

    OUT.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, tuple[int, ...]]] = [
        ("tok_emb.weight", (cfg.vocab_size, cfg.d_model)),
    ]
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        plan += [
            (p + "attn_norm.weight", (cfg.d_model,)),
            (p + "attn.qkv.weight", (3 * cfg.d_model, cfg.d_model)),
            (p + "attn.proj.weight", (cfg.d_model, cfg.d_model)),
            (p + "ffn_norm.weight", (cfg.d_model,)),
            (p + "ffn.gate.weight", (cfg.ffn_hidden, cfg.d_model)),
            (p + "ffn.up.weight", (cfg.ffn_hidden, cfg.d_model)),
            (p + "ffn.down.weight", (cfg.d_model, cfg.ffn_hidden)),
        ]
    plan.append(("out_norm.weight", (cfg.d_model,)))

    blob = bytearray()
    blob += struct.pack("<I", MAGIC)
    blob += struct.pack("<6i", cfg.vocab_size, cfg.d_model, cfg.n_layers,
                        cfg.n_heads, cfg.ffn_hidden, cfg.seq_len)
    blob += struct.pack("<f", cfg.rope_theta)

    for name, shape in plan:
        t = sd[name]
        if tuple(t.shape) != shape:
            raise SystemExit(f"{name}: expected {shape}, got {tuple(t.shape)}")
        blob += t.detach().numpy().astype("<f4").tobytes()

    path = OUT / "model.bin"
    path.write_bytes(blob)
    print(f"wrote {path}  ({len(blob) / 1024:.0f} KB, {len(plan)} tensors)")

    # --- golden -------------------------------------------------------------
    # Last-position logits for a fixed prompt. verify.c reproduces them from
    # model.bin alone; agreement proves the C forward pass, independently of the
    # tokenizer (whose own gate is tok_check.py).
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ROOT / "data" / "bpe.json"))
    meta = json.loads((ROOT / "data" / "meta.json").read_text())
    ids = tok.encode(GOLDEN_PROMPT).ids + [meta["go"]]

    with torch.no_grad():
        logits, _ = model(torch.tensor([ids]))
    row = logits[0, -1].numpy()

    gold = OUT / "golden.txt"
    with gold.open("w") as fh:
        fh.write(f"{len(ids)}\n")
        fh.write(" ".join(str(i) for i in ids) + "\n")
        fh.write(" ".join(f"{v:.8e}" for v in row) + "\n")
    print(f"wrote {gold}  (prompt {GOLDEN_PROMPT!r}, {len(ids)} ids, "
          f"{len(row)} logits)")

    inv = {v: k for k, v in tok.get_vocab().items()}
    print(f"argmax -> {inv.get(int(np.argmax(row)), '?')!r}")


if __name__ == "__main__":
    main()
