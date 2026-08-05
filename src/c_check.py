"""End-to-end gate: the C pipeline must emit exactly what PyTorch emits.

verify.c proves the forward pass on one golden prompt. tok_check.py proves the
tokenizer. Neither covers the sampling loop, the stopping rule, or the prompt
truncation -- and those are where a port quietly diverges: the parse stays
plausible, so nothing looks broken, it just gets a bit worse.

So this runs every held-out utterance through both and diffs the *emitted token
id sequence*. Identical ids mean identical commands by construction, for the
whole chain from typed bytes to symbols.

  cc -O3 -o /tmp/repl firmware/host_verify/repl.c -lm
  uv run python src/c_check.py [--run runs/cmd-v1-s0.pt]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from evaluate import Decoder  # noqa: E402

BIN = "/tmp/repl"
SPLITS = ["v1_gemini2_dev", "v1_gemini2_locked", "v1_massive"]


@torch.no_grad()
def torch_ids(dec: Decoder, text: str, max_new: int = 40) -> list[int]:
    """The same loop as Decoder.parse, but returning raw ids so it can be
    compared with the C rather than with a Frame."""
    ids = dec.tok.encode(text).ids
    budget = dec.cfg.seq_len - max_new - 1
    if len(ids) > budget:
        ids = ids[-budget:]
    ids = ids + [dec.go]
    gen: list[int] = []
    for _ in range(max_new):
        x = torch.tensor([ids + gen], device=dec.device)
        logits, _ = dec.model(x)
        nxt = int(logits[0, -1].argmax())
        gen.append(nxt)
        if nxt == dec.end_id:
            break
    return gen


def main() -> int:
    ap = argparse.ArgumentParser()
    # Kept in step with scripts/release.sh's RUN_TAG. release.sh passes --run
    # explicitly, so this default only affects manual invocation -- but a stale
    # default here is the same trap as a stale tokenizer: it compares against
    # the wrong checkpoint and says PASS or FAIL about something you did not ask.
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "cmd-v2-s1.pt")
    ap.add_argument("--model", default=str(ROOT / "firmware" / "model" / "model.bin"))
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()

    if not Path(BIN).exists():
        sys.exit(f"missing {BIN} -- build it:\n"
                 f"  cc -O3 -o {BIN} firmware/host_verify/repl.c -lm")

    texts: list[str] = []
    for s in a.splits:
        p = ROOT / "data" / "eval" / f"{s}.jsonl"
        if p.exists():
            texts += [json.loads(l)["text"] for l in p.open()][:a.limit]
    # Newlines would break the line-based harness; there are none in the sets,
    # but a stray one would silently shift every later comparison by one.
    texts = [t for t in texts if "\n" not in t and t.strip()]
    # The eval sets are gitignored, so a fresh clone has none of them. Without
    # this the gate compares zero utterances and reports PASS -- a release gate
    # that passes because it checked nothing is worse than no gate at all.
    if not texts:
        sys.exit(f"no utterances found for {', '.join(a.splits)} in "
                 f"{ROOT / 'data' / 'eval'} -- the held-out sets are not "
                 f"published; regenerate or copy them in before releasing.")
    print(f"{len(texts)} utterances from {', '.join(a.splits)}")

    proc = subprocess.run([BIN, a.model], input="\n".join(texts) + "\n",
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"repl failed: {proc.stderr[:400]}")
    got = proc.stdout.strip().split("\n")

    dec = Decoder(a.run)
    bad = 0
    for i, t in enumerate(texts):
        want = torch_ids(dec, t)
        mine = [int(x) for x in got[i].split()] if i < len(got) and got[i].strip() else []
        if want != mine:
            bad += 1
            if bad <= 6:
                print(f"MISMATCH {t!r}")
                print(f"  torch: {want}")
                print(f"  c    : {mine}")
                print(f"  torch symbols: {' '.join(dec.inv.get(x, '?') for x in want)}")
                print(f"  c     symbols: {' '.join(dec.inv.get(x, '?') for x in mine)}")

    n = len(texts)
    print(f"\n{n - bad}/{n} identical ({100.0 * (n - bad) / n:.2f}%)")
    if bad:
        print("FAIL: the C pipeline and PyTorch disagree")
        return 1
    print("PASS: C reproduces PyTorch end to end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
