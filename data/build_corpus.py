"""Assemble the training corpus.

Writes JSONL with one row per example:

    {"text": ..., "symbols": [...], "action": ..., "source": ..., "split": ...}

`source` is carried per row on purpose. MASSIVE is CC-BY-4.0, so if this corpus
is ever published the MASSIVE-derived rows need attribution while the synthetic
rows do not. Tagging at build time is cheap; reconstructing provenance later is
not.

Usage:
    uv run python data/build_corpus.py --n 150000 --out data/corpus
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import frames
import negatives
import realize
from frames import Action

# Negatives as a share of the whole corpus. High enough that <unknown> is not a
# rare class, low enough that the model does not learn to reject by default.
# Up from v0's 0.22 because v1 refuses a whole class it used to accept -- every
# command aimed at a name -- and that class is a large share of what people
# actually say to a device like this.
NEG_FRACTION = 0.30

# Within the negatives. MASSIVE near-misses are weighted far above their raw
# count: 680 real requests for dimming and colour are worth more than any
# number of generated gibberish strings, because they are what the false-accept
# metric actually measures.
#
# The `deferred` entry is gone as of v2.1. It held capability the device did
# not have yet -- name-targeted commands until v2.0, alias creation until v2.1
# -- and both are positives now. It was a separate entry precisely so that
# emptying it would be a one-line change.
#
# Its share was 0.30 in v1 and 0.08 in v2.0. Each time the freed share went
# back to the other four in their existing proportions rather than to one of
# them, which would have quietly re-weighted what false-accept measures.
NEG_MIX = {"massive_near": 0.23, "massive_far": 0.26, "synthetic": 0.34,
           "truncated": 0.17}


def build(n: int, seed: int, use_massive: bool) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()

    def add(text: str, frame: frames.Frame, source: str) -> bool:
        key = text.lower().strip()
        if not key or key in seen:
            return False
        seen.add(key)
        rows.append({
            "text": text,
            "symbols": frames.to_symbols(frame),
            "action": frame.action.value,
            "source": source,
        })
        return True

    n_neg = int(n * NEG_FRACTION)
    n_pos = n - n_neg

    # --- positives -----------------------------------------------------------
    got = tries = 0
    while got < n_pos:
        tries += 1
        if tries > n_pos * 40:
            print(f"  ! template space exhausted at {got:,} positives "
                  f"({tries:,} draws)")
            break
        f = frames.sample_frame(rng)
        got += add(realize.realize(f, rng), f, "template")

    # --- negatives -----------------------------------------------------------
    unknown = frames.Frame(Action.UNKNOWN)
    pools: dict[str, list[str]] = {}
    if use_massive:
        import massive
        pools["massive_near"] = massive.near_miss_negatives()
        pools["massive_far"] = massive.far_negatives()

    for kind, share in NEG_MIX.items():
        want = int(n_neg * share)
        if kind == "synthetic":
            got, tries = 0, 0
            while got < want and tries < want * 60:
                tries += 1
                got += add(negatives.sample_negative(rng), unknown, "synthetic")
            continue
        if kind == "truncated":
            # Cut real commands short. Half-finished input was the largest
            # false-accept category, and it has to be derived from the same
            # surface forms the positives use or the cut lands in the wrong
            # places.
            got, tries = 0, 0
            while got < want and tries < want * 60:
                tries += 1
                f = frames.sample_frame(rng)
                cut = negatives.truncate(realize.realize(f, rng), rng)
                if cut:
                    got += add(cut, unknown, "truncated")
            continue
        pool = pools.get(kind)
        if not pool:
            continue
        rng.shuffle(pool)
        got = 0
        for text in pool[:want]:
            got += add(text, unknown, kind)

        # MASSIVE only has 680 unique near-misses, far short of their share.
        # They are the negatives the false-accept metric actually measures, so
        # rather than leave them at 0.5% of the corpus, re-dress them with the
        # same politeness and punctuation variation the positives get. This adds
        # surface variety, not new information -- the alternative, duplicating
        # identical strings, adds neither.
        tries = 0
        while got < want and tries < want * 20:
            tries += 1
            got += add(negatives.vary(rng.choice(pool), rng), unknown, kind)

    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "corpus")
    ap.add_argument("--no-massive", action="store_true")
    a = ap.parse_args()

    rows = build(a.n, a.seed, use_massive=not a.no_massive)
    n_val = int(len(rows) * a.val_frac)
    a.out.mkdir(parents=True, exist_ok=True)

    for name, chunk in (("val", rows[:n_val]), ("train", rows[n_val:])):
        for r in chunk:
            r["split"] = name
        with (a.out / f"{name}.jsonl").open("w") as fh:
            for r in chunk:
                fh.write(json.dumps(r) + "\n")

    from collections import Counter
    print(f"{len(rows):,} rows -> {a.out}  (train {len(rows) - n_val:,} / val {n_val:,})")
    print("\nby action:")
    for k, v in Counter(r["action"] for r in rows).most_common():
        print(f"  {k:<9} {v:7,}  {v / len(rows):5.1%}")
    print("\nby source:")
    for k, v in Counter(r["source"] for r in rows).most_common():
        print(f"  {k:<14} {v:7,}  {v / len(rows):5.1%}")


if __name__ == "__main__":
    main()
