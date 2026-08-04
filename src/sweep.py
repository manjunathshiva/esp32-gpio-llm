"""Score one config across several training seeds and report the spread.

Every number through v1-v5 came from a single seed. By v5 the improvements
being chased were 5-8 items on a 150-item set, which is the same size as the
run-to-run variation nobody had measured -- so it was impossible to tell a real
gain from a lucky initialisation. This measures the variation directly.

The corpus is held fixed and only the training seed moves (weight init, batch
order, dropout draw). That isolates optimisation noise, which is the thing that
makes a single-seed comparison untrustworthy. Corpus seed is a separate and
larger source of variation, deliberately not folded in here -- mixing them
would give one number that answers neither question.

**How to read it.** The spread column is the honest resolution of the setup:
a change smaller than it is not evidence. Compare configs by mean, and only
believe a difference that clears the spread of both.

    uv run python src/sweep.py --tag v5 --seeds 0 1 2
    uv run python src/sweep.py --tag v5 --splits gemini2_dev massive
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from evaluate import Decoder, load_split, score, wilson

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

DEFAULT_SPLITS = ["v1_gemini2_dev", "v1_gemini2_locked", "v1_massive",
                  "v1_gemini"]


def fmt(vals: list[float]) -> str:
    """mean and half-range. Half-range rather than stdev: at n=3 a standard
    deviation is barely estimated, while the observed spread is exactly what it
    claims to be -- how far apart the runs actually landed."""
    if len(vals) == 1:
        return f"{vals[0]:6.1%}     --   "
    lo, hi = min(vals), max(vals)
    return f"{statistics.fmean(vals):6.1%}  +-{(hi - lo) / 2:5.1%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--constrained", action="store_true")
    a = ap.parse_args()

    runs = []
    for s in a.seeds:
        p = RUNS / f"cmd-{a.tag}-s{s}.pt"
        if not p.exists():
            raise SystemExit(f"missing {p}\n  uv run python src/train.py "
                             f"--steps 14000 --seed {s} --tag {a.tag}")
        runs.append((s, p))

    splits = {name: load_split(name) for name in a.splits}
    METRICS = [("exact", "pos_ok", "pos_n"), ("false", "neg_bad", "neg_n"),
               ("pin", "pin_ok", "pin_n"), ("sub", "sub_bad", "sub_n")]
    results: dict[str, dict[str, list[float]]] = {
        name: {m: [] for m, _, _ in METRICS} for name in a.splits}

    for seed, path in runs:
        dec = Decoder(path)
        for name, rows in splits.items():
            s = score(dec, rows, constrained=a.constrained)
            for m, num, den in METRICS:
                results[name][m].append(s[num] / max(1, s[den]))
            print(f"  {path.name:<18} {name:<16} "
                  f"exact {s['pos_ok']:4d}/{s['pos_n']:<4d} "
                  f"false {s['neg_bad']:3d}/{s['neg_n']:<4d} "
                  f"pin {s['pin_ok']:4d}/{s['pin_n']:<4d} "
                  f"sub {s['sub_bad']:3d}/{s['sub_n']:<4d}", flush=True)

    seeds = " ".join(str(s) for s in a.seeds)
    print(f"\ntag {a.tag}   seeds {seeds}   n={len(runs)}\n")
    print(f"{'split':<18}{'exact-match':<18}{'false-accept':<18}"
          f"{'pin copy':<18}{'substitution':<18}{'in-domain n':>12}")
    for name in a.splits:
        rows = splits[name]
        pos_n = sum(1 for r in rows if r["action"] != "unknown")
        cells = "".join(f"{fmt(results[name][m]):<18}" for m, _, _ in METRICS)
        print(f"{name:<18}{cells}{pos_n:>12}")

    # The interval on a single seed and the spread across seeds answer
    # different questions -- how precisely this set measures one model, and how
    # much the model moves. Both have to be smaller than a difference before it
    # is worth acting on.
    print("\n95% interval on the mean exact-match, single split, for reference:")
    for name in a.splits:
        rows = splits[name]
        pos_n = sum(1 for r in rows if r["action"] != "unknown")
        p = statistics.fmean(results[name]["exact"])
        _, lo, hi = wilson(round(p * pos_n), pos_n)
        print(f"  {name:<18} [{lo:.1%}, {hi:.1%}]")


if __name__ == "__main__":
    main()
