"""Re-score the existing held-out sets against the v1 grammar.

The wider eval batch was built for v0, and about 60% of its in-domain items
target a name -- which v1 refuses. Rebuilding those sets from scratch would
throw away the dev/locked stratification and the review work behind them, so
they are migrated instead: same texts, same split, relabelled.

Three things happen, and only the third involves judgement.

  1. **Name-targeted commands and alias requests become refusals.** They are
     tagged `v1_deferred` rather than merged into the permanent negatives,
     because v2 flips them back and the tag is what makes that one operation.
     `massive` is entirely name-targeted, so it stops being the in-domain
     benchmark and becomes 282 human-written refusals -- a better instrument
     than it was, and the only one in the project no model wrote.

  2. **Pin-targeted commands carry over unchanged**, re-emitted with digit-wise
     pins.

  3. **Refusals are re-read.** v0 labelled "switch off pin 100" and "blink pin 5
     every 20 milliseconds" <unknown>, because the grammar could not represent
     them. v1 can: they are commands the runtime refuses by range. Without
     moving them the substitution metric has an empty denominator -- the eval
     would be blind to the exact failure v1 exists to remove.

Step 3 is gated so it cannot quietly rewrite gold. A refusal is only promoted
when the rules read it as a frame that `range_check` **refuses**. If the rules
read it as something that would *run*, that is a claim the old gold was wrong,
which is a different and much stronger claim -- those go to the review file and
stay refusals.

The v0 files are left untouched; output is `v1_*.jsonl` beside them.

    uv run python data/migrate_eval.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import frames
import label_eval
from frames import Action, All, Frame, Level, Pin

HERE = Path(__file__).parent
EVAL = HERE / "eval"

SETS = ["gemini2_dev", "gemini2_locked", "massive", "gemini"]


def read_v0(syms: list[str]) -> tuple[str, list, str | None, int | None,
                                      int | None, str | None]:
    """Parse a v0 symbol list. Kept here rather than in frames.py: it is a
    migration detail with a finite life, not part of the grammar."""
    action = syms[0][1:-1]
    i, targets = 1, []
    level = interval = count = alias = None

    def digits(j: int) -> tuple[int, int]:
        d = ""
        while syms[j].isdigit():
            d, j = d + syms[j], j + 1
        return int(d), j + 1                      # skip <nend>

    while i < len(syms):
        s = syms[i]
        if s == "<all>":
            targets.append(All())
            i += 1
        elif s.startswith("<p") and s[2:-1].isdigit():
            targets.append(Pin(int(s[2:-1])))
            i += 1
        elif s == "<name>":
            targets.append(s)                     # marker; text at syms[i + 1]
            i += 3
        elif s in ("<high>", "<low>", "<toggle>"):
            level, i = s[1:-1], i + 1
        elif s == "<int>":
            interval, i = digits(i + 1)
        elif s == "<cnt>":
            count, i = digits(i + 1)
        else:
            i += 1
    if action == "alias" and targets and targets[-1] == "<name>":
        alias = "yes"
    return action, targets, level, interval, count, alias


def to_v1(row: dict) -> tuple[Frame, str, str | None]:
    """-> (v1 gold frame, class, review note)."""
    action, targets, level, interval, count, _ = read_v0(row["symbols"])
    named = any(t == "<name>" for t in targets)

    if action == "alias":
        return Frame(Action.UNKNOWN), "deferred_alias", None
    if named:
        return Frame(Action.UNKNOWN), "deferred_name", None

    if action != "unknown":
        f = Frame(Action(action), targets,
                  level=Level(level) if level else None,
                  interval_ms=interval, count=count)
        frames.validate(f)
        return f, "command", None

    # A v0 refusal. Re-read it -- see the module docstring, step 3.
    got = label_eval.label_gemini(row["text"])
    if got == label_eval.DEFERRED:
        return Frame(Action.UNKNOWN), "deferred_name", None
    if isinstance(got, str):
        return Frame(Action.UNKNOWN), "refusal", None
    if frames.executable(got):
        return (Frame(Action.UNKNOWN), "refusal",
                f"rules read a runnable command in a v0 refusal: "
                f"{' '.join(frames.to_symbols(got))}")
    return got, "range_command", None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=SETS)
    a = ap.parse_args()

    review: list[str] = []
    summary: dict[str, Counter] = {}

    for name in a.sets:
        src = EVAL / f"{name}.jsonl"
        if not src.exists():
            print(f"skip {name}: no {src}")
            continue
        rows = [json.loads(l) for l in src.open()]
        out, kinds = [], Counter()

        for r in rows:
            f, kind, note = to_v1(r)
            kinds[kind] += 1
            if note:
                review.append(f"[{name}] {r['text']}\n    {note}")
            out.append({**r, "symbols": frames.to_symbols(f),
                        "action": f.action.value, "v1_class": kind,
                        "v1_deferred": kind.startswith("deferred")})

        with (EVAL / f"v1_{name}.jsonl").open("w") as fh:
            for r in out:
                fh.write(json.dumps(r) + "\n")
        summary[name] = kinds

    (EVAL / "v1_migration_review.txt").write_text("\n".join(review))

    hdr = ["command", "range_command", "deferred_name", "deferred_alias",
           "refusal"]
    print(f"{'set':<18}" + "".join(f"{h:>15}" for h in hdr) + f"{'in-domain':>12}")
    for name, k in summary.items():
        pos = k["command"] + k["range_command"]
        print(f"{name:<18}" + "".join(f"{k[h]:>15,}" for h in hdr) + f"{pos:>12,}")

    print(f"\nwrote {EVAL}/v1_*.jsonl")
    print(f"review: {len(review)} lines where the rules disagree with v0 gold "
          f"-> {EVAL}/v1_migration_review.txt")
    print("\nin-domain n is now small by design -- names left the numerator. "
          "docs/V1-SCOPE.md step 3 (a v1-scoped batch) is what restores it.")


if __name__ == "__main__":
    main()
