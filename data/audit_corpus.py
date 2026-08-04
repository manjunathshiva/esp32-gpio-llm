"""Composition and label-consistency report for a built corpus.

The v1 gate from docs/V1-SCOPE.md, step 4: out-of-range values have to appear
often enough in pin-bearing positives that the model's prior is not "a number
after 'pin' is a legal pin". That prior is what produced the substitution bug --
"switch off pin 100" answered with <p10> -- and the only thing that removes it
is having seen plenty of pin numbers that are not on the board.

The second half is a labelling check the v0 corpus never had a reason to run.
Now that a pin is copied digits rather than a looked-up symbol, a label whose
digits are absent from the utterance is not merely odd, it is unlearnable: the
model would have to invent the number. `realize._terse` was dropping the count
from chase forms exactly this way.

    uv run python data/audit_corpus.py
    uv run python data/audit_corpus.py --corpus data/corpus --show 15
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import frames
import realize
from frames import Action, Pin

HERE = Path(__file__).parent

# Words that stand in for a number, so a label digit with no literal match in
# the text is explained rather than broken.
_WORDY = set(realize.NUM_WORDS.values()) | {
    "once", "one time", "a single time", "twice", "two times", "thrice",
    "forever", "continuously", "indefinitely", "non-stop", "endlessly",
    "a second", "half a second", "quarter second", "couple of seconds",
}


def digit_runs(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=HERE / "corpus")
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()

    rows = []
    for split in ("train", "val"):
        p = a.corpus / f"{split}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in p.open()]
    if not rows:
        raise SystemExit(f"no corpus at {a.corpus}")

    pins: Counter = Counter()
    verdicts: Counter = Counter()
    iv_lit = Counter()
    bad_pin_copy: list[tuple[str, int]] = []
    bad_cnt_copy: list[tuple[str, int]] = []
    n_pos = 0

    for r in rows:
        f = frames.from_symbols(r["symbols"])
        if f.action is Action.UNKNOWN:
            continue
        n_pos += 1
        runs = digit_runs(r["text"])
        low = r["text"].lower()

        for t in f.targets:
            if not isinstance(t, Pin):
                continue
            pins["legal" if t.n in frames.PINS_S3 else
                 ("3-digit" if t.n >= 100 else "out-of-range")] += 1
            if str(t.n) not in runs and not any(w in low for w in _WORDY):
                bad_pin_copy.append((r["text"], t.n))

        verdicts[frames.range_check(f)[0].value] += 1

        if f.interval_ms is not None:
            iv_lit["literal" if str(f.interval_ms) in runs else "derived"] += 1
        if f.count:                       # 0 is legitimately silent ("forever")
            if str(f.count) not in runs and not any(w in low for w in _WORDY):
                bad_cnt_copy.append((r["text"], f.count))

    tot = sum(pins.values())
    print(f"{len(rows):,} rows   {n_pos:,} positives   "
          f"{len(rows) - n_pos:,} negatives\n")

    print("pin references in positives")
    for k in ("legal", "out-of-range", "3-digit"):
        print(f"  {k:<14}{pins[k]:7,}  {pins[k] / tot:6.1%}")
    oor = (pins["out-of-range"] + pins["3-digit"]) / tot
    print(f"  {'-> not on board':<14}{'':7}  {oor:6.1%}"
          f"   {'OK' if oor > 0.05 else 'TOO LOW -- the prior stays legal-only'}")

    print("\nruntime verdict on positives (all of these must parse, "
          "and only `execute` may run)")
    for k, v in verdicts.most_common():
        print(f"  {k:<16}{v:7,}  {v / n_pos:6.1%}")

    print("\ninterval digits present verbatim in the utterance")
    for k, v in iv_lit.most_common():
        print(f"  {k:<16}{v:7,}  {v / max(1, sum(iv_lit.values())):6.1%}")
    print("  ('derived' is fine and expected: 'every 3 seconds' -> 3000)")

    print(f"\nunlearnable labels  pin {len(bad_pin_copy)}   "
          f"count {len(bad_cnt_copy)}")
    for label, bad in (("pin", bad_pin_copy), ("count", bad_cnt_copy)):
        for text, n in bad[:a.show]:
            print(f"  {label} {n} not in: {text}")
    if bad_pin_copy or bad_cnt_copy:
        print("  ^ each of these asks the model to emit a number it was never "
              "shown. Fix realize.py, do not tune around it.")


if __name__ == "__main__":
    main()
