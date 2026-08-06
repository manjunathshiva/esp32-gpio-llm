"""Write the name-targeted near-miss held-out stratum from MASSIVE.

**Why this exists.** Until now every one of MASSIVE's 501 near-miss rows went
into training, and the held-out sets contained 26 name-targeted near-misses
between them -- against a seed spread of +-1.2pp, which cannot resolve
anything. A corpus fix aimed at exactly that shape (branch `v22-near-miss`)
moved a purpose-built probe by 19 points and every held-out number by less than
noise. The model was not the thing that needed fixing; the instrument was.

`massive.is_held_out` reserves 4 rows in 10 of the three hue intents, hashed on
the cleaned text so the partition survives a rebuild, a reordering, or a new
filter upstream. Those rows are excluded from `near_miss_negatives()`, so the
corpus has to be rebuilt and every arm retrained before this stratum means
anything -- a model trained before the split has seen these rows.

Gold is `<unknown>` for all of them: every row asks for brightness, colour or a
scene, none of which this board can do. That is a claim about the whole file,
so DROPPED below records the ones it is not true of, by hand, rather than
letting a rule decide -- the rule that flags "name-shaped" fires just as
readily on six kinds of genuine refusal (see docs, part 4).

    uv run python data/build_nearmiss_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import massive

OUT = Path(__file__).parent / "eval"

# Read by hand, all 111. These two are dropped rather than labelled, which is
# this project's standing rule for an ambiguous row: "white" is a bare colour
# word that could equally be someone's name for a device, and "make a contrast
# one" is garbled past the point of having a correct answer. Everything else
# asks for brightness, colour or a scene and is unambiguously <unknown>.
DROPPED = {
    "white": "bare colour word -- could be a device name",
    "make a contrast one": "garbled; no correct label exists",
}

# The dev/locked split, hashed on the same key as the train/held-out one so a
# row cannot drift between halves either.
LOCKED_SHARE = 5        # in ten, of the held-out rows


def main() -> None:
    rows = massive.held_out_near_miss()
    kept = [t for t in rows if t.lower() not in DROPPED]
    dropped = [t for t in rows if t.lower() in DROPPED]

    dev, locked = [], []
    for t in kept:
        rec = {
            "text": t,
            "symbols": ["<unknown>", "<end>"],
            "action": "unknown",
            "source": "massive",
            "stratum": "neg_nearmiss_named",
            "in_train": False,
            "v2_class": "refusal",
            "v2_deferred": False,
        }
        # Reuse is_held_out on a salted key: a second independent draw, still
        # deterministic, still keyed on the text rather than on position.
        if massive.is_held_out("locked:" + t.lower()):
            rec["locked"] = True
            locked.append(rec)
        else:
            rec["locked"] = False
            dev.append(rec)

    for name, recs in (("v2_nearmiss_dev", dev), ("v2_nearmiss_locked", locked)):
        p = OUT / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        print(f"wrote {p.name:26s} {len(recs):3d} rows")

    review = OUT / "v2_nearmiss_review.txt"
    review.write_text(
        "Held-out name-targeted near-miss negatives, from MASSIVE.\n"
        f"{len(rows)} rows in the held-out partition, {len(kept)} kept, "
        f"{len(dropped)} dropped.\n\n"
        "Dropped by hand, with the reason:\n"
        + "".join(f"  {t!r}\n    {DROPPED[t.lower()]}\n" for t in dropped)
        + "\nEverything else is <unknown>: brightness, colour or scene, none of\n"
          "which this board can do. Hard cases deliberately kept:\n"
          "  'turn the lights down to seven'   -- a level that reads as a pin\n"
          "  'dim the lights to level two'     -- same\n"
          "  'turn the lights blue at three p. m.'  -- colour plus schedule\n"
          "  'turn up the lights'              -- brightness, not <set> high\n")
    print(f"wrote {review.name}")
    print(f"\n{len(rows)} held out, {len(kept)} labelled, {len(dropped)} dropped")


if __name__ == "__main__":
    main()
