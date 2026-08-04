"""Ingest the v1 eval batch and merge it into the held-out sets.

The migrated v1 sets hold 95 and 96 in-domain items -- smaller than the set
abandoned for being too small, and blind by construction: collected against v0,
they contain no utterance v0 could not express, so no multi-pin command and no
out-of-range value. Both v1 bugs found so far were invisible to them.

This reads six AI Studio runs (`data/eval_prompts/v1_*.md`) and **adds** to the
existing sets rather than replacing them -- a refusal is still a refusal, so the
old rows keep their value and their locked/dev assignment.

Gold is Gemini's annotation of its own sentences, re-checked three ways:
`frames.validate` for structural legality, `level_check` for on/off agreement,
and `label_eval.label_gemini` as an independent second opinion. In v1 that
second opinion is usable on *every* in-domain line, because every in-domain
line targets a pin number -- the restriction that made it unreliable on v0 was
name-targeted phrasing, and there is none left. Disagreements are quarantined,
never resolved automatically.

**Out-of-range values are positives here.** "switch off pin 100" gets the frame
it says, `[100]`, not a corrected `[10]`. The whole point of the numeric file is
to measure whether the model transcribes or substitutes, and an ingest that
clamped would delete the measurement.

    uv run python data/ingest_eval.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import frames
from frames import Action, All, Frame, Level, Pin
from label_eval import DEFERRED, label_gemini

HERE = Path(__file__).parent
RAW = HERE / "eval_raw"
OUT = HERE / "eval"
CORPUS = HERE / "corpus"

LEVELS = {l.value: l for l in Level}
ACTIONS = {a.value: a for a in Action}

# Sets the new rows are deduplicated against. Their own outputs are excluded so
# re-running after a raw-file edit is idempotent rather than empty.
SEEN_SETS = ["v1_gemini2_dev", "v1_gemini2_locked", "v1_massive", "v1_gemini"]
SELF = {"v1_dev.jsonl", "v1_locked.jsonl"}


def read_jsonl_ish(path: Path) -> list[dict]:
    """Pull JSON objects out of whatever AI Studio produced: a bare JSONL body,
    a ```json fence, or a {"response": "..."} wrapper. Anything else on a line
    is skipped rather than guessed at."""
    text = path.read_text()
    stripped = text.strip()
    if stripped.startswith("{") and '"response"' in stripped[:40]:
        try:
            text = json.loads(stripped)["response"]
        except (json.JSONDecodeError, KeyError):
            pass
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_lines(path: Path) -> list[str]:
    """Plain-text refusals, one per line."""
    text = path.read_text()
    stripped = text.strip()
    if stripped.startswith("{") and '"response"' in stripped[:40]:
        try:
            text = json.loads(stripped)["response"]
        except (json.JSONDecodeError, KeyError):
            pass
    out = []
    for line in text.splitlines():
        line = line.strip().strip("-*").strip()
        line = re.sub(r"^\d+[.)]\s*", "", line)      # stray numbering
        if line and not line.startswith(("#", "```", "|")):
            out.append(line)
    return out


# --------------------------------------------------------------------------
# in-domain
# --------------------------------------------------------------------------

ALL_PHRASE = re.compile(
    r"\b(?:everything|every\s+(?:pin|output|channel|line)|"
    r"all\s+(?:the\s+|of\s+)?(?:pins?|outputs?|channels?|lines?|them))\b", re.I)


def build_frame(o: dict) -> Frame:
    """Gemini's annotation -> a Frame. Raises on anything malformed.

    Deliberately does *not* range-check. A pin outside the allowlist is a valid
    v1 label; `frames.range_check` refuses it downstream, and that refusal is
    what the numeric file exists to measure.
    """
    action = ACTIONS.get(str(o.get("action", "")).lower())
    if action is None or action is Action.UNKNOWN:
        raise frames.ParseError(f"bad action {o.get('action')!r}")
    text = str(o.get("text", ""))

    targets: list[frames.Target] = []
    for t in o.get("targets") or []:
        if isinstance(t, bool):
            raise frames.ParseError("bool target")
        if isinstance(t, int):
            targets.append(Pin(t))
        elif isinstance(t, str) and t.strip().lower() in ("all", "*", ""):
            # The literal sentinel came back as "" on most all-pins items in the
            # first batch, so an empty target is repaired -- but only when the
            # text actually says so, which makes it a check and not a guess.
            if t.strip().lower() != "all" and not ALL_PHRASE.search(text):
                raise frames.ParseError(f"bad target {t!r}")
            targets.append(All())
        elif isinstance(t, str) and t.strip().isdigit():
            targets.append(Pin(int(t.strip())))
        else:
            # A name. v1 has no Name target, so this is either the wrong file or
            # a mis-annotation; either way it is not gold.
            raise frames.ParseError(f"non-numeric target {t!r}")

    lvl = o.get("level")
    interval, count = o.get("interval_ms"), o.get("count")
    for v in (interval, count):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
            raise frames.ParseError("non-integer timing")

    f = Frame(action, targets,
              level=LEVELS.get(str(lvl).lower()) if lvl else None,
              interval_ms=interval, count=count)
    frames.validate(f)
    return f


_ON_W = re.compile(r"\b(on|high|1|enable|activate|energi[sz]e)\b", re.I)
_OFF_W = re.compile(r"\b(off|low|0|disable|deactivate|kill|cut)\b", re.I)


def level_check(f: Frame, text: str) -> str | None:
    """Phrasing-independent and therefore safe on anything, unlike a reparse."""
    if f.action is not Action.SET or f.level is Level.TOGGLE:
        return None
    on, off = bool(_ON_W.search(text)), bool(_OFF_W.search(text))
    if on == off:                              # both or neither: no evidence
        return None
    want = Level.HIGH if on else Level.LOW
    if f.level is not want:
        return f"text reads {want.value} but the annotation says {f.level.value}"
    return None


def digits_present(f: Frame, text: str) -> str | None:
    """Every number in the frame must appear in the utterance.

    The one label error that cannot be tolerated in this batch: a gold pin the
    text never mentions makes the item unscoreable, and a *clamped* one ("pin
    100" annotated as 10) silently inverts the measurement the numeric file
    exists for. Word-numbers and derived rates are exempt.
    """
    runs = set(re.findall(r"\d+", text))
    missing = [str(t.n) for t in f.targets
               if isinstance(t, Pin) and str(t.n) not in runs]
    if missing and not re.search(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
            r"twenty)\b", text, re.I):
        return f"pin {', '.join(missing)} does not appear in the text"
    return None


def cross_check(f: Frame, text: str) -> str | None:
    """Reparse with the rule labeller as an independent second opinion.

    v0 restricted this to pin-numbered utterances, because on natural
    name-targeted phrasing the rules were wrong more often than Gemini and
    quarantined 67 correct items. v1 has no names, so the restriction is gone
    and it applies to everything -- but only where the rules actually fire.
    """
    rule = label_gemini(text)
    if isinstance(rule, str):                  # NEEDS_REVIEW or DEFERRED
        return None
    try:
        frames.validate(rule)
    except frames.ParseError:
        return None
    if rule.key() != f.key():
        return (f"rules read {' '.join(frames.to_symbols(rule))}, "
                f"annotation says {' '.join(frames.to_symbols(f))}")
    return None


# --------------------------------------------------------------------------
# off-domain
# --------------------------------------------------------------------------

# Categories, inferred rather than asked for: the prompt deliberately forbids
# numbering, so the label comes from the text. Only used for the diagnosis
# breakdown, never to accept or reject a row.
STRATA = [
    ("range",     re.compile(r"\bpins?\s*\d+\s*(?:-|–|to|through|thru)\s*\d+", re.I)),
    ("duration",  re.compile(r"\bfor\s+(?:a|an|\d+|half|the next)\s*\w*\s*"
                             r"(?:second|sec|minute|min|hour)s?\b", re.I)),
    ("except",    re.compile(r"\b(except|but not|apart from|other than|all but|"
                             r"every other|the odd|the even|the rest|remaining|"
                             r"the first \w+|the last \w+)\b", re.I)),
    ("analog",    re.compile(r"\b(dim|brighten|brightness|intensity|percent|%|"
                             r"pwm|analog|analogue|voltage|volts?|v\b|duty|fade|"
                             r"adc)\b", re.I)),
    ("colour",    re.compile(r"\b(colou?r|red|green|blue|yellow|purple|rgb|hue|"
                             r"white|orange|pink|cyan|magenta)\b", re.I)),
    ("schedule",  re.compile(r"\b(at \d|am\b|pm\b|o'?clock|tomorrow|tonight|"
                             r"sunset|sunrise|midnight|morning|evening|schedule|"
                             r"in \d+ (?:second|minute|hour)|every (?:day|night|"
                             r"morning|monday))", re.I)),
    ("condition", re.compile(r"\b(if|when|whenever|while|unless|until|sensor|"
                             r"button|pressed|detect)\b", re.I)),
    ("relative",  re.compile(r"\b(that one|it back|again|undo|the other|"
                             r"previous|last command|same for)\b", re.I)),
    ("meta",      re.compile(r"^\s*(how many|what can|which pins|what is a|"
                             r"what's a|list the|do you support|can you control)",
                             re.I)),
]
# Anything left is either a named target or a truncation; a command-shaped line
# with no numbers is the former, a short fragment the latter.
_TRUNC = re.compile(r"^(?:turn|set|blink|flash|read|check|toggle|chase|stop|"
                    r"switch|make|cancel|halt)\b[\w\s]{0,18}$", re.I)


def stratum_of(text: str) -> str:
    for name, rx in STRATA:
        if rx.search(text):
            return name
    if _TRUNC.match(text.strip()):
        return "truncated"
    return "named"


def accept_refusal(text: str) -> str | None:
    """Reject a 'refusal' that is really a command.

    The failure mode this guards: a line like "turn on pin 4" landing in the
    refusals file poisons the false-accept metric directly, and there is nothing
    downstream that would notice.
    """
    got = label_gemini(text)
    if got == DEFERRED or isinstance(got, str):
        return None                            # deferred or unreadable: fine
    try:
        frames.validate(got)
    except frames.ParseError:
        return None
    # The rules read a real command. If every value is legal it is not a
    # refusal at all; if some value is out of range it belongs in the numeric
    # file, which is also not here.
    where = "executable" if frames.executable(got) else "out-of-range"
    return f"rules read a {where} command: {' '.join(frames.to_symbols(got))}"


# --------------------------------------------------------------------------

def load_seen() -> set[str]:
    seen = set()
    for name in SEEN_SETS:
        p = OUT / f"{name}.jsonl"
        if p.exists() and p.name not in SELF:
            seen |= {json.loads(l)["text"].lower().strip() for l in p.open()}
    return seen


def train_texts() -> set[str]:
    p = CORPUS / "train.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["text"].lower().strip() for l in p.open()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seen, review = load_seen(), []
    rows: list[dict] = []
    dropped: Counter = Counter()

    def add(text: str, f: Frame, stratum: str) -> None:
        key = text.lower().strip()
        if not key or key in seen:
            dropped["duplicate"] += 1
            return
        seen.add(key)
        rows.append({"text": text, "symbols": frames.to_symbols(f),
                     "action": f.action.value, "source": "v1batch",
                     "stratum": stratum})

    # --- in-domain and numeric-edge ------------------------------------------
    for kind in ("indomain", "numeric"):
        for i in (1, 2):
            path = RAW / f"v1_{kind}_{i}.txt"
            if not path.exists():
                print(f"missing {path.name}")
                continue
            for o in read_jsonl_ish(path):
                text = str(o.get("text", "")).strip()
                try:
                    f = build_frame(o)
                except (frames.ParseError, ValueError) as e:
                    review.append(f"[{kind}] {text}\n    unusable: {e}")
                    dropped["malformed"] += 1
                    continue
                for check in (level_check, digits_present, cross_check):
                    msg = check(f, text)
                    if msg:
                        review.append(f"[{kind}] {text}\n    {msg}")
                        dropped["disagreement"] += 1
                        break
                else:
                    edge = not frames.executable(f)
                    add(text, f, "numeric" if edge else f.action.value)

    # --- off-domain -----------------------------------------------------------
    unknown = Frame(Action.UNKNOWN)
    for i in (1, 2):
        path = RAW / f"v1_offdomain_{i}.txt"
        if not path.exists():
            print(f"missing {path.name}")
            continue
        for text in read_lines(path):
            msg = accept_refusal(text)
            if msg:
                review.append(f"[offdomain] {text}\n    {msg}")
                dropped["not a refusal"] += 1
                continue
            add(text, unknown, "neg_" + stratum_of(text))

    if not rows:
        raise SystemExit("nothing ingested -- are the raw files in data/eval_raw/?")

    # --- stratified split -----------------------------------------------------
    # Same composition in both halves, so a dev-minus-locked gap after tuning
    # measures overfitting rather than a difference in what the halves contain.
    rng = random.Random(a.seed)
    by_stratum: dict[str, list[dict]] = {}
    for r in rows:
        by_stratum.setdefault(r["stratum"], []).append(r)
    new_dev, new_locked = [], []
    for _, group in sorted(by_stratum.items()):
        rng.shuffle(group)
        half = len(group) // 2
        new_dev += group[:half]
        new_locked += group[half:]

    in_train = train_texts()
    for half, name, old in ((new_dev, "v1_dev", "v1_gemini2_dev"),
                            (new_locked, "v1_locked", "v1_gemini2_locked")):
        locked = name.endswith("locked")
        for r in half:
            r["locked"] = locked
            r["in_train"] = r["text"].lower().strip() in in_train
        carried = [json.loads(l) for l in (OUT / f"{old}.jsonl").open()]
        merged = carried + half
        with (OUT / f"{name}.jsonl").open("w") as fh:
            for r in merged:
                fh.write(json.dumps(r) + "\n")
        pos = sum(1 for r in merged if r["action"] != "unknown")
        print(f"{name:<12} {len(merged):5,} rows  ({pos} in-domain / "
              f"{len(merged) - pos} refusals)   +{len(half)} new")

    (OUT / "v1_batch_review.txt").write_text("\n".join(review))
    print(f"\nnew rows {len(rows):,}   dropped {dict(dropped)}")
    print("by stratum:")
    for k, v in Counter(r["stratum"] for r in rows).most_common():
        print(f"  {k:<16}{v:5,}")
    print(f"\nreview: {len(review)} lines -> {OUT}/v1_batch_review.txt")
    print("Read it. That is reading *gold*, not model failures -- it does not "
          "spend the locked half.")


if __name__ == "__main__":
    main()
