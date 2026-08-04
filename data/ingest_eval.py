"""Ingest the wider Gemini eval batch and split it into a dev half and a
locked half.

The first held-out sets were too small to steer by. 150 in-domain items make
one item worth 0.67%; MASSIVE's 19 refusals make one item worth 5.3%. Five
rounds of data tuning were scored against those numbers, and by the last round
the run-to-run spread was the same size as the improvements being chased. That
is fitting to noise.

Two things fix it, and this script does both.

**Size.** ~400 in-domain and ~500 off-domain, so one item is worth 0.25% and
0.2% respectively. The <2% false-accept gate becomes something a measurement
can actually resolve.

**A set nobody has looked at.** The pooled batch is shuffled and split in half,
stratified so the two halves have the same composition. `dev` is for
diagnosis -- read its failures, tune against it, burn it. `locked` rows carry
`"locked": true`, and `evaluate.py` refuses to print their failures without
`--unlock`. It is scored once, at the end, and the gap between the two halves
is the estimate of how much tuning went into the dev half rather than into the
model.

Gold comes from Gemini's own annotation of its own sentences, re-checked here
three ways: `frames.validate` for structural legality, a verbatim-span check so
names are copies rather than paraphrases, and `label_eval.label_gemini` as an
independent second opinion wherever its rules fire. Disagreements are
quarantined, not resolved automatically. Reading quarantined *gold* is fine --
it is reading model *failures* on the locked half that would burn it.

    uv run python data/ingest_eval.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import frames
from frames import Action, All, Frame, Level, Name, Pin
from label_eval import label_gemini

HERE = Path(__file__).parent
RAW = HERE / "eval_raw"
OUT = HERE / "eval"
CORPUS = HERE / "corpus"

PINSET = set(frames.PINS_S3)
LEVELS = {l.value: l for l in Level}
ACTIONS = {a.value: a for a in Action}


# --------------------------------------------------------------------------
# raw parsing
# --------------------------------------------------------------------------

def read_jsonl_ish(path: Path) -> list[dict]:
    """Pull JSON objects out of whatever AI Studio produced.

    Handles a bare JSONL body, a ```json fence, and the {"response": "..."}
    wrapper the first batch arrived in. Anything else on a line is skipped
    rather than guessed at.
    """
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


# --------------------------------------------------------------------------
# in-domain
# --------------------------------------------------------------------------

# "everything", "all the pins", "every output". The `"*"` sentinel the prompt
# asks for came back as `""` on 18 of the 20 all-pins items in the first batch,
# so an empty target is repaired rather than discarded -- but only when the
# text says so, which makes it a check rather than a guess.
ALL_PHRASE = re.compile(
    r"\b(?:everything|every\s+(?:pin|output|channel|line)|"
    r"all\s+(?:the\s+|of\s+)?(?:pins?|outputs?|channels?|lines?|them))\b", re.I)


def build_frame(o: dict) -> Frame:
    """Gemini's annotation -> a Frame. Raises on anything malformed."""
    action = ACTIONS.get(str(o.get("action", "")).lower())
    if action is None or action is Action.UNKNOWN:
        raise frames.ParseError(f"bad action {o.get('action')!r}")
    text = str(o.get("text", ""))

    targets: list[frames.Target] = []
    for t in o.get("targets") or []:
        if isinstance(t, bool):
            raise frames.ParseError("bool target")
        if isinstance(t, int):
            if t not in PINSET:
                raise frames.ParseError(f"pin {t} not on the allowlist")
            targets.append(Pin(t))
        elif isinstance(t, str) and t.strip() in ("*", "all", "ALL", ""):
            if t.strip() != "*" and not ALL_PHRASE.search(text):
                raise frames.ParseError(f"bad target {t!r}")
            targets.append(All())
        elif isinstance(t, str) and t.strip():
            targets.append(Name(t.strip()))
        else:
            raise frames.ParseError(f"bad target {t!r}")

    # "stop everything" is trained as a bare <stop>: sample_frame draws STOP
    # targets with allow_all=False, so <stop> <all> is a form the model has
    # never seen and gold must not ask for it.
    if action is Action.STOP and any(isinstance(t, All) for t in targets):
        targets = []

    lvl = o.get("level")
    level = LEVELS[lvl.lower()] if isinstance(lvl, str) and lvl.strip() else None
    if lvl and level is None:
        raise frames.ParseError(f"bad level {lvl!r}")

    def as_int(k: str) -> int | None:
        v = o.get(k)
        if v is None or v == "":
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise frames.ParseError(f"bad {k} {v!r}")
        return int(v)

    alias = o.get("alias_name")
    f = Frame(action, targets, level, as_int("interval_ms"), as_int("count"),
              alias.strip() if isinstance(alias, str) and alias.strip() else None)
    strip_articles(f)
    frames.validate(f)
    return f


def _spans(f: Frame) -> list[str]:
    out = [t.text for t in f.targets if isinstance(t, Name)]
    if f.alias_name:
        out.append(f.alias_name)
    return out


_ARTICLE = re.compile(r"^(?:the|my|a|an|our|that|this)\s+", re.I)


def strip_articles(f: Frame) -> None:
    """Drop a leading article from every name, in place.

    Mechanical and unambiguous, and `label_eval._target` already does it for
    the MASSIVE gold -- leaving it to the annotator produced "the greenhouse
    mister" alongside "greenhouse mister" for identical phrasings, and no
    single model output could satisfy both.
    """
    f.targets = [Name(_ARTICLE.sub("", t.text).strip()) if isinstance(t, Name)
                 else t for t in f.targets]
    if f.alias_name:
        f.alias_name = _ARTICLE.sub("", f.alias_name).strip()


def span_problem(f: Frame, text: str) -> str | None:
    """The model's job on a name is to *copy* a span. If the annotation
    paraphrases or hallucinates it, the gold is asking for something the
    architecture cannot do and the item is unscoreable."""
    low = " ".join(text.lower().split())
    for s in _spans(f):
        if not s.strip():
            return "empty name"
        if s.lower() not in low:
            return f"name {s!r} is not a span of the text"
    return None


_OFF_W = re.compile(r"\b(?:off|low|shut|kill|disable|deactivate|extinguish|"
                    r"douse|cut)\b", re.I)
_ON_W = re.compile(r"\b(?:on|high|enable|activate|energi[sz]e|light\s+up)\b", re.I)


def level_check(f: Frame, text: str) -> str | None:
    """Does the level agree with the words in the sentence?

    Phrasing-independent and therefore safe on anything, unlike a full reparse.
    Skipped when the text carries both polarities ("is it on or off") or when
    the level is toggle, neither of which this can rule on.
    """
    if f.action is not Action.SET or f.level is Level.TOGGLE:
        return None
    on, off = bool(_ON_W.search(text)), bool(_OFF_W.search(text))
    if on == off:                          # both or neither: no evidence
        return None
    want = Level.HIGH if on else Level.LOW
    if f.level is not want:
        return f"text reads {want.value} but the annotation says {f.level.value}"
    return None


def cross_check(f: Frame, text: str) -> str | None:
    """Reparse with the rule labeller -- but only for pin-numbered utterances.

    The rules were written against the first batch, which was almost entirely
    terse machine-style forms. On this batch's natural phrasing they are simply
    worse than the annotation: run as a general second opinion they quarantined
    67 items, and all 67 were the rules misreading the sentence ("drive pin 42
    high" -> a name literally called "drive pin 42", "read warning_led state" ->
    a name called "warning_led state"). A second opinion that is wrong more
    often than the first is not a check, it is noise, and acting on it would
    have deleted 67 correct items.

    Where the rules *are* reliable is exactly what they were built for: an
    utterance whose targets are literal pin numbers. There a disagreement means
    a wrong pin or a wrong level, which is a real annotation bug. So the
    reparse runs only when both sides resolve to pins, and `level_check` above
    carries the phrasing-independent part everywhere else.
    """
    rule = label_gemini(text)
    if isinstance(rule, str):
        return None                       # rules do not cover this phrasing
    strip_articles(rule)                  # compare on the same convention
    try:
        frames.validate(rule)
    except frames.ParseError:
        return None
    all_pins = (bool(f.targets) and all(isinstance(t, Pin) for t in f.targets)
                and bool(rule.targets)
                and all(isinstance(t, Pin) for t in rule.targets))
    if not all_pins:
        return None
    if rule.key() != f.key():
        return (f"rules read it as {' '.join(frames.to_symbols(rule))}, "
                f"annotation says {' '.join(frames.to_symbols(f))}")
    return None


# --------------------------------------------------------------------------
# off-domain
# --------------------------------------------------------------------------

# Markers of a capability the grammar genuinely lacks. A sentence carrying one
# is a refusal whatever else it looks like -- "turn on the lamp in ten minutes"
# opens exactly like a legal command and is one only up to the last three words.
# Checked first so the collision detector below never fires on it.
#
# Deliberately narrow. Every word here switches the collision detector off for
# that item, so a word that can legitimately appear inside a command opens a
# hole. Four were cut for exactly that reason:
#   bare colour words -- "turn on the red led" is a normal alias
#   "bright"          -- likewise "the bright lamp"
#   "seconds"         -- realize.py renders intervals as "every two seconds"
#   "sensor/motion"   -- "the motion sensor light" is a name, not a capability
# Their categories are still caught: a colour request needs colour/hue/rgb, a
# conditional needs when/if/while, a sensor read needs measure/temperature.
UNSUPPORTED = re.compile(
    r"\b(?:"
    r"dim(?:mer|ming)?|brighten|brightness|fade|fading|dimly|percent|"
    r"pwm|duty\s*cycle|analog|analogue|volts?|voltage|half\s+power|intensity|"
    r"colou?r|rgb|hue|saturation|warm\s+white|cool\s+white|kelvin|"
    r"when|whenever|after|before|until|unless|while|schedule[ds]?|"
    r"tomorrow|tonight|morning|evening|midnight|noon|later|countdown|delay|"
    r"minutes?|hours?|o'?\s*clock|a\.?m\.?|p\.?m\.?|every\s+day|daily|"
    r"temperature|humidity|current\s+draw|power\s+usage|wattage|amperage|"
    r"measure|again|undo|previous|last\s+one|earlier|just\s+(?:did|turned|said)|"
    r"mins?|sunrise|sunset|dawn|dusk|timer"
    r")\b"
    # "check if pin 18 is on" is a legal read, not a conditional.
    r"|(?<!check )(?<!see )\bif\b"
    # Digit-anchored units. \b does not fire between "6" and "pm", so the word
    # list above misses "6pm", "2.5v", "75pc" and "10:30" entirely.
    r"|\d\s*(?:%|pc\b|percent|v\b|volts?\b|[ap]\.?m\.?\b)"
    r"|\d{1,2}:\d{2}"
    # "in 45 seconds" is a schedule; "every 2 seconds" is a legal blink rate.
    # The distinguishing word is the preposition, not the unit, so "seconds"
    # cannot go in the list above.
    r"|\b(?:in|after|within)\s+\w+\s+(?:seconds?|secs?|minutes?|mins?|hours?)\b",
    re.I)


def accept_problem(text: str, category: int) -> str | None:
    """Reject anything that is actually a legal command.

    This is the failure that cost the most in the first corpus: 128 MASSIVE
    rows of the form "turn on my <thing>" were carried in as <unknown>, which
    trains the model to reject unfamiliar nouns. docs/GRAMMAR.md: structure
    decides the parse, the alias table decides the execution.

    Order matters. The capability check runs first, because categories 1-6 are
    *supposed* to open with an ordinary command verb and then ask for something
    the device cannot do -- running the collision detector on them first
    quarantined "cut power to the mister an hour from now", a perfectly good
    scheduling refusal.
    """
    if UNSUPPORTED.search(text):
        return None

    rule = label_gemini(text)
    if not isinstance(rule, str):
        try:
            frames.validate(rule)
        except frames.ParseError:
            rule = "unparseable"
    if not isinstance(rule, str) and rule.action is not Action.UNKNOWN:
        return f"rules parse it as {' '.join(frames.to_symbols(rule))}"

    # `massive.collides` is deliberately not used here. It tests the opening
    # verb and ignores the rest of the sentence, which is the right trade for
    # mining MASSIVE -- there, dropping a borderline row costs nothing. Here it
    # quarantined 40-odd perfectly good refusals ("switch pin 8 to blue", "put
    # pin 21 at 2.5v", "tie pin 8 to pin 9"), all of which open like a command
    # and then ask for something the grammar cannot express, and it still
    # missed "what is the weather like today" in the other direction because
    # READ_Q_T contains "what is {r}".
    #
    # label_gemini above accounts for the whole sentence instead, which is what
    # the question actually is: can the grammar express this, or not?
    return None


# --------------------------------------------------------------------------

SELF = {"gemini2_dev.jsonl", "gemini2_locked.jsonl"}


def load_seen() -> set[str]:
    """Texts already used in an eval set. Overlap between held-out sets would
    double-count the same item across two reported numbers.

    This script's own outputs are excluded, or a second run -- which the
    protocol calls for after fixing anything the review file flagged -- would
    dedupe the whole batch against the previous pass and emit nothing.
    """
    seen = set()
    for p in OUT.glob("*.jsonl"):
        if p.name in SELF:
            continue
        for line in p.open():
            seen.add(" ".join(json.loads(line)["text"].lower().split()))
    return seen


def load_train() -> set[str]:
    p = CORPUS / "train.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["text"].lower().strip() for l in p.open()}


def stratified_halves(rows: list[dict], key, seed: int) -> tuple[list, list]:
    """Split in half so both sides have the same composition. A dev/locked gap
    then measures tuning, not a difference in what the two halves contain."""
    rng = random.Random(seed)
    buckets: dict = defaultdict(list)
    for r in rows:
        buckets[key(r)].append(r)
    dev, locked = [], []
    for k in sorted(buckets, key=str):
        b = buckets[k]
        rng.shuffle(b)
        for i, r in enumerate(b):
            (dev if i % 2 == 0 else locked).append(r)
    rng.shuffle(dev)
    rng.shuffle(locked)
    return dev, locked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", type=Path, default=RAW)
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if not a.raw.exists():
        raise SystemExit(
            f"{a.raw} does not exist. See data/eval_prompts/README.md -- run the "
            "two prompts twice each and save the replies there.")

    seen = load_seen()
    in_train = load_train()
    kept: list[dict] = []
    quarantine: list[str] = []
    stats: Counter = Counter()

    def dedupe(text: str) -> bool:
        k = " ".join(text.lower().split())
        if not k or k in seen:
            stats["duplicate"] += 1
            return False
        seen.add(k)
        return True

    # --- in-domain -----------------------------------------------------------
    for path in sorted(a.raw.glob("indomain*")):
        for o in read_jsonl_ish(path):
            text = str(o.get("text", "")).strip()
            stats["indomain_raw"] += 1
            if not text or not dedupe(text):
                continue
            try:
                f = build_frame(o)
            except (frames.ParseError, ValueError, KeyError, TypeError) as e:
                quarantine.append(f"[invalid {path.name}] {text}\n    {e}")
                stats["indomain_invalid"] += 1
                continue
            for problem in (span_problem(f, text), level_check(f, text),
                            cross_check(f, text)):
                if problem:
                    quarantine.append(f"[review {path.name}] {text}\n    {problem}")
                    stats["indomain_review"] += 1
                    break
            else:
                kept.append({"text": text, "symbols": frames.to_symbols(f),
                             "action": f.action.value, "source": "gemini2",
                             "stratum": f.action.value,
                             "in_train": " ".join(text.lower().split()) in in_train})

    # --- off-domain ----------------------------------------------------------
    unknown = Frame(Action.UNKNOWN)
    for path in sorted(a.raw.glob("offdomain*")):
        for o in read_jsonl_ish(path):
            text = str(o.get("text", "")).strip()
            stats["offdomain_raw"] += 1
            if not text or not dedupe(text):
                continue
            try:
                cat = int(o.get("category", 0))
            except (TypeError, ValueError):
                cat = 0
            problem = accept_problem(text, cat)
            if problem:
                quarantine.append(f"[not-a-refusal {path.name}] {text}\n    {problem}")
                stats["offdomain_rejected"] += 1
                continue
            kept.append({"text": text, "symbols": frames.to_symbols(unknown),
                         "action": "unknown", "source": "gemini2",
                         "stratum": f"neg{cat}",
                         "in_train": " ".join(text.lower().split()) in in_train})

    if not kept:
        raise SystemExit(f"nothing usable in {a.raw}")

    dev, locked = stratified_halves(kept, lambda r: r["stratum"], a.seed)
    for r in dev:
        r["locked"] = False
    for r in locked:
        r["locked"] = True

    for name, chunk in (("gemini2_dev", dev), ("gemini2_locked", locked)):
        with (OUT / f"{name}.jsonl").open("w") as fh:
            for r in chunk:
                fh.write(json.dumps(r) + "\n")
    (OUT / "gemini2_review.txt").write_text("\n".join(quarantine))

    def summarise(label: str, chunk: list[dict]) -> None:
        pos = [r for r in chunk if r["action"] != "unknown"]
        neg = [r for r in chunk if r["action"] == "unknown"]
        leak = sum(r["in_train"] for r in chunk)
        print(f"\n{label}: {len(chunk)} rows  "
              f"({len(pos)} in-domain / {len(neg)} refusals)  "
              f"verbatim in train: {leak}")
        for act, n in Counter(r["action"] for r in pos).most_common():
            print(f"    {act:<8} {n}")

    print(f"parsed {stats['indomain_raw']} in-domain and "
          f"{stats['offdomain_raw']} off-domain raw lines")
    print(f"  duplicates dropped     {stats['duplicate']}")
    print(f"  malformed annotations  {stats['indomain_invalid']}")
    print(f"  gold disagreements     {stats['indomain_review']}")
    print(f"  refusals that were not {stats['offdomain_rejected']}")
    summarise("dev   ", dev)
    summarise("locked", locked)
    print(f"\nwrote {OUT}/gemini2_{{dev,locked}}.jsonl")
    print(f"      {OUT}/gemini2_review.txt  -- {len(quarantine)} items to eyeball")
    print("\nlocked rows are scored but not diagnosed: evaluate.py will not "
          "print their failures without --unlock.")


if __name__ == "__main__":
    main()
