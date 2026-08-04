"""Real human utterances from MASSIVE, used as negatives and as a lexicon.

MASSIVE (Amazon Science, CC-BY-4.0) is 11.5k en-US utterances over 60 intents,
localized from SLURP. Every line was written by a person, which is exactly what
a fully synthetic corpus lacks.

**It is not imported wholesale, and its IoT rows are not relabelled.** Three
things go wrong if you try:

  * The labels are noisy for our purposes. "lights out" is tagged
    `iot_hue_lighton`; "time to sleep" is tagged `iot_hue_lightoff`. Implicit
    commands need world knowledge we do not have and must not be trained on.
  * There is no pin concept. Every target is a room or a device, so a row like
    "turn off the light in the bathroom" has no correct label here unless the
    user happens to have made that alias.
  * Some rows collide head-on with our command language. "stop" is tagged
    `audio_volume_mute` in MASSIVE and is a valid <stop> command here. Bulk
    import would train the model to reject its own grammar.

So rows are used three ways, and anything ambiguous is **dropped rather than
labelled**:

  1. `near_miss_negatives()` -- dimming, colour and brightness requests. Real
     phrasings for capabilities we genuinely lack. The highest-value negatives
     in the project.
  2. `far_negatives()`  -- other domains entirely, minus anything that collides.
  3. `mine_onoff()` -- on/off phrasings dumped for a human to fold into
     realize.py's banks. Lexicon, not labels.

Attribution is required by CC-BY-4.0. The *training* corpus is gitignored and
rebuilt from this script, so the repo ships the recipe rather than the data.
The held-out sets under `data/eval/` are the exception and are committed: they
are the measurement instrument, and a number nobody else can reproduce is not
worth much. See `data/eval/README.md` for the attribution that requires.
"""

from __future__ import annotations

import functools
import re

HF_DATASET = "mteb/amazon_massive_intent"
CITATION = (
    "MASSIVE: FitzGerald et al., Amazon Science, CC-BY-4.0. "
    "https://huggingface.co/datasets/AmazonScience/massive"
)

# Capabilities we deliberately do not have. Real users asking for them is the
# single best signal for <unknown>.
NEAR_MISS_INTENTS = {
    "iot_hue_lightdim",     # "dim the lights", "turn down the brightness"
    "iot_hue_lightup",      # "brighten the lights"
    "iot_hue_lightchange",  # "set lights to twenty percent", "make them green"
    "iot_coffee",
    "iot_cleaning",
}

# On/off rows: mined for phrasings, never relabelled. See the module docstring.
ONOFF_INTENTS = {
    "iot_hue_lighton", "iot_hue_lightoff", "iot_wemo_on", "iot_wemo_off",
}

# Voice-assistant wake words in the SLURP recordings. They are an artifact of
# how the data was collected, not something a user types at a serial port.
WAKE_WORDS = re.compile(r"\b(olly|ollie|hey olly)\b", re.I)

# Any of these makes a row ambiguous for us: it may well be a valid command in
# our grammar, so it must not become a negative.
_COLLIDE_WORDS = re.compile(
    r"\b(pin|pins|gpio|gpios|led|leds|blink|blinking|flash|flashing|strobe|"
    r"pulse|toggle|chase|sequence|relay|buzzer|siren|solenoid|"
    r"high|low|lamp|lamps|light|lights|socket|plug|fan|fans|"
    r"stop|halt|cancel|freeze|abort|quit)\b", re.I)

_LEAD_PREFIX = re.compile(
    r"^(please |can you |could you |would you |i want you to |hey |ok |"
    r"go ahead and |just |now )+", re.I)


@functools.lru_cache(maxsize=1)
def _command_openings() -> tuple[str, ...]:
    """Leading verb phrases of our own templates, derived from realize.py so the
    filter cannot drift when a verb is added there.

    A row that opens with one of these is a structurally valid command whose
    object happens to be a device we do not control. Per docs/GRAMMAR.md it must
    be **dropped**, never labelled <unknown>: the model decides structure, and
    the alias table decides whether the target exists. Teaching it to reject
    "turn on my playlist" also teaches it to reject "turn on my aquarium pump",
    which is a legitimate alias it has never seen.
    """
    import realize

    banks = (realize.ON_T + realize.OFF_T + realize.TOGGLE_T + realize.BLINK_T
             + realize.SEQ_T + realize.STOP_ONE_T + realize.READ_CMD_T
             + realize.READ_Q_T)
    out = set()
    for t in banks:
        # Only templates ending in the target give an unambiguous opening;
        # "turn {r} on" would reduce to a bare "turn" and over-match.
        if t.endswith("{r}"):
            out.add(t[:-3].strip().lower())
    return tuple(sorted(out, key=len, reverse=True))


def _clean(s: str) -> str:
    s = WAKE_WORDS.sub(" ", s)
    return " ".join(s.split()).strip(" ,")


def collides(text: str) -> bool:
    """True if the row might be a legal command here. Such rows are dropped --
    labelling them <unknown> would teach the model to reject its own grammar."""
    if _COLLIDE_WORDS.search(text):
        return True
    body = _LEAD_PREFIX.sub("", text.lower()).strip()
    return any(body.startswith(v + " ") for v in _command_openings())


@functools.lru_cache(maxsize=1)
def _rows() -> list[tuple[str, str]]:
    from datasets import load_dataset  # imported lazily: only prepare needs it

    out: list[tuple[str, str]] = []
    for split in ("train", "validation", "test"):
        d = load_dataset(HF_DATASET, "en", split=split)
        out += [(r["label_text"], r["text"]) for r in d]
    return out


def near_miss_negatives() -> list[str]:
    """Real requests for capabilities we do not have -- dimming, colour, %."""
    seen, out = set(), []
    for intent, text in _rows():
        if intent not in NEAR_MISS_INTENTS:
            continue
        t = _clean(text)
        # The word filter is deliberately skipped -- these are near-misses
        # *because* they mention lights. But the verb-opening filter still
        # applies: iot_hue_lightup contains "bring up the lights", which is a
        # brightness request phrased exactly like our ON command. Labelling that
        # <unknown> contradicts the positives and the model rightly ignored it.
        body = _LEAD_PREFIX.sub("", t.lower()).strip()
        if any(body.startswith(v + " ") for v in _command_openings()):
            continue
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def far_negatives() -> list[str]:
    """Other domains entirely, with anything command-shaped removed."""
    seen, out = set(), []
    for intent, text in _rows():
        if intent.startswith("iot"):
            continue
        t = _clean(text)
        if not t or collides(t) or t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out


def mine_onoff() -> list[str]:
    """On/off phrasings, for folding into realize.py's banks by hand.
    Deliberately not wired into the corpus builder -- these need a human to
    decide which are real phrasings and which are implicit requests.

    **v1: do not wire this up, and the reason changed.** These rows are all
    on/off against a room or device name, which under v1 is exactly the
    deferred-capability refusal -- so they now look like free, human-written
    negatives. They are not free: `data/eval/massive.jsonl` is drawn from this
    same pool, and under v1 scoring it is the refusal benchmark. Training on
    them would leak the only held-out set in the project that no model wrote.
    negatives.name_targeted() covers the class from generated names instead.
    """
    seen, out = set(), []
    for intent, text in _rows():
        if intent not in ONOFF_INTENTS:
            continue
        t = _clean(text)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


if __name__ == "__main__":
    near, far, mined = near_miss_negatives(), far_negatives(), mine_onoff()
    print(f"near-miss negatives : {len(near):5d}")
    print(f"far negatives       : {len(far):5d}")
    print(f"on/off to mine      : {len(mined):5d}   (lexicon only)")
    print(f"\ndropped as colliding: "
          f"{sum(1 for i, t in _rows() if not i.startswith('iot') and collides(_clean(t)))}")
    print("\n--- near-miss sample ---")
    for s in near[:10]:
        print("  ", s)
    print("\n--- far sample ---")
    for s in far[:10]:
        print("  ", s)
