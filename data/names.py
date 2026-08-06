"""Open-ended alias names.

The first corpus drew every Name target from a fixed list of 36 nouns, so the
model learned to *recognise* those 36 rather than to copy an arbitrary span. On
held-out data it answered <unknown> to "turn on the plug" and "turn off
warning_led" -- names it had never seen -- which is precisely the failure
data/frames.py warns about: a model that rejects unfamiliar nouns will also
reject a legitimate alias the user just created.

So names are generated compositionally from large pools, plus invented words
that appear nowhere else. The space is tens of thousands of forms rather than
36, and no individual name is frequent enough to memorise. The model has no
option but to learn the *slot*.

**That was not enough, and the way it failed is worth keeping.** With the pools
below alone, a held-out name whose words all came from them was copied 89% of
the time and one containing any other word 56% -- and the gap was not span
length, which does nothing once vocabulary is held fixed (90.0% for 5+ token
names against 88.4% for short ones). Two separate causes:

  - **Negative-only words become refusal triggers.** "coffee" occurred 2,080
    times in the corpus, every one of them in an utterance labelled <unknown>,
    so "check coffee maker state" answered <unknown> on all three seeds. Every
    word that worked -- lamp, fan, relay, laser -- appeared on both sides,
    75-90% positive.

  - **Absent words truncate the span.** "cutter" occurred nowhere at all, and
    "enable laser cutter" came back as "laser": the model stopped at the last
    noun it recognised.

Both are fixed by WORDS (see data/wordlist.txt), which supplies ordinary
English -- including every content word the negatives use -- so that no word is
evidence for refusing and no word is evidence for stopping.

The invented words stay, but they were never the hard case: pronounceable
nonsense was copied 90% of the time while real English the model had not seen
in this slot managed 43%. Nothing in "zibmuk" competes for another reading,
which is exactly what makes it easy and what made it a misleading measure of
whether the slot had been learned.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

# Split out, not removed. A colour is a perfectly ordinary thing to call a
# device after ("the red lamp"), so these stay in ADJ and keep building names.
# They are named separately because negatives.unsupported_capability needs the
# same list to build the one utterance that teaches the distinction -- a colour
# inside a name *and* a colour as the thing being set, in one sentence. Keeping
# two hand-maintained copies is how the two sides drift apart.
ADJ_COLOUR = [
    "red", "green", "blue", "white", "amber", "yellow", "purple", "orange",
    "pink", "teal",
]

ADJ = ADJ_COLOUR + [
    "big", "small", "little", "main", "spare", "front", "back",
    "left", "right", "top", "bottom", "upper", "lower", "inner", "outer",
    "north", "south", "east", "west", "primary", "secondary", "backup", "old",
    "new", "first", "second", "third", "hot", "cold", "warm", "quiet", "loud",
    "smart", "tall", "short", "wide", "narrow", "corner", "centre", "side",
    "emergency", "safety", "night", "day", "work", "test", "spare", "extra",
]

PLACE = [
    "kitchen", "bedroom", "bathroom", "hallway", "porch", "garage", "garden",
    "shed", "attic", "basement", "office", "studio", "workshop", "lab",
    "closet", "pantry", "balcony", "patio", "driveway", "stairs", "corridor",
    "lobby", "entry", "deck", "yard", "barn", "greenhouse", "cellar", "loft",
    "hall", "landing", "utility", "garage", "porch", "terrace", "conservatory",
    "bench", "desk", "table", "shelf", "cabinet", "rack", "wall", "ceiling",
    # Multi-word places. Without these "living room lights" is structurally out
    # of distribution, and the copy degraded into invented subwords ("liviva",
    # "livange light") rather than reproducing the span.
    "living room", "dining room", "bed room", "game room", "utility room",
    "front porch", "back porch", "spare room", "laundry room", "store room",
    "front door", "back door", "side gate", "top floor", "ground floor",
    "meeting room", "server room", "boot room", "play room", "wash room",
]

DEVICE = [
    "lamp", "light", "led", "bulb", "fan", "motor", "pump", "valve", "relay",
    "buzzer", "siren", "bell", "alarm", "heater", "cooler", "vent",
    "sprinkler", "solenoid", "speaker", "strip", "sign", "beacon", "indicator",
    "socket", "plug", "outlet", "switch", "coil", "servo", "compressor",
    "blower", "chime", "horn", "strobe", "laser", "display", "panel", "gate",
    "latch", "lock", "curtain", "blind", "shutter", "feeder", "mister",
    "humidifier", "filter", "aerator", "charger", "dispenser", "conveyor",
    "spindle", "actuator", "damper", "injector", "kettle", "toaster", "radio",
    "projector", "printer", "scanner", "camera", "sensor", "monitor", "torch",
    "lantern", "floodlight", "spotlight", "downlight", "uplight", "nightlight",
    # Compound devices, likewise -- "smart plug socket" came back as
    # "smart plug", a span-boundary error, because every device noun was one
    # word.
    "plug socket", "smart plug", "wall socket", "power socket", "light strip",
    "led strip", "strip light", "ceiling fan", "extractor fan", "desk lamp",
    "floor lamp", "table lamp", "wall light", "porch light", "status led",
    "power led", "warning light", "call bell", "door chime", "smoke alarm",
    "water pump", "air pump", "heat mat", "grow light", "fairy lights",
]

@lru_cache(maxsize=1)
def _words() -> tuple[list[str], list[str], list[int]]:
    """Load data/wordlist.txt -> (all words, one-sided words, their weights).

    A line is "word" or "word<TAB>n", where n is how many times the negatives
    use a word the positives never do. Those get sampled in proportion to n,
    because the imbalance is the thing being corrected and it is wildly uneven:
    "coffee" carries 2,302 negative occurrences and most of the list carries
    three. Sampling the list uniformly would give "coffee" two appearances in
    40,000 names and leave the decision rule exactly where it was.
    """
    p = Path(__file__).parent / "wordlist.txt"
    every: list[str] = []
    heavy: list[str] = []
    weight: list[int] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        word, _, n = line.partition("\t")
        every.append(word)
        if n:
            heavy.append(word)
            weight.append(int(n))
    if len(every) < 1000:
        raise SystemExit(f"{p} has only {len(every)} words; the whole point is "
                         f"a vocabulary too large to memorise")
    return every, heavy, weight


WORDS, _HEAVY, _HEAVY_W = _words()

# How often an English name word is drawn from the one-sided pool rather than
# the flat one. High, because the two pools fix different failures and only
# this one is self-limiting: once a word appears on both sides it stops being
# evidence for anything, and no amount of further exposure helps.
ONE_SIDED_PROB = 0.55


def _word(rng: random.Random) -> str:
    if _HEAVY and rng.random() < ONE_SIDED_PROB:
        return rng.choices(_HEAVY, weights=_HEAVY_W)[0]
    return rng.choice(WORDS)


# Pronounceable nonsense, so some fraction of names are guaranteed to be
# outside any word list the model could have memorised.
_CONS = "bcdfghjklmnpqrstvwz"
_VOW = "aeiou"


def _invented(rng: random.Random) -> str:
    syll = rng.randint(2, 3)
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) +
                   (rng.choice(_CONS) if rng.random() < 0.5 else "")
                   for _ in range(syll))


def _english(rng: random.Random) -> str:
    """A name built from ordinary English instead of device vocabulary.

    The head noun is deliberately often *not* a device word -- "coffee maker",
    "fish tank", "laser cutter" -- because that is the shape the model could
    not copy: it stopped at the last noun it recognised. Naming things after
    words that are not in a device list is what people do, and until this
    existed the corpus said otherwise.
    """
    w = lambda: _word(rng)
    r = rng.random()
    if r < 0.34:                                    # "coffee maker"
        return f"{w()} {w()}"
    if r < 0.55:                                    # "exhaust fan"
        return f"{w()} {rng.choice(DEVICE)}"
    if r < 0.70:                                    # "spare cutter"
        return f"{rng.choice(ADJ)} {w()}"
    if r < 0.82:                                    # "faucet"
        return w()
    if r < 0.92:                                    # "dish_washer"
        return f"{w()}_{w()}"
    return f"{rng.choice(PLACE)} {w()}"             # "kitchen kettle"


def sample_name(rng: random.Random) -> str:
    """One alias name, single-target flavour.

    Shares were rebalanced after measuring which names the model could copy.
    Ordinary English gets 32% because it was the entire failure -- names built
    only from the curated pools already scored 89%. The invented share dropped
    from 12% to 6% for the opposite reason: nonsense was copied 90% of the time
    and was never teaching what it looked like it was teaching.
    """
    r = rng.random()
    if r < 0.32:                                    # ordinary English
        return _english(rng)
    if r < 0.47:                                    # "blue led", "spare pump"
        return f"{rng.choice(ADJ)} {rng.choice(DEVICE)}"
    if r < 0.60:                                    # "kitchen light"
        return f"{rng.choice(PLACE)} {rng.choice(DEVICE)}"
    if r < 0.68:                                    # "lamp"
        return rng.choice(DEVICE)
    if r < 0.75:                                    # "front porch light"
        return f"{rng.choice(ADJ)} {rng.choice(PLACE)} {rng.choice(DEVICE)}"
    if r < 0.82:                                    # "warning_led", "pump_2"
        base = rng.choice(DEVICE)
        head = rng.choice(ADJ + PLACE)
        return (f"{head}_{base}" if rng.random() < 0.7
                else f"{base}_{rng.randint(1, 9)}")
    if r < 0.89:                                    # "led2", "relay 3"
        sep = "" if rng.random() < 0.5 else " "
        return f"{rng.choice(DEVICE)}{sep}{rng.randint(1, 12)}"
    if r < 0.97:                                    # invented word
        w = _invented(rng)
        return w if rng.random() < 0.6 else f"{w} {rng.choice(DEVICE)}"
    return f"{_invented(rng)}_{rng.choice(DEVICE)}"  # invented identifier


def sample_group_name(rng: random.Random) -> str:
    """A name that plausibly covers several pins -- plural or collective."""
    r = rng.random()
    if r < 0.22:                            # "coffee makers", "garden lamps"
        w = _word(rng)
        return (f"{w} {rng.choice(DEVICE)}s" if rng.random() < 0.6
                else f"{rng.choice(ADJ)} {w}s")
    if r < 0.48:
        return f"{rng.choice(PLACE)} {rng.choice(DEVICE)}s"
    if r < 0.68:
        return f"{rng.choice(ADJ)} {rng.choice(DEVICE)}s"
    if r < 0.84:
        return f"{rng.choice(DEVICE)}s"
    return rng.choice([
        f"{rng.choice(PLACE)} row", f"{rng.choice(ADJ)} bank",
        f"{rng.choice(DEVICE)} bank", f"{rng.choice(DEVICE)} array",
        f"{rng.choice(PLACE)} group", f"{rng.choice(DEVICE)} cluster",
    ])


if __name__ == "__main__":
    rng = random.Random(0)
    print("single:", ", ".join(sample_name(rng) for _ in range(18)))
    print("group :", ", ".join(sample_group_name(rng) for _ in range(10)))
