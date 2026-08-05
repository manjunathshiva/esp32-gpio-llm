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
"""

from __future__ import annotations

import random

ADJ = [
    "red", "green", "blue", "white", "amber", "yellow", "purple", "orange",
    "pink", "teal", "big", "small", "little", "main", "spare", "front", "back",
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

# Pronounceable nonsense, so some fraction of names are guaranteed to be
# outside any word list the model could have memorised.
_CONS = "bcdfghjklmnpqrstvwz"
_VOW = "aeiou"


def _invented(rng: random.Random) -> str:
    syll = rng.randint(2, 3)
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) +
                   (rng.choice(_CONS) if rng.random() < 0.5 else "")
                   for _ in range(syll))


def sample_name(rng: random.Random) -> str:
    """One alias name, single-target flavour."""
    r = rng.random()
    if r < 0.26:                                    # "blue led", "spare pump"
        return f"{rng.choice(ADJ)} {rng.choice(DEVICE)}"
    if r < 0.48:                                    # "kitchen light"
        return f"{rng.choice(PLACE)} {rng.choice(DEVICE)}"
    if r < 0.60:                                    # "lamp"
        return rng.choice(DEVICE)
    if r < 0.70:                                    # "front porch light"
        return f"{rng.choice(ADJ)} {rng.choice(PLACE)} {rng.choice(DEVICE)}"
    if r < 0.80:                                    # "warning_led", "pump_2"
        base = rng.choice(DEVICE)
        head = rng.choice(ADJ + PLACE)
        return (f"{head}_{base}" if rng.random() < 0.7
                else f"{base}_{rng.randint(1, 9)}")
    if r < 0.88:                                    # "led2", "relay 3"
        sep = "" if rng.random() < 0.5 else " "
        return f"{rng.choice(DEVICE)}{sep}{rng.randint(1, 12)}"
    if r < 0.96:                                    # invented word
        w = _invented(rng)
        return w if rng.random() < 0.6 else f"{w} {rng.choice(DEVICE)}"
    return f"{_invented(rng)}_{rng.choice(DEVICE)}"  # invented identifier


def sample_group_name(rng: random.Random) -> str:
    """A name that plausibly covers several pins -- plural or collective."""
    r = rng.random()
    if r < 0.35:
        return f"{rng.choice(PLACE)} {rng.choice(DEVICE)}s"
    if r < 0.60:
        return f"{rng.choice(ADJ)} {rng.choice(DEVICE)}s"
    if r < 0.80:
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
