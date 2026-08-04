"""Semantic frames: the canonical meaning of a command.

This module is the single definition of what a command *is*. The sampler
produces frames, the realizer renders them to English, the evaluator compares
them, and `firmware/common/command.h` reconstructs them from emitted symbols.

The symbol encoder here and the decoder in command.h must agree exactly --
there are no names in the token stream, only ordered slots. Change one, change
the other. (Same contract as export.py's tensor `plan` and llm_load().)

**v1 targets pins by number only.** No names, no aliases -- see docs/V1-SCOPE.md
for why, and section "Pins are copied digits" in docs/GRAMMAR.md for the part
that matters most: pin numbers are emitted digit-wise, exactly like intervals,
and the *runtime* decides whether a number is a GPIO on this board. The model
transcribes; it does not validate.

That split is load-bearing. v0 gave each allowlisted pin its own symbol, which
made an illegal pin unrepresentable -- and an unrepresentable pin does not
produce a refusal, it produces the nearest legal one. "switch off pin 100" came
back as <set> <p10> <low>: the wrong physical pin, actuated silently. Range is
a hardware fact, so it belongs with the hardware.

See docs/GRAMMAR.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

# Pin allowlists, copied from femtoclaw's tool_gpio.c. Unlike v0 these no longer
# constrain what the model can emit -- they are what range_check() tests against
# and what gpio_control.c enforces on device. The two must stay in step.
PINS_S3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           21, 38, 39, 40, 41, 42, 48]
PINS_WROOM = [2, 4, 5, 12, 13, 14, 15, 18, 19, 22, 23, 25, 26, 27, 32, 33]

INTERVAL_MIN, INTERVAL_MAX = 50, 10000      # gpio_control.c bounds
MAX_SEQ_PINS = 6                            # see docs/GRAMMAR.md section 4

# Longest digit run the parser will accept in any numeric slot. Not a range
# check -- a range check has a value to report. This rejects a generation that
# has run away, which is a structural failure.
MAX_DIGITS = 6


class Action(str, Enum):
    SET = "set"
    READ = "read"
    BLINK = "blink"
    SEQ = "seq"
    STOP = "stop"
    UNKNOWN = "unknown"


class Level(str, Enum):
    HIGH = "high"
    LOW = "low"
    TOGGLE = "toggle"


@dataclass(frozen=True)
class Pin:
    n: int


@dataclass(frozen=True)
class All:
    pass


Target = Pin | All


@dataclass
class Frame:
    action: Action
    targets: list[Target] = field(default_factory=list)
    level: Level | None = None
    interval_ms: int | None = None
    count: int | None = None        # 0 == infinite

    def key(self) -> tuple:
        """Hashable identity, for exact-match scoring."""
        return (
            self.action,
            tuple(("pin", t.n) if isinstance(t, Pin) else ("all",)
                  for t in self.targets),
            self.level,
            self.interval_ms,
            self.count,
        )


# --------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------

S_END = "<end>"
S_ALL = "<all>"
S_PIN = "<pin>"
S_INT = "<int>"
S_CNT = "<cnt>"

ACTION_SYM = {a: f"<{a.value}>" for a in Action}
LEVEL_SYM = {l: f"<{l.value}>" for l in Level}

# Every marker that introduces a run of digits. A run ends at the next symbol
# that is not a digit, which is why v1 has no <nend>: with names gone, the only
# thing <nend> terminated was a number, and a number already terminates itself.
NUM_SYM = (S_PIN, S_INT, S_CNT)


def special_symbols() -> list[str]:
    """Every reserved id, in a stable order. gen_assets.py emits symbols.h from
    this, so the order is part of the model format -- do not sort it later.

    No longer parameterised by the pin allowlist: v1 has one <pin> symbol and
    the board is not baked into the vocabulary. Changing boards is now a change
    to gpio_control.c alone, not a retrain."""
    return (
        [ACTION_SYM[a] for a in Action]
        + [S_PIN, S_ALL]
        + [LEVEL_SYM[l] for l in Level]
        + [S_INT, S_CNT, S_END]
    )


def to_symbols(f: Frame) -> list[str]:
    """Frame -> emitted symbol sequence. Order is fixed:
    ACTION TARGET* [LEVEL] [<int> digits] [<cnt> digits] <end>"""
    out = [ACTION_SYM[f.action]]

    for t in f.targets:
        if isinstance(t, Pin):
            out += [S_PIN, *str(t.n)]
        else:
            out.append(S_ALL)

    if f.level is not None:
        out.append(LEVEL_SYM[f.level])

    if f.interval_ms is not None:
        out += [S_INT, *str(f.interval_ms)]

    if f.count is not None:
        out += [S_CNT, *str(f.count)]

    out.append(S_END)
    return out


class ParseError(ValueError):
    pass


def from_symbols(syms: list[str]) -> Frame:
    """Emitted symbols -> Frame. Strict: raises ParseError on anything the
    grammar cannot produce, which is how a malformed generation is rejected
    rather than half-interpreted.

    Note what this deliberately does *not* reject: a pin that is not on the
    board, an interval outside the device bounds, a chase longer than the
    hardware runs. Those parse cleanly and are refused by range_check(). A
    number the model heard is a number it should be allowed to say."""
    i = 0

    def peek() -> str | None:
        return syms[i] if i < len(syms) else None

    if peek() is None:
        raise ParseError("empty")
    sym_action = {v: k for k, v in ACTION_SYM.items()}
    if syms[0] not in sym_action:
        raise ParseError(f"expected an action, got {syms[0]!r}")
    action = sym_action[syms[0]]
    i = 1

    f = Frame(action=action)
    if action is Action.UNKNOWN:
        if peek() != S_END:
            raise ParseError("<unknown> takes no slots")
        return f

    sym_level = {v: k for k, v in LEVEL_SYM.items()}

    def read_digits(what: str) -> int:
        nonlocal i
        digits = ""
        while (s := peek()) is not None and s.isdigit():
            digits += s
            i += 1
        if not digits:
            raise ParseError(f"empty {what}")
        if len(digits) > MAX_DIGITS:
            raise ParseError(f"{what} has {len(digits)} digits")
        return int(digits)

    # targets
    while (s := peek()) is not None:
        if s == S_ALL:
            f.targets.append(All())
            i += 1
        elif s == S_PIN:
            i += 1
            f.targets.append(Pin(read_digits("pin")))
        else:
            break

    if (s := peek()) in sym_level:
        f.level = sym_level[s]
        i += 1

    if peek() == S_INT:
        i += 1
        f.interval_ms = read_digits("interval")

    if peek() == S_CNT:
        i += 1
        f.count = read_digits("count")

    if peek() != S_END:
        raise ParseError(f"expected {S_END}, got {peek()!r}")
    if i + 1 != len(syms):
        raise ParseError("trailing symbols")

    validate(f)
    return f


def validate(f: Frame) -> None:
    """Enforce the per-action shape in docs/GRAMMAR.md -- which slots an action
    takes, and how many. The sampler and the parser both go through this, so
    they cannot drift apart.

    Structure only. Whether a *value* is legal on this board is range_check()'s
    job, and keeping them apart is what lets the model emit "pin 100" and the
    runtime refuse it, instead of the model silently emitting "pin 10"."""
    a, n = f.action, len(f.targets)
    has_all = any(isinstance(t, All) for t in f.targets)

    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ParseError(f"{a.value}: {msg}")

    # Timing is optional for blink and seq. "chase 2 3 4 5" specifies none, and
    # gpio_control.c already has defaults (500ms blink / 200ms sequence, count 0)
    # -- so an omitted slot means "device default", not a malformed command.
    #
    # A count with no rate is legal: "blink pin 4 five times" means five cycles
    # at the default rate, which is unambiguous. It used to be rejected, and the
    # model answered it by inventing a 100ms rate and running forever -- a number
    # nobody said, and the wrong end condition. A *rate* with no count stays
    # illegal, because that would discard a number the speaker did say.
    def timing_ok() -> None:
        need(not (f.interval_ms is not None and f.count is None),
             "a rate needs a count")

    # Every action that takes a pin takes a *list* of pins. The asymmetry was
    # the bug class: `set` accepted one target, `read` and `blink` one, `stop`
    # one, and each of those limits produced a different silent failure when a
    # speaker named two. Uniformity is the fix, not four separate patches.
    if a is Action.SET:
        # Several pins at one level is ordinary speech -- "turn on pins 4, 5 and
        # 6" -- and v0 could not say it. One target was not a considered choice,
        # it was an oversight, and the failure it caused is the one this grammar
        # exists to prevent: with no way to represent three pins the model
        # emitted the first and dropped the rest, so two pins the speaker asked
        # for stayed put and nothing reported it. Found on hardware with LEDs
        # attached, which is the only place it is visible.
        #
        # No upper bound: unlike a chase, which is limited by animation slots,
        # setting a level is a loop the board can run over as many pins as were
        # named. cmd_parse's CMD_MAX_PINS is the practical ceiling.
        need(n >= 1, "needs at least one target")
        need(not (has_all and n > 1), "<all> is the whole board, not a list item")
        need(f.level is not None, "needs a level")
        need(f.interval_ms is None and f.count is None, "takes no timing")
    elif a is Action.READ:
        need(n >= 1 and not has_all, "needs at least one concrete target")
        need(f.level is None, "takes no level")
    elif a is Action.BLINK:
        need(n >= 1, "needs at least one target")
        need(not (has_all and n > 1), "<all> is the whole board, not a list item")
        timing_ok()
    elif a is Action.SEQ:
        # Only the lower bound is structural: a chase of one pin is not a chase.
        # The upper bound is how many the hardware will run, so it moved to
        # range_check().
        need(n >= 2, "needs at least 2 targets")
        need(not has_all, "cannot chase <all>")
        timing_ok()
    elif a is Action.STOP:
        # Zero targets means everything. Any other number is a list of pins to
        # stop -- "stop pins 4 and 5" used to be unrepresentable and came back
        # as <seq> pin4 pin5, which *started* an animation instead of ending
        # one. The most dangerous shape found so far: not a wrong pin, a wrong
        # verb, from the one command a user reaches for when something is wrong.
        need(not has_all, "<stop> with no targets already means everything")
        need(f.level is None and f.interval_ms is None, "takes no slots")

    if f.count is not None:
        need(f.count >= 0, "negative count")


# --------------------------------------------------------------------------
# Range checking -- the hardware's half of the contract
# --------------------------------------------------------------------------

class Verdict(str, Enum):
    """What the runtime does with a parsed frame. Mirrored in gpio_control.c."""
    EXECUTE = "execute"
    BAD_PIN = "bad_pin"
    BAD_INTERVAL = "bad_interval"
    TOO_MANY_PINS = "too_many_pins"


def range_check(f: Frame, pins: list[int] = PINS_S3) -> tuple[Verdict, str]:
    """Is this frame executable on this board? Returns a verdict and a message
    fit to show the user.

    This is the only place the board is described, and it is deliberately not
    the model's problem. "turn on pin 100" is a well-formed command that this
    board cannot run -- which is a better thing to tell someone than "I didn't
    understand that", and a far better thing than turning on pin 10."""
    if f.action is Action.UNKNOWN:
        return Verdict.EXECUTE, ""

    for t in f.targets:
        if isinstance(t, Pin) and t.n not in pins:
            return Verdict.BAD_PIN, f"pin {t.n} is not a GPIO on this board"

    if f.action is Action.SEQ and len(f.targets) > MAX_SEQ_PINS:
        return (Verdict.TOO_MANY_PINS,
                f"a chase runs at most {MAX_SEQ_PINS} pins, got {len(f.targets)}")

    if f.interval_ms is not None and not (
            INTERVAL_MIN <= f.interval_ms <= INTERVAL_MAX):
        return (Verdict.BAD_INTERVAL,
                f"interval {f.interval_ms}ms is outside "
                f"{INTERVAL_MIN}-{INTERVAL_MAX}ms")

    return Verdict.EXECUTE, ""


def executable(f: Frame, pins: list[int] = PINS_S3) -> bool:
    return range_check(f, pins)[0] is Verdict.EXECUTE


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

# Kept only for negatives.py, which needs plausible device nouns to build
# unsupported-capability phrases ("dim the X to half brightness").
EXAMPLE_NAMES = ["lamp", "fan", "buzzer", "heater", "valve", "relay", "siren",
                 "red led", "kitchen light", "desk lamp", "status led"]

# Intervals with a natural English form, sampled more often than arbitrary
# values because that is what people actually say. The arbitrary tail is what
# forces the model to copy digits rather than memorize buckets.
ROUND_INTERVALS = [50, 100, 150, 200, 250, 300, 400, 500, 750,
                   1000, 1500, 2000, 2500, 3000, 5000]
COMMON_COUNTS = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 10, 12, 20]

# --- out-of-range sampling ---------------------------------------------------
#
# The whole point of digit-wise pins is that the model transcribes a number
# without judging it. It only learns that if out-of-range numbers are ordinary
# things it has seen said. In v0 these were *negatives*, which trained exactly
# the reflex being removed -- so they do not merely get deleted, they change
# side. See docs/V1-SCOPE.md, "Negatives flip label".
#
# The rate is a balance: too low and the prior stays "a number after 'pin' is
# legal", which is the substitution bug; too high and the corpus stops looking
# like real usage, where almost every pin named is a real one.
OOR_PIN_PROB = 0.18
OOR_INTERVAL_PROB = 0.10
LONG_SEQ_PROB = 0.10

# Pins adjacent to legal ones are the dangerous case and the reason this exists:
# v0 answered "turn on pin 19" with <p9>, acting on a different pin than asked.
NEAR_MISS_PINS = [0, 19, 20, 22, 23, 24, 27, 30, 35, 36, 37, 43, 44, 45, 46,
                  47, 49, 50]


def sample_pin_number(rng: random.Random, pins: list[int],
                      oor_prob: float = OOR_PIN_PROB) -> int:
    """A pin number as a speaker would say it: usually real, sometimes not."""
    if rng.random() >= oor_prob:
        return rng.choice(pins)
    roll = rng.random()
    if roll < 0.55:
        return rng.choice(NEAR_MISS_PINS)
    if roll < 0.80:
        return rng.choice([n for n in range(0, 60) if n not in pins])
    # Three digits. Under-sampled by everything else here, and the arm of the
    # distribution the truncation bug lived on -- "pin 100" -> <p10>.
    return rng.randint(100, 199)


def sample_interval(rng: random.Random,
                    oor_prob: float = OOR_INTERVAL_PROB) -> int:
    if rng.random() < oor_prob:
        return (rng.randint(1, INTERVAL_MIN - 1) if rng.random() < 0.4
                else rng.randint(INTERVAL_MAX + 1, 200000))
    if rng.random() < 0.75:
        return rng.choice(ROUND_INTERVALS)
    return rng.randint(INTERVAL_MIN, INTERVAL_MAX)


def sample_count(rng: random.Random) -> int:
    if rng.random() < 0.85:
        return rng.choice(COMMON_COUNTS)
    return rng.randint(1, 99)


def sample_target(rng: random.Random, pins: list[int], *,
                  allow_all: bool = True) -> Target:
    if allow_all and rng.random() < 0.08:
        return All()
    return Pin(sample_pin_number(rng, pins))


# How often a pin-taking action names more than one. Low, because most commands
# really do address a single pin -- but never zero, which is what v0 trained and
# what made "stop pins 4 and 5" come back as a chase.
MULTI_PIN_PROB = 0.18


def _pin_list(rng: random.Random, pins: list[int], *,
              allow_all: bool = True) -> list[Target]:
    """One target, or several pins. Shared by set/read/blink/stop so the four
    cannot drift apart again."""
    t = sample_target(rng, pins, allow_all=allow_all)
    if isinstance(t, All) or rng.random() >= MULTI_PIN_PROB:
        return [t]
    n = rng.randint(2, MAX_SEQ_PINS)
    return [Pin(sample_pin_number(rng, pins)) for _ in range(n)]


# Weighted so the corpus is not dominated by the rarest actions. `set` is what
# people actually type. `alias`'s v0 share is redistributed rather than kept:
# there is no alias action in v1.
ACTION_WEIGHTS = {
    Action.SET: 38,
    Action.BLINK: 22,
    Action.READ: 14,
    Action.STOP: 12,
    Action.SEQ: 14,
}


def sample_frame(rng: random.Random, pins: list[int] = PINS_S3) -> Frame:
    action = rng.choices(list(ACTION_WEIGHTS), weights=list(ACTION_WEIGHTS.values()))[0]

    if action is Action.SET:
        level = rng.choices([Level.HIGH, Level.LOW, Level.TOGGLE],
                            weights=[45, 45, 10])[0]
        f = Frame(action, _pin_list(rng, pins), level=level)

    elif action is Action.READ:
        f = Frame(action, _pin_list(rng, pins, allow_all=False))

    elif action is Action.BLINK:
        roll = rng.random()
        if roll < 0.70:                      # a rate, and a count with it
            iv, ct = sample_interval(rng), sample_count(rng)
        elif roll < 0.84:                    # "blink pin 4 five times"
            iv, ct = None, rng.randint(1, 20)
        else:                                # "blink pin 4" -- device default
            iv, ct = None, None
        f = Frame(action, _pin_list(rng, pins), interval_ms=iv, count=ct)

    elif action is Action.SEQ:
        # Over-long chases stop at +4. Past that the utterance runs beyond
        # seq_len and prepare.py drops it, which does not remove the case
        # evenly -- it removes the wordiest phrasings of it, leaving the model
        # trained on terse over-long chases only. Ten pins already covers what
        # held-out data contains, and growing seq_len to fit twelve would cost
        # KV cache and a millisecond a token on device for input nobody sends.
        long = rng.random() < LONG_SEQ_PROB
        n = (rng.randint(MAX_SEQ_PINS + 1, MAX_SEQ_PINS + 4) if long
             else rng.randint(2, MAX_SEQ_PINS))
        # Real chases run along physically adjacent LEDs, so a consecutive
        # ascending run is the common case; scattered pins are the tail.
        roll = rng.random()
        if roll < 0.5 and len(pins) > n:
            start = rng.randrange(len(pins) - n + 1)
            chosen = pins[start:start + n]
        elif roll < 0.7 and n <= len(pins):
            chosen = sorted(rng.sample(pins, n))
        else:
            chosen = [sample_pin_number(rng, pins) for _ in range(n)]
        timed = rng.random() < 0.72          # "chase 2 3 4 5" carries no timing
        f = Frame(action, [Pin(p) for p in chosen],
                  interval_ms=sample_interval(rng) if timed else None,
                  count=sample_count(rng) if timed else None)

    else:  # STOP
        # No target means everything, which is most of what people say.
        targets = ([] if rng.random() < 0.45
                   else _pin_list(rng, pins, allow_all=False))
        f = Frame(action, targets)

    validate(f)
    return f


if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(16):
        fr = sample_frame(rng)
        syms = to_symbols(fr)
        assert from_symbols(syms).key() == fr.key(), fr
        v, msg = range_check(fr)
        note = "" if v is Verdict.EXECUTE else f"   -> {v.value}: {msg}"
        print(f"{' '.join(syms):<52}{note}")
    print(f"\n{len(special_symbols())} reserved symbols")
