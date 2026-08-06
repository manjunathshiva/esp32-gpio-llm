"""Semantic frames: the canonical meaning of a command.

This module is the single definition of what a command *is*. The sampler
produces frames, the realizer renders them to English, the evaluator compares
them, and `firmware/common/command.h` reconstructs them from emitted symbols.

The symbol encoder here and the decoder in command.h must agree exactly --
there are no names in the token stream, only ordered slots. Change one, change
the other. (Same contract as export.py's tensor `plan` and llm_load().)

**Targets are pin numbers or names.** Pin numbers are emitted digit-wise,
exactly like intervals, and the *runtime* decides whether a number is a GPIO on
this board. The model transcribes; it does not validate.

That split is load-bearing. v0 gave each allowlisted pin its own symbol, which
made an illegal pin unrepresentable -- and an unrepresentable pin does not
produce a refusal, it produces the nearest legal one. "switch off pin 100" came
back as <set> <p10> <low>: the wrong physical pin, actuated silently. Range is
a hardware fact, so it belongs with the hardware.

**v2 applies the same rule to names**, which is the whole reason they were
worth deferring until after the pin fix was understood. A name is copied from
the utterance as a span and resolved against an alias table the model has never
seen; an unknown name is refused by name. The tempting alternative -- teaching
the model which names exist -- rebuilds the pin-100 bug one level up: a model
that can only say names it was trained on answers "turn on the aquarium pump"
with the nearest name it knows, and actuates the wrong device. It must be able
to say a name that does not resolve, so that something downstream can say so.

v1 shipped without names and answered <unknown> to all of them; see
data/names.py for why the name space is generated rather than listed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

import names

# Pin allowlists, copied from femtoclaw's tool_gpio.c. Unlike v0 these no longer
# constrain what the model can emit -- they are what range_check() tests against
# and what gpio_control.c enforces on device. The two must stay in step.
PINS_S3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           21, 38, 39, 40, 41, 42, 48]
PINS_WROOM = [2, 4, 5, 12, 13, 14, 15, 18, 19, 22, 23, 25, 26, 27, 32, 33]

INTERVAL_MIN, INTERVAL_MAX = 50, 10000      # gpio_control.c bounds
MAX_SEQ_PINS = 6                            # see data/frames.py (MAX_SEQ_PINS)

# Longest digit run the parser will accept in any numeric slot. Not a range
# check -- a range check has a value to report. This rejects a generation that
# has run away, which is a structural failure.
MAX_DIGITS = 6

# Same idea for a copied name span. A name is arbitrary text, so unlike a digit
# run it has no natural end and a degenerate generation could copy forever; this
# is the structural stop, and it sizes the buffer in command.h. Generous on
# purpose -- "the light above the workbench in the garage" is a name someone
# will type, and truncating it silently would resolve to a *different* alias,
# which is the one outcome this design exists to prevent.
MAX_NAME_CHARS = 48


class Action(str, Enum):
    SET = "set"
    READ = "read"
    BLINK = "blink"
    SEQ = "seq"
    STOP = "stop"
    UNKNOWN = "unknown"
    # Reserved by v2.0, sampled from v2.1. It costs one id out of 1024 to carry
    # it now, and adding a reserved symbol later would shift every token id and
    # force a second full retrain of a model that is otherwise finished.
    ALIAS = "alias"


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


@dataclass(frozen=True)
class Name:
    """A target named rather than numbered, copied verbatim from the utterance.

    Deliberately just a string. Nothing here knows whether the name resolves --
    the alias table lives on the device and is edited at runtime, so resolution
    cannot be a property of the frame without baking a snapshot of the user's
    table into the training data."""
    s: str

    def norm(self) -> str:
        """The form the alias table matches on. Case and run-length of spaces
        are not meaningful in a name someone typed, and the device folds them
        the same way -- so scoring must too, or a correct copy of "Desk Lamp"
        reads as a miss."""
        return " ".join(self.s.split()).casefold()


Target = Pin | All | Name


def _target_key(t: Target) -> tuple:
    if isinstance(t, Pin):
        return ("pin", t.n)
    if isinstance(t, Name):
        return ("name", t.norm())
    return ("all",)


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
            tuple(_target_key(t) for t in self.targets),
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
S_NAME = "<name>"
S_NEND = "<nend>"

ACTION_SYM = {a: f"<{a.value}>" for a in Action}
LEVEL_SYM = {l: f"<{l.value}>" for l in Level}

# Every marker that introduces a run of digits. A run ends at the next symbol
# that is not a digit, so none of these needs a terminator: a number terminates
# itself. <nend> exists for the one slot where that is not true -- a name is
# arbitrary text, and "kitchen light" followed by <high> is indistinguishable
# from a name that happens to contain the word "high" unless something closes
# the span explicitly.
NUM_SYM = (S_PIN, S_INT, S_CNT)


def special_symbols() -> list[str]:
    """Every reserved id, in a stable order. gen_assets.py emits symbols.h from
    this, so the order is part of the model format -- do not sort it later.

    Not parameterised by the pin allowlist: there is one <pin> symbol and the
    board is not baked into the vocabulary. Changing boards is a change to
    gpio_control.c alone, not a retrain.

    v2 appends <name>/<nend> and <alias>, which moves every id after them. That
    is a tokenizer change and therefore a retrain -- src/train.py's fingerprint
    is what stops a v1 checkpoint from being scored against this table and
    reading as a model that forgot which action was which."""
    return (
        [ACTION_SYM[a] for a in Action]
        + [S_PIN, S_ALL, S_NAME, S_NEND]
        + [LEVEL_SYM[l] for l in Level]
        + [S_INT, S_CNT, S_END]
    )


# Membership test for "is this element a marker rather than content". Only a
# name span can contain arbitrary text, so this is what tells a copied name from
# the symbol that ends it.
RESERVED_SET = frozenset(special_symbols())


def to_symbols(f: Frame) -> list[str]:
    """Frame -> emitted symbol sequence. Order is fixed:
    ACTION TARGET* [LEVEL] [<int> digits] [<cnt> digits] <end>

    An element of the returned list is one of three things: a reserved symbol, a
    single digit, or -- between <name> and <nend> -- a whole name as one string.
    The name is *not* pre-split into tokens here. prepare.encode_completion is
    the only place that decides how a span becomes ids, and evaluate.py
    reassembles the run back into one element before parsing, so both ends of
    the pipeline agree on what a "symbol" is."""
    out = [ACTION_SYM[f.action]]

    for t in f.targets:
        if isinstance(t, Pin):
            out += [S_PIN, *str(t.n)]
        elif isinstance(t, Name):
            out += [S_NAME, t.s, S_NEND]
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

    def read_name() -> Name:
        """The run between <name> and <nend>, which arrives as exactly one
        element -- see to_symbols. Reassembling model output into that one
        element is evaluate.py's job, so anything else here means the two halves
        have drifted and the right response is to fail, not to guess."""
        nonlocal i
        s = peek()
        if s is None or s in RESERVED_SET:
            raise ParseError("empty name")
        i += 1
        if peek() != S_NEND:
            raise ParseError(f"expected {S_NEND} after a name, got {peek()!r}")
        i += 1
        if not s.strip():
            raise ParseError("blank name")
        if len(s) > MAX_NAME_CHARS:
            raise ParseError(f"name is {len(s)} chars")
        return Name(s)

    # targets
    while (s := peek()) is not None:
        if s == S_ALL:
            f.targets.append(All())
            i += 1
        elif s == S_PIN:
            i += 1
            f.targets.append(Pin(read_digits("pin")))
        elif s == S_NAME:
            i += 1
            f.targets.append(read_name())
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
    """Enforce the per-action shape in data/frames.py -- which slots an action
    takes, and how many. The sampler and the parser both go through this, so
    they cannot drift apart.

    Structure only. Whether a *value* is legal on this board is range_check()'s
    job, and keeping them apart is what lets the model emit "pin 100" and the
    runtime refuse it, instead of the model silently emitting "pin 10"."""
    a, n = f.action, len(f.targets)
    has_all = any(isinstance(t, All) for t in f.targets)
    has_name = any(isinstance(t, Name) for t in f.targets)

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
        #
        # A name suspends even the lower bound. "chase the porch lights" is one
        # target that resolves to four pins, and how many pins a name covers is
        # a fact about the alias table -- which the model has never seen. Demand
        # two targets here and the only way to satisfy it is to invent a second
        # one. The device refuses a chase that resolves to fewer than two pins,
        # where the count is actually known.
        need(n >= 2 or has_name, "needs at least 2 targets, or a name")
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
    else:
        # Reached by <alias>, whose id is reserved from v2.0 but which nothing
        # is trained to emit yet. Without this the if/elif chain would fall
        # through and validate an <alias> frame by checking nothing at all --
        # the quiet kind of hole, since the parse would succeed and the device
        # would act on a command with no shape rules behind it.
        raise ParseError(f"{a.value}: no shape rule, and none is emitted yet")

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
    # Only the device can return these: both need the alias table, which lives
    # in NVS and is edited at runtime. They are listed here so the two Verdict
    # enums stay a mirror of each other rather than diverging quietly.
    UNKNOWN_NAME = "unknown_name"
    NAME_TOO_FEW_PINS = "name_too_few_pins"


def range_check(f: Frame, pins: list[int] = PINS_S3) -> tuple[Verdict, str]:
    """Is this frame executable on this board? Returns a verdict and a message
    fit to show the user.

    This is the only place the board is described, and it is deliberately not
    the model's problem. "turn on pin 100" is a well-formed command that this
    board cannot run -- which is a better thing to tell someone than "I didn't
    understand that", and a far better thing than turning on pin 10."""
    if f.action is Action.UNKNOWN:
        return Verdict.EXECUTE, ""

    # Name targets pass through untouched, and that is not an omission. Whether
    # a name resolves is a fact about the user's alias table, which lives on the
    # device; the nearest thing this process could do is check against a
    # snapshot, and a stale snapshot would reject aliases that exist. The device
    # returns UNKNOWN_NAME. Here, a name is simply not a range question.
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
#
# The whole-second values above 3000 are here for a specific reason: they are
# the only in-range intervals a speaker says in seconds, and 10000 is the only
# legal one whose seconds form crosses into five digits. See SECONDS_INTERVALS.
ROUND_INTERVALS = [50, 100, 150, 200, 250, 300, 400, 500, 750,
                   1000, 1500, 2000, 2500, 3000, 4000, 5000,
                   6000, 7000, 8000, 9000, 10000]
COMMON_COUNTS = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 10, 12, 20]

# --- out-of-range sampling ---------------------------------------------------
#
# The whole point of digit-wise pins is that the model transcribes a number
# without judging it. It only learns that if out-of-range numbers are ordinary
# things it has seen said. In v0 these were *negatives*, which trained exactly
# the reflex being removed -- so they do not merely get deleted, they change
# side: an out-of-range number is a positive that range_check() refuses, not a
# negative the model learns to reject.
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


# Out-of-range intervals a speaker states in seconds or minutes, rather than as
# a millisecond count. These exist because of a measured hole, not for variety.
#
# The high branch used to be `randint(10001, 200000)` alone. Uniform draws land
# on a multiple of 1000 about once in a thousand, and realize.interval_phrase
# only offers a "seconds" wording when `ms % 1000 == 0` -- so across 147,000
# rows, *every* unit conversion in the corpus resolved to 3 or 4 digits and none
# to 5. The model learned the length, not the operation: it answered "every 60
# seconds" with 6000 and "every 90 seconds" with 9000, dropping exactly one
# zero, every time, on every seed.
#
# Copying is not the weak side -- five- and six-digit intervals written out in
# full were already transcribed correctly ~95% of the time. So this reallocates
# the high branch toward round values rather than widening it: the arbitrary
# tail that trains copying shrinks to a quarter, which costs nothing measurable,
# and buys the only thing missing, a conversion whose answer is long.
SECONDS_INTERVALS = [n * 1000 for n in
                     (11, 12, 15, 20, 25, 30, 40, 45, 50, 60, 75, 90, 120, 180)]
MINUTES_INTERVALS = [n * 60000 for n in (2, 3, 4, 5, 10)]   # 6 digits, phrased
                                                            # "every 3 minutes"


def sample_interval(rng: random.Random,
                    oor_prob: float = OOR_INTERVAL_PROB) -> int:
    if rng.random() < oor_prob:
        if rng.random() < 0.4:
            return rng.randint(1, INTERVAL_MIN - 1)
        roll = rng.random()
        if roll < 0.55:
            return rng.choice(SECONDS_INTERVALS)
        if roll < 0.75:
            return rng.choice(MINUTES_INTERVALS)
        # The arbitrary tail. Kept because it is what forces digit-by-digit
        # copying of a number with no round structure to lean on. Capped at six
        # digits: frames.MAX_DIGITS and command.h's CMD_MAX_DIGITS both refuse
        # a seventh, so generating one would train an unparseable output.
        return rng.randint(INTERVAL_MAX + 1, 200000)
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
# Of those multi-pin lists, how many run past what a chase can drive.
#
# Zero, and the zero is a measurement. Long lists were under-covered -- 1.76% of
# pin-bearing rows against 6.08% of the held-out set -- and raising this to 0.25
# closed that gap exactly (to 5.90%) while moving exact-match +0.5 against a
# +-1.4 spread: nothing. It also cost 1.5 points of false-accept, which does
# clear its spread, because lists of numbers in the positives make the model
# readier to read any list of numbers as a command.
#
# The derailment on long lists was never a coverage problem. It was the model
# losing track of position through a long digit run, and two extra layers plus
# four extra heads fixed what 3.4x the training examples did not. Kept at zero
# rather than deleted so the finding is not rediscovered.
LONG_LIST_PROB = 0.0


# Share of pin-taking frames that address their target by name instead. Higher
# than the ~12% of the held-out set that is name-shaped, because the two are
# answering different questions: held-out share measures what people type,
# this measures how much evidence the *slot* needs. A pin is one of 25 values
# and a name is one of tens of thousands, so the same coverage costs more rows.
NAME_TARGET_PROB = 0.30
# Of named frames, how many name several devices ("the lamp and the fan").
NAME_LIST_PROB = 0.14
# ...and how many mix a name with a bare pin number. Rare in speech, but never
# zero: a shape absent from the corpus becomes a shape the model refuses, and
# "turn on pin 4 and the desk lamp" is a sentence someone will type.
MIXED_TARGET_PROB = 0.06


def _named_list(rng: random.Random, pins: list[int]) -> list[Target]:
    """Targets for a frame that addresses devices by name."""
    if rng.random() < NAME_LIST_PROB:
        return [Name(names.sample_name(rng)) for _ in range(rng.randint(2, 3))]
    if rng.random() < MIXED_TARGET_PROB:
        ts: list[Target] = [Name(names.sample_name(rng)),
                            Pin(sample_pin_number(rng, pins))]
        rng.shuffle(ts)         # either order occurs; neither should be special
        return ts
    return [Name(names.sample_name(rng))]


def _pin_list(rng: random.Random, pins: list[int], *,
              allow_all: bool = True, named: bool = False) -> list[Target]:
    """One target, or several pins. Shared by set/read/blink/stop so the four
    cannot drift apart again."""
    if named:
        return _named_list(rng, pins)
    t = sample_target(rng, pins, allow_all=allow_all)
    if isinstance(t, All) or rng.random() >= MULTI_PIN_PROB:
        return [t]
    # Lists longer than a chase can run. Only `seq` ever produced these, at
    # LONG_SEQ_PROB, so lists of 7+ were 1.76% of pin-bearing rows while the
    # held-out set needs them for 6% -- and that is where the emission derails,
    # dropping or repeating a pin partway through a long digit run. Capped at
    # +4 for the same reason seq is: past ten pins the utterance runs beyond
    # seq_len and prepare.py drops it, which removes the wordiest phrasings
    # first and biases what is left.
    n = (rng.randint(MAX_SEQ_PINS + 1, MAX_SEQ_PINS + 4)
         if rng.random() < LONG_LIST_PROB else rng.randint(2, MAX_SEQ_PINS))
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
    # Decided once per frame, not per target: an utterance that says "the desk
    # lamp and pin 7" is a real but unusual shape, and sampling name-ness per
    # target would make it the common one.
    named = rng.random() < NAME_TARGET_PROB

    if action is Action.SET:
        level = rng.choices([Level.HIGH, Level.LOW, Level.TOGGLE],
                            weights=[45, 45, 10])[0]
        f = Frame(action, _pin_list(rng, pins, named=named), level=level)

    elif action is Action.READ:
        f = Frame(action, _pin_list(rng, pins, allow_all=False, named=named))

    elif action is Action.BLINK:
        roll = rng.random()
        if roll < 0.70:                      # a rate, and a count with it
            iv, ct = sample_interval(rng), sample_count(rng)
        elif roll < 0.84:                    # "blink pin 4 five times"
            iv, ct = None, rng.randint(1, 20)
        else:                                # "blink pin 4" -- device default
            iv, ct = None, None
        f = Frame(action, _pin_list(rng, pins, named=named),
                  interval_ms=iv, count=ct)

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
        if named:
            # A chase named collectively -- "cycle the porch lights" -- is one
            # target that the alias table expands into several. It is the shape
            # that forces the >=2 rule in validate() to be about *targets* and
            # not about pins; see the note there.
            seq_targets: list[Target] = (
                [Name(names.sample_group_name(rng))] if rng.random() < 0.7
                else [Name(names.sample_name(rng))
                      for _ in range(rng.randint(2, 3))])
        else:
            seq_targets = [Pin(p) for p in chosen]
        f = Frame(action, seq_targets,
                  interval_ms=sample_interval(rng) if timed else None,
                  count=sample_count(rng) if timed else None)

    else:  # STOP
        # No target means everything, which is most of what people say.
        targets = ([] if rng.random() < 0.45
                   else _pin_list(rng, pins, allow_all=False, named=named))
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
