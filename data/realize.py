"""Surface realization: Frame -> English.

The handwritten half of the corpus. It guarantees coverage of the core ways
each action is phrased; the LLM paraphrase pass (paraphrase.py) adds the
diversity nobody thinks of. Both are needed -- templates alone teach the model
to match templates.

Three invariants this file must not break:

  * A pin number appears **verbatim** in the utterance, whatever the style. The
    model copies those digits into <pin> ... , so a surface form that rounds or
    reformats the number makes the label unlearnable -- and a model that cannot
    copy a number is one that substitutes a plausible one, which is the failure
    v1 exists to remove.
  * A fuzzy quantity maps to exactly one canonical value. "quickly" is always
    200ms, never sometimes 100. Contradictory labels for the same words are
    worse than missing coverage -- they teach the model that the slot is noise.
  * A template is decorated only with prefixes its mood accepts. "i want you to
    status of the fan" is what happens when an imperative prefix meets an
    interrogative template.

v1 realizes pin targets only. `Speaker.name()` survives because
negatives.name_targeted() renders name-targeted commands through this same
machinery -- they have to be decorated identically to real commands, or the
model learns to reject them on punctuation rather than on the target.
"""

from __future__ import annotations

import random

import names
from frames import Action, All, Frame, Level, Name, Pin, Target

NUM_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 21: "twenty one",
}

# How one speaker writes a pin number. Chosen once per utterance -- mixing
# "p39, #4, GPIO 42" inside a single sentence is a generator artifact, not
# something a person does.
PIN_STYLES = [
    lambda n: f"pin {n}",
    lambda n: f"pin {n}",
    lambda n: f"pin {n}",
    lambda n: f"gpio {n}",
    lambda n: f"GPIO {n}",
    lambda n: f"gpio{n}",
    lambda n: f"pin number {n}",
    lambda n: f"#{n}",
    lambda n: f"io {n}",
    lambda n: f"p{n}",
    # The one style that is not a digit copy: "pin ten" has to be translated to
    # 1,0. Kept because people do write it, but it is a memorization island in
    # an otherwise copy-only grammar (NUM_WORDS covers 1-18 and 21, no more),
    # and it is why the transcription metric only scores utterances whose digits
    # appear literally.
    lambda n: f"pin {NUM_WORDS[n]}" if n in NUM_WORDS else f"pin {n}",
    # Bare number. People typing at a serial port write "off 12", not "turn off
    # pin 12" -- and a corpus without this scored 0% on that whole register.
    lambda n: f"{n}",
    lambda n: f"{n}",
]


class Speaker:
    """Per-utterance style, so references stay internally consistent."""

    def __init__(self, rng: random.Random, terse: bool = False):
        self.rng = rng
        self.pin_style = rng.choice(PIN_STYLES)
        # Terse input drops the determiner, which puts the name at position 0.
        # ByteLevel gives "status" and "Ġstatus" different ids, so a corpus
        # where names are almost always mid-sentence never teaches the copy at
        # the start -- "status_pin off" came back as a garbled span.
        self.det = rng.choices(["the ", "my ", "", "that "],
                               weights=[15, 5, 78, 2] if terse else [70, 12, 14, 4])[0]

    def pin(self, n: int) -> str:
        return self.pin_style(n)

    def name(self, text: str) -> str:
        return f"{self.det}{text}"

    def where(self) -> str:
        """A trailing location phrase, sometimes.

        "turn off the light in the bathroom" is the dominant shape in real
        smart-home speech and the first corpus produced none of it, so the model
        answered <unknown> to the whole pattern. The location is noise the model
        must learn to ignore: the target is the head noun, and this device has no
        concept of rooms.
        """
        if self.rng.random() > 0.28:
            return ""
        place = self.rng.choice(names.PLACE)
        return self.rng.choice([
            f" in the {place}", f" in {place}", f" in my {place}",
            f" on the {place}", f" at the {place}", f" upstairs",
            f" downstairs", f" out {place}", f" over there",
        ])

    def target(self, t: Target) -> str:
        if isinstance(t, All):
            return self.rng.choice(
                ["everything", "all pins", "all of them", "every pin",
                 "all the pins", "them all", "all gpio", "all the leds",
                 "the lot", "everything at once"])
        if isinstance(t, Name):
            # Verbatim, and only the determiner is added around it. prepare.py
            # finds this span in the finished utterance to slice the label's
            # token ids out of the prompt, so anything that rewrites the name
            # itself -- pluralising, trimming, expanding an abbreviation --
            # breaks the copy and the row is dropped with a SpanError.
            return self.name(t.s)
        return self.pin(t.n)

    # What joins the last two items of a list. "and" is not the only one people
    # write, and the others were absent: "turn on pin 4 then pin 5" came back as
    # pin 4 alone, dropping the pin after the unknown connector. Ordering words
    # are included deliberately -- for a level change the order is cosmetic, so
    # the right reading of "4 then 5" is both, not the first one.
    _JOIN = ["and", "and", "and", "then", "and then", "plus", "as well as", "&"]

    def target_list(self, ts: list[Target]) -> str:
        rng = self.rng
        join = rng.choice(self._JOIN)
        if all(isinstance(t, Pin) for t in ts) and rng.random() < 0.55:
            ns = [t.n for t in ts]
            head = rng.choice(["pins", "pins", "gpio", "gpios", "the pins"])
            style = rng.random()
            if style < 0.4:
                body = ", ".join(str(n) for n in ns[:-1]) + f" {join} {ns[-1]}"
            elif style < 0.7:
                body = ", ".join(str(n) for n in ns)
            else:
                body = " ".join(str(n) for n in ns)
            return f"{head} {body}"

        refs = [self.target(t) for t in ts]
        if rng.random() < 0.75:
            return ", ".join(refs[:-1]) + f" {join} {refs[-1]}"
        return ", ".join(refs)


# --------------------------------------------------------------------------
# Quantities
# --------------------------------------------------------------------------

# Fuzzy speed words, each pinned to exactly one interval, and only emitted when
# the sampled interval is that value.
SPEED_WORDS: dict[int, list[str]] = {
    100: ["really fast", "very quickly", "rapidly", "super fast"],
    200: ["quickly", "fast", "quick"],
    1000: ["slowly", "slow", "at a slow pace"],
    2000: ["very slowly", "really slowly", "nice and slow"],
}

RATE_WORDS: dict[int, list[str]] = {
    100: ["ten times a second"],
    200: ["five times a second"],
    250: ["four times a second", "every quarter second"],
    500: ["twice a second", "every half second", "every half a second"],
    1000: ["once a second", "every second", "one per second"],
    1500: ["every second and a half"],
    2000: ["every two seconds", "every couple of seconds"],
    3000: ["every three seconds"],
}


def interval_phrase(ms: int, rng: random.Random) -> str:
    forms = [f"every {ms}ms", f"every {ms} ms", f"every {ms} milliseconds",
             f"at {ms}ms", f"with a {ms}ms interval", f"{ms}ms apart",
             f"every {ms} millis"]
    if ms % 1000 == 0:
        s = ms // 1000
        unit = "second" if s == 1 else "seconds"
        # The abbreviations were absent entirely: "30s" and "2 sec" matched zero
        # of 147,000 rows while the held-out set used them freely. A form the
        # corpus never contains is not a rare form, it is an unknown one.
        forms += [f"every {s} {unit}", f"at {s} {unit} intervals",
                  f"every {NUM_WORDS.get(s, s)} {unit}",
                  f"every {s}s", f"at {s}s", f"{s}s apart",
                  f"every {s} sec", f"at {s} sec", f"every {s} secs",
                  f"speed {s} {unit}", f"interval {s} {unit}"]
    if ms % 1000 == 500:
        forms.append(f"every {ms / 1000:g} seconds")
    # Minutes are the only way a speaker states an interval this long, and the
    # only surface form whose conversion is x60000. Without it the six-digit
    # values sample_interval now draws would only ever appear written out in
    # milliseconds, which is not how anyone says them.
    if ms % 60000 == 0:
        m = ms // 60000
        unit = "minute" if m == 1 else "minutes"
        forms += [f"every {m} {unit}", f"at {m} {unit} intervals",
                  f"every {NUM_WORDS.get(m, m)} {unit}", f"at {m} {unit} rate"]
    # Hertz. Also absent -- three matches in the whole corpus, all of them
    # inside MASSIVE negatives, against six in the dev set alone. Only emitted
    # when the frequency is a whole number, because "at 6.7Hz" is not something
    # a person types and a non-integer would teach the model to round.
    if ms and 1000 % ms == 0:
        hz = 1000 // ms
        forms += [f"at {hz}Hz", f"at {hz} Hz", f"{hz}Hz", f"at {hz}hz"]
    forms += RATE_WORDS.get(ms, [])
    forms += SPEED_WORDS.get(ms, [])
    return rng.choice(forms)


def count_phrase(n: int, rng: random.Random, unit: str = "cycles") -> str:
    """`unit` is action-aware: a chase runs cycles, it does not run blinks."""
    if n == 0:
        return rng.choice(["", "", "forever", "continuously", "indefinitely",
                           "non-stop", "endlessly", "until i tell you to stop",
                           "until i say stop", "and keep going"])
    if n == 1:
        return rng.choice(["once", "one time", "1 time", "a single time"])
    if n == 2:
        return rng.choice(["twice", "two times", "2 times"])
    if n == 3:
        return rng.choice(["three times", "3 times", "thrice"])
    forms = [f"{n} times", f"for {n} cycles", f"{n}x", f"for {n} {unit}"]
    if n in NUM_WORDS:
        forms.append(f"{NUM_WORDS[n]} times")
    return rng.choice(forms)


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------
# `{r}` is the target reference. No template supplies its own determiner --
# Speaker.name() already does, and "blink the the desk lamp" is how that goes
# wrong. Templates are grouped by mood so _decorate can pick a legal prefix.

ON_T = [
    "turn on {r}", "turn {r} on", "switch on {r}", "switch {r} on",
    "power on {r}", "power up {r}", "enable {r}", "activate {r}",
    "light up {r}", "fire up {r}", "start {r}", "energize {r}",
    "set {r} high", "set {r} to high", "drive {r} high", "assert {r}",
    "raise {r}", "bring up {r}", "put {r} on", "flip {r} on", "flip on {r}",
    "kick on {r}", "get {r} going", "set {r} to 1", "make {r} high",
    "pull {r} high",
]
ON_BARE = ["{r} on", "{r} high"]           # not verb phrases; no prefix

OFF_T = [
    "turn off {r}", "turn {r} off", "switch off {r}", "switch {r} off",
    "power off {r}", "power down {r}", "disable {r}", "deactivate {r}",
    "shut off {r}", "shut down {r}", "shut {r} off", "kill {r}",
    "cut {r}", "cut power to {r}", "extinguish {r}", "set {r} low",
    "set {r} to low", "drive {r} low", "deassert {r}", "drop {r}",
    "take down {r}", "put out {r}", "flip {r} off", "flip off {r}",
    "douse {r}", "set {r} to 0", "make {r} low", "pull {r} low",
]
OFF_BARE = ["{r} off", "{r} low"]

TOGGLE_T = [
    "toggle {r}", "flip {r}", "invert {r}", "switch {r}", "reverse {r}",
    "toggle the state of {r}", "flip the state of {r}", "change {r}",
    "swap {r}", "toggle {r} over", "invert the state of {r}",
]

BLINK_T = [
    "blink {r}", "flash {r}", "strobe {r}", "pulse {r}", "make {r} blink",
    "make {r} flash", "have {r} blink", "get {r} blinking", "wink {r}",
    "blip {r}", "start blinking {r}", "set {r} blinking", "flash up {r}",
    "make {r} pulse", "start flashing {r}",
]

SEQ_T = [
    "chase across {r}", "run a chase across {r}", "cycle through {r}",
    "sweep across {r}", "scan across {r}", "run a sequence on {r}",
    "sequence {r}", "chase {r}", "cycle {r}", "run {r} in sequence",
    "marquee across {r}", "knight rider on {r}",
    "go back and forth across {r}", "light {r} one at a time",
    "step through {r}", "walk across {r}", "ripple across {r}",
]

# A bare verb here is a *complete* command -- stop-all needs no target -- while
# a bare verb in negatives.TRUNCATED ("blink", "set") is an incomplete one. The
# model has to tell those apart from very few examples, so the single-word forms
# are listed generously rather than left as a handful.
STOP_ALL_T = [
    "stop", "stop everything", "halt", "cancel", "quit", "freeze",
    "stop all animations", "knock it off", "cut it out", "enough",
    "stop them all", "cease", "make it stop", "stop all of it",
    "stop the blinking", "all stop", "abort",
    # "everything off" was here and is not a stop. SET/All/LOW realizes the very
    # same string through OFF_BARE ("{r} off" with r = "everything"), so the
    # corpus carried one utterance under two labels and the model learned to
    # read the whole shape as <stop>. Contradictory labels are worse than
    # missing coverage: they teach that the slot is noise.
    "stop it", "stop them", "stop all", "halt everything", "cancel everything",
    "freeze everything", "quit it", "give it a rest", "that's enough",
    "stop the animation", "stop all the blinking", "kill the animation",
    "stop whatever you're doing", "settle down", "calm down", "pack it in",
    # Verb + bare noun, no target. "halt chase" was refused because only
    # "kill chase" existed.
    "halt chase", "stop chase", "cancel chase", "end chase", "kill chase",
    "stop chasing", "stop sequence", "cancel sequence", "end sequence",
    "stop blink", "cancel blink", "end blink", "kill blink", "halt blink",
    "stop flashing", "cancel animation", "stop animation", "halt animation",
]

# Naming something. `{r}` is the pin reference, `{n}` the name being assigned.
# Two registers: an imperative ("call pin 4 the desk lamp") and a declarative
# ("pin 4 is the desk lamp"). The declarative matters more than its share
# suggests -- it has no command verb at all, so a model that keys on the verb
# to decide the action has nothing to key on and must read the sentence.
ALIAS_T = [
    "call {r} the {n}", "name {r} the {n}", "call {r} {n}", "name {r} {n}",
    "rename {r} to {n}", "remember {r} as the {n}", "label {r} as the {n}",
    "set the name of {r} to {n}", "refer to {r} as the {n}",
    "alias {r} to {n}", "tag {r} as {n}", "map {r} to the {n}",
]
ALIAS_DECL_T = [
    "{r} is the {n}", "{r} is called the {n}", "let's call {r} the {n}",
    "{r} should be called the {n}", "from now on {r} is the {n}",
    "{r} is my {n}", "i call {r} the {n}", "{r} = {n}",
    "{r} is now the {n}", "treat {r} as the {n}",
]

STOP_ONE_T = [
    "stop {r}", "stop blinking {r}", "halt {r}", "cancel {r}",
    "stop the blinking on {r}", "quit blinking {r}", "freeze {r}",
    "stop flashing {r}", "leave {r} alone", "stop animating {r}",
    "cancel the animation on {r}", "stop the flashing on {r}",
    # "cancel blink 13" was read as a blink command -- the bare verb after
    # cancel/stop/end needs to be covered, not just the -ing form.
    "cancel blink {r}", "stop blink {r}", "end blink {r}", "kill blink {r}",
    "cancel blink on {r}", "stop blink on {r}", "end blink on {r}",
    "disable blink {r}", "cancel chase {r}", "stop chase {r}",
    "stop sequence {r}", "end flash {r}", "cancel flash {r}",
]

# Imperative: "read X", "check X" -- takes "please", "can you", "go ahead and".
READ_CMD_T = [
    "read {r}", "check {r}", "get {r}", "read the level of {r}",
    "show me {r}", "tell me about {r}", "check on {r}",
    "give me the state of {r}", "report {r}",
    # Instrument register. Held-out data asked to "query pin 15" and "poll 22"
    # and got <unknown>: the read bank was conversational throughout, so a whole
    # vocabulary people reach for at a serial port was missing.
    "query {r}", "poll {r}", "sample {r}", "probe {r}", "inspect {r}",
    "read back {r}", "readback {r}", "dump {r}", "measure {r}",
    "what's the value of {r}", "give me the value of {r}",
]
# Interrogative: already a question -- an imperative prefix breaks it.
READ_Q_T = [
    "what is {r}", "what's {r}", "what is the state of {r}",
    "what's the status of {r}", "what is the status of {r}",
    "is {r} on", "is {r} off", "is {r} high", "is {r} low",
    "how is {r}", "what's {r} doing", "is {r} running",
    "what level is {r}", "what state is {r} in",
]
# Fragment: a bare noun phrase, stands alone.
READ_FRAG_T = ["status of {r}", "state of {r}", "{r} status", "{r} state",
               "{r} value", "{r} level", "{r} reading", "value of {r}",
               "level of {r}"]

IMPERATIVE_PREFIX = ["", "", "", "", "", "please ", "can you ", "could you ",
                     "would you ", "i want you to ", "hey ", "ok ",
                     "please can you ", "go ahead and ", "just ", "now ",
                     # Abbreviations people actually type. "pls high pin 18"
                     # was refused outright for want of three characters.
                     "pls ", "plz ", "pls can you ", "can u ", "cud u ",
                     "kindly ", "gimme ", "lemme ",
                     # Disfluency. "um, sweep 38 42" was refused outright --
                     # people type filler at a prompt, and every one of these
                     # reaches negatives.sample_negative too, so it stays a
                     # register rather than becoming a cue.
                     "um ", "um, ", "uh ", "er ", "hmm ", "ah "]
QUESTION_PREFIX = ["", "", "", "", "hey ", "ok ", "so ", "quick question, ",
                   "tell me, ", "um ", "um, ", "uh ", "hmm "]
NEUTRAL_PREFIX = ["", "", "", "", "hey ", "ok ", "so ", "note: ", "fyi ",
                  "um ", "uh ", "hmm "]

SUFFIX = ["", "", "", "", "", " please", " now", " for me", " thanks",
          " right now", " ok?"]
NEUTRAL_SUFFIX = ["", "", "", "", "", " ok?", " got it?"]


def case_and_punctuate(s: str, rng: random.Random, q: float = 0.05) -> str:
    """Capitalisation and terminal punctuation.

    **Shared with negatives.py, and it has to be shared rather than merely
    similar.** While only the positives could come out shouting, negatives were
    30% of the corpus but 15% of the ALL-CAPS rows -- so upper case became a cue
    for "this is a command". The model accepted "SET PIN 10 TO 25 PERCENT
    BRIGHTNESS" and refused the same sentence in lower case. Any register that
    one side of the corpus gets and the other does not becomes a shortcut.
    """
    r = rng.random()
    if r < 0.30:
        s = s[0].upper() + s[1:]
    elif r < 0.34:
        s = s.upper()               # "TURN OFF 39" -- people do shout
    if rng.random() < q:
        s += "?"
    elif rng.random() < 0.18:
        s += rng.choice([".", "!"])
    return s


def _decorate(s: str, rng: random.Random, mood: str = "imperative") -> str:
    prefix, suffix, q = {
        "imperative": (IMPERATIVE_PREFIX, SUFFIX, 0.05),
        "question": (QUESTION_PREFIX, NEUTRAL_SUFFIX, 0.65),
        "neutral": (NEUTRAL_PREFIX, NEUTRAL_SUFFIX, 0.10),
        "bare": ([""], NEUTRAL_SUFFIX, 0.0),
        "terse": ([""], [""], 0.0),
    }[mood]

    s = rng.choice(prefix) + s + rng.choice(suffix)
    return case_and_punctuate(" ".join(s.split()), rng, q)


# Compact machine-like forms. A held-out set written by a second model was full
# of these ("off 12", "blink 1 500ms 10", "chase 2 3 4 5") and the conversational
# templates above covered none of them.
def _terse(f: Frame, rng: random.Random, sp: Speaker) -> str | None:
    a = f.action
    if a is Action.SET and f.level is not Level.TOGGLE:
        w = ("on", "high", "1") if f.level is Level.HIGH else ("off", "low", "0")
        lvl, r = rng.choice(w), _ref(f, sp)
        return f"{lvl} {r}" if rng.random() < 0.5 else f"{r} {lvl}"
    if a is Action.SET:
        return f"toggle {_ref(f, sp)}"
    if a is Action.READ:
        r = _ref(f, sp)
        return rng.choice([f"read {r}", f"{r} state", f"{r} status", f"check {r}"])
    if a is Action.BLINK:
        r = _ref(f, sp)
        if f.interval_ms is None:
            # A count with no rate: "blink 4 five times". The count must still
            # be said -- it is the only number in the frame -- and it needs a
            # marker. A bare trailing number ("blink pin 9 10") cannot be told
            # from part of the target, and an ambiguous label teaches the model
            # that the boundary is guesswork.
            if f.count:
                head = rng.choice(["blink", "flash"])
                return rng.choice([f"{head} {r} x{f.count}",
                                   f"{head} {r} {f.count} times",
                                   f"{head} {r} count {f.count}"])
            return f"{rng.choice(['blink', 'flash', 'strobe'])} {r}"
        head = rng.choice(["blink", "flash"])
        ct = "" if f.count == 0 else f" {f.count}"
        # "blink pin 41 speed 300ms count 20" -- keyword-labelled slots, which
        # the conversational templates never produce.
        if rng.random() < 0.3:
            k = "" if f.count == 0 else f" count {f.count}"
            return (f"{head} {r} {rng.choice(['speed', 'interval', 'rate'])} "
                    f"{f.interval_ms}ms{k}")
        return f"{head} {r} {f.interval_ms}ms{ct}"
    if a is Action.SEQ:
        # The bare-number form is what makes this register terse ("chase 2 3 4
        # 5"), and it only exists for pins. A named chase still has a terse
        # form -- "chase porch lights 200ms 5" -- but it has to go through the
        # reference renderer, which is the only thing that writes a name
        # verbatim; str(t.n) on a Name is how this first crashed.
        body = (" ".join(str(t.n) for t in f.targets)
                if all(isinstance(t, Pin) for t in f.targets) else _ref(f, sp))
        head = rng.choice(["chase", "cycle", "sweep", "seq"])
        if f.interval_ms is None:
            return f"{head} {body}"
        # The count has to be said. Dropping it (as this branch did) labelled
        # "chase 38 39 40 41 50ms" with <cnt> 3 -- a number nowhere in the text,
        # so the only way to score it is to guess. Same shape as the blink form
        # above, where a zero count means "forever" and is legitimately silent.
        ct = "" if f.count == 0 else f" {f.count}"
        return f"{head} {body} {f.interval_ms}ms{ct}"
    if a is Action.STOP:
        return (f"stop {_ref(f, sp)}" if f.targets
                else rng.choice(["stop", "stop all", "halt", "kill chase"]))
    return None


def _ref(f: Frame, sp: Speaker) -> str:
    """The target phrase. Every pin-taking action may name several pins, so all
    of them go through here rather than reaching for targets[0]."""
    return (sp.target(f.targets[0]) if len(f.targets) == 1
            else sp.target_list(f.targets))


def realize(f: Frame, rng: random.Random) -> str:
    """One surface form for a frame. Call repeatedly for variety."""
    a = f.action

    if rng.random() < 0.26:
        t = _terse(f, rng, Speaker(rng, terse=True))
        if t is not None:
            return _decorate(t, rng, "terse")
    sp = Speaker(rng)

    if a is Action.SET:
        if f.level is Level.TOGGLE:
            bank, mood = TOGGLE_T, "imperative"
        elif rng.random() < 0.14:
            bank = ON_BARE if f.level is Level.HIGH else OFF_BARE
            mood = "bare"
        else:
            bank = ON_T if f.level is Level.HIGH else OFF_T
            mood = "imperative"
        s = bank[rng.randrange(len(bank))].format(r=_ref(f, sp))
        return _decorate(s + sp.where(), rng, mood)

    if a is Action.READ:
        r = _ref(f, sp)
        kind = rng.random()
        if kind < 0.45:
            return _decorate(rng.choice(READ_Q_T).format(r=r) + sp.where(),
                             rng, "question")
        if kind < 0.85:
            return _decorate(rng.choice(READ_CMD_T).format(r=r) + sp.where(),
                             rng, "imperative")
        return _decorate(rng.choice(READ_FRAG_T).format(r=r), rng, "bare")

    if a is Action.BLINK:
        s = rng.choice(BLINK_T).format(r=_ref(f, sp))
        return _decorate(_with_timing(s, f, rng, "blinks"), rng)

    if a is Action.SEQ:
        s = rng.choice(SEQ_T).format(r=sp.target_list(f.targets))
        return _decorate(_with_timing(s, f, rng, "cycles"), rng)

    if a is Action.ALIAS:
        # The name is the last target and everything before it is the pins
        # being named -- see frames.validate. Rendered verbatim through
        # Speaker.name so prepare.py can still find the span; the templates
        # supply their own determiner, so this one must not add a second.
        pins = [t for t in f.targets if isinstance(t, Pin)]
        nm = next(t for t in f.targets if isinstance(t, Name)).s
        r = (sp.pin(pins[0].n) if len(pins) == 1
             else sp.target_list(list(pins)))
        decl = rng.random() < 0.42
        bank = ALIAS_DECL_T if decl else ALIAS_T
        return _decorate(rng.choice(bank).format(r=r, n=nm), rng,
                         "neutral" if decl else "imperative")

    # STOP
    s = (rng.choice(STOP_ONE_T).format(r=_ref(f, sp))
         if f.targets else rng.choice(STOP_ALL_T))
    return _decorate(s, rng)


def _with_timing(s: str, f: Frame, rng: random.Random, unit: str) -> str:
    """Attach interval and count. Order varies because both orders are natural
    and a fixed one would let the model key on position instead of meaning."""
    if f.interval_ms is None:
        # No rate. If a count was given it is the only number in the frame, so
        # it has to be said -- a silent one would be a label asking the model to
        # invent it. With neither, the device default applies.
        return s if not f.count else f"{s} {count_phrase(f.count, rng, unit)}"
    iv = interval_phrase(f.interval_ms, rng)
    ct = count_phrase(f.count, rng, unit)
    if not ct:
        return f"{s} {iv}"
    return f"{s} {ct} {iv}" if rng.random() < 0.35 else f"{s} {iv} {ct}"


if __name__ == "__main__":
    import frames

    rng = random.Random(1)
    for _ in range(28):
        fr = frames.sample_frame(rng)
        print(f"{realize(fr, rng):<64}  {' '.join(frames.to_symbols(fr))}")
