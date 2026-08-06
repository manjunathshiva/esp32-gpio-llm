"""Off-domain utterances that must map to <unknown>.

There is no cloud fallback, so a misparse has nowhere to go: "what's the
weather in Paris" becoming <set> <p2> <high> turns on a light. <unknown> is a
trained output class, not a confidence heuristic bolted on afterwards, and this
file is its training signal.

The valuable negatives are the near-misses. Generic chit-chat is easy to reject
and teaches almost nothing; "turn on the" and "how many pins are there" are the
ones that decide the false-accept rate.

**What stopped being a negative in v1.** "blink pin 4 every 5ms", "turn on pin
100", "chase 8 pins" used to live here. They are now *positives* -- the model
transcribes the number and gpio_control.c refuses it. Keeping them as <unknown>
would train the exact reflex v1 removes, so they changed side rather than being
deleted; the sampling now happens in frames.sample_pin_number and
frames.sample_interval. See the out-of-range sampling comment in data/frames.py.

**What stopped being a negative in v2.0.** name_targeted() used to live here,
because v1 had no alias table and so a name had nothing to resolve against.
A named command is now a positive with a Name target and the *device* answers
for whether the name resolves; keeping it here as well would train two answers
for one sentence. The generator is retained, unused by sample_deferred, because
it is still the cleanest way to produce name-shaped utterances for probing.

**What is still deferred.** alias_request() -- asking to *create* a name -- is
debt until v2.1. It carries source "deferred_alias" so it drops in one move,
and the thing being rejected is the *shape* (assign a name to a pin), never a
particular noun. See data/frames.py on why a model that learns to reject nouns
is a model that rejects legitimate new aliases.
"""

from __future__ import annotations

import random

import names
import realize
from frames import EXAMPLE_NAMES, PINS_S3, ROUND_INTERVALS

# --- far off-domain: different task entirely ---------------------------------

OTHER_DOMAIN = [
    "what's the weather in Paris", "what's the weather like today",
    "set an alarm for 7am", "play some music", "play the next song",
    "send a text to mom", "call dad", "what time is it", "what's the date",
    "add milk to my shopping list", "how do i get to the station",
    "tell me a joke", "what's 17 times 4", "translate hello into spanish",
    "who won the game last night", "read me the news", "order a pizza",
    "start a timer for 10 minutes", "remind me to call the dentist",
    "how tall is mount everest", "what's the capital of France",
    "turn up the volume", "make the screen brighter", "open the garage door",
    "lock the front door", "what's my heart rate", "book a table for two",
    "how many calories in an apple", "spell necessary", "define entropy",
    "write me a poem", "summarise this article", "what's the stock price",
    "set the thermostat to 21", "vacuum the living room",
]
# Deliberately absent: "turn on the tv", "switch off the heater" and friends.
# Those are structurally valid commands whose target is simply not in the alias
# table, so they must fail at resolution, not at parse. See data/frames.py.

CHITCHAT = [
    "hello", "hi", "hey there", "good morning", "good night", "how are you",
    "who are you", "what can you do", "thanks", "thank you", "cheers",
    "never mind", "forget it", "sorry", "oops", "cool", "nice", "ok",
    "yes", "no", "maybe", "what", "huh", "help", "are you there",
    "say something", "test", "testing", "ping", "hello?", "you there?",
]

# --- near-miss: pin-shaped but not an executable command ----------------------

META_QUESTIONS = [
    "how many pins are there", "what pins can i use", "which pins are safe",
    "what can you control", "list the pins", "show me the pin map",
    "what's a gpio", "how do pins work", "what does blink mean",
    "is pin 4 safe to use", "what pins are free", "how many leds",
    "what's connected to pin 4", "what names have i set",
    "what are my aliases", "what did i call pin 4",
    "how fast can you blink", "what's the maximum interval",
    "can you control more than one pin", "do you support pwm",
    "what voltage is pin 4", "how much current can pin 4 take",
]

OPINIONS = [
    "i like pin 4", "pin 4 is my favourite", "that led is too bright",
    "the fan is noisy", "this is cool", "that worked", "that didn't work",
    "the light is already on", "i think pin 2 is broken",
    "the buzzer is annoying", "nice, the lamp works now",
    "the kitchen light keeps flickering", "pin 18 seems dead",
]

# --- near-miss: truncated or slot-less commands -------------------------------

TRUNCATED = [
    "turn on", "turn off", "switch on", "switch off", "blink", "flash",
    "set", "read", "check", "toggle", "chase", "sequence", "name", "call",
    "turn on the", "turn off my", "blink the", "set pin", "set pin to",
    "read the", "call pin", "name pin", "blink pin", "chase across",
    "turn the", "make the", "please turn on", "can you blink",
    "i want you to", "go ahead and", "every 300ms", "5 times", "forever",
    "high", "low", "on", "off", "the desk lamp", "pin", "gpio",
    # Cut-off prepositional tails. truncate() cannot reach these: it only cuts
    # after a word in _DANGLING, and the words that end these ("on", "of") are
    # excluded there because a cut ending in "on" can land on "pin 4 on", a
    # valid command. So the family is listed rather than derived.
    "stop blinking on", "stop the blinking on", "cancel blink on",
    "read the state of", "give me the state of", "check on",
    "what is the status of", "toggle the state of", "run a chase across",
]


def pin_range(rng: random.Random) -> str:
    """"Turn on pins 4-8." A range of pins, which the grammar cannot express.

    Refused rather than expanded, and the reason is what it did instead: "pins
    4-8" came back as **pin 48** -- the two digits read as one number, and 48 is
    a real pin on this board. "pins 4 to 8" fared no better, turning on 4 and 8
    and skipping everything between. Expanding ranges is a reasonable feature
    and belongs in a grammar change (a <range> slot the runtime enumerates), not
    in a model left to guess.
    """
    lo = rng.choice(PINS_S3[:-4])
    hi = lo + rng.randint(2, 8)
    verb = rng.choice(["turn on", "turn off", "switch on", "switch off",
                       "blink", "toggle", "chase", "cycle through", "read",
                       "set high", "sweep across", "flash"])
    ref = rng.choice([f"pins {lo}-{hi}", f"pins {lo} to {hi}",
                      f"pins {lo} through {hi}", f"pin {lo} to pin {hi}",
                      f"gpio {lo}-{hi}", f"{lo}-{hi}",
                      f"pins {lo} thru {hi}", f"the range {lo} to {hi}"])
    return f"{verb} {ref}"


def exception_set(rng: random.Random) -> str:
    """"Turn on all pins except 4 and 5." A set minus a subset.

    The grammar has <all> or an explicit list, and nothing in between. This was
    refused before it was ever trained -- purely because the phrasing was out of
    distribution -- and the moment multi-pin commands were added properly,
    <all> became the better match and "all pins except 4 and 5" started turning
    on 4 and 5. The exact opposite of the request, from a shape that had one
    example in 147,000 rows.

    Accidental correctness is not correctness. It survives until the
    distribution moves, and then it fails in whichever direction the model
    happens to lean.
    """
    ps = rng.sample(PINS_S3, rng.randint(1, 3))
    lst = (str(ps[0]) if len(ps) == 1
           else ", ".join(map(str, ps[:-1])) + f" and {ps[-1]}")
    head = rng.choice(["all pins", "everything", "every pin", "all the pins",
                       "all gpio", "all of them"])
    verb = rng.choice(["turn on", "turn off", "switch on", "switch off",
                       "blink", "toggle", "stop"])
    return rng.choice([
        f"{verb} {head} except {'pin ' if rng.random() < 0.5 else ''}{lst}",
        f"{verb} {head} but not {lst}",
        f"{verb} {head} apart from pin {ps[0]}",
        f"{verb} {head} other than pin {ps[0]}",
        f"{verb} everything but pin {ps[0]}",
        f"{verb} all but pin {ps[0]}",
        f"{verb} the rest of the pins",
        f"{verb} the other pins",
        f"{verb} the remaining pins",
        f"{verb} the first {rng.randint(2, 5)} pins",
        f"{verb} the last {rng.randint(2, 5)} pins",
        f"{verb} the odd pins", f"{verb} the even pins",
        f"{verb} every other pin",
    ])


def duration(rng: random.Random) -> str:
    """"Blink pin 4 for 10 seconds." A duration as an end condition.

    data/frames.py has always listed this as unexpressible -- the tool counts
    cycles, not time -- but nothing trained the refusal, so the model read "10
    seconds" as a 1000ms rate and blinked forever. Wrong rate and wrong end
    condition, from a sentence that names neither.
    """
    p = rng.choice(PINS_S3)
    # Small second-counts on purpose. "for 10 seconds" collides head-on with
    # interval_phrase's "every 10 seconds" (10,000ms) -- the only thing telling
    # them apart is `for` against `every`, and with the tail weighted to 90 the
    # model never saw enough of the overlap. It answered "blink pin 4 for 10
    # seconds" with a 1000ms rate running forever.
    n = rng.randint(2, 12) if rng.random() < 0.6 else rng.randint(2, 90)
    unit = rng.choice(["seconds", "secs", "minutes", "mins", "hours",
                       "seconds", "secs"])
    verb = rng.choice(["blink", "flash", "strobe", "pulse"])
    return rng.choice([
        f"{verb} pin {p} for {n} {unit}",
        f"{verb} pin {p} for the next {n} {unit}",
        f"turn on pin {p} for {n} {unit}",
        f"keep pin {p} high for {n} {unit}",
        f"{verb} pin {p} over the next {n} {unit}",
        f"run pin {p} for {n} {unit}",
        f"{verb} pin {p} for half a minute",
        f"turn on pin {p} for a couple of {unit}",
    ])


def relative_reference(rng: random.Random) -> str:
    """"Turn *that* one off." Every utterance is independent by design, so
    there is no previous turn to refer to -- deliberately unsupported, see
    data/frames.py (MAX_SEQ_PINS)."""
    return rng.choice([
        "turn that one off", "turn it back on", "do that again",
        "undo that", "the other one", "same for the next pin",
        "now the other led", "put it back", "revert that",
        "do the same thing again", "and the one next to it",
        "repeat what you just did", "undo the last command",
        "what was the last pin i changed", "switch back to how it was",
        "run the previous sequence again", "reverse the last operation",
    ])


# --- v1 debt: deferred capability, not permanent refusal ----------------------
# Both generators below produce commands v2 will *accept*. They are decorated
# through realize's own machinery on purpose: a negative that differs from a
# positive in punctuation or politeness teaches the model to key on punctuation.
# The only thing that may distinguish these from a positive is the target.

def name_targeted(rng: random.Random) -> str:
    """A well-formed command aimed at a name instead of a pin number.

    v1 has no alias table, so there is nothing for a name to resolve against.
    v2 deletes this generator and restores Name targets to the grammar."""
    sp = realize.Speaker(rng)
    nm = sp.name(names.sample_group_name(rng) if rng.random() < 0.18
                 else names.sample_name(rng))
    kind = rng.random()

    if kind < 0.42:                                   # set on / off / toggle
        bank = rng.choices([realize.ON_T, realize.OFF_T, realize.TOGGLE_T,
                            realize.ON_BARE, realize.OFF_BARE],
                           weights=[38, 38, 12, 6, 6])[0]
        mood = "bare" if bank in (realize.ON_BARE, realize.OFF_BARE) else "imperative"
        s = bank[rng.randrange(len(bank))].format(r=nm) + sp.where()
        return realize._decorate(s, rng, mood)

    if kind < 0.62:                                   # blink
        s = rng.choice(realize.BLINK_T).format(r=nm)
        if rng.random() < 0.55:
            ms = rng.choice(ROUND_INTERVALS)
            s = f"{s} {realize.interval_phrase(ms, rng)}"
            if rng.random() < 0.4:
                s = f"{s} {realize.count_phrase(rng.randint(2, 20), rng, 'blinks')}"
        return realize._decorate(s, rng)

    if kind < 0.80:                                   # read, all three moods
        r = rng.random()
        if r < 0.45:
            return realize._decorate(rng.choice(realize.READ_Q_T).format(r=nm)
                                     + sp.where(), rng, "question")
        if r < 0.85:
            return realize._decorate(rng.choice(realize.READ_CMD_T).format(r=nm)
                                     + sp.where(), rng, "imperative")
        return realize._decorate(rng.choice(realize.READ_FRAG_T).format(r=nm),
                                 rng, "bare")

    if kind < 0.92:                                   # stop one
        return realize._decorate(rng.choice(realize.STOP_ONE_T).format(r=nm), rng)

    # chase across several names
    refs = [sp.name(names.sample_name(rng)) for _ in range(rng.randint(2, 4))]
    body = ", ".join(refs[:-1]) + f" and {refs[-1]}"
    return realize._decorate(rng.choice(realize.SEQ_T).format(r=body), rng)


# Moved to realize.py in v2.1, where the positives live. Re-exported here so
# alias_request() below still reads the same list -- it is kept as a probe
# generator even though nothing samples it as a negative any more.
_ALIAS_T = realize.ALIAS_T
_ALIAS_DECL_T = realize.ALIAS_DECL_T


def alias_request(rng: random.Random) -> str:
    """Asking to *create* a name. The target is a perfectly good pin -- it is
    the action that v1 does not have. Kept separate from name_targeted because
    it is a different reason to refuse, and because v2 restores it as `alias`."""
    sp = realize.Speaker(rng)
    n = 1 if rng.random() < 0.8 else rng.randint(2, 4)
    nm = names.sample_group_name(rng) if n > 1 else names.sample_name(rng)
    ps = rng.sample(PINS_S3, n)
    r = (sp.pin(ps[0]) if n == 1
         else ", ".join(sp.pin(p) for p in ps[:-1]) + f" and {sp.pin(ps[-1])}")
    if rng.random() < 0.45:
        return realize._decorate(rng.choice(_ALIAS_DECL_T).format(r=r, n=nm),
                                 rng, "neutral")
    if rng.random() < 0.25:                            # the terse register
        head = rng.choice(["name", "alias", "call", "label", "rename", "tag"])
        return realize._decorate(f"{head} {r} {nm}", rng, "terse")
    return realize._decorate(rng.choice(_ALIAS_T).format(r=r, n=nm), rng)


def unsupported_capability(rng: random.Random) -> str:
    """Analogue output, colour, scheduling and conditionals.

    These leaked through at 20% on a held-out set, so they get their own
    generator rather than a handful of literals. The categories are drawn from
    what a held-out set contained, but every string here is generated -- copying
    the eval lines into training would destroy the measurement.
    """
    p, nm = rng.choice(PINS_S3), rng.choice(EXAMPLE_NAMES)
    pct, secs = rng.choice([10, 20, 25, 30, 50, 60, 75, 80, 90]), rng.randint(2, 90)
    colour = rng.choice(["red", "green", "blue", "yellow", "purple", "cyan",
                         "magenta", "warm white", "orange", "pink"])
    return rng.choice([
        # analogue / PWM
        f"set pin {p} to {pct} percent", f"set pin {p} brightness to {pct}%",
        f"dim {nm} to half brightness", f"dim pin {p}", f"dim {nm}",
        f"make pin {p} {pct}% bright", f"fade {nm} in over {secs} seconds",
        f"fade out pin {p}", f"ramp up brightness on pin {p}",
        f"set pin {p} duty cycle to {pct}%", f"pulse pin {p} with smooth dimming",
        f"adjust intensity of {nm} to medium", f"set analog value 128 on pin {p}",
        f"output analog signal on pin {p}", f"turn down the brightness on {nm}",
        f"brighten {nm}", f"make {nm} dimmer", f"set pin {p} to 2.5 volts",
        f"output 3.3v on pin {p}",
        f"scale pin {p} down to half intensity",
        f"scale {nm} up to full intensity",
        f"adjust the intensity of pin {p} to {pct}%",
        # Analog *read*. These open with our own read verbs, so the thing being
        # learned is the object: a voltage or an ADC count is unsupported, the
        # digital level of a pin is not. Without them "READ ANALOG VOLTAGE ON
        # PIN 10" came back as a command.
        f"read the analog voltage on pin {p}", f"read analog value of pin {p}",
        f"get the voltage on pin {p}", f"what voltage is on pin {p}",
        f"adc read pin {p}", f"read the adc on pin {p}",
        f"measure the current on pin {p}", f"sample the analog input on pin {p}",
        # colour
        f"change pin {p} color to {colour}", f"set {nm} to {colour}",
        f"make pin {p} glow {colour}", f"cycle through colors on pin {p}",
        f"set pin {p} hue to {colour}", f"set color temperature of {nm} to 3000k",
        f"switch the rgb led on pin {p} to {colour}",
        f"change pin {p} rgb to {rng.randint(0, 255)} {rng.randint(0, 255)} "
        f"{rng.randint(0, 255)}",
        f"set pin {p} to rgb {rng.randint(0, 255)},{rng.randint(0, 255)},"
        f"{rng.randint(0, 255)}",
        # scheduling
        f"turn on pin {p} in {secs} minutes", f"turn off {nm} after an hour",
        f"schedule pin {p} to turn off at 5pm", f"turn on {nm} at sunset",
        f"wait {secs} seconds then toggle pin {p}",
        f"turn on pin {p} every day at 8am", f"keep {nm} off until midnight",
        f"auto turn off pin {p} after {secs} minutes",
        f"delay {secs} seconds and switch pin {p} high",
        # conditionals / sensors
        f"if pin {p} is high turn on pin {rng.choice(PINS_S3)}",
        f"when the button on pin {p} is pressed blink {nm}",
        f"toggle {nm} whenever pin {p} goes high",
        f"monitor pin {p} and turn on {nm} if it changes",
        f"turn off {nm} when the temperature sensor triggers",
        f"sound the alarm on pin {p} if pin {rng.choice(PINS_S3)} goes low",
        f"read pin {p} and copy its state to pin {rng.choice(PINS_S3)}",
        f"link pin {p} and pin {rng.choice(PINS_S3)} so they turn on together",
        f"wait until pin {p} is low then set pin {rng.choice(PINS_S3)} high",
        f"while pin {p} is high keep blinking {nm}",
        # morse / patterns we do not have
        f"blink {nm} in morse code", f"make {nm} pulse smoothly",
        f"play a heartbeat pattern on pin {p}",
    ])


_Q_OBJECT = [
    "the capital of france", "the meaning of life", "the square root of 144",
    "the population of japan", "the boiling point of water", "the time in tokyo",
    "the tallest mountain", "the speed of light", "the price of bitcoin",
    "the answer to everything", "the best programming language",
    "the distance to the moon", "the weather like tomorrow",
    "the offside rule", "the plot of hamlet", "the airspeed of a swallow",
]


def offdomain_question(rng: random.Random) -> str:
    """General questions. These open exactly like a read command ("what is X"),
    so structure alone cannot reject them -- an early model answered
    "what is the capital of France" with <read> <name> capitals fan.

    The discriminator being taught here is the shape of the object: a long
    of-phrase or an arithmetic expression, never a short device noun. That is
    still structure, not a judgement about whether the noun is a known alias.
    """
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    return rng.choice([
        f"what is {rng.choice(_Q_OBJECT)}",
        f"what's {rng.choice(_Q_OBJECT)}",
        f"tell me {rng.choice(_Q_OBJECT)}",
        f"what is {a} times {b}", f"what is {a} plus {b}",
        f"calculate {a} divided by {b}", f"how much is {a} percent of {b}",
        f"what's {a} minus {b}", f"compute the square root of {a * a}",
        f"how many {rng.choice(['days', 'weeks', 'litres', 'miles'])} "
        f"in {rng.choice(['a year', 'a marathon', 'a gallon'])}",
        f"who {rng.choice(['wrote hamlet', 'won the world cup', 'invented the transistor'])}",
        f"where is {rng.choice(['the nearest station', 'my phone', 'mount everest'])}",
    ])


def conditional_wrapper(rng: random.Random) -> str:
    """A legal command wrapped in a condition or a delay, which the grammar has
    no way to express. Both orders, because both leaked."""
    p, q = rng.sample(PINS_S3, 2)
    cmd = rng.choice([
        f"blink pin {p}", f"turn on pin {p}", f"turn off pin {p}",
        f"toggle pin {p}", f"stop blink on pin {p}",
        f"chase on {p} {q}", f"set pin {p} high", f"read pin {p}",
    ])
    cond = rng.choice([
        f"if pin {q} is high", f"if pin {q} turns off", f"when pin {q} goes low",
        f"whenever pin {q} changes", f"once pin {q} is on",
        f"if the button on pin {q} is pressed", f"unless pin {q} is already on",
        f"in {rng.randint(2, 90)} seconds", f"in {rng.randint(2, 59)} minutes",
        "at sunset", "at 5pm", "tomorrow morning", "every day at 8am",
        f"after {rng.randint(2, 30)} minutes", "when i get home",
        "when the temperature sensor triggers", "while the door is open",
        # Clock times, in the forms people actually write them. A bare "0800"
        # or "07:30" reads to the slot rules as a rate, and it read that way to
        # the model too: "TURN ON PIN 42 AT 0800 HOURS" was executed.
        f"at {rng.randint(1, 12)}{rng.choice(['am', 'pm'])}",
        f"at {rng.randint(0, 23):02d}{rng.randint(0, 59):02d} hours",
        f"at {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
        "every evening", "every morning", "every night", "first thing tomorrow",
        "at the end of the day", "on weekdays", "at dawn", "at dusk",
    ])
    if rng.random() < 0.5:
        return f"{cond} {cmd}"
    return f"{cmd} {cond}"


# Words that cannot legally end an utterance. A cut is only kept when it lands
# on one of these -- otherwise truncating "blink pin 4 every 300ms 5 times"
# yields "blink pin 4", which is a perfectly valid command, and the negative
# would contradict the positives.
_DANGLING = {
    "the", "a", "an", "my", "that", "this", "to", "at", "of", "in", "for",
    "every", "and", "with", "from", "is", "are", "was", "turn", "set", "make",
    "blink", "flash", "strobe", "pulse", "read", "check", "chase", "cycle",
    "sweep", "scan", "name", "call", "label", "alias", "rename", "toggle",
    "flip", "switch", "power", "shut", "please", "can", "could", "would",
    "want", "go", "just", "state", "status", "level", "interval", "speed",
    "count", "pin", "gpio", "pins", "gpios", "across", "through", "up", "rate",
    "start", "get", "give", "put", "have", "let", "run", "step", "walk",
    "cancel", "halt", "leave", "keep", "i", "you", "me", "we", "it's",
}


def truncate(text: str, rng: random.Random) -> str | None:
    """Cut a real command short, or None if no cut leaves it incomplete.

    Half-finished input was the largest false-accept category -- "blink 12 at",
    "read state of", "set line 15 to" were all executed rather than refused.
    Deriving truncations from the same surface forms the positives use covers
    the shape systematically, which a hand-written list never did.
    """
    words = text.rstrip("?.!,").split()
    cuts = [i for i in range(1, len(words))
            if words[i - 1].lower().strip(",") in _DANGLING]
    if not cuts:
        return None
    return " ".join(words[:rng.choice(cuts)])


GIBBERISH_CHARS = "abcdefghijklmnopqrstuvwxyz"


def gibberish(rng: random.Random) -> str:
    n = rng.randint(1, 4)
    return " ".join(
        "".join(rng.choice(GIBBERISH_CHARS) for _ in range(rng.randint(2, 9)))
        for _ in range(n))


# Weighted toward near-misses: they are what the false-accept metric measures.
# The third element is the mood realize._decorate dresses the line in.
#
# Negatives are decorated through the *positives'* machinery on purpose. When
# only positives could carry "please", "can you", " for me" or upper case,
# those markers became evidence of a command in their own right -- measurably
# so: negatives were 30% of the corpus and 15% of its ALL-CAPS rows. Every
# register either side gets, both sides must get.
_BANKS = [
    (OTHER_DOMAIN, 14, "imperative"),
    (CHITCHAT, 10, "neutral"),
    (META_QUESTIONS, 16, "question"),
    (OPINIONS, 10, "neutral"),
    (TRUNCATED, 22, "imperative"),
]
# out_of_range's share is not redistributed here -- it moved to the positives
# entirely (frames.sample_pin_number / sample_interval).
_GEN = [(unsupported_capability, 18, "imperative"),
        (conditional_wrapper, 12, "imperative"),
        (offdomain_question, 10, "question"),
        (relative_reference, 8, "imperative"),
        # Weighted like the near-misses they are: both were silent wrong actions
        # before, not merely unhandled input.
        (pin_range, 10, "imperative"),
        (duration, 12, "imperative"),
        (exception_set, 10, "imperative"),
        (gibberish, 4, "terse")]


def sample_deferred(rng: random.Random) -> str:
    """Negatives for capability this device is *going* to have.

    Sampled and tagged apart from the permanent negatives so a version bump
    drops them in one move rather than picking them out of a mixed pool.
    Already decorated by realize._decorate, so unlike sample_negative this
    needs no further dressing.

    v2.0 dropped name_targeted() from here: a command aimed at a name is now a
    positive with a Name target, and leaving it in the negative pool would
    train both answers for one sentence. v2.1 dropped alias_request() for the
    same reason -- naming a pin is the <alias> action now.

    Nothing is deferred any more, so this returns None and build_corpus.py has
    no deferred share left to fill. Both generators stay in this file: they are
    still the cleanest way to produce name-shaped and alias-shaped utterances
    for probing a trained model, which is a different job from training it.
    """
    return None


def sample_negative(rng: random.Random) -> str:
    banks_w = sum(w for _, w, _ in _BANKS)
    gen_w = sum(w for _, w, _ in _GEN)
    if rng.random() < banks_w / (banks_w + gen_w):
        bank, mood = rng.choices([(b, m) for b, _, m in _BANKS],
                                 weights=[w for _, w, _ in _BANKS])[0]
        s = rng.choice(bank)
    else:
        fn, mood = rng.choices([(f, m) for f, _, m in _GEN],
                               weights=[w for _, w, _ in _GEN])[0]
        s = fn(rng)

    return realize._decorate(s, rng, mood)


_VARY_PREFIX = ["", "please ", "can you ", "could you ", "hey ", "ok ",
                "i want you to ", "go ahead and ", "just ", "now ", "so "]
_VARY_SUFFIX = ["", " please", " now", " for me", " thanks", " ok?"]


def vary(text: str, rng: random.Random) -> str:
    """Re-dress an utterance with the same decoration the positives get.

    Used to stretch a small pool of real near-miss negatives across a larger
    share of the corpus without duplicating identical strings.
    """
    s = text[0].lower() + text[1:] if text else text
    s = s.rstrip("?.!")
    s = rng.choice(_VARY_PREFIX) + s + rng.choice(_VARY_SUFFIX)
    return realize.case_and_punctuate(" ".join(s.split()), rng, q=0.10)


if __name__ == "__main__":
    rng = random.Random(3)
    print("--- permanent ---")
    for _ in range(20):
        print(" ", sample_negative(rng))
    print("--- v1 deferred (v2 accepts these) ---")
    for _ in range(20):
        print(" ", sample_deferred(rng))
