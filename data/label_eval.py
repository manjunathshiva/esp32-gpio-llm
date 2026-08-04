"""Label the held-out evaluation sets.

Two sources, kept separate on purpose because they support different claims:

  massive  374 on/off utterances written by MASSIVE crowdworkers. Genuinely
           human, genuinely independent -- they predate this project entirely.
           Narrow: on/off only, and targets are rooms and devices, not pins.
  gemini   ~250 lines from Gemini given only the capability list, never any
           phrasing from realize.py. An independence check, NOT a human
           baseline: Gemini and this project's templates both draw on the same
           underlying distribution of English commands.

**Labelling is done here, by the same author as the generator, and that is
fine.** The rule is that the *phrasing* must be independent. What would
invalidate the set is writing the sentences, not scoring them.

Anything the rules cannot label confidently is marked NEEDS_REVIEW and excluded
rather than guessed -- wrong gold is worse than a smaller eval set.

**v1 changed what these rules are allowed to reject.** A pin number outside the
board allowlist used to make a line unlabellable, because the grammar had no
symbol for it. It is now an ordinary label: the model transcribes the digits and
frames.range_check() refuses the frame. So `_pin` no longer filters, the
interval bounds no longer gate, and a chase of nine pins gets a nine-pin frame.
Without that these rules would score the *old* behaviour -- treating "turn on
pin 100" as unlabellable is how a substitution stays invisible.

The other direction is DEFERRED: a name target or an alias request is
well-formed and simply not in v1. It is not NEEDS_REVIEW (the rules understood
it fine) and it is not a permanent refusal (v2 accepts it), so it gets its own
sentinel and the caller decides.

    uv run python data/label_eval.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import frames
from frames import Action, All, Frame, Level, Pin

HERE = Path(__file__).parent
OUT = HERE / "eval"

# Capability v2 restores. Distinct from NEEDS_REVIEW, which means the rules
# could not read the sentence at all.
DEFERRED = "DEFERRED"

# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

ON_W = r"(?:on|high|1)"
OFF_W = r"(?:off|low|0)"

_NUM = r"(\d+)"
_PINREF = r"(?:pin|gpio|io|p)?\s*" + _NUM


def _pin(n: int) -> Pin:
    """Any number a speaker put after "pin" is a pin reference. Whether the
    board has it is range_check()'s call, not the labeller's -- see the module
    docstring."""
    return Pin(n)


# Names may contain digits ("light2"), so the guard is that the target is not
# *purely* a number -- a bare number is a pin, not a name.
#
# "is alarm_buzzer on" is a read question, not a set on a name called "is
# alarm_buzzer"; an earlier version labelled it the latter and marked the model
# wrong for answering correctly. A capture starting with one of our own action
# verbs is the same mistake: "run a chase on" and "can you turn off" are
# truncated commands, and reading them as aliases called "run a chase" and
# "turn" put them in the eval gold as valid commands.
_NOT_A_NAME = re.compile(
    r"^(?:is|are|was|check|what|whats|read|get|show|tell|how|does|do|"
    r"run|turn|switch|set|make|shut|blink|flash|strobe|pulse|stop|halt|"
    r"cancel|kill|toggle|flip|name|alias|call|label|rename|chase|sweep|"
    r"sequence)\b")

# A capture made entirely of function words is a truncated utterance, not a
# name -- without this, "toggle the", "switch off my" and "name pin 3 to", the
# whole truncated-input negative class, parse as commands on aliases called
# "the", "my" and "to".
_STOPWORDS = {"the", "a", "an", "my", "our", "that", "this", "it", "its",
              "to", "as", "on", "off", "in", "of", "at", "for", "if",
              "and", "or", "me", "pin", "pins", "gpio", "io", "please",
              "up", "down", "back", "some", "any", "all"}

# A name ending in a preposition or article is a cut-off slot: "read the state
# of", "stop blinking on".
_DANGLING_TAIL = {"the", "a", "an", "my", "our", "that", "this", "to", "as",
                  "of", "in", "at", "for", "and", "or", "with", "from"}


def _named(x: str) -> bool:
    x = x.strip()
    if not x or re.fullmatch(r"[\d\s]+", x) or _NOT_A_NAME.match(x):
        return False
    words = re.findall(r"[a-z0-9_]+", x)
    if not words or words[-1] in _DANGLING_TAIL:
        return False
    return any(w not in _STOPWORDS for w in words)


_WORD_NUM = {
    "once": 1, "twice": 2, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50,
    "a hundred": 100, "hundred": 100, "a couple": 2, "a few": 3,
}
# Words that announce a repeat count. If one is present and no number came out,
# the rules misread the utterance and must say so.
_COUNT_WORD = re.compile(r"\b(times?|cycles?|blinks?|flashes|repeats?|rounds?|"
                         r"passes|sweeps?|loops?)\b", re.I)
# Words that announce a rate. Same idea as _COUNT_WORD, for the interval slot.
_RATE_WORD = re.compile(r"\b(every|at|interval|speed|rate|per)\b", re.I)

# Count could not be determined. Callers turn this into NEEDS_REVIEW rather than
# a guess -- an earlier version silently defaulted to 0 (forever), which
# mislabelled every spelled-out count as infinite and marked the model wrong for
# reading "four times" correctly.
UNREADABLE = -1


def _interval_count(rest: str) -> tuple[int | None, int | None]:
    """Pull an interval and a cycle count out of Gemini's compact tail forms:
    '500ms 10', 'at 800ms for 10 times', 'speed 300ms count 20', '100ms forever',
    'every 250ms four times'.
    """
    ms = re.search(r"(\d+)\s*ms", rest)
    if not ms:
        ms = re.search(r"(?:interval|speed|every|at)\s*(\d+)", rest)
    interval = int(ms.group(1)) if ms else None

    if re.search(r"\b(forever|indefinitely|continuous(?:ly)?|non-?stop)\b", rest):
        return interval, 0

    c = re.search(r"(?:for\s+)?(\d+)\s*(?:times?|x|count)", rest)
    if not c:
        c = re.search(r"count\s*(\d+)", rest)
    if not c and ms:
        c = re.search(r"\b(\d+)\b", rest[ms.end():])
    if c:
        return interval, int(c.group(1))

    for word, n in _WORD_NUM.items():
        if re.search(rf"\b{re.escape(word)}\b", rest, re.I):
            return interval, n
    if _COUNT_WORD.search(rest):
        return interval, UNREADABLE
    # "blink pin 5 every", "blink 10 at speed" -- the rate slot is announced and
    # then cut off. Reading these as an untimed blink turned the whole truncated
    # class into valid commands.
    if interval is None and _RATE_WORD.search(rest):
        return UNREADABLE, None
    return interval, None


# Scheduling and conditionals -- neither is in the grammar, at any version, so
# these are permanent refusals. The guard has to run *before* the slot rules,
# because a clock time reads as a rate: "blink pin 4 every morning at 8am" came
# out of _interval_count as an 8ms interval, which would have filed a schedule
# request in the eval gold as a command the runtime merely range-refuses.
#
# "every" alone is not a trigger -- "every 300ms" and "every two seconds" are
# the normal way to say a rate. Only "every <calendar word>" is. Likewise
# "until" is left out: "until i say stop" is realize.py's phrasing for count 0.
_SCHEDULE = re.compile(
    r"\b(?:if|when|whenever|while|unless|tomorrow|tonight|sunset|sunrise|"
    r"midnight|noon|o'?clock|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)\b"
    r"|\bevery\s+(?:morning|evening|night|day|hour|week|month|other)\b"
    r"|\b(?:in|after|within)\s+(?:a|an|\d+)\s*"
    r"(?:second|minute|hour|day|week)s?\b"
    # A duration as an end condition -- "blink pin 4 for 10 seconds". The tool
    # counts cycles, not time. Units are required so "for 5 cycles", which is a
    # legal count, does not match.
    r"|\bfor\s+(?:a|an|\d+|half|the\s+next)\s*\w*\s*"
    r"(?:second|sec|minute|min|hour)s?\b"
    r"|\d\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b", re.I)


def label_gemini(s: str) -> Frame | str:
    """Rule-based label for one Gemini line, or NEEDS_REVIEW."""
    t = s.strip().rstrip("?.!").strip()
    low = re.sub(r"\bpls\b|\bplease\b|\bcan u\b|\bcan you\b|\bnow\b", " ", t.lower())
    low = " ".join(low.split())
    if not low:
        return "NEEDS_REVIEW"
    if _SCHEDULE.search(low):
        return "NEEDS_REVIEW"

    # --- alias: name/alias/call/label/rename <pin> <name> --------------------
    # v1 has no alias action, so a well-formed alias request is DEFERRED, not
    # NEEDS_REVIEW. Still matched precisely: a cut-off alias ("name pin 3 to")
    # is a truncated utterance and belongs in the permanent refusals.
    m = re.match(r"^(?:name|alias|call|label|rename)\s+"
                 r"((?:(?:pin|gpio|io|p)?\s*\d+[\s,]*(?:and\s+)?)+)"
                 r"(?:to\s+|as\s+|name\s+)?(.+)$", low)
    if not m:
        m = re.match(rf"^assign\s+name\s+(.+?)\s+to\s+{_PINREF}$", low)
        if m:
            return DEFERRED if _named(m.group(1)) else "NEEDS_REVIEW"
    if m:
        ps = re.findall(r"\d+", m.group(1))
        return DEFERRED if ps and _named(m.group(2).strip()) else "NEEDS_REVIEW"
    m = re.match(rf"^set\s+{_PINREF}\s+name\s+(.+)$", low)
    if m:
        return DEFERRED if _named(m.group(2)) else "NEEDS_REVIEW"

    # --- stop ----------------------------------------------------------------
    # The stop verb is required. With it optional, a bare "chase" or "blink" --
    # a truncated command -- matched the empty tail and became <stop>.
    if re.fullmatch(r"(?:stop|halt|cancel|kill|end|disable)\s*"
                    r"(?:all|everything|chase|chasing|sequence|blink(?:ing)?)?", low):
        return Frame(Action.STOP)
    m = re.match(r"^(?:stop|halt|cancel|kill|end|disable)\s+"
                 rf"(?:blink(?:ing)?|chase|sequence)?\s*(?:on\s+)?{_PINREF}$", low)
    if m:
        return Frame(Action.STOP, [_pin(int(m.group(1)))])

    # --- chase ---------------------------------------------------------------
    m = re.match(r"^(?:chase|cycle|sweep|sequence|run)\s+(?:across\s+|through\s+)?"
                 r"(?:pins?\s+|gpios?\s+)?((?:\d+[\s,]*)+)$", low)
    if m:
        ps = [_pin(int(x)) for x in re.findall(r"\d+", m.group(1))]
        # Only the lower bound gates the label. A chase of nine pins is a frame
        # the model must produce so range_check can refuse it; calling it
        # unlabellable is how "chase 1 2 3 4 5 6 7 8 9" silently became a
        # six-pin chase that ran.
        if len(ps) < 2:
            return "NEEDS_REVIEW"
        # No timing given, so the frame carries none -- the device default
        # applies. Inventing 200/0 here would demand the model hallucinate
        # numbers the user never said.
        return Frame(Action.SEQ, ps)

    # --- blink ---------------------------------------------------------------
    m = re.match(rf"^(?:blink|flash|strobe|pulse)\s+{_PINREF}\s*(.*)$", low)
    if not m:
        m2 = re.match(rf"^make\s+{_PINREF}\s+(?:blink|flash)\s*(.*)$", low)
        m = m2
    if m:
        p = _pin(int(m.group(1)))
        iv, ct = _interval_count(m.group(2))
        if UNREADABLE in (iv, ct):
            return "NEEDS_REVIEW"
        # A count with no rate ("blink pin 9 a few times") is now representable:
        # it means that many cycles at the device default rate. It used to be
        # NEEDS_REVIEW, which kept the whole shape out of the eval while the
        # model was answering it by inventing a rate.
        if iv is None:
            # A count with no rate means that many cycles at the device
            # default. Returning a bare frame here dropped the count entirely,
            # so "blink pin 9 five times" reparsed as an untimed blink and the
            # cross-check flagged correct gold as a disagreement.
            return Frame(Action.BLINK, [p], count=ct) if ct is not None \
                else Frame(Action.BLINK, [p])
        # An interval outside the device bounds is labelled, not rejected: 12000
        # is what the speaker said, so it is what the model must emit.
        return Frame(Action.BLINK, [p], interval_ms=iv, count=ct if ct is not None else 0)

    # --- read ----------------------------------------------------------------
    m = re.match(rf"^(?:read|check|get\s+status|status(?:\s+of)?|state(?:\s+of)?)\s+"
                 rf"(?:if\s+)?{_PINREF}(?:\s+is\s+\w+)?$", low)
    if not m:
        m = re.match(rf"^{_PINREF}\s+(?:state|status)$", low)
    if not m:
        m = re.match(rf"^(?:is|check\s+if)\s+{_PINREF}\s+(?:on|off|high|low|active)$", low)
    if m:
        return Frame(Action.READ, [_pin(int(m.group(1)))])

    # --- toggle --------------------------------------------------------------
    m = re.match(rf"^(?:toggle|flip)\s+{_PINREF}$", low)
    if m:
        return Frame(Action.SET, [_pin(int(m.group(1)))], level=Level.TOGGLE)

    # --- set on/off, in every order Gemini produced --------------------------
    # "trun off 15" -- Gemini included typos, which is exactly the register a
    # generated corpus lacks. Worth labelling rather than discarding.
    verb = r"(?:turn|trun|tunr|switch|set|make|shut|swithc)"
    tail = r"(?:\s+(?:now|fast|quick(?:ly)?|immediately|asap))?"
    for pat, lvl in (
        (rf"^{verb}?\s*{ON_W}\s+{_PINREF}{tail}$", Level.HIGH),
        (rf"^{verb}?\s*{OFF_W}\s+{_PINREF}{tail}$", Level.LOW),
        (rf"^{verb}\s+{_PINREF}\s+(?:to\s+|go\s+)?{ON_W}{tail}$", Level.HIGH),
        (rf"^{verb}\s+{_PINREF}\s*(?:to\s+|go\s+)?{OFF_W}{tail}$", Level.LOW),
        (rf"^shut\s+down\s+{_PINREF}{tail}$", Level.LOW),
        (rf"^{_PINREF}\s+(?:to\s+|go\s+)?{ON_W}{tail}$", Level.HIGH),
        (rf"^{_PINREF}\s+(?:to\s+|go\s+)?{OFF_W}{tail}$", Level.LOW),
    ):
        m = re.match(pat, low)
        if m:
            return Frame(Action.SET, [_pin(int(m.group(1)))], level=lvl)

    # --- every pin at once ---------------------------------------------------
    # Must come before the name-targeted block below, which would otherwise
    # read "everything off" as a set on an alias literally called "everything".
    # The first batch contained no such utterance so the gap went unnoticed;
    # indomain.md now asks for 8-12 per run.
    _ALL = (r"(?:everything|every\s+pin|all\s+(?:the\s+|of\s+)?(?:pins?|"
            r"channels?|outputs?|them|lines?)|all)")
    for pat, lvl in ((rf"^{verb}?\s*{ON_W}\s+{_ALL}{tail}$", Level.HIGH),
                     (rf"^{verb}?\s*{OFF_W}\s+{_ALL}{tail}$", Level.LOW),
                     (rf"^{verb}\s+{_ALL}\s+(?:to\s+|go\s+)?{ON_W}{tail}$", Level.HIGH),
                     (rf"^{verb}\s+{_ALL}\s+(?:to\s+|go\s+)?{OFF_W}{tail}$", Level.LOW),
                     (rf"^{_ALL}\s+(?:to\s+|go\s+)?{ON_W}{tail}$", Level.HIGH),
                     (rf"^{_ALL}\s+(?:to\s+|go\s+)?{OFF_W}{tail}$", Level.LOW)):
        if re.match(pat, low):
            return Frame(Action.SET, [All()], level=lvl)
    if re.match(rf"^(?:toggle|flip)\s+{_ALL}$", low) or \
       re.match(rf"^{_ALL}\s+toggle$", low):
        return Frame(Action.SET, [All()], level=Level.TOGGLE)
    m = re.match(rf"^(?:blink|flash|strobe|pulse)\s+{_ALL}\s*(.*)$", low)
    if m:
        iv, ct = _interval_count(m.group(1))
        if UNREADABLE in (iv, ct):
            return "NEEDS_REVIEW"
        # A count with no rate ("blink pin 9 a few times") is now representable:
        # it means that many cycles at the device default rate. It used to be
        # NEEDS_REVIEW, which kept the whole shape out of the eval while the
        # model was answering it by inventing a rate.
        if iv is None:
            return Frame(Action.BLINK, [All()])
        return Frame(Action.BLINK, [All()], interval_ms=iv,
                     count=ct if ct is not None else 0)

    # --- name-targeted forms: "pump on", "turn off warning_led", "light2 high"
    # Names may contain digits ("light2"), so the guard is that the target is
    # not *purely* a number -- a bare number is a pin, not a name.
    # "is alarm_buzzer on" is a read question, not a set with the name
    # "is alarm_buzzer" -- an earlier version labelled it the latter and marked
    # the model wrong for answering correctly.
    # (see _named at module scope)

    # Every branch below recognises the sentence perfectly well and then returns
    # DEFERRED, because v1 has no Name target. They are kept -- rather than
    # deleted so the lines fall through to NEEDS_REVIEW -- for two reasons: the
    # caller has to be able to tell "a command v2 will accept" from "the rules
    # could not read this", and v2 restores these bodies unchanged.
    for pat in (rf"^([a-z][a-z0-9_ ]*?)\s+{ON_W}{tail}$",
                rf"^([a-z][a-z0-9_ ]*?)\s+{OFF_W}{tail}$",
                rf"^{verb}\s+{ON_W}\s+([a-z][a-z0-9_ ]*?){tail}$",
                rf"^{verb}\s+{OFF_W}\s+([a-z][a-z0-9_ ]*?){tail}$",
                r"^(?:toggle|flip)\s+([a-z][a-z0-9_ ]*)$",
                r"^([a-z][a-z0-9_ ]*?)\s+toggle$",
                r"^(?:read|check)\s+([a-z][a-z0-9_ ]*)$",
                r"^is\s+([a-z][a-z0-9_ ]*?)\s+(?:on|off|high|low)$",
                r"^(?:stop|halt|cancel|kill|end|disable)\s+"
                r"(?:blink(?:ing)?|chase|sequence)?\s*(?:on\s+)?(?:the\s+)?"
                r"([a-z][a-z0-9_ ]*)$"):
        m = re.match(pat, low)
        if m and _named(m.group(1)):
            return DEFERRED

    # Blink on a name. The timing tail still has to parse: "blink the porch
    # light every" is a truncated utterance, which is a permanent refusal, not
    # deferred capability.
    m = re.match(r"^(?:blink|flash|strobe|pulse)\s+(?:the\s+)?"
                 r"([a-z][a-z0-9_ ]*?)"
                 r"(\s+(?:\d.*|forever.*|every\b.*|at\b.*|for\b.*))?$", low)
    if m and _named(m.group(1)):
        iv, ct = _interval_count(m.group(2) or "")
        if UNREADABLE in (iv, ct) or (iv is None and ct is not None):
            return "NEEDS_REVIEW"
        return DEFERRED

    return "NEEDS_REVIEW"


def parse_gemini(path: Path) -> tuple[list[str], list[str]]:
    raw = json.loads(path.read_text())["response"]
    head, _, tail = raw.partition("REFUSED REQUESTS:")
    pos = [l.strip() for l in head.splitlines() if l.strip()]
    neg = [l.strip() for l in tail.splitlines() if l.strip()]
    return pos, neg


# --------------------------------------------------------------------------
# MASSIVE
# --------------------------------------------------------------------------

_M_LEAD = re.compile(
    r"^(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|will\s+you\s+|"
    r"i\s+want\s+(?:you\s+to\s+)?|i\s+need\s+(?:you\s+to\s+)?|i'?d\s+like\s+"
    r"(?:you\s+to\s+)?|hey\s+|ok\s+|just\s+|go\s+ahead\s+and\s+|kindly\s+|"
    r"could\s+u\s+|can\s+u\s+|pls\s+|plz\s+)+", re.I)
# Politeness and purpose clauses that are not part of the target name. Leaving
# them in produced gold like "kitchen light for me".
_M_TRAIL = re.compile(
    r"(?:\s+(?:please|thanks|thank\s+you|for\s+me|now|right\s+now|"
    r"immediately|ok|okay))+$", re.I)
_M_DET = re.compile(r"^(?:the|my|a|an|our|that|this)\s+", re.I)

_ON_V = r"(?:turn|switch|put|power|flip|kick|shift|make|set|get)"
_OFF_V = r"(?:turn|switch|put|power|shut|flip|kick|cut|shift|make|set)"
# Single-word verbs that carry the level themselves.
_M_ON_SOLO = re.compile(r"^(?:activate|enable|energi[sz]e|start)\s+(?P<t>.+)$", re.I)
_M_OFF_SOLO = re.compile(
    r"^(?:deactivate|disable|extinguish|kill|douse)\s+(?P<t>.+)$", re.I)
_M_PUTOUT = re.compile(r"^put\s+out\s+(?P<t>.+)$", re.I)

_M_ON_PRE = re.compile(rf"^{_ON_V}\s+on\s+(?P<t>.+)$", re.I)
_M_OFF_PRE = re.compile(rf"^{_OFF_V}\s+off\s+(?P<t>.+)$", re.I)
# "turn the lights off in the bedroom" -- level mid-sentence, location trailing.
_M_ON_MID = re.compile(rf"^{_ON_V}\s+(?P<t>.+?)\s+on(?:\s+(?:in|at)\s+.+)?$", re.I)
_M_OFF_MID = re.compile(rf"^{_OFF_V}\s+(?P<t>.+?)\s+off(?:\s+(?:in|at)\s+.+)?$", re.I)
_M_BARE = re.compile(r"^(?P<t>[a-z][a-z ]*?)\s+(?P<lvl>on|off)$", re.I)

# Conditions and schedules. The grammar has neither.
_M_UNSUPPORTED = re.compile(
    r"\b(when|after|before|until|if|while|tomorrow|tonight|morning|"
    r"minutes?|hours?|o'?\s*clock|a\.?\s*m\.?|p\.?\s*m\.?|schedule|later)\b", re.I)

# Implicit requests: real instructions to a voice assistant, but they need world
# knowledge this model does not have, so refusing is the correct behaviour.
# Curated rather than inferred -- everything not matched here is excluded, not
# assumed to be a refusal.
_M_IMPLICIT = re.compile(
    r"^(?:it'?s?\s+(?:too\s+)?dark|it'?s?\s+dark|time\s+(?:to|for)\s+(?:sleep|bed)|"
    r"good\s*night|i'?m?\s+going\s+to\s+(?:sleep|bed)|i\s+(?:don'?t|do\s+not)\s+"
    r"(?:need|want)|i\s+can'?t\s+see|too\s+dark|too\s+bright|"
    r"i\s+am\s+going\s+to\s+(?:sleep|bed)|let'?s\s+go\s+to\s+(?:sleep|bed))", re.I)


# A trailing prepositional phrase is a location, not part of the device name.
# Without stripping it uniformly the gold contradicts itself: "turn off the
# light in the bathroom" would keep the location while "turn lights off in
# kitchen" drops it, and no single answer could satisfy both.
# "of the bathroom" is a location just as much as "in the bathroom". Omitting
# "of" made the gold inconsistent: "light in the bathroom" reduced to "light"
# while "light of the bathroom" kept its phrase, and no single answer satisfied
# both.
_M_LOC = re.compile(
    r"\s+(?:in|at|on|inside|from|of)\s+(?:the\s+|my\s+|our\s+|this\s+)?\S.*$", re.I)
# "...socket to plug in my dongle" -- a purpose clause, likewise not a name.
_M_PURPOSE = re.compile(r"\s+(?:to|so|for)\s+\S+\s+\S.*$", re.I)


def _target(raw: str) -> str | None:
    t = _M_TRAIL.sub("", raw.strip().strip(",")).strip()
    t = _M_DET.sub("", t).strip()
    for rx in (_M_PURPOSE, _M_LOC):
        stripped = rx.sub("", t).strip()
        if stripped:                   # keep the head noun phrase only
            t = stripped
    t = _M_DET.sub("", t).strip()
    if not t or re.search(r"\d", t) or len(t.split()) > 6:
        return None
    return t


def label_massive(s: str) -> Frame | str:
    """Label one MASSIVE on/off line, or NEEDS_REVIEW.

    **Never defaults to UNKNOWN.** An earlier version did, and it labelled
    "put out the lights" and "activate the wemo plug socket" as refusals -- both
    are valid commands whose verbs appear in realize.py's own banks. Penalising
    the model for getting those right would have made the eval worse than
    useless. Only the curated implicit patterns become refusals; everything
    unmatched is excluded.

    **Under v1 every line this matches is DEFERRED.** MASSIVE targets are rooms
    and devices, never pin numbers, so the whole set moves from being the
    in-domain benchmark to being the refusal benchmark -- 282 name-targeted
    commands written by people who never saw this project. That is a better
    instrument than what it replaced, and it is why massive.mine_onoff() must
    stay out of the training corpus.
    """
    t = " ".join(s.split()).rstrip("?.!").strip()
    if not t:
        return "NEEDS_REVIEW"
    if _M_IMPLICIT.match(t):
        return Frame(Action.UNKNOWN)
    if _M_UNSUPPORTED.search(t):
        return Frame(Action.UNKNOWN)

    # The level each pattern carries no longer matters -- every match is
    # DEFERRED in v1 -- but the patterns themselves still have to be exact, or
    # a line that is merely unreadable would be reported as deferred capability.
    body = _M_LEAD.sub("", t).strip()
    for rx in (_M_ON_PRE, _M_ON_SOLO, _M_OFF_PRE, _M_OFF_SOLO, _M_PUTOUT,
               _M_ON_MID, _M_OFF_MID, _M_BARE):
        m = rx.match(body)
        if not m:
            continue
        return DEFERRED if _target(m.group("t")) is not None else "NEEDS_REVIEW"

    return "NEEDS_REVIEW"


# --------------------------------------------------------------------------

def rows_for(texts: list[str], labeller, source: str,
             force_unknown: bool = False) -> tuple[list[dict], list[str]]:
    """Rows plus the lines the rules would not label.

    DEFERRED becomes a refusal row carrying `v1_deferred`, not a dropped one.
    The distinction is the whole point: these must be *scored* -- v1 has to
    refuse them -- while staying flagged, because v2 flips them back to
    commands and the eval must move with the grammar rather than be rebuilt.
    """
    out, review = [], []
    for t in texts:
        f = Frame(Action.UNKNOWN) if force_unknown else labeller(t)
        deferred = f == DEFERRED
        if deferred:
            f = Frame(Action.UNKNOWN)
        elif isinstance(f, str):
            review.append(t)
            continue
        try:
            frames.validate(f)
        except frames.ParseError:
            review.append(t)
            continue
        out.append({"text": t, "symbols": frames.to_symbols(f),
                    "action": f.action.value, "source": source,
                    "v1_deferred": deferred})
    return out, review


def main() -> None:
    OUT.mkdir(exist_ok=True)
    report = {}

    # --- Gemini --------------------------------------------------------------
    gem = HERE.parent / "ai_studio_code.txt"
    if gem.exists():
        pos, neg = parse_gemini(gem)
        rp, review_p = rows_for(pos, label_gemini, "gemini")
        rn, _ = rows_for(neg, None, "gemini", force_unknown=True)
        with (OUT / "gemini.jsonl").open("w") as fh:
            for r in rp + rn:
                fh.write(json.dumps(r) + "\n")
        (OUT / "gemini_review.txt").write_text("\n".join(review_p))
        report["gemini"] = {"in_domain": len(rp), "refusals": len(rn),
                            "needs_review": len(review_p), "raw_in_domain": len(pos)}

    # --- MASSIVE -------------------------------------------------------------
    import massive
    mined = massive.mine_onoff()
    rm, review_m = rows_for(mined, label_massive, "massive")
    with (OUT / "massive.jsonl").open("w") as fh:
        for r in rm:
            fh.write(json.dumps(r) + "\n")
    (OUT / "massive_review.txt").write_text("\n".join(review_m))
    n_cmd = sum(1 for r in rm if r["action"] != "unknown")
    report["massive"] = {"in_domain": n_cmd, "refusals": len(rm) - n_cmd,
                         "needs_review": len(review_m), "raw": len(mined)}

    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}/  -- review files list what the rules would not label")


if __name__ == "__main__":
    main()
