# Off-domain eval generation prompt

Paste everything below the line into a fresh Google AI Studio chat. Run it
**twice**, in two separate chats, and save the replies to
`data/eval_raw/offdomain_1.txt` and `data/eval_raw/offdomain_2.txt`.

These are the items the false-accept number is computed over. The current set
has 19 usable refusals on the MASSIVE side, where one item moves the metric by
5.3% — the number is not measuring anything. 500 of these fixes that.

The hard part of this prompt is the exclusion in "What is NOT a refusal". A
generator asked for "things the device can't do" reliably produces
`turn on the TV` and `switch off the playlist`, which are valid commands whose
targets happen not to exist. Training or scoring those as refusals teaches the
model to reject unfamiliar nouns, which is precisely the failure mode
`docs/GRAMMAR.md` is written to prevent. `ingest_eval.py` re-checks this
automatically and quarantines anything command-shaped.

---

I am testing a natural-language command parser for a microcontroller. It has a
deliberately narrow capability surface, and when a request falls outside it the
correct behaviour is to refuse rather than guess — there is no cloud fallback
and no way to ask a clarifying question. I need a held-out set of requests it
**should** refuse.

## What the device can do

It drives 25 GPIO pins on an ESP32-S3. Legal pin numbers are exactly:

    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 21 38 39 40 41 42 48

Each pin is fully on or fully off — nothing in between. It can: switch a target
on/off/toggle it, report a target's state, blink a target, run a chase across
2–6 pins, stop a running blink or chase, and assign a name to pins. Blink and
chase rates are in milliseconds, 50 to 10000 inclusive.

A target may be named rather than numbered. The parser does **not** know which
names exist — name resolution happens later in a lookup table.

## What is NOT a refusal — read this before generating

Any request that fits the six actions above is in-domain **no matter what the
target is called**. The parser's job is to recognise the structure; whether the
name resolves to a real pin is decided elsewhere.

So none of these are refusals, and none may appear in your output:

- `turn on the television` — a `set` on a name that may or may not exist
- `switch off my espresso machine` — same
- `is the garage door open` — a `read` on a name
- `flash the beacon every second, ten times` — a legal `blink`
- anything else of the shape *verb + target + on/off*, or *blink/flash/chase/
  stop/name + target*, regardless of how implausible the target is

If you find yourself writing "the device doesn't control that kind of thing" —
stop, it does, or at least it will happily try. The refusal must come from the
**verb or the structure**, not from the noun.

## What to generate

**250 requests**, one per line, in these categories:

| # | category | count | note |
|---|---|---:|---|
| 1 | Partial brightness / analog output | 30 | dimming, percentages, fading, PWM duty cycles, specific voltages, "half power" |
| 2 | Colour | 25 | RGB, hue, colour temperature, "make it warm white" |
| 3 | Scheduling and delays | 30 | "in ten minutes", "at 6pm", "every morning", "after an hour", countdowns |
| 4 | Conditionals and automation | 30 | "if/when/while X then Y", sensor triggers, linking two pins, loops |
| 5 | Reference to earlier turns | 20 | "do that again", "undo", "the one I just turned on" — there is no memory between commands |
| 6 | Sensing the device cannot do | 20 | temperature, humidity, current draw, measuring a voltage, motion, "how much power" |
| 7 | Numerically illegal but well-formed | 30 | a pin number outside the list above; a blink rate below 50ms or above 10000ms; a chase across 7 or more pins. Keep the phrasing completely ordinary — the only thing wrong is the number. |
| 8 | Incomplete input | 30 | cut off mid-slot: a trailing preposition, a verb with no target, a rate with no number. Must be genuinely unfinished, not merely terse — "blink 4" is a complete command and must not appear here. |
| 9 | Not a device request at all | 35 | weather, arithmetic, trivia, translation, recipes, jokes, writing tasks, general chit-chat |

Requirements:

- Mix registers as in real speech: curt, polite, spoken, run-on, all-caps,
  occasional typos. Roughly a fifth should read as spoken rather than typed.
- Keep every request under about 15 words.
- Categories 1–7 should sound like a person genuinely trying to use *this*
  device and overreaching, not like a test case.
- Do not repeat yourself. 250 distinct requests.

## Output format

One JSON object per line, no surrounding prose, no markdown fence:

    {"text": ..., "category": ...}

`category` is the integer 1–9 from the table. Format examples only, do not
reuse them:

    {"text": "take the workshop lamp down to about a third", "category": 1}
    {"text": "cut power to the mister an hour from now", "category": 3}
    {"text": "chase 3 4 5 6 7 8 9 10", "category": 7}
    {"text": "switch the", "category": 8}

Output the 250 lines and nothing else.
