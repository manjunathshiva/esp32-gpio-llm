# In-domain eval generation prompt

Paste everything below the line into a fresh Google AI Studio chat. Run it
**twice**, in two separate chats, and save the replies to
`data/eval_raw/indomain_1.txt` and `data/eval_raw/indomain_2.txt`.

Two separate chats matters: a follow-up turn in the same chat sees the first
200 lines and will steer away from them, which correlates the two runs. They
must be independent draws so the dev/locked split is exchangeable.

---

I am testing a natural-language command parser for a microcontroller. I need a
held-out evaluation set of realistic user utterances. Generate the utterances
yourself — do not ask me for examples, and do not reuse the sample sentences
below for anything except learning the output format.

## What the device can do

It drives 25 GPIO pins on an ESP32-S3. Legal pin numbers are exactly:

    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 21 38 39 40 41 42 48

A pin can be referred to by number, or by an alias the user has previously
given it ("kitchen light", "warning_led", "pump 2"). The parser does not know
which aliases exist — resolving a name to a pin happens later, in a lookup
table. So **any device or room name is a legal target**, including ones you
invent.

Six things it can be asked to do:

| action | what it does | slots |
|---|---|---|
| `set`   | drive one target high, low, or toggle it | one target, a level |
| `read`  | report the current state of one target | one target |
| `blink` | flash one target on and off repeatedly | one target, optional rate + repeat count |
| `seq`   | run a chase/sweep across 2–6 pins in order | 2–6 pins, optional rate + repeat count |
| `stop`  | stop a running blink or chase | zero or one target |
| `alias` | give a name to 1–6 pins | 1–6 pins, a name |

Constraints:

- Blink/chase rate is in milliseconds, between **50 and 10000** inclusive.
- Repeat count `0` means "keep going forever".
- Rate and count travel together. If an utterance states a rate, it states a
  count too (use `0` for forever). If it states neither, both are null.
  **Never write an utterance that gives a repeat count without a rate** —
  "blink pin 4 five times" is outside this grammar and must not appear.
- A chase runs across **pin numbers only**, never names, and 2–6 of them.
- `alias` targets are pin numbers only.
- A target can also be *every* pin at once ("all the pins", "everything").
  `read` cannot take that; the others can where it makes sense.

## What to generate

**200 utterances**, one per line, in this distribution:

| action | count |
|---|---:|
| `set`   | 70 |
| `read`  | 30 |
| `blink` | 30 |
| `seq`   | 25 |
| `stop`  | 20 |
| `alias` | 25 |

Cross-cutting requirements:

- **At least 90 of the 200 must target a name rather than a pin number.**
  Real users mostly say "the porch light", not "pin 41". Names should be
  varied: rooms, appliances, colours, underscored identifiers, names with
  digits in them, one- and multi-word, and some you invent outright.
- 8–12 utterances should target all pins at once.
- Mix registers heavily: curt fragments, full polite sentences, questions,
  spoken-aloud phrasing with filler, shouted all-caps, missing punctuation,
  and roughly 10 with realistic typos. Do not make them uniformly clean —
  clean text is the easy case and I already measure it.
- Vary sentence *structure*, not just vocabulary. Put the level before the
  target sometimes and after it other times; front some with the target.
- Keep every utterance under about 15 words.
- Every utterance must be unambiguously one of the six actions. If a sentence
  could plausibly be two different frames, do not include it.

## Output format

One JSON object per line, no surrounding prose, no markdown fence, no commas
between lines. Every object has all seven keys, `null` where unused:

    {"text": ..., "action": ..., "targets": [...], "level": ..., "interval_ms": ..., "count": ..., "alias_name": ...}

- `action` — one of `set` `read` `blink` `seq` `stop` `alias`
- `targets` — a list. Each entry is an **integer** for a pin number, a
  **string** for an alias name, or the exact string `"ALL"` for all pins.
  `"ALL"` is a literal — never an empty string, never a paraphrase.
- `level` — `"high"`, `"low"`, `"toggle"`, or null. Only `set` uses it.
- `interval_ms` — integer 50–10000, or null
- `count` — integer, `0` for forever, or null
- `alias_name` — string, only for `alias`

**A name in `targets` or `alias_name` must appear verbatim in `text`.** Copy
the exact span, same spelling and case-insensitively identical — including any
typo you introduced. Do not normalise, expand, or correct it. If the user says
"the porch lite", the target is `porch lite`. Strip the leading article: "the
porch light" gives `porch light`, not `the porch light`. Strip trailing
politeness: "kitchen fan please" gives `kitchen fan`.

Format examples only — these are deliberately dull, do not imitate their
phrasing or reuse them:

    {"text": "set pin 12 low", "action": "set", "targets": [12], "level": "low", "interval_ms": null, "count": null, "alias_name": null}
    {"text": "what is the attic vent doing", "action": "read", "targets": ["attic vent"], "level": null, "interval_ms": null, "count": null, "alias_name": null}
    {"text": "blink 7 every 250ms four times", "action": "blink", "targets": [7], "level": null, "interval_ms": 250, "count": 4, "alias_name": null}
    {"text": "sweep 38 39 40 41", "action": "seq", "targets": [38, 39, 40, 41], "level": null, "interval_ms": null, "count": null, "alias_name": null}
    {"text": "stop", "action": "stop", "targets": [], "level": null, "interval_ms": null, "count": null, "alias_name": null}
    {"text": "pin 5 is now the shed heater", "action": "alias", "targets": [5], "level": null, "interval_ms": null, "count": null, "alias_name": "shed heater"}

Output the 200 lines and nothing else.
