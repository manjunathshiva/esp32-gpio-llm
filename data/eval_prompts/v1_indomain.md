# v1 in-domain prompt — run in a fresh AI Studio chat, twice

---

You are helping build a held-out test set for a small on-device command parser.

The device is an ESP32 that controls **GPIO pins by number**. It has no screen,
no network, and no concept of rooms or device names. Usable pins are
1–18, 21, 38–42 and 48.

It understands exactly five things:

| action | meaning |
|---|---|
| `set` | drive one or more pins high, low, or toggle them |
| `read` | report the current level of one or more pins |
| `blink` | flash pins on and off |
| `seq` | run a chase across 2 or more pins in order |
| `stop` | end any running blink or chase |

## What I need

Write **200 lines** that a real person might type at a serial console to do one
of those five things, each with its parse.

One JSON object per line, nothing else — no numbering, no commentary, no code
fences:

```
{"text": "turn on pin 4", "action": "set", "targets": [4], "level": "high", "interval_ms": null, "count": null}
```

Fields:

- **`text`** — what the person typed.
- **`action`** — one of `set`, `read`, `blink`, `seq`, `stop`.
- **`targets`** — a list of pin numbers, e.g. `[4]` or `[4,5,6]`. For "everything"
  / "all pins", write the single literal `["ALL"]`. For a bare `stop` with no
  target, write `[]`.
- **`level`** — `"high"`, `"low"` or `"toggle"` for `set`; `null` otherwise.
- **`interval_ms`** — the rate in milliseconds, or `null` if none was said.
  Convert units: "every 2 seconds" → `2000`, "twice a second" → `500`.
- **`count`** — how many times, or `null` if not said. "forever" / "keep going"
  → `0`.

## Rules per action

- `set` — needs a level. Never has timing.
- `read` — no level, no timing. Never targets `["ALL"]`.
- `blink` — a rate **and** a count, **or** a count alone ("blink pin 4 five
  times" → `interval_ms: null, count: 5`), or neither. Never a rate with no
  count: if a rate is said and no count, use `count: 0`.
- `seq` — at least 2 pins. Same timing rules as `blink`. Never `["ALL"]`.
- `stop` — `targets` is `[]` (everything) or a list of pins. No level, no timing.

## Quotas

| | lines |
|---|---|
| `set`, single pin | 45 |
| `set`, several pins | 30 |
| `set`, all pins | 12 |
| `read` | 25 |
| `blink`, with a rate | 25 |
| `blink`, count only | 12 |
| `blink`, no timing | 8 |
| `seq` | 25 |
| `stop` | 18 |

## Make it look like real typing

Vary hard. Roughly a third should be terse ("off 12", "4 5 6 on", "blink 9 x3"),
a third conversational ("could you please turn pin 21 off"), a third somewhere
between. Include:

- different ways to name a pin: `pin 4`, `GPIO4`, `gpio 4`, `#4`, `io 4`, `p4`,
  bare `4`, and occasionally spelled out (`pin seven`)
- typos and abbreviations: `trun on`, `pls`, `plz`, `u`, `swithc`
- ALL CAPS lines, and lines with no capitals or punctuation at all
- filler and politeness: `um,`, `hey`, `thanks`, `for me`, `right now`
- trailing noise the parser should ignore: `in the garage`, `upstairs`

## Do not

- **Do not use device or room names.** No "the desk lamp", no "kitchen light".
  Every target is a pin number, or "all pins". Name-targeted commands are a
  *separate* test set — putting them here would corrupt both.
- **Do not invent numbers.** Every number in `targets`, `interval_ms` and
  `count` must be traceable to something in `text`. If the text says "quickly"
  with no number, use `interval_ms: null`.
- **Do not use pins outside 1–18, 21, 38–42, 48** in this file. Out-of-range
  pins are a separate set.
- **Do not repeat a sentence pattern more than three times.** If you notice
  yourself writing "turn on pin N" over and over, switch register.
