# v1 numeric-edge prompt — run in a fresh AI Studio chat, twice

---

You are helping build a held-out test set for a small on-device command parser.

The device is an ESP32 that controls **GPIO pins by number**. Usable pins are
**1–18, 21, 38–42 and 48** — nothing else. Blink and chase rates must be between
**50 and 10000 ms**. A chase runs at most **6 pins**.

## What this file is for

Commands that are **perfectly well-formed English, and ask for a value the board
cannot do**. "Switch off pin 100" is a normal sentence about a pin that does not
exist. The parser must understand it exactly as said — pin *one hundred* — so
the hardware layer can then refuse it by name.

The failure being tested is subtle and worth stating: a parser that cannot
represent "pin 100" does not refuse, it silently answers with the nearest pin it
*can* represent, and switches pin 10 instead. So the label here is the command
**as spoken**, never a corrected or clamped version.

## What I need

**200 lines**, one JSON object per line, no numbering, no commentary, no code
fences:

```
{"text": "switch off pin 100", "action": "set", "targets": [100], "level": "low", "interval_ms": null, "count": null}
{"text": "blink pin 7 at 12000ms", "action": "blink", "targets": [7], "level": null, "interval_ms": 12000, "count": 0}
```

Fields are the same as an ordinary command: `text`, `action` (`set` / `read` /
`blink` / `seq` / `stop`), `targets` (list of pin numbers), `level`
(`high`/`low`/`toggle` or `null`), `interval_ms`, `count`.

**Record what was said.** If the text says pin 100, `targets` is `[100]`. If it
says 12000 ms, `interval_ms` is `12000`. Never substitute a legal value.

## Quotas

| category | lines | examples of the *kind* (write your own) |
|---|---|---|
| pin just outside the list | 60 | 19, 20, 22–37, 43–47, 49, 50 |
| pin far outside | 30 | 60, 99, 100, 128, 200 |
| pin 0 | 10 | |
| rate below 50 ms | 30 | 1 ms, 10 ms, 20 ms, 45 ms |
| rate above 10000 ms | 30 | 12000 ms, 20000 ms, 50000 ms, 60 seconds |
| chase longer than 6 pins | 30 | 7 to 12 pins, all legal pins |
| several pins, one illegal | 10 | "turn on pins 4, 5 and 100" |

Two-digit and three-digit pins matter most — those are where a parser drops a
digit. Make sure "100", "128" and "200" all appear several times.

## Register

Same variety as real typing: terse and conversational, `pin 19` / `gpio19` /
`#19` / bare `19`, some ALL CAPS, some with no punctuation, occasional typos.
A wrong pin number is just as likely to be typed casually as carefully.

## Do not

- **Do not clamp, round or correct.** "pin 100" is `[100]`, not `[10]`. "50000ms"
  is `50000`, not `10000`. That correction is exactly the bug under test.
- **Do not use device or room names.** Every target is a number.
- **Do not write commands the board *can* do.** Every line here must be illegal
  in at least one value — otherwise it belongs in the ordinary in-domain file.
- **Do not make them ungrammatical.** These are normal sentences. Broken or
  half-finished input belongs in the refusals file.
- **Do not mention seconds as a duration** ("for 10 seconds"). That is a
  different, unsupported thing and belongs in the refusals file. A *rate* of
  "every 60 seconds" is fine here — it is a rate, just out of bounds.
