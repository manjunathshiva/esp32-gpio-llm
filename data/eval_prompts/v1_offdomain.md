# v1 off-domain prompt — run in a fresh AI Studio chat, twice

---

You are helping build a held-out test set for a small on-device command parser.

The device is an ESP32 that controls **GPIO pins by number**. Usable pins are
1–18, 21, 38–42 and 48. It can do exactly five things: set a pin high/low/toggle,
read a pin, blink a pin, chase across pins, and stop. It has no screen, no
network, no clock, no sensors, no memory of previous commands, and no names for
anything.

## What I need

**260 lines** it must **refuse** — inputs where the right answer is "I don't
understand that". One per line, plain text, no JSON, no numbering, no
commentary, no code fences.

## Categories and quotas

| # | category | lines | the shape |
|---|---|---|---|
| 1 | named targets | 45 | "turn on the desk lamp", "switch off the bedroom light" — a real command aimed at a *name* rather than a pin number |
| 2 | dimming, brightness, analog, PWM | 30 | "set pin 4 to 50%", "dim pin 9", "read the analog voltage on pin 10", "output 2.5v" |
| 3 | colour | 15 | "make pin 4 red", "set pin 12 rgb to 0 255 0" |
| 4 | scheduling and clock | 25 | "turn on pin 4 at 8am", "switch off pin 7 in 10 minutes", "every morning" |
| 5 | duration as an end condition | 25 | "blink pin 4 for 10 seconds", "keep pin 9 high for 2 minutes" |
| 6 | pin ranges | 25 | "turn on pins 4-8", "chase pins 1 through 6" |
| 7 | all-except / subsets | 20 | "turn on all pins except 4", "the odd pins", "every other pin", "the first three pins" |
| 8 | conditionals and sensors | 20 | "if pin 4 is high turn on pin 5", "when the button is pressed" |
| 9 | truncated or half-finished | 25 | "turn on", "blink pin", "set pin 4 to", "chase 1 2 3 rate", "read the state of" |
| 10 | referring to earlier turns | 15 | "do that again", "turn it back on", "undo that" |
| 11 | questions about the device | 15 | "how many pins are there", "what can you control", "is pin 4 safe" |

## What is NOT a refusal — read this before writing

**Anything the five actions can express, on a pin number, belongs in a different
file. Do not put it here.**

- "turn on pin 4" — a command.
- "turn on pins 4, 5 and 6" — a command. Several pins is fine.
- "blink pin 4 five times" — a command. A count without a rate is fine.
- "stop pins 4 and 5" — a command.
- **"turn on pin 100"** — a command. The pin does not exist, but the *sentence*
  is fine, and the device refuses it by checking the number, not by failing to
  understand. Out-of-range values have their own file.
- **"blink pin 7 every 12000ms"** — same. A rate out of bounds is still a rate.

If you find yourself writing a refusal because the *number* is wrong, stop —
that is the other file.

## Why category 1 is a refusal, and how to write it

This device genuinely has no names yet. Not "it does not know that particular
lamp" — it has no way to attach a name to a pin at all. So "turn on the desk
lamp" is refused because **naming is not built**, not because a desk lamp is an
odd thing to control.

Write these as ordinary, reasonable smart-home requests. Vary the nouns widely —
lamps, fans, pumps, buzzers, relays, made-up device names like `warning_led` or
`pump_2`, plural groups like "the porch lights". Do **not** cluster around a few
nouns; the point is the *shape*, not the vocabulary.

## Register

Same as real typing: terse and conversational, ALL CAPS lines, missing
punctuation, typos, politeness (`pls`, `thanks`, `could you`), filler (`um,`,
`hey`). A refusal is just as likely to be typed carelessly as a command — if
every refusal here is neatly written, the parser learns to reject on punctuation.

## Do not

- **Do not number the lines or add headers.** Plain text, one per line.
- **Do not repeat a noun or a sentence pattern more than three times.**
- **Do not write gibberish or random characters.** These should all be things a
  real person might plausibly type at this device.
