# espcontrol

A natural-language interface to GPIO pins that runs entirely on an ESP32. Type
*"blink pin 4 twice a second"* and the pin blinks. There is no network of any
kind — no WiFi, no cloud, no API key.

The language model is 312K parameters — a 1.2 MB fp32 payload in its own flash
partition, mapped and read in place.

**Status: running on an ESP32-S3.** Typed commands drive real pins; out-of-range
input is refused by the hardware layer. Latency 150 ms – 1.5 s per command,
measured on the board.

### How often does it move the wrong pin?

That is the only question worth leading with for something wired to hardware,
and this README used to lead with accuracy instead. Locked half of the held-out
set, 612 valid commands and 417 utterances it must refuse, three seeds:

| | rate | |
|---|---|---|
| **moves the wrong pin on a valid command** | **4.5% ± 0.7%** | the real failure |
| **moves a pin on input it should refuse** | **5.1% ± 1.3%** | the other real failure |
| answers correctly | 84.4% ± 1.5% | |
| refuses, or is refused downstream — nothing moves | 11.1% | wrong but harmless |

Accuracy numbers, for completeness: exact-match 84.4%, false-accept 13.3%, pin
copy 91.3%, substitution 1.8%, against gates of >95%, <2%, >99% and zero. **All
four are unmet.** Substitution used to be reported as passing; it was reading
0/21 per seed on the v1 set, and at n=201 it is 1.8%.

Note the gap between "false-accept 13.3%" and "moves a pin 5.1%". Most rejected
input that gets parsed anyway resolves to a *name* the device does not have, so
it is refused by name and nothing happens. Exact-match cannot see any of this —
it charges one point for "I don't understand" and one point for switching off
the wrong pin. `src/evaluate.py`'s `outcome()` is what separates them, and it
exists because a reader pointed out that an evaluation which cannot is
measuring the wrong thing.

Not comparable with the v0.2.x numbers below: v2 made names a positive, which
roughly doubled the in-domain set and added its hardest half. The pin-numbered
path is 88–90% either way.

### Would a hand-written parser be better?

For this command grammar, on the metric that matters: **yes**, and it is worth
saying so at the top rather than leaving it implied. Same held-out set, against
the rule-based parser already in this repo (`data/label_eval.py`):

| | model | parser |
|---|---|---|
| correct | **84.4%** | 41.7% |
| said "I don't understand" | 4.8% | 58.3% |
| **did the wrong thing** | 10.8% | **0.0%** |
| accepted off-domain input | 13.3% | **0.0%** |

The parser never mis-parses; it abstains. It meets the <2% safety target this
model has missed in every release, and since that parser was written to label
gold and is deliberately conservative, 41.7% is a floor rather than a ceiling.

What the model buys is coverage — roughly double, over surface variation nobody
enumerated (`kill the blinking on 13`, `solarium fan off`, `reding patio
lights`). What it costs is that it always answers. The interesting architecture
is neither on its own: parser first, model only on what the parser abstains
from, hardware refusing out-of-range values regardless.

Flashing: [`firmware/device/README.md`](firmware/device/README.md).

**Naming.** *"Call pin 4 the desk lamp"*, then *"turn on the desk
lamp"*. The model copies the name out of what you typed and the device resolves
it against a table you built, so a name it does not know is refused **by name**
rather than swapped for the nearest one it does know. Names persist across a
power cycle.

## Quickstart — flash it and talk to it

No toolchain, no Python, no training run. Grab the latest
`espcontrol-esp32s3-*.bin` from
[Releases](https://github.com/manjunathshiva/esp32-gpio-llm/releases) — it
contains the bootloader, partition table, firmware and the model itself — and
write it at offset 0:

```sh
pip install esptool
esptool.py --chip esp32s3 --port /dev/cu.usbmodemXXXX --baud 921600 \
  write_flash 0x0 espcontrol-esp32s3-v0.3.0.bin
```

Open a serial monitor at **115200** and type — this is a real session on the
board, timings included:

```
> turn on pin 4
pin 4 high   (515ms)

> blink pin 4 every 5 seconds 3 times
blinking 1 pin every 5000ms, 3 times   (1155ms)

> chase pins 4 5 6 every 1 second
chasing 3 pins every 1000ms   (1429ms)

> turn on pin 4 5 and 6
pin 4, 5, 6 high   (992ms)

> switch off pin 100
refused: pin 100 is not a GPIO on this board   (779ms)

> blink pin 4 every 60 seconds
refused: interval 60000ms is outside 50-10000ms   (1101ms)

> call pin 4 the desk lamp
"desk lamp" -> 4   (833ms)

> turn on the desk lamp
pin 4 high   (673ms)

> name gpios 5, 6, 7 and 8 the porch lights
"porch lights" -> 5 6 7 8   (1651ms)

> chase the porch lights 200ms 5
chasing 4 pins every 200ms   (1210ms)

> switch off the aquarium pump
refused: I don't know "aquarium pump"   (1101ms)
```

The three refusals are the design working rather than the model failing, and
they are all the same mechanism. The model transcribes what it heard — `100`,
`60000`, `aquarium pump` — and a layer that owns the relevant facts refuses it
*by value* or *by name*. An earlier grammar gave each legal pin its own symbol,
which made "pin 100" unsayable; the model then answered with pin 10 and switched
off a real pin, silently. Doing the same to names would have rebuilt that bug
one level up, so the model must be able to say a name that does not resolve.
See [Where it falls down](#where-it-falls-down).

To see something happen, put an LED and a **220Ω–1kΩ resistor** in series
between GPIO 4 and GND — long leg to the pin side. Never wire an LED without the
resistor. Twenty-five pins are usable: 1–18, 21, 38–42, 48.

Requires an **ESP32-S3 with PSRAM** (the KV cache lives there). On a devkit with
two USB ports use the **native USB** port — the image is built with
`CDCOnBoot=cdc`, so the CH340 port will run correctly but print nothing.

Building from source, and what to do when it misbehaves:
[`firmware/device/README.md`](firmware/device/README.md).

## What it is

A 312K-parameter transformer, trained from scratch on synthetic pin-control
commands, that maps English onto a small symbol grammar. The runtime assembles
the symbols into a command struct and drives the pin. The model never writes
JSON, so malformed output is not representable.

## What it is not

Not an assistant. It does one thing: parse pin commands. Anything off-domain
returns `<unknown>` and the device says it did not understand — there is no
fallback to ask, which is why the reject path is a trained output class rather
than a confidence heuristic bolted on afterwards.

**Not an argument that a transformer is the right tool for this job.** It is
not, and the table above says so with numbers: a deterministic parser is
smaller, exhaustively testable, and cannot hallucinate a GPIO action. The model
costs 1.2 MB of flash and 302 KB of PSRAM for the KV cache; the parser path's
entire runtime state is a 288-byte command struct and a 1.3 KB alias table.
The 1.2 MB partition exists solely to hold weights.

What the model is, is an *instrument*. Making invalid outputs unrepresentable —
which is what everyone reaches for when they want a model to be safe — turned
out to produce substitution rather than refusal: a grammar in which "pin 100"
could not be said answered "pin 10" and switched off a real pin. A parser
cannot exhibit that failure, which is exactly why a parser cannot teach you
about it. The finding transfers to GBNF grammars, JSON-schema mode and
function-calling APIs, which are used where the input space is open and a
parser genuinely is not an option.

That claim is an argument from mechanism, not a measurement. It has not been
tested against a real constrained-decoding stack, and it is the strongest thing
this project asserts on the least evidence.

On-device NLU for closed command domains is also well-trodden ground
commercially. The point here is not novelty: it is that the entire stack —
tokenizer, training, quantization, inference, pin control — is a few thousand
lines you can read end to end, including the parts that are wrong.

## Plan

The full design and build order is in
[`../esp32llm/PINLM-PLAN.md`](../esp32llm/PINLM-PLAN.md) (outside this repo).
Phases 1–3 are host-only and gate everything else: no C is written and no
hardware is touched until the model clears **>95% exact-match on in-domain
commands and <2% false-accepts on off-domain input**, measured against a
held-out set written by a human who never read the data generator.

## Provenance

This repo copies code from two MIT-licensed projects. Neither is modified in
place; both remain the authoritative source for their own results.

**[esp32-tinyllm](https://github.com/manjunathshiva/esp32-tinyllm)** — supplies
the portable C inference runtime (`llm.h`), the on-device BPE encoder
(`tokenizer.h`), the training/export pipeline, and the host-verification
discipline. That project in turn builds on
**[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)** by Viacheslav Sierbov,
whose copyright notice is retained in [`LICENSE`](LICENSE).

| file here | from |
|---|---|
| `firmware/common/tokenizer.h` | esp32-tinyllm, verbatim |
| `firmware/common/llm.h` | esp32-tinyllm, PLE removed and head restructured |
| `firmware/host_verify/verify.c`, `tok_test.c` | esp32-tinyllm |
| `src/model.py`, `train.py`, `export.py`, `gen_assets.py`, `tok_check.py` | esp32-tinyllm |
| `data/prepare.py` | esp32-tinyllm, retargeted to the command corpus |

**femtoclaw** — supplies the GPIO tool: the safe-pin allowlist, the eight-slot
`esp_timer` animation engine, and the range validation.

| file here | from |
|---|---|
| `firmware/common/gpio_control.c/.h` | femtoclaw `main/tools/tool_gpio.c/.h`, cJSON layer removed |

femtoclaw has its own upstream, so the credit does not stop there — see
[Acknowledgments](#acknowledgments).

That validation is kept deliberately intact, and v1 made it load-bearing rather
than a second opinion. It checks the pin allowlist and
`interval_ms ∈ [50,10000]` independently of the model, so a misparse is refused
by the hardware layer rather than trusted. It is what makes a 312K-parameter
model safe to ship with no fallback.

The first grammar quietly defeated it. Giving each allowlisted pin its own
symbol made an illegal pin *unrepresentable*, which sounds like a stronger
guarantee and is a weaker one: a model that cannot say "pin 100" does not
refuse, it says the nearest thing it can. On held-out data *"switch off pin
100"* came back as pin 10 — a real pin, switched off, and by the time the frame
reached the allowlist it was already valid. v1 emits pin numbers digit by digit
so the check has something to reject. `data/frames.py` is the grammar: the
symbol list, what `validate()` accepts, and what `range_check()` refuses.
`firmware/common/command.h` mirrors it on the device side.

## Where it falls down

All four gates are unmet. The v0.2.x interval work below is kept because the method
is the point: splitting the held-out rows by what the model has to *do* with the
number, rather than by what the number looks like, three seeds pooled.

| what the utterance asks for | n | v0.1.0 | v0.2.1 |
|---|---:|---:|---:|
| copy a number that is written down (`every 500ms`) | 372 | 97% | 98% |
| convert a unit (`every 2 seconds`, `at 5Hz`) | 63 | 48% | 86% |
| convert a unit to a 5-digit result (`every 60 seconds`) | 36 | **0%** | **94%** |

The last row used to fail the same way every time — dropping exactly one zero:
`60 seconds → 6000`, `90 seconds → 9000`, `15 seconds → 1500`. The corpus was
why. Out-of-range intervals were drawn uniformly, so they almost never landed on
a round multiple of 1000, and a "seconds" phrasing is only ever generated for a
round value. The result was that **every unit conversion in 147,000 training
rows resolved to 3 or 4 digits, and none to 5**. The model had learned a length
prior — pad with zeros, stop at four digits — and applied it faithfully.

It was a coverage hole, not a capacity limit: digit copying was already at 97%
on the very number lengths the conversions failed on. Fixing the generator
closed it without touching the architecture, the parameter count or the training
schedule. Three neighbouring forms turned out to have the same problem: `at 5Hz`
occurred 3 times in the corpus (all inside refusals), `every 2 sec` and
`every 30s` zero times, and "minutes" appeared **only** in refusals — so the
model had learned that minutes mean "reject", which reads as caution rather than
as a gap.

What that bought, three seeds on the held-out set: substitution 2.8% → **0.8%**
(p = 0.010). Exact-match, false-accept and pin copy moved by less than noise
(p = 0.23, 0.42) and are not claimed as improvements.

**All four gates remain unmet in v2** (locked half, three seeds): exact-match
84.4% against >95%, false-accept 13.3% against <2%, pin copy 91.3% against
>99%, substitution 1.8% against zero.

Substitution was previously reported as passing here, which was wrong twice
over: 1.8% is not zero, and the v1 reading it was compared against was 0/21 per
seed — a denominator `docs/V1-SCOPE.md` already flagged as too small to be
evidence. The v2 set carries 201 substitution probes per seed. The rate did not
necessarily rise; the instrument got big enough to see it.

**False-accept regressed on purpose, and it is the cost of the feature.** It
was 8.6% with names alone and 13.3% once naming-by-voice was added; a fifth of
the false accepts parse as `<alias>`. Losing 25 easy refusals from the
denominator explains about half a point of that, so most of it is real:
**adding positives of a shape makes the model readier to accept that shape.**
The same tax showed up twice before — closing a long-pin-list coverage gap cost
1.5 points, and tripling the corpus cost 3.6 — which makes it the most reliable
finding in the project and the one to design around next.

Named targets are the weak half: 72.2% against 88.9% for pin-numbered ones on
the locked set, with a 5–7 point seed spread where pin-numbered sits under 1.
The cause of the *original* 20-point gap was vocabulary rather than model
capacity. Held-out names built only from the generator's own word pools were
copied 89% of the time and names containing any other word 56%, while span
length did nothing once vocabulary was held fixed (90.0% for 5+ token names
against 88.4% for short ones). A controlled probe made it plain: the model
copied invented nonsense (`zibmuk valve`) 90% of the time and ordinary English
it had not seen in that slot (`coffee maker`) 43%.

Counting words by which side of the corpus they appear on said why. `coffee`
occurred 2,080 times, every one inside a refusal, so the model had learned it
meant *reject*; `cutter` occurred nowhere at all, so `enable laser cutter` came
back as `laser`. Every word that copied correctly — `lamp`, `fan`, `relay` —
appeared on both sides at 75–90% positive. Fixing the generator's vocabulary,
with no change to the architecture, took out-of-pool name copy from 56.4% to
78.2% and the English probe from 43.3% to 80.0%.

## Note on the dataset

Training data is synthetic. Paraphrase diversity is generated **once, offline,
on a laptop**, using a large model. The shipping device never contacts anything
— the same relationship it has with the compiler that built it.

## Acknowledgments

FemtoClaw is inspired by and forked from MimiClaw by Ziboyan Wang. FemtoClaw
extends MimiClaw with ESP-WROOM-32 support, zero-config web search, SNTP time
sync, and dual-target builds.

This applies here because `firmware/common/gpio_control.c/.h` is taken from
femtoclaw's GPIO tool — the safe-pin allowlist, the eight-slot `esp_timer`
animation engine and the range validation are that lineage's work, not this
project's. Only the entry point was rewritten, to take a parsed `Command`
instead of JSON.

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Viacheslav Sierbov and
Copyright (c) 2026 Manjunath Janardhan.
