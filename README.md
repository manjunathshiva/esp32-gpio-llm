# espcontrol

A natural-language interface to GPIO pins that runs entirely on an ESP32. Type
*"blink pin 4 twice a second"* and the pin blinks. There is no network of any
kind — no WiFi, no cloud, no API key.

The language model is 312K parameters — a 1.2 MB fp32 payload in its own flash
partition, mapped and read in place.

**Status: running on an ESP32-S3.** Typed commands drive real pins; out-of-range
input is refused by the hardware layer. Latency 113 ms – 2 s per command.
On a 466-item held-out set, three seeds: exact-match 90.4% ±2.0, false-accept
5.0% ±1.7, pin copy 93.9% ±1.7, substitution 2.8% ±1.5. The gates are >95% and
<2%, so **two of the four are not met** and this is not finished. The known
cause of the largest remaining bucket is a corpus hole, not model capacity: see
[Where it falls down](#where-it-falls-down). Flashing:
[`firmware/device/README.md`](firmware/device/README.md).

**v1 targets pins by number.** Names and aliases — *"blink the desk lamp"* — are
deferred to v2 and currently return `<unknown>`.

## Quickstart — flash it and talk to it

No toolchain, no Python, no training run. Grab the latest
`espcontrol-esp32s3-*.bin` from
[Releases](https://github.com/manjunathshiva/esp32-gpio-llm/releases) — it
contains the bootloader, partition table, firmware and the model itself — and
write it at offset 0:

```sh
pip install esptool
esptool.py --chip esp32s3 --port /dev/cu.usbmodemXXXX --baud 921600 \
  write_flash 0x0 espcontrol-esp32s3-v0.1.0.bin
```

Open a serial monitor at **115200** and type:

```
> turn on pin 4
pin 4 high   (378ms)

> blink pin 4 every 500ms
blinking pin 4 every 500ms   (765ms)

> turn on pins 4, 5 and 6
pin 4, 5, 6 high   (726ms)

> switch off pin 100
refused: pin 100 is not a GPIO on this board   (571ms)
```

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

On-device NLU for closed command domains is well-trodden ground commercially.
The point here is not novelty: it is that the entire stack — tokenizer,
training, quantization, inference, pin control — is a few thousand lines you can
read end to end.

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

Two gates are unmet, and the largest single bucket has a known cause. Splitting
the held-out interval rows by what the model has to do with the number, three
seeds pooled:

| what the utterance asks for | n | correct |
|---|---:|---:|
| copy a number that is written down (`every 500ms`) | 372 | 97% |
| convert a unit (`every 2 seconds`, `at 5Hz`) | 63 | 48% |
| convert a unit to a 5-digit result (`every 60 seconds`) | 36 | **0%** |

The last row fails the same way every time — it drops exactly one zero:
`60 seconds → 6000`, `90 seconds → 9000`, `15 seconds → 1500`. The corpus is
why. Out-of-range intervals are drawn uniformly, so they are almost never round
multiples of 1000, and a "seconds" phrasing is only ever generated for a round
value. The result is that **every unit conversion in 147,000 training rows
resolves to 3 or 4 digits, and none to 5**. The model learned a length prior —
pad with zeros, stop at four digits — and applies it faithfully.

It is a coverage hole, not a capacity limit: digit copying is at 97% on the same
number lengths the conversions fail on.

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
