# espcontrol

A natural-language interface to GPIO pins that runs entirely on an ESP32. Type
*"blink pin 4 twice a second"* and the pin blinks. There is no network of any
kind — no WiFi, no cloud, no API key.

The language model is ~230 KB and lives in the firmware binary.

**Status: running on an ESP32-S3.** Typed commands drive real pins; out-of-range
input is refused by the hardware layer. Latency 113 ms – 2 s per command.
Held-out exact-match is 95.8% ±1.0 and substitution 0 across three seeds, but
false-accept (2.8%) and pin-copy (97.2%) still miss their gates, and the
in-domain set is only 96 items — too small to resolve further tuning, so a wider
one comes before more corpus work. Numbers and caveats:
[`docs/V1-SCOPE.md`](docs/V1-SCOPE.md). Flashing:
[`firmware/device/README.md`](firmware/device/README.md).

**v1 targets pins by number.** Names and aliases — *"blink the desk lamp"* — are
deferred to v2 and currently return `<unknown>`; see
[`docs/V1-SCOPE.md`](docs/V1-SCOPE.md) for why, and what it bought.

## What it is

A ~230K-parameter transformer, trained from scratch on synthetic pin-control
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
by the hardware layer rather than trusted. It is what makes a 230 KB model safe
to ship with no fallback.

The first grammar quietly defeated it. Giving each allowlisted pin its own
symbol made an illegal pin *unrepresentable*, which sounds like a stronger
guarantee and is a weaker one: a model that cannot say "pin 100" does not
refuse, it says the nearest thing it can. On held-out data *"switch off pin
100"* came back as pin 10 — a real pin, switched off, and by the time the frame
reached the allowlist it was already valid. v1 emits pin numbers digit by digit
so the check has something to reject. See [`docs/GRAMMAR.md`](docs/GRAMMAR.md)
§1.

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
