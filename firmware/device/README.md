# Flashing espcontrol

Two artifacts go on the board and they flash separately: the **sketch** (~1 MB)
and the **model** (898 KB, its own partition). Firmware changes only need the
first; the model is rewritten only when `src/export.py` runs again.

## 0. Build the assets (host)

```sh
uv run python src/export.py v1-s0        # firmware/model/model.bin + golden.txt
uv run python src/gen_assets.py          # firmware/generated/{bpe.h,symbols.h}
./firmware/device/sync.sh                # flatten sources into the sketch dir
```

`sync.sh` copies `firmware/common/*` and `firmware/generated/*` next to the
`.ino`, because the Arduino build compiles what sits beside the sketch and
follows no include paths of its own. **Edit the originals in `firmware/common`,
never the copies** — the next sync overwrites them.

## 1. Pass the host gates first

None of these need hardware, and all three should pass before you flash. They
catch different things and none subsumes another.

```sh
cc -O3 -o /tmp/verify firmware/host_verify/verify.c -lm
/tmp/verify firmware/model/model.bin firmware/model/golden.txt   # forward pass

cc -O3 -o /tmp/tok_test firmware/host_verify/tok_test.c
uv run python src/tok_check.py                                   # tokenizer

cc -O3 -Ifirmware/generated -o /tmp/repl firmware/host_verify/repl.c -lm
uv run python src/c_check.py                                     # whole chain
```

`repl.c` is the same pipeline the sketch runs, so you can also drive it by hand
before touching a board:

```sh
/tmp/repl firmware/model/model.bin --pretty
```

## 2. Compile and upload the sketch

```sh
arduino-cli compile \
  --fqbn esp32:esp32:esp32s3:PartitionScheme=custom,PSRAM=opi,CDCOnBoot=cdc \
  firmware/device/espcontrol

arduino-cli upload -p /dev/cu.usbmodemXXXX \
  --fqbn esp32:esp32:esp32s3:PartitionScheme=custom,PSRAM=opi,CDCOnBoot=cdc \
  firmware/device/espcontrol
```

`PartitionScheme=custom` is what makes it read the `partitions.csv` in the
sketch directory — without it the `model` partition does not exist and the
sketch stops at "no 'model' partition".

**Which port, and `CDCOnBoot`.** An S3 devkit exposes two: the USB-serial bridge
(CH340, `/dev/cu.usbserial-*` or `/dev/cu.wchusbserial*`) and the chip's native
USB (`/dev/cu.usbmodem*`). They are not interchangeable, and the setting has to
match the one you open — `CDCOnBoot=cdc` routes `Serial` to native USB, while
the CH340 port wants `CDCOnBoot=default`. Mismatch it and the board runs
correctly with a silent console, which reads exactly like a hang.

`PSRAM=opi` matters: the ~200 KB KV cache is allocated there first. Without
PSRAM the sketch falls back to internal RAM, which may or may not fit alongside
the Arduino core.

## 3. Flash the model

Once per export, to the offset in `partitions.csv`:

```sh
esptool.py --chip esp32s3 --port /dev/cu.usbmodemXXXX --baud 921600 \
  write_flash 0x210000 firmware/model/model.bin
```

## 4. Talk to it

```sh
arduino-cli monitor -p /dev/cu.usbmodemXXXX --config baudrate=115200
```

Measured on an ESP32-S3 at 240 MHz, `PSRAM=opi`, model mapped from flash:

```
> turn on pin 4
pin 4 high   (378ms)

> blink pin 38 every 300ms 5 times
blinking pin 38 every 300ms, 5 times   (961ms)

> switch off pin 100
refused: pin 100 is not a GPIO on this board   (571ms)

> chase 1 2 3 4 5 6 7 8 9
refused: a chase runs at most 6 pins, got 9   (1523ms)

> turn on the desk lamp
refused: I don't understand that   (264ms)

> TURN OFF 39
pin 39 low   (570ms)
```

Latency runs **113 ms to ~2 s**, and it is dominated by emitted symbols rather
than by prompt length: `stop` is one symbol, an eight-pin chase is twenty-five.
Every token is a full forward pass over the mapped weights with no batching and
no SIMD, so the obvious lever if this needs to be faster is the matvec in
`llm.h`, not the model size.

`turn on the desk lamp` is v1 behaving as designed, not failing: pin numbers
only, names deferred to v2. See `docs/V1-SCOPE.md`.

Note the two refusals in the middle. Pin 100 and the nine-pin chase both reached
the runtime *intact* — the chase was refused with a count of 9, not quietly
trimmed to the six it could run. That is the property v1 was rebuilt for, and it
is visible here rather than only in the eval.

## If it does not work

| symptom | cause |
|---|---|
| `no 'model' partition` | `PartitionScheme=custom` missing from the FQBN |
| `bad magic` | model partition empty, or flashed to the wrong offset |
| `model exceeds the compile-time ceilings` | `model.bin` is from a bigger config than `llm.h`'s `LLM_MAX_*` |
| nothing on serial | `CDCOnBoot` does not match the port you opened |
| `malformed generation` | `symbols.h` and `model.bin` are from different tokenizer runs — re-run `gen_assets.py` **and** `export.py`, then `sync.sh` |
