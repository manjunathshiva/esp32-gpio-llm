#!/usr/bin/env bash
# Build a flashable release image, and refuse to build one that has not passed
# the host gates.
#
#   ./scripts/release.sh v0.2.0              # build + verify, publish nothing
#   ./scripts/release.sh v0.2.0 --publish    # also create the GitHub release
#   ./scripts/release.sh v0.2.0 --run v1-s1  # a checkpoint other than the default
#
# The output is one file: dist/espcontrol-esp32s3-<version>.bin, containing
# bootloader + partition table + app + model at their correct offsets. A user
# flashes it with a single esptool command and needs no toolchain, no Python and
# no training run.
#
# The three gates below are the point of this script. They are cheap, they catch
# different things, and a release that skips them is a release nobody can trust:
# a tokenizer mismatch or a symbols.h regenerated without a matching export
# produces confident nonsense on device, with nothing in the logs.
set -euo pipefail

VERSION="${1:-}"
[ -n "$VERSION" ] || { echo "usage: $0 <version> [--publish] [--run <tag>]" >&2; exit 1; }
shift

RUN_TAG="v1-s0"
PUBLISH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --publish) PUBLISH=1 ;;
    --run) RUN_TAG="$2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
DIST="$ROOT/dist"
BUILD="$ROOT/dist/build"
IMAGE="$DIST/espcontrol-esp32s3-$VERSION.bin"
FQBN='esp32:esp32:esp32s3:PartitionScheme=custom,PSRAM=opi,CDCOnBoot=cdc'
mkdir -p "$DIST"
rm -rf "$BUILD"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

step "export weights from runs/cmd-$RUN_TAG.pt"
uv run python src/export.py "$RUN_TAG"

step "generate device headers"
uv run python src/gen_assets.py | head -2

step "gate 1/3: C forward pass vs PyTorch golden"
cc -O3 -o /tmp/verify firmware/host_verify/verify.c -lm
/tmp/verify firmware/model/model.bin firmware/model/golden.txt | tail -2

step "gate 2/3: device tokenizer vs training tokenizer"
cc -O3 -o /tmp/tok_test firmware/host_verify/tok_test.c
uv run python src/tok_check.py | tail -1

step "gate 3/3: whole chain vs PyTorch over held-out text"
cc -O3 -Ifirmware/generated -o /tmp/repl firmware/host_verify/repl.c -lm
uv run python src/c_check.py | tail -1

step "compile sketch"
./firmware/device/sync.sh > /dev/null
arduino-cli compile --fqbn "$FQBN" --output-dir "$BUILD" \
  firmware/device/espcontrol | tail -2

step "merge into a single flashable image"
# Offsets come from firmware/device/espcontrol/partitions.csv. The model lands
# at 0x210000; change one and change the other, or the sketch will map whatever
# happens to be sitting there.
BOOT_APP0=$(find "$HOME/Library/Arduino15/packages/esp32/hardware/esp32" \
              -name boot_app0.bin | sort | tail -1)
esptool.py --chip esp32s3 merge_bin -o "$IMAGE" \
  --flash_mode dio --flash_freq 80m --flash_size 16MB \
  0x0      "$BUILD/espcontrol.ino.bootloader.bin" \
  0x8000   "$BUILD/espcontrol.ino.partitions.bin" \
  0xe000   "$BOOT_APP0" \
  0x10000  "$BUILD/espcontrol.ino.bin" \
  0x210000 "$ROOT/firmware/model/model.bin" | tail -1

shasum -a 256 "$IMAGE" | tee "$IMAGE.sha256"
printf '\n\033[1m%s\033[0m  (%s)\n' "$IMAGE" "$(du -h "$IMAGE" | cut -f1)"

if [ "$PUBLISH" -eq 1 ]; then
  step "publish $VERSION"
  gh release create "$VERSION" \
    "$IMAGE" "$IMAGE.sha256" "$ROOT/firmware/model/model.bin" \
    --title "$VERSION" --notes-file - <<NOTES
Flash and use — no toolchain, no Python, no training run:

\`\`\`sh
esptool.py --chip esp32s3 --port /dev/cu.usbmodemXXXX --baud 921600 \\
  write_flash 0x0 espcontrol-esp32s3-$VERSION.bin
\`\`\`

Then open a serial monitor at 115200 and type a command.

**Built from \`runs/cmd-$RUN_TAG.pt\`.** All three host gates passed before this
image was produced: C-vs-PyTorch forward pass, device tokenizer vs training
tokenizer, and the whole chain over every held-out utterance.

\`model.bin\` is attached separately for anyone building the sketch themselves.
NOTES
fi
