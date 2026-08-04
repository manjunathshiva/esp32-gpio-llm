#!/usr/bin/env sh
# Copy the shared sources into the sketch directory.
#
# The Arduino build compiles exactly what sits beside the .ino and follows no
# include paths of its own, so `firmware/common` and `firmware/generated` have
# to be flattened in. Copying rather than symlinking because arduino-cli's
# behaviour on symlinked sources differs between versions, and a sketch that
# builds stale code is a very confusing afternoon.
#
# Everything this writes is generated -- see .gitignore. Re-run it after
# gen_assets.py, after editing anything in firmware/common, and before flashing.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
sketch="$here/espcontrol"

[ -f "$root/firmware/generated/symbols.h" ] || {
  echo "missing firmware/generated/symbols.h -- run: uv run python src/gen_assets.py" >&2
  exit 1
}

cp "$root/firmware/common/llm.h" \
   "$root/firmware/common/tokenizer.h" \
   "$root/firmware/common/command.h" \
   "$root/firmware/common/gpio_control.h" \
   "$root/firmware/common/gpio_control.c" \
   "$root/firmware/generated/symbols.h" \
   "$root/firmware/generated/bpe.h" \
   "$sketch/"

echo "synced -> $sketch"
ls -1 "$sketch"
