"""Prove the device BPE encoder matches the training tokenizer.

The device tokenizes typed text itself (firmware/common/tokenizer.h). If its
output diverges from `tokenizers`, prompts land out of distribution and the only
symptom is worse parses -- no error, nothing in the logs. So this diffs the two
over real corpus lines, the same way verify.c diffs the C runtime against the
PyTorch golden.

**Numbers are the point of this check in v1.** Pin numbers are copied digit by
digit, so a device that grouped "100" into one chunk, or let the preceding space
attach to it, would feed the model a token sequence it never saw for exactly the
inputs where being wrong moves a physical pin. The edge cases below lead with
that; the corpus sample covers the rest.

  cc -O3 -o /tmp/tok_test firmware/host_verify/tok_test.c
  uv run python src/tok_check.py
"""

import json
import os
import subprocess
import sys

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TOK = os.path.join(ROOT, "data", "bpe.json")
CORPUS = os.path.join(ROOT, "data", "corpus", "train.jsonl")
BIN = "/tmp/tok_test"
N_LINES = 3000

EDGE_CASES = [
    # digits: single, multi, boundary, and the out-of-range values v1 must copy
    "turn on pin 4",
    "turn on pin 38",
    "turn on pin 100",
    "switch off pin 100",
    "chase 1 2 3 4 5 6 7 8 9 10",
    "blink pin 38 every 300ms",
    "blink pin 7 at 12000ms",
    "blink pin 12 at 1ms",
    "flash 5 500ms 10",
    "0", "1", "10", "100", "1000", "12345",
    " 100", "100 ", "pin100", "gpio11", "p7", "#48", "io 21",
    "1,2,3", "1, 2, 3", "1;2", "1.5", "2.5v", "07:30", "6pm", "75%",
    # the registers the corpus carries
    "TURN OFF 39",
    "pls can you blink gpio 4 every 500ms thanks",
    "um, sweep 38 42",
    "what's the weather in Paris",
    "she didn't know; they're here!",
    "double  space and   triple",
    " leading space", "trailing space ", "tabs\tand\tmore",
    "UPPER lower MiXeD", "a", " ", "", "!!!???...---",
]


def main() -> int:
    if not os.path.exists(BIN):
        sys.exit(f"missing {BIN} -- build it first:\n"
                 f"  cc -O3 -o {BIN} firmware/host_verify/tok_test.c")

    tok = Tokenizer.from_file(TOK)

    lines = list(EDGE_CASES)
    if os.path.exists(CORPUS):
        with open(CORPUS, encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)["text"]
                # The harness is line-based, so anything with a tab is skipped.
                if "\t" not in t:
                    lines.append(t)
                if len(lines) >= N_LINES:
                    break
    else:
        print(f"note: {CORPUS} absent, running edge cases only")

    proc = subprocess.run(BIN, input="\n".join(lines) + "\n",
                          capture_output=True, text=True)
    got = proc.stdout.split("\n")

    bad = 0
    for i, line in enumerate(lines):
        want = tok.encode(line).ids
        mine = [int(x) for x in got[i].split()] if i < len(got) and got[i].strip() else []
        if want != mine:
            bad += 1
            if bad <= 6:
                print(f"MISMATCH line {i}: {line!r}")
                print(f"  python: {want}")
                print(f"  c     : {mine}")
                print(f"  python tokens: {tok.encode(line).tokens}")

    total = len(lines)
    print(f"\n{total - bad}/{total} lines match "
          f"({100.0 * (total - bad) / total:.2f}%)")
    if bad:
        print(f"FAIL: {bad} mismatched")
        return 1
    print("PASS: device encoder matches the training tokenizer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
