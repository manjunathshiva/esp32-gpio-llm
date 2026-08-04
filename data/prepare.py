"""Train the BPE and encode the corpus into fixed-width training arrays.

**Digits are split individually** (`Digits(individual_digits=True)`), so "300ms"
tokenizes as `Ġ`,`3`,`0`,`0`,`ms` in the input and the label emits `3`,`0`,`0`.
A number is then a literal token copy rather than a decomposition, which is the
difference between an easy attention task and a memorization one. It costs a few
tokens per number and requires the matching change in
`firmware/common/tokenizer.h` -- GPT-2's split groups `\\p{N}+` into one chunk,
this does not. `src/tok_check.py` is the gate for that.

In v1 this is the *only* decision that shapes the completion, and it carries far
more weight than it did: pin numbers are copied digits too now, so a tokenizer
that merged "10" into one token would make "pin 100" -> "pin 10" a single-token
slip. The v0 note about encoding name spans with a leading space is gone with
names -- every completion symbol is now either reserved or a lone digit, which
`encode_completion` asserts rather than handles.

Usage:
    uv run python data/prepare.py --seq-len 96 --vocab 1024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

import frames

HERE = Path(__file__).parent

PAD = "<pad>"
GO = "<go>"          # end of prompt; the next token is the action symbol


def reserved_symbols() -> list[str]:
    """Reserved ids in a fixed order. gen_assets.py emits symbols.h from this,
    so the order is part of the model format. No longer takes a pin allowlist:
    v1 has one <pin> symbol and the board lives in gpio_control.c."""
    return [PAD, GO] + frames.special_symbols()


def train_tokenizer(texts: list[str], vocab: int, reserved: list[str],
                    path: Path) -> Tokenizer:
    if path.exists():
        print(f"already have {path}")
        return Tokenizer.from_file(str(path))

    print(f"training BPE vocab={vocab} ({len(reserved)} reserved)...")
    tok = Tokenizer(models.BPE(unk_token=None))
    # Digits first so no merge can ever span two digits, then ByteLevel for the
    # GPT-2 split and byte fallback.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=reserved,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    ))
    tok.save(str(path))
    return tok


def encode_completion(tok: Tokenizer, syms: list[str],
                      reserved: set[str]) -> list[int]:
    """Symbol list -> token ids.

    Every v1 completion symbol is either reserved or a single digit, so this is
    a straight lookup with no BPE in the loop. The assert is the guard: if a
    multi-character non-reserved symbol ever appears here it means to_symbols()
    grew a span slot again, and silently BPE-ing it would produce a label the
    model has to translate rather than copy.
    """
    out: list[int] = []
    for s in syms:
        assert s in reserved or (s.isdigit() and len(s) == 1), \
            f"completion symbol {s!r} is neither reserved nor a digit"
        out.append(tok.token_to_id(s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--corpus", type=Path, default=HERE / "corpus")
    a = ap.parse_args()

    reserved = reserved_symbols()
    reserved_set = set(reserved)

    splits = {}
    for name in ("train", "val"):
        with (a.corpus / f"{name}.jsonl").open() as fh:
            splits[name] = [json.loads(line) for line in fh]
        print(f"{name}: {len(splits[name]):,} rows")

    tok = train_tokenizer([r["text"] for r in splits["train"]],
                          a.vocab, reserved, HERE / "bpe.json")
    real_vocab = tok.get_vocab_size()
    print(f"vocab {real_vocab}")

    go_id, pad_id = tok.token_to_id(GO), tok.token_to_id(PAD)
    assert pad_id == 0, "pad must be id 0 so padding is inert in the mask"

    meta = {"vocab": real_vocab, "seq_len": a.seq_len, "pad": pad_id,
            "go": go_id, "reserved": {s: tok.token_to_id(s) for s in reserved}}

    for name, rows in splits.items():
        toks = np.zeros((len(rows), a.seq_len), dtype=np.uint16)
        mask = np.zeros((len(rows), a.seq_len), dtype=np.uint8)
        kept = dropped = 0
        longest = 0

        for r in rows:
            body = tok.encode(r["text"]).ids
            prompt = body + [go_id]
            comp = encode_completion(tok, r["symbols"], reserved_set)
            seq = prompt + comp
            longest = max(longest, len(seq))
            if len(seq) > a.seq_len:
                dropped += 1
                continue
            toks[kept, :len(seq)] = seq
            # Loss only on the completion. The prompt is context, not a target:
            # training the model to predict the user's own words wastes capacity
            # on a task nobody needs.
            mask[kept, len(prompt):len(seq)] = 1
            kept += 1

        toks, mask = toks[:kept], mask[:kept]
        toks.tofile(HERE / f"{name}_tok.u16")
        mask.tofile(HERE / f"{name}_mask.u8")
        meta[f"n_{name}"] = int(kept)
        print(f"{name}: kept {kept:,}  dropped {dropped:,} over {a.seq_len}  "
              f"(longest {longest})")

    (HERE / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {HERE}/meta.json")


if __name__ == "__main__":
    main()
