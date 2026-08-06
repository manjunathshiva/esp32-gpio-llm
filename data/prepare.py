"""Train the BPE and encode the corpus into fixed-width training arrays.

**Digits are split individually** (`Digits(individual_digits=True)`), so "300ms"
tokenizes as `Ġ`,`3`,`0`,`0`,`ms` in the input and the label emits `3`,`0`,`0`.
A number is then a literal token copy rather than a decomposition, which is the
difference between an easy attention task and a memorization one. It costs a few
tokens per number and requires the matching change in
`firmware/common/tokenizer.h` -- GPT-2's split groups `\\p{N}+` into one chunk,
this does not. `src/tok_check.py` is the gate for that.

This carries a lot of weight: pin numbers are copied digits too, so a tokenizer
that merged "10" into one token would make "pin 100" -> "pin 10" a single-token
slip.

v2 adds the other copied slot, a name. Its label ids are **sliced out of the
encoded prompt** rather than re-encoded from the name string -- see
`name_span_ids` for why re-encoding cannot work under ByteLevel. That makes the
completion a literal subsequence of the input, which is the property that turns
copying into an attention task instead of a memorization one.

Usage:
    uv run python data/prepare.py --seq-len 96 --vocab 1024
"""

from __future__ import annotations

import argparse
import json
import sys
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


class SpanError(ValueError):
    """A name in the label is not present in the utterance. A corpus bug, not a
    tokenizer one: the realizer must render the name it recorded."""


def name_span_ids(enc, text: str, name: str) -> list[int]:
    """The prompt's own token ids for `name`, sliced out of an encoded prompt.

    Not `tok.encode(name)`. A name is *copied*, so the label has to be the same
    ids the model can see in front of it -- and re-encoding does not give those:

      - ByteLevel makes "status" and "Ġstatus" different ids, so a name at
        position 0 ("status_pin off") re-encodes to tokens that never appear in
        the prompt. v0 hit exactly this and the span came back garbled.
      - Utterances get case-mangled after the frame is built, so "desk lamp"
        may surface as "DESK LAMP", which tokenizes completely differently.

    Taking the ids from the prompt makes the label a literal subsequence of the
    input, so copying is correct by construction rather than by coincidence, and
    every surface form is handled without enumerating them.

    Selection is by *overlap*, not containment: ByteLevel folds the preceding
    space into the following token, so "Ġdesk" spans (11,16) while the name
    starts at char 12. Requiring containment silently drops a name's first
    token -- which still trains, still looks plausible, and resolves to the
    wrong alias.
    """
    lo = text.casefold().find(name.casefold())
    if lo < 0:
        raise SpanError(f"{name!r} does not occur in {text!r}")
    hi = lo + len(name)
    ids = [i for i, (s, e) in zip(enc.ids, enc.offsets) if e > lo and s < hi]
    if not ids:
        raise SpanError(f"{name!r} selected no tokens in {text!r}")
    return ids


def encode_completion(tok: Tokenizer, syms: list[str], reserved: set[str],
                      enc=None, text: str = "") -> list[int]:
    """Symbol list -> token ids.

    Outside a name span every symbol is reserved or a single digit, so it is a
    straight lookup with no BPE in the loop. The assert is the guard: a
    multi-character non-reserved symbol anywhere else means to_symbols() grew a
    slot this function does not know about, and silently BPE-ing it would
    produce a label the model has to translate rather than copy.
    """
    out: list[int] = []
    i = 0
    while i < len(syms):
        s = syms[i]
        if s == frames.S_NAME:
            # The one slot whose content is arbitrary text. It is bounded by
            # <nend> because, unlike a digit run, a name does not end itself.
            out.append(tok.token_to_id(s))
            i += 1
            if i >= len(syms):
                raise SpanError(f"{frames.S_NAME} with nothing after it")
            out += name_span_ids(enc, text, syms[i])
            i += 1
            continue
        assert s in reserved or (s.isdigit() and len(s) == 1), \
            f"completion symbol {s!r} is neither reserved nor a digit"
        out.append(tok.token_to_id(s))
        i += 1
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

    unspanned: list[str] = []
    n_unspanned = 0

    for name, rows in splits.items():
        toks = np.zeros((len(rows), a.seq_len), dtype=np.uint16)
        mask = np.zeros((len(rows), a.seq_len), dtype=np.uint8)
        kept = dropped = 0
        longest = 0

        for r in rows:
            enc = tok.encode(r["text"])
            prompt = enc.ids + [go_id]
            try:
                comp = encode_completion(tok, r["symbols"], reserved_set,
                                         enc, r["text"])
            except SpanError as e:
                # The realizer rendered a name the label does not contain, so
                # the row cannot teach a copy. Collected rather than raised on
                # the spot: one example tells you almost nothing, and the count
                # is what says whether it is a typo or a broken template.
                if len(unspanned) < 5:
                    unspanned.append(str(e))
                n_unspanned += 1
                continue
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

    # Fatal, and deliberately at the end so the report is the whole count rather
    # than the first row that tripped. A name the model cannot copy from the
    # prompt is not a smaller training set, it is a silently different task.
    if n_unspanned:
        for e in unspanned:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(
            f"{n_unspanned:,} rows had a name absent from their own utterance. "
            f"data/realize.py must render the name it recorded in the frame.")


if __name__ == "__main__":
    main()
