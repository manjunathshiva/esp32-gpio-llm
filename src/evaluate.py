"""Free-running evaluation: generate a command, parse it, compare frames.

This is the metric that matters. `train.py`'s symbol accuracy is teacher-forced
-- it grades each symbol given all the correct ones before it, so it hides
exactly the failure that breaks a device: one wrong symbol early derailing the
rest. Here the model runs on its own output.

Four numbers are reported, and all four gate the C runtime work:

  exact-match   fraction of in-domain commands whose parsed frame equals the
                gold frame -- action, targets, level, interval, count, all of it
  false-accept  fraction of off-domain inputs that produced an *executable*
                command instead of <unknown>. With no cloud fallback this is the
                number that decides whether the thing is safe to ship.
  pin copy      fraction of pin-bearing commands whose emitted pin list is
                exactly the one in the utterance
  substitution  the failure this design exists to remove: an utterance naming a
                pin the board does not have, answered with a pin it does

The last two are new in v1 and the last one is not tradeable. v0 gave each
allowlisted pin its own symbol, so an illegal pin was unrepresentable -- which
does not yield a refusal, it yields the nearest legal pin. "switch off pin 100"
came back as <set> <p10> <low>: a real pin, switched off, silently. Now the pin
is copied digit by digit and frames.range_check() refuses it, so the same input
produces a message instead of an action, and any regression shows up here as a
substitution rather than hiding inside exact-match.

Note what false-accept counts: *executable*, not merely non-<unknown>. An
off-domain input answered with <set> <pin> 1 0 0 <high> is refused downstream
and actuates nothing, so it is not a false accept. `not-unknown` is reported
next to it because the two answer different questions -- one is safety, the
other is whether the model knows it is out of its depth.

All rates carry a 95% Wilson interval. Without it a 150-item set reads like a
precise number when one item moves it by 0.67%, and the first five rounds of
data tuning were steered by differences well inside the interval.

Sets whose rows carry `"locked": true` are scored but not diagnosed: no failure
listing, no per-source breakdown. Seeing *which* items fail is what turns a
held-out set into a training set, and the locked half exists to still be
held-out at the end. `--unlock` overrides it and says so in the output.

    uv run python src/evaluate.py --run runs/cmd-s0.pt
    uv run python src/evaluate.py --run runs/cmd-s0.pt --split gemini2_dev
    uv run python src/evaluate.py --run runs/cmd-s0.pt --constrained
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

import frames  # noqa: E402  (needs the data/ path above)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and 95% interval for k successes in n. Wilson rather than
    the normal approximation because the interesting rates here are near 0 --
    at 2/200 the normal interval runs negative."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def rate(label: str, k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{label}: {k}/{n} = {p:7.2%}   [{lo:.2%}, {hi:.2%}]"


class Decoder:
    """Wraps model + tokenizer and turns text into a Frame."""

    def __init__(self, run: Path, device: str = "cpu"):
        ck = torch.load(run, map_location=device, weights_only=False)
        self.cfg = Config(**ck["cfg"])
        self.model = TinyLM(self.cfg).to(device).eval()
        self.model.load_state_dict(ck["state"])
        self.device = device

        self.tok = Tokenizer.from_file(str(ROOT / "data" / "bpe.json"))
        self.meta = json.loads((ROOT / "data" / "meta.json").read_text())

        # A checkpoint stores ids, and prepare.py reassigns them every time the
        # corpus is rebuilt. Scoring across that boundary does not fail loudly:
        # digits keep decoding correctly while reserved symbols swap places, so
        # the run reads as a model that mysteriously forgot which action was
        # which. Refuse instead of reporting a number that means nothing.
        from train import tokenizer_fingerprint  # noqa: E402  (needs src/ path)
        want, have = ck.get("tok_fp"), tokenizer_fingerprint()
        if want is None:
            print(f"warning: {run.name} predates the tokenizer fingerprint; "
                  f"its ids may not match data/bpe.json. Retrain to be sure.",
                  file=sys.stderr)
        elif want != have:
            sys.exit(f"{run.name} was trained against tokenizer {want}, but "
                     f"data/bpe.json is {have}. The corpus has been rebuilt "
                     f"since; retrain, or restore the matching bpe.json.")

        self.inv = {v: k for k, v in self.tok.get_vocab().items()}
        self.reserved = set(self.meta["reserved"])
        self.go = self.meta["go"]
        self.end_id = self.tok.token_to_id(frames.S_END)

        # Legal first symbols, for constrained decoding.
        self.action_ids = [self.tok.token_to_id(s)
                           for s in frames.ACTION_SYM.values()]

    def _to_symbols(self, ids: list[int]) -> list[str]:
        """Token ids -> the symbol list frames.from_symbols expects.

        A straight lookup in v1. Every completion token is either a reserved
        symbol or a lone digit -- there are no spans to reassemble now that
        names are gone, and digits stay separate because prepare.py's
        Digits(individual_digits=True) never merges them."""
        return [self.inv.get(i, "") for i in ids]

    @torch.no_grad()
    def parse(self, text: str, max_new: int = 40,
              constrained: bool = False) -> tuple[frames.Frame | None, list[str], float]:
        ids = self.tok.encode(text).ids + [self.go]
        ids = ids[-(self.cfg.seq_len - max_new):]
        gen: list[int] = []
        logps: list[float] = []

        for step in range(max_new):
            x = torch.tensor([ids + gen], device=self.device)[:, -self.cfg.seq_len:]
            logits, _ = self.model(x)
            row = logits[0, -1]
            if constrained and step == 0:
                # The first emitted symbol is always an action. Nothing else is
                # reachable, so masking costs nothing and removes a whole class
                # of malformed output.
                keep = torch.full_like(row, float("-inf"))
                keep[self.action_ids] = row[self.action_ids]
                row = keep
            lp = torch.log_softmax(row, -1)
            nxt = int(row.argmax())
            logps.append(float(lp[nxt]))
            gen.append(nxt)
            if nxt == self.end_id:
                break

        conf = sum(logps) / max(1, len(logps))
        syms = self._to_symbols(gen)
        try:
            return frames.from_symbols(syms), syms, conf
        except frames.ParseError:
            return None, syms, conf


def load_split(split: str) -> list[dict]:
    """An eval/ set by name, falling back to a corpus/ split."""
    held = ROOT / "data" / "eval" / f"{split}.jsonl"
    src = held if held.exists() else ROOT / "data" / "corpus" / f"{split}.jsonl"
    return [json.loads(l) for l in src.open()]


def _pins(f: frames.Frame | None) -> list[int]:
    return [] if f is None else [t.n for t in f.targets
                                 if isinstance(t, frames.Pin)]


def _runs(f: frames.Frame | None) -> bool:
    """Would the device actually do something? Anything else -- <unknown>, a
    malformed generation, a frame range_check refuses -- actuates no pin."""
    return (f is not None and f.action is not frames.Action.UNKNOWN
            and frames.executable(f))


def score(dec: "Decoder", rows: list[dict], constrained: bool = False,
          show: int = 0) -> dict:
    """Run the decoder over one split and tally. Split out of main() so
    sweep.py scores identically rather than reimplementing the comparison."""
    pos_ok = pos_n = neg_bad = neg_loose = neg_n = malformed = 0
    pin_ok = pin_n = sub_bad = sub_n = 0
    fails: list[tuple[str, str, str]] = []
    subs: list[tuple[str, str, str]] = []
    by_action: Counter = Counter()
    by_action_ok: Counter = Counter()
    by_source_bad: Counter = Counter()

    for r in rows:
        gold = frames.from_symbols(r["symbols"])
        got, syms, _ = dec.parse(r["text"], constrained=constrained)
        if got is None:
            malformed += 1

        if gold.action is frames.Action.UNKNOWN:
            neg_n += 1
            if got is None or got.action is not frames.Action.UNKNOWN:
                neg_loose += 1
            if _runs(got):
                neg_bad += 1
                by_source_bad[r.get("stratum") or r["source"]] += 1
                if len(fails) < show:
                    fails.append((r["text"], "<unknown>", " ".join(syms)))
        else:
            pos_n += 1
            by_action[gold.action.value] += 1
            if got is not None and got.key() == gold.key():
                pos_ok += 1
                by_action_ok[gold.action.value] += 1
            elif len(fails) < show:
                fails.append((r["text"], " ".join(r["symbols"]), " ".join(syms)))

            if _pins(gold):
                pin_n += 1
                pin_ok += _pins(got) == _pins(gold)

            # Substitution: the utterance names something this board cannot do
            # -- an absent pin, an interval past the bounds, too long a chase --
            # and the model answered with something that runs anyway. Refusing
            # for the wrong reason still counts as safe here; only actuation is
            # the failure.
            if not frames.executable(gold):
                sub_n += 1
                if _runs(got):
                    sub_bad += 1
                    if len(subs) < show:
                        subs.append((r["text"], " ".join(r["symbols"]),
                                     " ".join(syms)))

    return {"pos_ok": pos_ok, "pos_n": pos_n, "neg_bad": neg_bad,
            "neg_loose": neg_loose, "neg_n": neg_n, "pin_ok": pin_ok,
            "pin_n": pin_n, "sub_bad": sub_bad, "sub_n": sub_n,
            "malformed": malformed, "fails": fails, "subs": subs,
            "by_action": by_action, "by_action_ok": by_action_ok,
            "by_source_bad": by_source_bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "cmd-s0.pt")
    ap.add_argument("--split", default="val",
                    help="'val' (generated, circular) or an eval/ set: "
                         "'massive' (human-written) / 'gemini' (second model)")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--show", type=int, default=12, help="sample failures to print")
    ap.add_argument("--unlock", action="store_true",
                    help="print failures for a locked set. Doing so spends it.")
    a = ap.parse_args()

    dec = Decoder(a.run)
    rows = load_split(a.split)[:a.limit]
    if not (ROOT / "data" / "eval" / f"{a.split}.jsonl").exists():
        print("NOTE: the generated split measures the generator against itself.\n"
              "      Use --split massive or --split gemini2_dev for a held-out number.\n")

    locked = any(r.get("locked") for r in rows)
    if locked and not a.unlock:
        a.show = 0

    s = score(dec, rows, constrained=a.constrained, show=a.show)

    print(f"run: {a.run.name}   split: {a.split}   "
          f"constrained={a.constrained}   n={len(rows)}")
    if locked:
        print("LOCKED set" + ("  -- --unlock given, this set is now spent"
                              if a.unlock else "  -- aggregate only"))
    leak = sum(1 for r in rows if r.get("in_train"))
    if leak:
        print(f"note: {leak}/{len(rows)} of these texts appear verbatim in the "
              f"training corpus")
    print()
    print(rate("exact-match  (in-domain) ", s["pos_ok"], s["pos_n"]))
    print(rate("false-accept (off-domain)", s["neg_bad"], s["neg_n"]))
    print(rate("  .. not <unknown>       ", s["neg_loose"], s["neg_n"]))
    print(rate("pin copy     (in-domain) ", s["pin_ok"], s["pin_n"]))
    print(rate("substitution (off-board) ", s["sub_bad"], s["sub_n"]))
    print(f"malformed generations    : {s['malformed']}")

    print("\nby action:")
    by_action, by_action_ok = s["by_action"], s["by_action_ok"]
    for act, n in by_action.most_common():
        print(f"  {act:<8} {by_action_ok[act]:5d}/{n:<5d} {by_action_ok[act] / n:7.2%}")
    if s["by_source_bad"] and (not locked or a.unlock):
        print("\nfalse accepts by source:")
        for src, n in s["by_source_bad"].most_common():
            print(f"  {src:<14} {n}")

    for title, items in (("sample failures", s["fails"]),
                         ("substitutions -- these actuate the wrong pin",
                          s["subs"])):
        if not items:
            continue
        print(f"\n{title}:")
        for text, want, got in items:
            print(f"  in   {text}")
            print(f"  want {want}")
            print(f"  got  {got}\n")


if __name__ == "__main__":
    main()
