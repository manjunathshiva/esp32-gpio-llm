"""Train the command parser.

Adapted from esp32-tinyllm's train.py. The substantive change is the loss: that
project trained a plain next-token LM over a continuous token stream, this one
trains fixed-width (prompt, completion) pairs and scores **only the
completion**. `data/prepare.py` writes the mask.

Reported val loss is therefore per emitted symbol, not per token of English, and
is not comparable to anything from the other project.

    uv run python src/train.py --steps 6000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import Config, make_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"


def tokenizer_fingerprint() -> str:
    """Identify the tokenizer a checkpoint was trained against.

    A checkpoint stores token *ids*. `prepare.py` retrains the BPE whenever the
    corpus is rebuilt, and the symbol ids move when it does -- so a checkpoint
    scored against a tokenizer it was not trained on decodes one reserved symbol
    as another. It does not crash and it does not look like a bug: pins copy
    perfectly, actions come out wrong, and the run reads like a model that
    failed to learn. That cost a full evaluation round, so the fingerprint is
    written into every run and checked on load.
    """
    import hashlib

    h = hashlib.sha256()
    for p in (DATA / "bpe.json", DATA / "meta.json"):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Batcher:
    """Whole examples, not windows into a stream. Each row is one command."""

    def __init__(self, split: str, meta: dict, batch_size: int, device: str,
                 seed: int = 0):
        n, s = meta[f"n_{split}"], meta["seq_len"]
        self.tok = np.fromfile(DATA / f"{split}_tok.u16", dtype=np.uint16).reshape(n, s)
        self.mask = np.fromfile(DATA / f"{split}_mask.u8", dtype=np.uint8).reshape(n, s)
        self.bs, self.n, self.device = batch_size, n, device
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        i = self.rng.integers(0, self.n, self.bs)
        t = torch.from_numpy(self.tok[i].astype(np.int64)).to(self.device)
        m = torch.from_numpy(self.mask[i].astype(np.int64)).to(self.device)
        # Next-token prediction: position j predicts token j+1, and counts only
        # if token j+1 belongs to the completion.
        return t[:, :-1], t[:, 1:], m[:, 1:]


@torch.no_grad()
def evaluate(model, b: Batcher, iters: int) -> tuple[float, float]:
    """Returns (masked CE, teacher-forced per-symbol accuracy).

    The accuracy is a cheap training-time proxy. It is NOT the project's gate,
    which is exact-match on whole commands from a human-written held-out set.
    """
    model.eval()
    total, right, seen = 0.0, 0, 0
    for _ in range(iters):
        x, y, m = b()
        logits, loss = model(x, y, m)
        total += loss.item()
        pred = logits.argmax(-1)
        right += ((pred == y) & (m > 0)).sum().item()
        seen += int((m > 0).sum().item())
    model.train()
    return total / iters, right / max(1, seen)


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=40)
    _d = Config()
    ap.add_argument("--d-model", type=int, default=_d.d_model)
    ap.add_argument("--n-layers", type=int, default=_d.n_layers)
    ap.add_argument("--n-heads", type=int, default=_d.n_heads)
    ap.add_argument("--ffn-hidden", type=int, default=_d.ffn_hidden)
    ap.add_argument("--untied", action="store_true",
                    help="separate output head instead of tying it to tok_emb")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    meta = json.loads((DATA / "meta.json").read_text())
    device = get_device()
    print(f"device: {device}   corpus: {meta['n_train']:,} train / {meta['n_val']:,} val")

    cfg = Config(vocab_size=meta["vocab"], d_model=a.d_model, n_layers=a.n_layers,
                 n_heads=a.n_heads, ffn_hidden=a.ffn_hidden, seq_len=meta["seq_len"],
                 tie_embeddings=not a.untied)
    model = make_model(cfg).to(device)

    decay = [p for _, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for _, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95))

    train_b = Batcher("train", meta, a.batch_size, device, a.seed)
    val_b = Batcher("val", meta, a.batch_size, device, a.seed + 1)

    RUNS.mkdir(exist_ok=True)
    name = f"cmd{'-' + a.tag if a.tag else ''}-s{a.seed}"
    hist, best, t0 = [], float("inf"), time.time()

    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)

        x, y, m = train_b()
        _, loss = model(x, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.eval_every == 0 or step == a.steps - 1:
            vl, acc = evaluate(model, val_b, a.eval_iters)
            best = min(best, vl)
            hist.append({"step": step, "train": loss.item(), "val": vl, "sym_acc": acc})
            print(f"step {step:6d}  train {loss.item():.4f}  val {vl:.4f}  "
                  f"symbol-acc {acc:.4f}  {time.time() - t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    fp = tokenizer_fingerprint()
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict(), "tok_fp": fp},
               RUNS / f"{name}.pt")
    (RUNS / f"{name}.json").write_text(json.dumps({
        "cfg": cfg.__dict__, "params": model.param_count(), "steps": a.steps,
        "batch_size": a.batch_size, "lr": a.lr, "seed": a.seed, "tok_fp": fp,
        "best_val": best, "seconds": elapsed, "history": hist}, indent=2))
    print(f"\n{elapsed:.0f}s   best val {best:.4f}   -> runs/{name}.pt")


if __name__ == "__main__":
    main()
