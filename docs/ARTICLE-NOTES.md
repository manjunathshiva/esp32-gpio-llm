# Notes for the write-up

Raw material, in the order things actually happened, including the wrong turns.
The wrong turns are the article — a post that only lists what worked is a
tutorial, and there are enough of those.

Every number here is three seeds on `v1_locked` unless said otherwise. Where a
claim is unresolved it says so.

---

## 1. The setup

A transformer trained from scratch, on a laptop, that parses English pin
commands entirely on an ESP32-S3. No WiFi, no cloud, no API key. 312,128
parameters, fp32, 1.2 MB mapped from flash, 113 ms–2 s per command.

The model emits a **symbol grammar**, not JSON:

```
turn on pin 4   ->   <set> <pin> 4 <high> <end>
```

`firmware/common/command.h` reassembles those symbols into a struct. There is no
string to mis-quote and no field name to misspell, so a malformed command is not
representable. That much was the plan and it worked.

## 2. The bug that reframed the project

The first grammar gave every allowlisted pin its own token — `<p4>`, `<p18>`,
25 of them. An illegal pin was therefore *unrepresentable*, which sounds like a
safety guarantee.

It is the exact opposite. Measured on held-out data:

```
switch off pin 100        ->  <set> <p10> <low>     switches pin 10
name pin 35 as back door  ->  <alias> <p39> ...     renames pin 39
blink pin 8 at 12000 ms   ->  <blink> <p8> 1200     runs at 1200ms
```

A model that *cannot say* "pin 100" does not refuse. It says the nearest thing
it can. And pin 10 is a real pin on the board, so the command validated cleanly
and actuated the wrong hardware, silently, with nothing in the logs.

**An unrepresentable request does not produce a refusal. It produces a plausible
substitute.** That single sentence reorganised the whole design: pin numbers are
now emitted digit by digit, and `gpio_control.c` — which already owned the board
allowlist — does the refusing. The model transcribes; the hardware judges.

Same failure then turned up four more times in slots nobody had checked:

| input | did | should |
|---|---|---|
| `stop pins 4 and 5` | **started a chase** | stop both |
| `turn on pins 4-8` | pin **48** | refuse |
| `turn on pin 4 5 and 6` | pin 4 only | all three |
| `blink pin 4 five times` | invented 100 ms, forever | 5 cycles |

Each had a different cause and the same shape. `set` accepted one target,
`read` one, `blink` one, `stop` one — four separate limits, four separate silent
failures. The fix was a rule, not four patches: *every action that takes a pin
takes a list of pins.*

## 3. Three LEDs beat the entire eval suite

The multi-pin bug was found by wiring LEDs to pins 4, 5 and 6 and typing
"turn on pin 4 5 and 6". One lit.

It was invisible to every metric in the project, and *necessarily* so: the
held-out set had been collected against the old grammar, so it contained no
utterance that grammar could not express. **An eval derived from a grammar
cannot find what that grammar cannot say.** Hardware is not redundant with a
held-out set; it covers the blind spot the set has by construction.

## 4. What did not work — five things

**More parameters.** Twice tested, twice refuted. d128/L6/F256 — 1.1M params,
4.9× — read 88.1% exact-match against the 230K baseline's 87.0%, and *below* the
312K config that won. Training loss halved; free-running accuracy did not move;
false-accept was identical to the decimal. Width and FFN buy nothing here.

**Untied embeddings.** Speculated about in a code comment since the first
commit. Measured: 86.7% exact and 6.0% substitution against 87.0% and 2.8%
tied — worse. Digits are the one token appearing in both input and output roles,
and a copy task apparently wants them sharing a vector.

**More training data for the failing shape.** Long pin lists were genuinely
under-covered — 1.76% of corpus rows against 6.08% of the eval. Closing that gap
exactly moved exact-match **+0.5 against a ±1.4 spread** and cost 1.5 points of
false-accept. A real gap, correctly identified, and not the cause.

**My own predictions.** I predicted 8 heads at d_model 64 would hurt, because
head_dim drops to 8. It was among the best configs. What actually helped:

| config | params | exact | pin copy |
|---|---:|---:|---:|
| d64 L4 H4 | 229,952 | 87.0% | 90.5% |
| d64 L6 H8 | 312,128 | **90.4%** | **93.9%** |
| d128 L6 F256 | 1,115,776 | 88.1% | 93.0% |

The task is *tracking position* through a long run of near-identical
`<pin> d d` tokens. That wants more independent attention patterns, not fatter
ones. Head count and depth; not width.

**A held-out set of 96 items.** After rescoping, the eval had 96 in-domain rows
and reported 94.4% exact-match and *zero* substitutions. Rebuilt at 466 rows
with numeric edge cases as a first-class stratum, the same model read 87.0% and
2.8% substitution. Nothing regressed — the small set had inherited the old
grammar's composition and could not contain the hard shapes.

## 5. Measurement traps, each hit more than once

**Seed spread is not measurement precision.** Half-range across three seeds says
how much the model moves between training runs. It says nothing about how
precisely n items estimate a rate. I called a dev/locked gap "non-overlapping"
from ±0.5 and ±0.8 seed spreads and concluded the split was broken. The Wilson
intervals were 7.5% [4.6, 12.0] and 4.5% [2.4, 8.3] — heavily overlapping. There
was no imbalance; there was a rare event and n=200.

**"It already refuses that" is not evidence.** `except` phrasings appeared in
**1 of 147,000** corpus rows and refused correctly — by luck. The moment
multi-pin data made `<all>` a better match, "turn on all pins except 4 and 5"
turned them on. Accidental correctness survives exactly until the distribution
moves, then fails in whichever direction the model happens to lean.

**A second opinion that is wrong more often than the first is noise.** The rule
labeller flagged 51 gold disagreements in one batch. About half were the rules
being wrong — Hz never converted to ms, "twice a second" read as a *count* of
two, multi-pin blink capturing one pin, `disable pin 3` read as `<stop>` when
the corpus says set-low. Acting on the flags would have deleted correct gold and
biased the eval against the exact shapes under test.

**A register only one side of the corpus has becomes a shortcut.** Negatives
were 30% of the corpus but only **15% of its ALL-CAPS rows**, because the
positive realizer shouted 4% of the time and the negative generator never did.
The model learned that shouting means a command: it executed "SET PIN 10 TO 25
PERCENT BRIGHTNESS" and refused the same sentence in lower case.

**Two sources of truth for one fact will disagree silently.** `train.py`
repeated the model shape in its argparse defaults. I changed `model.py`, and
every run afterwards came out with the old architecture while all three
correctness gates passed — because the gates check that C matches PyTorch, not
that PyTorch is the model you intended. The only tell was the training time.

## 6. Where it actually stands

Honest, and the article should say so:

| metric | now | gate |
|---|---:|---:|
| exact-match | 90.4% ±2.0 | > 95% |
| false-accept | 5.0% ±1.7 | < 2% |
| pin copy | 93.9% ±1.7 | > 99% |
| substitution | 2.8% ±1.5 | 0 at n ≥ 300 |

Two gates fail. Substitution concentrates in out-of-bounds *rates* (10.9%
pooled) rather than out-of-range pins (4.6%) — five-digit literals like 50000 ms
lose a digit. That is the next thread.

## 7. The line to end on

The interesting claim is not that a 312 KB model runs on a microcontroller.
It is that **making illegal states unrepresentable is only safe when the model
can still say what it heard.** Take away its ability to express something wrong,
and it will express something plausible instead — and on hardware, plausible is
worse than wrong, because wrong announces itself.
