# Wider eval batch — protocol

## Why

The held-out sets used through v1–v5 are too small to steer by:

| set | in-domain | refusals | one item is worth |
|---|---:|---:|---|
| `gemini`  | 150 | 100 | 0.67% / 1.0% |
| `massive` | 282 |  19 | 0.35% / **5.3%** |

Five rounds of data tuning were scored against those. By v5 the run-to-run
spread was the same size as the differences being chased — the v4→v5 "Gemini
regression" was 8 items on one seed. And the same failures had been read five
times, so the sets had partly become training data.

This batch fixes both: bigger denominators, and a half that stays unread.

## Steps

1. Run `data/eval_prompts/indomain.md` in a **fresh** AI Studio chat.
   Save the reply to `data/eval_raw/indomain_1.txt`.
2. Run it again in **another fresh chat**. Save to `indomain_2.txt`.
3. Same for `offdomain.md` → `offdomain_1.txt`, `offdomain_2.txt`.
4. `uv run python data/ingest_eval.py`
5. Read `data/eval/gemini2_review.txt` and fix or delete what it flags. This is
   reading *gold*, not model failures — it does not spend the locked half.
6. Re-run `ingest_eval.py` after any edit to the raw files.

Fresh chats, not follow-up turns: a second turn in the same chat sees the first
200 lines and steers away from them, which correlates the runs and breaks the
assumption that the two halves are exchangeable.

## What came out

900 generated lines in, 848 kept: **`gemini2_dev` 427 rows** (193 in-domain /
234 refusals) and **`gemini2_locked` 421 rows** (192 / 229). Diagnose dev
freely; locked is scored and never diagnosed — `evaluate.py` suppresses its
failure listing unless `--unlock`.

52 items were dropped in ingest, and the split of *why* is the useful part:

| | n | verdict |
|---|---:|---|
| duplicates | 30 | already in `gemini`/`massive`, or repeated across the two runs |
| unrepresentable | 2 | "chase all pins at 150ms forever" — `seq` cannot take `<all>` |
| not actually refusals | 20 | see below |
| gold disagreements | 0 | after the rule fixes, none |

The 20 dropped refusals were all the same mistake, and it is the one
`docs/GRAMMAR.md` exists to prevent: "read the pressure sensor", "check the
battery level", "turn on the other one" are *structurally legal* — a `read` or
a `set` on a name. Whether a pressure sensor can answer a digital read is the
alias table's problem, not the parser's. Labelling them `<unknown>` would have
trained and scored the model to reject unfamiliar nouns.

### First v5 measurement

|  | dev | locked | old `gemini` |
|---|---:|---:|---:|
| exact-match | 73.6% [66.9, 79.3] | 75.0% [68.4, 80.6] | 86.7% |
| false-accept | 19.2% [14.7, 24.8] | 16.6% [12.3, 22.0] | 7.0% |

The two halves agree well inside their intervals, which is what the stratified
split was for — the gap that matters later is dev-minus-locked *after* tuning,
and it starts at zero.

Against the old set the model looks 13 points worse and its false-accept rate
nearly triples. Nothing regressed; the old set was easier. The first Gemini
batch was almost entirely terse pin-numbered forms, which sit very close to
`realize.py`'s compact templates, and it had 100 refusals drawn from far fewer
categories. The gate — >95% / <2% — is a long way off, and that is the honest
starting point rather than the flattering one.

The strongest evidence that the wider batch is sound is that it lands on
**73.6% / 75.0% against MASSIVE's 73.4%** — the two independent sets now agree,
one written by crowdworkers who predate this project and one by a second model.
It was the narrow first batch at 86.7% that was the outlier.

Report both. The dev number is the one to tune against; the gap between them is
the estimate of how much of the dev number is tuning rather than capability.
The gate — >95% exact-match in-domain, <2% false-accept — is judged on
**locked**.

`in_train` is set per row when the text appears verbatim in `data/corpus/
train.jsonl`. Such rows are kept, not dropped: "turn on pin 5" being in the
corpus is a fact about how common the phrasing is, and removing it would bias
the eval toward exotic surface forms. The count is printed so the claim can be
stated honestly.

## Seed sweep — how much of v1→v5 was real

`uv run python src/sweep.py --tag v5 --seeds 0 1 2`. Corpus held fixed, only
the training seed moves. Spread is the half-range across the three runs.

| split | exact-match | false-accept | in-domain n |
|---|---:|---:|---:|
| `gemini2_dev` | 73.4% ±1.3 | 17.9% ±1.5 | 193 |
| `gemini2_locked` | 75.2% ±0.3 | 15.6% ±1.1 | 192 |
| `massive` | 70.4% ±3.2 | 8.8% ±2.6 | 282 |
| `gemini` (old) | 87.1% ±2.7 | 7.3% ±0.5 | 150 |

**The v4→v5 comparison was noise in both directions.** On the old `gemini` set
v5 read 5.3 points *below* v4, and that was written up as a regression worth
explaining; the seed range on that set is 5.4 points (127–135 correct). On
`massive` v5 read 5.4 points *above* v4 and was written up as the payoff of the
multi-word name fix; the seed range there is 6.4 points (189–207), and seed 2
alone lands at 67.0%, below v4. Neither movement survives its own spread. Five
rounds of tuning were steered by differences this measurement cannot resolve.

The other result is that **the diversified set is the more stable instrument,
and not because it is bigger**. `gemini2_locked` has 192 in-domain items to
`massive`'s 282 and still moves an order of magnitude less (±0.3 vs ±3.2).
`massive` is entirely `set`-on-a-name, so it measures one capability — name
copying — which is exactly the noisiest one. Spreading the same number of items
across six actions lets the per-action noise partly cancel. Size fixed the
false-accept denominator; composition fixed the variance.

So the resolution to work with from here: a change under ~1.5 points on
`gemini2` is not evidence, and one under ~6 points on `massive` never was.

These numbers belong in `RESULTS.md` when it is written.

## What ingesting it changed in `label_eval.py`

The rule labeller was written against the first batch, which was terse and
pin-numbered. Run as a general second opinion on natural phrasing it flagged 67
items — and **all 67 were the rules misreading the sentence**, not Gemini
mislabelling it ("drive pin 42 high" read as an alias literally called "drive
pin 42"; "read warning_led state" as one called "warning_led state"). A second
opinion that is wrong more often than the first is not a check. Acting on it
would have deleted 67 correct items.

So `cross_check` now reparses only utterances whose targets are literal pin
numbers, which is what the rules were built for and where a disagreement means
a real wrong pin or wrong level. `level_check` carries the phrasing-independent
part everywhere else. Ten genuine rule bugs surfaced along the way and were
fixed: spelled-out counts ("four times" read as forever), `<all>` targets
(unsupported entirely), multi-pin aliases, the `as` connector, names made of
function words ("toggle the"), names starting with our own verbs ("run a chase
on"), a bare `switch 12` read as low, bare "blink" read as `<stop>`, an
announced-but-cut-off rate ("blink pin 5 every"), and a count with no rate.

Every change was checked against `gemini.jsonl` and `massive.jsonl` byte-for-
byte, so the v1–v5 numbers remain comparable.

## Two things this batch deliberately does not measure

**Repeat count without a rate.** "blink pin 4 five times" is not representable —
`frames.validate` requires interval and count to travel together. It is absent
from the corpus in both directions, so scoring it either way would measure an
untrained case rather than a capability. The prompts exclude it. It belongs in
a grammar change, not here.

**Composition shift between generator and evaluator.** Gemini and `realize.py`
both draw on the same underlying distribution of English commands. This is an
independence check on phrasing, not a human baseline. `massive` remains the
only set written by people with no knowledge of this project.
