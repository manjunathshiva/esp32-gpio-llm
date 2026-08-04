# v1 scope — pin numbers only

Status: **steps 1–5 built** (grammar, eval re-scope, corpus, training).
Step 3 (a v1-scoped in-domain batch) and step 6 (the C runtime) are open.
Measured results are at the end; `docs/GRAMMAR.md` describes what was built.

## The cut

v1 accepts commands that name a **pin by number**. It has no aliases, no names,
no `alias` action.

```
turn on pin 5              ->  execute
blink pin 7 every 500ms    ->  execute
turn on pin 100            ->  refuse: not a GPIO on this board
turn on the bedroom light  ->  refuse: v1 does not do names
what's the weather         ->  refuse
```

Five actions survive: `set`, `read`, `blink`, `seq`, `stop`. `alias` goes,
because it exists only to create names.

### Why

Exact-match on the wider eval, split by what the command targets (v5, 3 seeds,
half-range):

| target | n | exact-match |
|---|---:|---:|
| pin number | 133 | **93.5% ±4.1** |
| name | 237 | **62.4% ±1.9** |
| all pins | 15 | 91.1% ±6.7 |

Names are two-thirds of the eval and carry nearly all of the error. Removing
them moves the headline from ~75% to ~93.5% without touching the model. It also
removes `alias`, the worst action at 44%, and it dissolves the ambiguity class
that cost 20 items in ingest — with no Name slot, "read the pressure sensor" is
unambiguously a refusal instead of a judgement call.

## The part of the plan that does not work as written

"Pin 100 is invalid" is the *worst* thing this model currently does, not an
easy extra. Numerically-illegal-but-well-formed input (`neg7`) false-accepts at
**33.3%** against 16.8% overall. And the failure is not a missed refusal — it is
silent digit truncation into a valid command on the wrong pin:

```
switch off pin 100        ->  <set> <p10> <low>        turns off pin 10
name pin 35 as back door  ->  <alias> <p39> ...        renames pin 39
name pin 20 to front      ->  <alias> <p2>  ...        renames pin 2
blink pin 8 at 12000 ms   ->  <blink> <p8> <int> 1200  runs at 1200ms
blink pin 10 at 50000 ms  ->  <blink> <p10> <int> 5000
```

Over-long chases behave the same way: 9 of 13 truncate to 6 pins and are
accepted.

**Cause.** The pin allowlist is baked into the *symbol table* — 25 pin tokens,
no `<p100>`. The model physically cannot represent "pin 100", so it emits the
nearest thing it can. "Illegal pins are unrepresentable" reads like a safety
property and is the opposite of one: it converts a rejection into a
substitution, and for GPIO control a substitution is a wrong physical action
taken silently.

The C runtime cannot catch this. By the time the frame arrives, `<p10>` is a
perfectly legal symbol and the input digits are gone.

Intervals do not have this problem structurally — they are already emitted
digit-wise, so `12000` *is* representable. There the model simply never saw an
out-of-range interval in training. Two different bugs, two different fixes.

**A pins-only v1 does not reduce this. It makes it the whole problem.**

## The grammar change

Make pins digit-wise like intervals, and move range validation into the
runtime.

```
                       v0                          v1
turn on pin 5     <set> <p5> <high>           <set> <pin> 5 <high>
turn on pin 100   (unrepresentable)           <set> <pin> 1 0 0 <high>
chase 4 5 6       <seq> <p4> <p5> <p6>        <seq> <pin> 4 <pin> 5 <pin> 6
```

The model transcribes the number it heard. `gpio_control.c` — which already
carries the board allowlist — decides whether that number is a GPIO. This is
the same split `docs/GRAMMAR.md` already makes for names ("structure decides
the parse; the alias table decides the execution"), applied to numbers.

Three consequences worth stating plainly:

1. **The model never learns the board.** Swap S3 for C3 and only the C
   allowlist changes. Today a different pinout means a retrain.
2. **The failure becomes visible.** "Did the emitted digits match the digits in
   the text" is checkable against the input with no gold label, so wrong-pin
   errors can be counted directly instead of inferred.
3. **The user-facing error improves.** "pin 100 is not a GPIO on this board"
   instead of a generic "I didn't understand that."

The user-visible behaviour the plan asked for is unchanged — `turn on pin 100`
is still refused. Only the layer that refuses it moves.

### Symbol table

Removed: `<p1>`…`<p48>` (25), `<alias>`, `<name>`, `<nend>`.
Added: `<pin>`, taking the digit run that follows it.
Digits `0`–`9` are already in the vocabulary, shared with `<int>` and `<cnt>`.

Net vocabulary shrinks. A digit run terminates at the next non-digit symbol, so
`<pin> 1 0 0 <low>` and `<seq> <pin> 4 <pin> 5` both parse without a delimiter.

### The cost, honestly

Sequences get longer. A single-digit pin goes from 1 token to 2, a two-digit
pin to 3. A six-pin chase goes from 6 tokens to as many as 18.

`seq` is currently the only action at 100%. It is the one most exposed by this
change and the one to watch. `Config.seq_len` may need to grow, which costs
device compute in Phase 5.

## Runtime verdicts

Parsing and legality become separate answers. Three outcomes, not two:

| input | model emits | runtime |
|---|---|---|
| `turn on pin 5` | `<set> <pin> 5 <high>` | execute |
| `turn on pin 100` | `<set> <pin> 1 0 0 <high>` | refuse — pin not on board |
| `blink pin 7 at 12000ms` | `<blink> <pin> 7 <int> 1 2 0 0 0` | refuse — interval out of bounds |
| `chase 1 2 3 4 5 6 7 8` | `<seq>` + 8 pins | refuse — chase too long |
| `turn on the bedroom light` | `<unknown>` | refuse — v1 does not do names |
| `what's the weather` | `<unknown>` | refuse |

`frames.validate()` currently conflates these — it enforces both syntax and
range. Split it: `validate()` keeps the syntactic invariants (interval and
count travel together; `seq` never takes `<all>`), and a new range check owns
`PINS_S3`, `INTERVAL_MIN/MAX`, and the chase-length cap. The range check has a
mirror in C and is the only place the board is described.

## Data changes

**Positives.** `sample_frame` / `realize.py` stop drawing `Name` targets and
the `alias` action. Pin numbers are drawn from a range *wider than the
allowlist* — the legal 25 over-represented (real usage is legal), with a
deliberate minority out of range so out-of-range digits are ordinary things to
transcribe rather than a distribution the model has never seen. Same for
intervals, counts, and chase length. Gold for all of these is the verbatim
frame; the runtime does the refusing.

**Negatives flip label.** This matters more than adding data:
`negatives.py`'s `_NEAR_MISS_PINS` items are currently trained as `<unknown>`.
Under v1 they are *correct parses that the runtime refuses*. Leaving them as
negatives would train exactly the behaviour being fixed. They must move, not be
deleted.

What stays a negative is now purely structural: no actuation verb, a question
about the world, an unsupported modality (colour, brightness, temperature), a
schedule, chit-chat.

**Name rejection is v1 debt and must be marked as such.** With no alias table,
"turn on the bedroom light" has to go somewhere, and `<unknown>` is the only
honest answer. The model rejects the *shape* — verb plus noun phrase — not the
specific noun, so this does not violate the GRAMMAR.md principle. But v2 has to
unlearn it. Keep these rows in their own file with their own stratum tag so v2
drops them with one flag, and never fold them into the general negatives.

## The eval problem this creates

The wider batch was built for the full grammar, and **about 60% of its
in-domain items are name-targeted**. Under v1 scoring they stop being in-domain.
In-domain n falls from 385 to roughly 148 across both halves — about 74 each,
*smaller than the old `gemini` set that was abandoned for being too small*.

Two moves, both worth making:

**Free.** The 237 name items become v1 refusals as they stand, correctly
labelled, at no cost. Off-domain grows to roughly 700. This is a strong
name-rejection measurement that did not exist before.

By the same logic **`massive` flips from the in-domain benchmark to the refusal
benchmark** — it is entirely `set`-on-a-name, so under v1 all 282 of its
in-domain rows are refusals. A human-written, project-blind set of 282 name
rejections is a better instrument than what it was replacing.

**Not free.** A v1-scoped in-domain batch, reusing the protocol in
`data/eval_prompts/README.md` unchanged — fresh chats, two runs, stratified
dev/locked split, `ingest_eval.py`. Roughly 300 pins-only in-domain plus 150
numeric-edge items (out-of-range pins, out-of-bounds intervals and counts,
over-long chases), since numeric edges are now the whole game rather than one
category in nine.

`ingest_eval.py`'s `cross_check` gets *easier*: it was restricted to
pin-numbered utterances because the rule labeller misreads natural phrasing,
and under v1 every in-domain utterance is pin-numbered. It becomes a general
second opinion rather than a narrow one.

## Gates

The existing gate stays, with two additions that the old grammar could not
express:

| metric | gate |
|---|---|
| exact-match, in-domain | > 95% |
| false-accept, off-domain → executable | < 2% |
| **numeric transcription** — emitted digits equal the digits in the text | **> 99%** |
| **wrong-pin rate** — an input naming an illegal pin producing a legal one | **0 observed, n ≥ 300** |

Judged on the locked half, dev/locked discipline unchanged.

Numeric transcription is set high because it is pure copying, not judgement.
Wrong-pin rate is the one that cannot be traded against the others: it is the
failure that actuates hardware. At n=300 with 0 observed the Wilson upper bound
is ~1.2%, which is the honest way to state it.

Measurement resolution from the seed sweep still applies — a change under ~1.5
points on `gemini2` is not evidence. Every gate number is a 3-seed mean.

## Order of work

Eval before training, so there is a before-number and the scorer is validated
against a model that already exists.

1. **Grammar.** `frames.py`: `<pin>` digit runs, `alias`/`<name>`/`<nend>`
   removed, `validate()` split from the range check. Update `docs/GRAMMAR.md`.
   *Gate:* `from_symbols(to_symbols(f)) == f` over generated frames; every
   existing corpus row re-serializes.
2. **Eval re-scope.** Relabel the existing sets under v1 policy; teach
   `evaluate.py` the three-way verdict and the two new metrics.
   *Gate:* score the current v5 checkpoints under v1 scoring. This is the
   baseline and it costs nothing.
3. **New in-domain batch.** Prompts, two fresh chats each, ingest, review.
4. **Corpus.** Regenerate positives with wide numeric ranges; flip the
   near-miss negatives; add the marked name-rejection file.
   *Gate:* composition report — out-of-range values must appear often enough in
   pin-bearing positives that the prior is not "a number after 'pin' is legal".
5. **Retrain**, 3 seeds, `sweep.py`. Per-action table, watching `seq`.
6. **C runtime** (the old Phase 4) — `llm.h`, structured head, digit-split in
   `tokenizer.h`, `verify.c`, and the range check mirrored from step 1.

## Risks

- **`seq` regression** from longer sequences. It is at 100% now and has the
  most to lose. Per-action reporting, not just the headline.
- **Three-digit transcription** is the least-practised path and the one the
  safety gate depends on. It needs deliberate sampling weight, not incidental
  coverage.
- **v2 pays for v1.** Name rejection has to be unlearned. Contained by keeping
  it in its own file, but it is real and it is why that file is tagged.
- **The in-domain denominator shrinks** until step 3 lands. Numbers between
  steps 2 and 3 are directional only.

## Measured

3 seeds, spread is half-range. `uv run python src/sweep.py --tag v1`.

| split | exact-match | false-accept | pin copy | substitution | in-domain n |
|---|---:|---:|---:|---:|---:|
| `v1_gemini2_dev` | 95.4% ±1.6 | 0.7% ±0.3 | 98.1% ±1.1 | 0.0% ±0.0 | 95 |
| `v1_gemini2_locked` | **95.8% ±1.0** | 2.8% ±0.6 | 97.2% ±0.6 | 0.0% ±0.0 | 96 |
| `v1_massive` | — | 0.1% ±0.2 | — | — | 0 |

On `locked`, exact-match clears >95% and substitution is 0 across all three
seeds. **Two gates remain unmet**: false-accept 2.8% against <2%, and pin copy
97.2% against >99%. Substitution reads 0/21 per seed where the gate asks for
n ≥ 300, so it is not yet evidence of much.

The dev−locked gap that opened after the first round of tuning (96.5 / 93.8) has
closed to 95.4 / 95.8, which is the reading that gap should have had all along:
noise plus a little dev-fitting, not a durable difference.

**None of this measures the multi-pin fix.** The eval sets are migrated from v0,
and v0 could not represent a multi-pin `set`, so not one such utterance exists
in them to score. The movement here comes from the `everything off` label fix
and from seed variation. The fix is verified on hardware and by the C pipeline,
not by these numbers — which is the same blind spot described above, seen from
the other side.

### The headline number is mostly scope, not capability

v5 read 75.2% exact-match on the old locked half and v1 reads 93.8%, but the
two are not measuring the same items — v1's in-domain set is v0's *minus* every
name-targeted command. Split by class, against v5's 93.5% ±4.1 on pin-targeted
items:

| class | dev | locked | what v0 scored |
|---|---:|---:|---|
| `command` — pin-targeted, executable | 97.3% | 93.8% | 93.5% ±4.1 |
| `range_command` — names a value the board lacks | 93.9% | 93.7% | **0%, by construction** |

On like-for-like items the model did not get better; the difference sits inside
v5's own spread. **The real result is the second row.** Those utterances were
unrepresentable in v0 — not merely wrong, but wrong in the specific way that
turns a refusal into an action on a different pin. They are now parsed correctly
~94% of the time and refused by `range_check`, and the substitution rate that
measures the old failure runs 0–1.6%.

The other genuine movement is false-accept, 15.6% → 3.2% on locked, though that
set also changed composition.

### A dev−locked gap opened, and it is the expected kind

Before tuning: dev 95.4% / locked 95.1% exact, false-accept 3.4% / 3.8%. After
one round of corpus fixes diagnosed on dev: dev 96.5% / locked 93.8%, and
false-accept 1.7% ±0.6 / 3.2% ±0.5 — intervals that no longer overlap.

The fixes were principled (a register artifact confirmed in the corpus, not
phrasings copied from failures) and they still moved dev roughly twice as far as
locked. Locked exact-match went *down* 1.3 points, inside its own spread but not
in the direction claimed. That is precisely what the stratified split is for,
and the reading is: **stop tuning.** At n=95 in-domain and n=21 substitution the
instrument cannot resolve the changes now being attempted, which is the same
mistake v1–v5 made on the old sets. Step 3 — a v1-scoped batch, with the numeric
edge cases as a first-class stratum — is the blocking item, not more corpus
work.

### What the hardware test found that the eval could not

Wiring LEDs to pins 4, 5 and 6 and typing *"turn on pin 4 5 and 6"* turned on
pin 4 alone. Not a parse failure: `set` was defined as exactly one target, so
the corpus held **0 multi-pin `set` rows out of 37,226**, and the model emitted
the only thing it could represent. Two requested pins stayed dark and the device
reported success.

This is the substitution failure again, in a slot nobody checked. The numeric
version was found by building a metric for it; this one was invisible to every
metric here, because a held-out set collected against the v0 grammar contains no
utterance the v0 grammar could not express. **An eval derived from a grammar
cannot find what that grammar cannot say.** LEDs could, in about a minute.

Fixed by allowing `set` any number of pins (`frames.validate`, `sample_frame`,
`realize._set_ref`, `command.h`, and a loop in `gpio_control.c`); multi-pin sets
are now 18.9% of `set` rows. The general form of the lesson is in
`docs/GRAMMAR.md` §1: ask of every slot what a speaker might legitimately say
that the shape cannot hold, because the answer is never a refusal.

### What one round of corpus work did fix

Diagnosing dev found a real generator artifact rather than missing phrasings:
negatives were 30% of the corpus but only **15.1% of its ALL-CAPS rows**, because
`realize._decorate` shouts 4% of positives and `negatives.sample_negative` never
did. Upper case had become evidence of a command — "SET PIN 10 TO 25 PERCENT
BRIGHTNESS" was executed while the same sentence in lower case was refused. The
same asymmetry existed for politeness prefixes and suffixes. Negatives now go
through the positives' own `_decorate`, and the fix is structural: any register
one side of the corpus gets and the other does not becomes a shortcut.

## Out of scope, and when it comes back

- **Names and aliases** — v2. The full grammar, the alias table, and the
  `alias` action return; the v1 name-rejection file is dropped in the same
  change.
- **Repeat count without a rate** ("blink pin 4 five times") — still not
  representable, still absent from the corpus in both directions. A grammar
  change, unrelated to this one.
- **A human baseline.** `massive` is the only project-blind set and under v1 it
  measures refusals, not commands.
