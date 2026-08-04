# Semantic frames and the symbol grammar

The two artifacts everything downstream keys off. The frame is the canonical
meaning; the symbol grammar is how the model emits it. `data/frames.py` samples
frames, `data/realize.py` renders them to English, `src/evaluate.py` compares
parsed frames, and `firmware/common/command.h` reconstructs a frame from symbols.

**Change these before Phase 1, not after.**

---

## 1. The frame

**v1 targets pins by number only** — no names, no aliases. See
`docs/V1-SCOPE.md` for the scope decision; this section describes what is built.

```python
Target = Pin(n: int) | All

@dataclass
class Frame:
    action:      Action              # set read blink seq stop unknown
    targets:     list[Target]
    level:       Level | None        # high low toggle
    interval_ms: int | None
    count:       int | None          # 0 = infinite
```

Validity by action. The sampler only produces these shapes; the parser only
accepts them; anything else is a bug in one of the two.

| action | targets | level | interval_ms | count |
|---|---|---|---|---|
| `set` | ≥1 pins, or `All` | **required** | — | — |
| `read` | 1 | — | — | — |
| `blink` | 1, or `All` | — | together | together |
| `seq` | ≥ 2 | — | together | together |
| `stop` | 0 (= all), or 1 | — | — | — |
| `unknown` | — | — | — | — |

"together" means the two slots are supplied as a pair or not at all; omitting
both means the device default applies.

`set` takes any number of pins and `seq` at least two, so both are bounded by
`range_check`, not by shape — a chase is capped at six by the animation slots,
while setting a level is a loop with no such limit.

**`set` was one target until an LED test caught it.** "Turn on pins 4, 5 and 6"
had no representation, the corpus contained not one example in 37,226 `set`
rows, and the model did what an unrepresentable request always makes it do: it
emitted the nearest thing it could, `<set> <pin> 4 <high>`, and executed it. One
pin moved, two the speaker asked for did not, and nothing reported a problem.
Exactly the failure mode this grammar was rebuilt to remove — fixed for numbers
in the same pass that left it standing for target counts. The lesson generalises
past this instance: **an unrepresentable request does not produce a refusal, it
produces a plausible substitute**, so every slot needs asking what a speaker
might legitimately say that the shape cannot hold.

### Shape is the parser's job, range is the hardware's

`validate()` enforces the table above. It does **not** enforce
`interval_ms ∈ [50, 10000]`, `count ≥ 0`, the 25-pin ESP32-S3 allowlist, or the
six-pin chase limit. Those are `range_check()`, mirrored in `gpio_control.c`,
and a frame that fails them still parses:

```
turn on pin 100   ->  <set> <pin> 1 0 0 <high> <end>   ->  refused: not a GPIO
```

This split is the correction v1 exists to make. v0 put the allowlist in the
*symbol table* — one token per legal pin — so an illegal pin was
unrepresentable. That reads like a safety property and is the reverse of one: a
model that cannot say "pin 100" does not refuse, it says the nearest thing it
can. Measured on held-out data, "switch off pin 100" came back as
`<set> <p10> <low>` — a real pin, switched off, silently; "name pin 35 as the
back door" reached `<p39>`. A rejection failure had become a wrong-actuation
failure, which is strictly worse on hardware.

So the model transcribes and the runtime judges. Three consequences:

- the model no longer encodes the board, so ESP32-S3 → WROOM is a change to
  `gpio_control.c` and not a retrain;
- the error message improves — "pin 100 is not a GPIO on this board" rather
  than "I didn't understand that";
- the failure becomes *countable*: `evaluate.py`'s substitution metric asks
  whether an utterance naming an absent pin produced a present one, which was
  not expressible while illegal pins had no representation.

### Ambiguity is designed out, not resolved

An alias maps a name to **one or more pins**. "The lights" is not three pins
competing to answer — it is a single group alias.

This removes the disambiguation problem entirely: there is never a case where
the device must ask "which one?", so there is no cross-turn state, no clarifying
question, and no conversation. Every utterance is independently parseable.

`All` stays separate from a group alias: it means every allowed pin, and needs
no table entry.

### Structure decides the parse; the alias table decides the execution

The model never judges whether a name exists. *"Turn on the aquarium pump"*
parses to `<set> <name> aquarium pump <nend> <high> <end>` whether or not that
alias has been created; the runtime then either resolves it or reports that it
does not know the name.

This is why *"turn on the tv"* is **not** a training negative. It is a
structurally valid command with an unresolvable target. Labelling it `<unknown>`
would teach the model to reject commands based on the object noun — and a model
that rejects "turn on the tv" will also reject "turn on the aquarium pump",
which is a legitimate alias it has simply never seen.

`<unknown>` is for utterances with no command *structure*, or with structure the
grammar cannot express (dimming, colour, scheduling). Not for unfamiliar nouns.
`data/massive.py` enforces this by dropping command-shaped rows rather than
labelling them, using verb openings derived from `realize.py` so the two cannot
drift apart.

---

## 2. The symbol grammar

One shared BPE vocabulary covers input words, digits, and the special symbols,
so the model can copy name and number tokens straight out of the input.

| group | symbols | count |
|---|---|---|
| actions | `<set> <read> <blink> <seq> <stop> <unknown>` | 6 |
| targets | `<pin>` + digits, `<all>` | 2 |
| levels | `<high> <low> <toggle>` | 3 |
| numeric slots | `<int> <cnt>` | 2 |
| terminator | `<end>` | 1 |

**14 reserved ids**, down from ~66 — the 25 pin symbols and the three name
symbols are gone. Digits `0`–`9` are ordinary vocabulary entries, kept
un-mergeable by `prepare.py`'s `Digits(individual_digits=True)`. The rest of the
1,024-token vocabulary is BPE over the command corpus plus 256 byte tokens, so
any input tokenizes.

### Emission order is fixed

```
ACTION  TARGET*  [LEVEL]  [<int> digits]  [<cnt> digits]  <end>
```

Fixed order is deliberate. It makes constrained decoding a small DFA and removes
a degree of freedom the model would otherwise waste capacity learning.

### Every number is copied digits

There is one mechanism now, not two. `<pin>`, `<int>` and `<cnt>` each introduce
a run of digits that ends at the next symbol which is not a digit — so v1 has no
`<nend>` either. With names gone the only thing it terminated was a number, and
a number already terminates itself.

- **Pins are open.** `<pin> 1 0 0` is exactly as emittable as `<pin> 4`. What
  the board actually has is `range_check()`'s business (§1).
- **Intervals and counts are open.** 137 ms works without ever appearing in
  training. Bucketing could not express it.

The cost is length: a two-digit pin is 3 tokens where v0 used 1, so a six-pin
chase can reach 18 tokens against v0's 6. That is the price of the safety
property in §1, and `seq` is the action to watch for it.

### Examples

```
turn on pin 4
  <set> <pin> 4 <high> <end>

turn on pin 100                        parses; refused by range_check
  <set> <pin> 1 0 0 <high> <end>

blink pin 4 every 300ms
  <blink> <pin> 4 <int> 3 0 0 <cnt> 0 <end>

blink pin 7 at 12000ms                 parses; refused by range_check
  <blink> <pin> 7 <int> 1 2 0 0 0 <cnt> 0 <end>

chase across pins 2, 4 and 5 twice a second
  <seq> <pin> 2 <pin> 4 <pin> 5 <int> 5 0 0 <cnt> 0 <end>

turn everything off
  <set> <all> <low> <end>

stop
  <stop> <end>

is pin 18 on?
  <read> <pin> 1 8 <end>

turn off the desk lamp                 v1: no names, deferred to v2
  <unknown> <end>

call pin 4 the desk lamp               v1: no alias action, deferred to v2
  <unknown> <end>

what's the weather in Paris
  <unknown> <end>
```

Longest realistic emission is ~30 tokens (a ten-pin chase with a five-digit
interval and a two-digit count).

---

## 3. Consequences for the model config

Two changes from `PINLM-PLAN.md` §4, both from working the grammar through:

**`seq_len` 64 → 96.** A 6-pin chase with an explicit interval runs ~40 input
tokens plus ~20 emitted. 64 was too tight to be comfortable. Cost is KV cache
131 KB → 197 KB fp32 and roughly +1 ms/token — cheap insurance against a whole
class of truncation failure that would be invisible in aggregate accuracy.

**96 survives v1's longer emissions**, but only because the sampler caps
over-long chases at ten pins. Measured on the built corpus: longest row 162
tokens, 81 of 147,000 over the limit. Left uncapped at twelve pins the drop rate
was 247, and it was not an even sample — it removed the wordiest phrasings of
over-long chases and left the terse ones, training the model on a biased view of
the case it most needs to refuse. Raising `seq_len` instead would have cost KV
cache and a millisecond a token on input nobody sends.

**`<toggle>` needs 5 lines in `gpio_control.c`.** femtoclaw's tool has no toggle
action; it is a read-then-set-inverse. Worth having — "toggle the light" is a
natural thing to say — but note it as an addition beyond the copied file rather
than something inherited.

**`gpio_control.c` also owns range refusal**, which is new work beyond the
copied file: the pin allowlist, the 50–10000 ms interval bounds and the six-pin
chase cap, each with a message naming the offending value. `frames.range_check`
is the reference implementation and `Verdict` is the enum to mirror.

Parameter count is unchanged at **229,952** (vocab, d_model, layers and FFN are
all as planned; `seq_len` costs cache, not weights). The vocabulary is still
1,024 despite losing 52 reserved symbols — the freed ids go to BPE merges.

---

## 4. What this grammar cannot express

Recorded now so Phase 3 failures are not a surprise:

- **Conditionals and time.** "Turn the lamp on at sunset", "if the button is
  pressed". No cron, no sensors, no clock.
- **Relative reference.** "Turn *that* one off." Every utterance is independent
  by design (§1) — there is no previous turn to refer to.
- **Durations as an end condition.** "Blink for 10 seconds" must be generated as
  a count (10 s ÷ interval), not a duration. The underlying tool counts cycles.
- **Anything not about pins.** By construction: that is what `<unknown>` is for.

Note what is **no longer** on this list. "Sequences longer than 6 pins" used to
be here, returning `<unknown>`. They now parse into a frame of however many pins
were said, and `range_check` refuses it — same for an absent pin and an
out-of-bounds interval. The grammar expresses them; the hardware declines them.

### Deferred to v2, not absent

Distinct from the above, and tracked separately everywhere: `v1_deferred` in the
corpus, `DEFERRED` in `label_eval.py`, `v1_class` in the migrated eval sets.

- **Names and aliases.** "Turn on the desk lamp" and "call pin 4 the desk lamp"
  are `<unknown>` in v1 because there is no alias table for a name to resolve
  against. What is trained is rejection of the *shape* — verb plus noun phrase —
  never of a particular noun, which is why `names.py` generates from a space too
  large to memorise. §1's principle is intact and v2 restores the slots.
- **Repeat count without a rate.** "Blink pin 4 five times" is still not
  representable: `validate()` requires interval and count together. Absent from
  the corpus in both directions rather than labelled either way.
