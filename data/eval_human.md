# Human held-out evaluation set

**This file must be written by a person who has not read `realize.py`.**

Everything else in `data/` is generated, so measuring against it is circular:
the model will score ~99% and then fail on a stranger's first sentence. This
file is the only honest number in the project. It is the equivalent of
`verify.c` in esp32-tinyllm — the gate that decides whether anything proceeds.

## Rules for whoever writes it

1. **Do not read `data/realize.py` or `data/frames.py` first.** If you have,
   ask someone else. The point is phrasings the generator never saw.
2. Write how *you* would actually talk to the device, not how you imagine a
   parser wants to be addressed. Sloppy is good. Typos are good.
3. Aim for ~200 lines in section A and ~100 in section B.
4. Do not try to be clever or adversarial in section A — write ordinary
   requests. Section B is where the hard cases go.

## What the device can do

- Turn a pin on or off, or toggle it
- Read whether a pin is on or off
- Blink one pin at an interval, a set number of times or forever
- Chase across 2–6 pins in sequence
- Stop a blink or chase, on one pin or all of them
- Give a pin a name, then use that name instead of the number
  (a name may cover several pins, e.g. "the lights")

Pins available: 1–18, 21, 38–42, 48. Intervals 50–10000 ms.

It cannot dim, fade, set colour, set a percentage, schedule anything for later,
react to sensors, or remember what you said in a previous line.

---

## Section A — commands you expect to work

One per line: the utterance, then `|`, then what should happen in plain words.

```
turn on pin 4 | pin 4 high
```

<!-- write below this line -->


---

## Section B — things you expect it to refuse

Same format. Include off-domain requests, half-finished sentences, and things
that sound like commands but ask for something it cannot do.

```
dim the lamp a bit | refuse, no dimming
```

<!-- write below this line -->


---

## After it is written

`src/label_eval.py` converts these lines into frames for scoring. Labelling is
not contamination — authoring the phrasing is. It is fine for the labels to be
added by someone who has read the generator, as long as the sentences were not.

**Write a second set later and do not look at it.** Once you have read the
failures on set A three times you are fitting to it, and the number stops
meaning anything. Set B is what goes in `RESULTS.md`.
