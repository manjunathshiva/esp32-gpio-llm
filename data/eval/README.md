# Held-out evaluation sets

These are committed on purpose. Everything else generated is rebuilt from a
script, but a reported number nobody else can reproduce is not worth much, and
regenerating these needs a dataset download plus exact library versions.

| file | what it is | who wrote the sentences |
|---|---|---|
| `massive.jsonl` | 301 on/off commands | MASSIVE crowdworkers |
| `gemini.jsonl` | 250 lines, the first batch | Gemini |
| `gemini2_dev.jsonl` / `gemini2_locked.jsonl` | 848 lines, stratified halves | Gemini |
| `v1_*.jsonl` | the above relabelled for the v1 grammar | — |
| `*_review.txt` | lines the rule labeller would not label | — |

`v1_*` files are produced by `data/migrate_eval.py` from the v0 files beside
them, which are kept unchanged so v2 can go back to them.

**`gemini2_locked.jsonl` is scored but not diagnosed.** `src/evaluate.py`
suppresses its failure listing unless `--unlock` is given. Reading which items
fail is what turns a held-out set into a training set, and this half exists to
still be held-out at the end.

## Attribution

`massive.jsonl` and `v1_massive.jsonl` are derived from **MASSIVE** (FitzGerald
et al., Amazon Science), licensed **CC-BY-4.0**:
<https://huggingface.co/datasets/AmazonScience/massive>

Rows are cleaned and relabelled by `data/label_eval.py`; the utterance text is
MASSIVE's. Reuse of these two files carries the same CC-BY-4.0 attribution
requirement. Everything else here is generated and covered by the repo's MIT
licence.
