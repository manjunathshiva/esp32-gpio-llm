// The whole device pipeline, on the host: typed text -> BPE -> model -> emitted
// symbols -> Command -> range verdict. Nothing here is host-specific except
// reading model.bin from a file instead of mmapping flash, so getting this
// right is most of getting the sketch right.
//
// Two modes, and the default is the machine one on purpose. `verify.c` proves
// the forward pass against one golden prompt; this proves the *whole chain*
// against every held-out utterance, by emitting the generated token ids for
// each line so src/c_check.py can diff them against the PyTorch decoder. A
// mismatch anywhere -- tokenizer, sampling loop, stopping rule -- shows up as
// a differing id sequence rather than as a slightly worse parse nobody notices.
//
//   cc -O3 -o /tmp/repl firmware/host_verify/repl.c -lm
//   /tmp/repl firmware/model/model.bin --pretty      # interactive
//   uv run python src/c_check.py                     # the gate
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/llm.h"
#include "../common/tokenizer.h"
#include "../common/command.h"
#include "../common/alias.h"
#include "../generated/bpe.h"

#define MAX_LINE 1024
#define MAX_IDS 256
#define MAX_NEW 40

// The board. This is the one place the allowlist is written on the host side;
// on device it comes from gpio_control.h. Kept in step with frames.PINS_S3.
static const int PINS_S3[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                              15, 16, 17, 18, 21, 38, 39, 40, 41, 42, 48};
#define N_PINS_S3 ((int)(sizeof(PINS_S3) / sizeof(PINS_S3[0])))

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = (uint8_t *)malloc(*n);
  if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f); return b;
}

static Run run;   // static: the KV cache is ~200 KB

// text -> emitted symbol ids. Returns how many were emitted (<= MAX_NEW).
static int generate(const Model *m, const Bpe *bpe, const char *text,
                    int *out, int max_out) {
  int ids[MAX_IDS];
  int n = tok_encode(bpe, text, ids, MAX_IDS - 1);

  // Leave room for the completion, dropping the *oldest* prompt tokens. Same
  // rule as src/evaluate.py: the tail of a command carries the slots.
  int budget = m->c.seq_len - MAX_NEW - 1;
  if (n > budget) {
    memmove(ids, ids + (n - budget), (size_t)budget * sizeof(int));
    n = budget;
  }
  ids[n++] = SYM_GO;

  llm_reset(&run);
  for (int i = 0; i < n; i++) llm_forward(m, &run, ids[i]);

  int emitted = 0;
  while (emitted < max_out && emitted < MAX_NEW) {
    int next = llm_argmax(run.logits, m->c.vocab);
    out[emitted++] = next;
    if (next == SYM_END) break;
    llm_forward(m, &run, next);
  }
  return emitted;
}

int main(int argc, char **argv) {
  const char *bin = "firmware/model/model.bin";
  int pretty = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--pretty")) pretty = 1;
    else bin = argv[i];
  }

  size_t nbytes;
  uint8_t *buf = read_file(bin, &nbytes);
  Model m;
  int rc = llm_load(&m, buf);
  if (rc) { fprintf(stderr, "llm_load failed (%d)\n", rc); return 1; }

  Bpe bpe = BPE_INIT;

  // A few aliases so the pretty mode can show both halves of resolution: a
  // name that exists drives pins, one that does not is refused by name. The
  // device builds this table at runtime instead; these are only here so the
  // host repl is usable without one.
  AliasTable aliases;
  alias_init(&aliases);
  alias_set(&aliases, "desk lamp", (const int[]){4}, 1);
  alias_set(&aliases, "status led", (const int[]){2}, 1);
  alias_set(&aliases, "porch lights", (const int[]){5, 6, 7, 8}, 4);

  if (pretty)
    printf("loaded %s: V=%d D=%d L=%d  (%.0f KB)\ntype a command, ctrl-D to quit\n\n",
           bin, m.c.vocab, m.c.dim, m.c.n_layers, nbytes / 1024.0);

  static char line[MAX_LINE];
  int out[MAX_NEW];

  while (fgets(line, sizeof(line), stdin)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;

    int n = generate(&m, &bpe, line, out, MAX_NEW);

    if (!pretty) {
      for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", out[i]);
      putchar('\n');
      fflush(stdout);
      continue;
    }

    Command c;
    char msg[128], desc[192];
    if (cmd_parse(out, n, &bpe, &c) != 0) {
      printf("  -> MALFORMED (%d symbols)\n\n", n);
      continue;
    }
    // Names resolve before the board is consulted: an unknown name is not a
    // bad pin, and saying so is the whole point of copying the name.
    Verdict v = alias_resolve(&c, &aliases, msg, sizeof(msg));
    if (v != VERDICT_EXECUTE) {
      printf("  -> refused: %s\n\n", msg);
      fflush(stdout);
      continue;
    }
    v = cmd_range_check(&c, PINS_S3, N_PINS_S3, msg, sizeof(msg));
    cmd_describe(&c, desc, sizeof(desc));
    if (v == VERDICT_EXECUTE) printf("  -> %s   [execute]\n\n", desc);
    else                      printf("  -> %s   [refused: %s]\n\n", desc, msg);
    fflush(stdout);
  }
  return 0;
}
