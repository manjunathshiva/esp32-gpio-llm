// Host-side correctness gate: run the portable C inference over the golden
// prompt and compare its last-position logits to PyTorch's (dumped by
// src/export.py). Nothing reaches hardware until this passes.
//
// The tolerance is 1e-3, not the 0.02 an int4 port needs. Weights are exported
// fp32, so the only differences left are summation order and libm -- anything
// larger is a porting bug, and a loose bound would hide it. See src/export.py.
//
//   cc -O3 -o /tmp/verify firmware/host_verify/verify.c -lm
//   /tmp/verify firmware/model/model.bin firmware/model/golden.txt
#include <stdio.h>
#include <stdlib.h>
#include "../common/llm.h"

#define TOL 1e-3

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = (uint8_t *)malloc(*n);
  if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f); return b;
}

static Run run;   // static: the KV cache is ~200 KB, too big for the stack

int main(int argc, char **argv) {
  const char *bin = argc > 1 ? argv[1] : "firmware/model/model.bin";
  const char *gold = argc > 2 ? argv[2] : "firmware/model/golden.txt";

  size_t n;
  uint8_t *buf = read_file(bin, &n);
  Model m;
  int rc = llm_load(&m, buf);
  if (rc == -1) { fprintf(stderr, "bad magic in %s\n", bin); return 1; }
  if (rc == -2) { fprintf(stderr, "model exceeds compile-time ceilings\n"); return 1; }

  printf("loaded: V=%d D=%d L=%d H=%d F=%d seq=%d theta=%.0f  (%.0f KB)\n",
         m.c.vocab, m.c.dim, m.c.n_layers, m.c.n_heads, m.c.ffn, m.c.seq_len,
         (double)m.c.rope_theta, n / 1024.0);
  if (llm_size(&m) != n)
    printf("WARNING: header implies %zu bytes, file is %zu\n", llm_size(&m), n);

  FILE *gf = fopen(gold, "r");
  if (!gf) { perror(gold); return 1; }
  int plen;
  if (fscanf(gf, "%d", &plen) != 1) return 1;
  int *prompt = (int *)malloc((size_t)plen * sizeof(int));
  for (int i = 0; i < plen; i++)
    if (fscanf(gf, "%d", &prompt[i]) != 1) return 1;
  float *ref = (float *)malloc((size_t)m.c.vocab * sizeof(float));
  for (int i = 0; i < m.c.vocab; i++)
    if (fscanf(gf, "%f", &ref[i]) != 1) return 1;
  fclose(gf);

  llm_reset(&run);
  for (int i = 0; i < plen; i++) llm_forward(&m, &run, prompt[i]);

  double maxabs = 0, sum2 = 0;
  int c_top = 0, r_top = 0;
  for (int i = 0; i < m.c.vocab; i++) {
    double d = (double)run.logits[i] - (double)ref[i];
    if (fabs(d) > maxabs) maxabs = fabs(d);
    sum2 += d * d;
    if (run.logits[i] > run.logits[c_top]) c_top = i;
    if (ref[i] > ref[r_top]) r_top = i;
  }

  printf("argmax: C=%d  PyTorch=%d  %s\n", c_top, r_top,
         c_top == r_top ? "(agree)" : "(DISAGREE)");
  printf("max abs diff = %.3e   rms diff = %.3e\n", maxabs, sqrt(sum2 / m.c.vocab));

  int ok = maxabs < TOL && c_top == r_top;
  printf(ok ? "PASS: C matches the PyTorch golden\n"
            : "FAIL: numerics diverge -- this is a port bug, not rounding\n");
  return ok ? 0 : 2;
}
