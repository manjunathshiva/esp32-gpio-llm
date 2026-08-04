// Portable single-header inference for the command TinyLM. The same code runs
// on the host (verified against a PyTorch golden) and on the ESP32-S3. No
// dynamic allocation and no arch dispatch: dims come from the model.bin header
// and weights are bound in place from the mapped base.
//
// Matches src/model.py op-for-op -- split-half RoPE, SiLU-SwiGLU,
// RMSNorm(weight * x * rsqrt(mean(x^2)+eps)), tied input/output embedding.
// Change one side and you must change the other, re-export, and re-run
// firmware/host_verify/verify.c.
//
// **fp32 throughout.** This model is 229,952 parameters -- 898 KB, which fits
// flash without argument. esp32-tinyllm's int4 path exists because a 28.9M
// model does not fit; carrying it here would buy space nobody needs and give up
// an exact host check. See src/export.py.
//
// The tensor order in `llm_load` mirrors export.py's `plan` list exactly. There
// are no names in the file, only offsets.
#ifndef LLM_H
#define LLM_H
#include <stdint.h>
#include <math.h>
#include <string.h>

#define LLM_MAGIC 0x434D4431u   // "CMD1"
#define RMS_EPS 1e-6f

// Compile-time ceilings for the no-malloc state below. They default to the
// shipped config; llm_load() rejects a model.bin that exceeds them rather than
// writing past the buffers, because the alternative is memory corruption that
// looks like a bad parse.
#ifndef LLM_MAX_LAYERS
#define LLM_MAX_LAYERS 4
#endif
#ifndef LLM_MAX_DIM
#define LLM_MAX_DIM 64
#endif
#ifndef LLM_MAX_FFN
#define LLM_MAX_FFN 128
#endif
#ifndef LLM_MAX_SEQ
#define LLM_MAX_SEQ 96
#endif
#ifndef LLM_MAX_VOCAB
#define LLM_MAX_VOCAB 1024
#endif

typedef struct {
  int vocab, dim, n_layers, n_heads, ffn, seq_len;
  float rope_theta;
} Cfg;

typedef struct {
  Cfg c;
  int head_dim;
  const float *tok_emb;                    // [V, D], also the tied output head
  const float *attn_norm[LLM_MAX_LAYERS];  // [D]
  const float *qkv[LLM_MAX_LAYERS];        // [3D, D]
  const float *attn_proj[LLM_MAX_LAYERS];  // [D, D]
  const float *ffn_norm[LLM_MAX_LAYERS];   // [D]
  const float *gate[LLM_MAX_LAYERS];       // [F, D]
  const float *up[LLM_MAX_LAYERS];         // [F, D]
  const float *down[LLM_MAX_LAYERS];       // [D, F]
  const float *out_norm;                   // [D]
} Model;

// Per-generation state. Declare one (static, or in PSRAM on device) and pass it
// in; nothing here allocates.
typedef struct {
  float k[LLM_MAX_LAYERS][LLM_MAX_SEQ][LLM_MAX_DIM];
  float v[LLM_MAX_LAYERS][LLM_MAX_SEQ][LLM_MAX_DIM];
  float x[LLM_MAX_DIM], xb[LLM_MAX_DIM], xo[LLM_MAX_DIM];
  float qkv[3 * LLM_MAX_DIM];
  float att[LLM_MAX_SEQ];
  float h1[LLM_MAX_FFN], h2[LLM_MAX_FFN];
  float logits[LLM_MAX_VOCAB];
  int pos;
} Run;

// --- primitives --------------------------------------------------------------

static void rmsnorm(float *out, const float *x, const float *w, int n) {
  float ss = 0.0f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.0f / sqrtf(ss / (float)n + RMS_EPS);
  for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
}

// out[rows] = W[rows, cols] @ x[cols], row-major -- the layout torch.nn.Linear
// stores and export.py writes unchanged.
static void matvec(float *out, const float *w, const float *x,
                   int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    const float *wr = w + (size_t)r * cols;
    float s = 0.0f;
    for (int c = 0; c < cols; c++) s += wr[c] * x[c];
    out[r] = s;
  }
}

static void softmax(float *v, int n) {
  float m = v[0];
  for (int i = 1; i < n; i++) if (v[i] > m) m = v[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) { v[i] = expf(v[i] - m); sum += v[i]; }
  for (int i = 0; i < n; i++) v[i] /= sum;
}

// Split-half RoPE, matching model.py's apply_rope: the vector is halved and the
// two halves rotated against each other. Interleaved pairing -- the other
// common convention -- is a different function, not a different layout, and
// would silently degrade rather than fail.
static void rope(float *vec, int n_heads, int head_dim, int pos, float theta) {
  int half = head_dim / 2;
  for (int i = 0; i < half; i++) {
    float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
    float ang = (float)pos * freq;
    float c = cosf(ang), s = sinf(ang);
    for (int h = 0; h < n_heads; h++) {
      float *p = vec + h * head_dim;
      float a = p[i], b = p[i + half];
      p[i] = a * c - b * s;
      p[i + half] = b * c + a * s;
    }
  }
}

// --- load --------------------------------------------------------------------

// Binds weights in place from `base`. Returns 0 on success, or a negative code:
// -1 bad magic, -2 config exceeds the compile-time ceilings.
static int llm_load(Model *m, const void *base) {
  const uint8_t *p = (const uint8_t *)base;
  uint32_t magic;
  memcpy(&magic, p, 4); p += 4;
  if (magic != LLM_MAGIC) return -1;

  int32_t hdr[6];
  memcpy(hdr, p, sizeof(hdr)); p += sizeof(hdr);
  memcpy(&m->c.rope_theta, p, 4); p += 4;

  m->c.vocab = hdr[0]; m->c.dim = hdr[1]; m->c.n_layers = hdr[2];
  m->c.n_heads = hdr[3]; m->c.ffn = hdr[4]; m->c.seq_len = hdr[5];
  m->head_dim = m->c.dim / m->c.n_heads;

  if (m->c.n_layers > LLM_MAX_LAYERS || m->c.dim > LLM_MAX_DIM ||
      m->c.ffn > LLM_MAX_FFN || m->c.seq_len > LLM_MAX_SEQ ||
      m->c.vocab > LLM_MAX_VOCAB || m->head_dim % 2 != 0)
    return -2;

  const float *f = (const float *)p;
  int D = m->c.dim, F = m->c.ffn;

  m->tok_emb = f; f += (size_t)m->c.vocab * D;
  for (int l = 0; l < m->c.n_layers; l++) {
    m->attn_norm[l] = f; f += D;
    m->qkv[l] = f;       f += (size_t)3 * D * D;
    m->attn_proj[l] = f; f += (size_t)D * D;
    m->ffn_norm[l] = f;  f += D;
    m->gate[l] = f;      f += (size_t)F * D;
    m->up[l] = f;        f += (size_t)F * D;
    m->down[l] = f;      f += (size_t)D * F;
  }
  m->out_norm = f;
  return 0;
}

static size_t llm_size(const Model *m) {
  int D = m->c.dim, F = m->c.ffn;
  size_t n = (size_t)m->c.vocab * D + D;
  n += (size_t)m->c.n_layers * (D + 3 * D * D + D * D + D + 2 * F * D + D * F);
  return 4 + 6 * 4 + 4 + n * sizeof(float);
}

// --- forward -----------------------------------------------------------------

static void llm_reset(Run *r) { r->pos = 0; }

// One token in, logits out (into r->logits). Advances r->pos.
static void llm_forward(const Model *m, Run *r, int token) {
  const int D = m->c.dim, F = m->c.ffn, H = m->c.n_heads, Dh = m->head_dim;
  const int pos = r->pos;

  memcpy(r->x, m->tok_emb + (size_t)token * D, (size_t)D * sizeof(float));

  for (int l = 0; l < m->c.n_layers; l++) {
    rmsnorm(r->xb, r->x, m->attn_norm[l], D);
    matvec(r->qkv, m->qkv[l], r->xb, 3 * D, D);

    float *q = r->qkv, *k = r->qkv + D, *v = r->qkv + 2 * D;
    rope(q, H, Dh, pos, m->c.rope_theta);
    rope(k, H, Dh, pos, m->c.rope_theta);

    memcpy(r->k[l][pos], k, (size_t)D * sizeof(float));
    memcpy(r->v[l][pos], v, (size_t)D * sizeof(float));

    // Causal attention against everything cached so far, head by head.
    float scale = 1.0f / sqrtf((float)Dh);
    for (int h = 0; h < H; h++) {
      const float *qh = q + h * Dh;
      for (int t = 0; t <= pos; t++) {
        const float *kh = r->k[l][t] + h * Dh;
        float s = 0.0f;
        for (int i = 0; i < Dh; i++) s += qh[i] * kh[i];
        r->att[t] = s * scale;
      }
      softmax(r->att, pos + 1);
      float *oh = r->xo + h * Dh;
      for (int i = 0; i < Dh; i++) oh[i] = 0.0f;
      for (int t = 0; t <= pos; t++) {
        const float *vh = r->v[l][t] + h * Dh;
        float a = r->att[t];
        for (int i = 0; i < Dh; i++) oh[i] += a * vh[i];
      }
    }

    matvec(r->xb, m->attn_proj[l], r->xo, D, D);
    for (int i = 0; i < D; i++) r->x[i] += r->xb[i];

    rmsnorm(r->xb, r->x, m->ffn_norm[l], D);
    matvec(r->h1, m->gate[l], r->xb, F, D);
    matvec(r->h2, m->up[l], r->xb, F, D);
    for (int i = 0; i < F; i++) {
      float g = r->h1[i];
      r->h1[i] = (g / (1.0f + expf(-g))) * r->h2[i];   // SiLU * up
    }
    matvec(r->xb, m->down[l], r->h1, D, F);
    for (int i = 0; i < D; i++) r->x[i] += r->xb[i];
  }

  rmsnorm(r->xb, r->x, m->out_norm, D);
  matvec(r->logits, m->tok_emb, r->xb, m->c.vocab, D);   // tied head
  r->pos++;
}

static int llm_argmax(const float *v, int n) {
  int best = 0;
  for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
  return best;
}

#endif
