// espcontrol -- natural-language GPIO on an ESP32-S3, with no network.
//
// Type "blink pin 4 twice a second" over serial and pin 4 blinks. The whole
// path is on-chip: BPE encode (tokenizer.h) -> 230K-parameter transformer
// (llm.h) -> emitted symbols -> Command (command.h) -> GPIO (gpio_control.c).
//
// The model is *not* linked into the sketch. It lives in its own flash
// partition and is memory-mapped, so a firmware change does not mean reflashing
// 898 KB and the build stays fast. See README.md for the two flash commands.
//
// v1 understands pin numbers only. "turn on the desk lamp" answers <unknown> by
// design, not by failure -- there is no alias table yet. See README.md.

#include "esp_partition.h"
#include "esp_heap_caps.h"

#include "llm.h"
#include "tokenizer.h"
#include "command.h"
#include "gpio_control.h"
#include "bpe.h"

#define MAX_LINE 256
#define MAX_IDS  256
#define MAX_NEW  40

static Model model;
static Run  *run;                 // ~200 KB of KV cache: PSRAM, not the stack
static Bpe   bpe = {BPE_BYTE_TOK, BPE_PAIR_KEY, BPE_PAIR_RANK, BPE_PAIR_NEW,
                    BPE_N_PAIRS};
static const void *model_base = nullptr;

static bool map_model() {
  // Subtype 0x40 matches the `model` line in partitions.csv. Looked up by type
  // rather than by name so a renamed partition fails here, loudly, instead of
  // mapping whatever happens to sit at a hard-coded offset.
  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("no 'model' partition -- check partitions.csv"); return false; }

  esp_partition_mmap_handle_t handle;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &model_base, &handle);
  if (err != ESP_OK) { Serial.printf("mmap failed: %d\n", err); return false; }

  int rc = llm_load(&model, model_base);
  if (rc == -1) { Serial.println("bad magic -- is model.bin flashed?"); return false; }
  if (rc == -2) { Serial.println("model exceeds the compile-time ceilings in llm.h"); return false; }
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\nespcontrol");

  run = (Run *)heap_caps_malloc(sizeof(Run), MALLOC_CAP_SPIRAM);
  if (!run) run = (Run *)malloc(sizeof(Run));      // no PSRAM: try internal
  if (!run) { Serial.println("out of memory for the KV cache"); while (1) delay(1000); }

  if (!map_model()) { while (1) delay(1000); }

  Serial.printf("model: V=%d D=%d L=%d H=%d  (%u KB mapped)\n",
                model.c.vocab, model.c.dim, model.c.n_layers, model.c.n_heads,
                (unsigned)(llm_size(&model) / 1024));

  int n_allow;
  const int *allow = gpio_allowed_pins(&n_allow);
  Serial.printf("pins: %d usable (%d..%d)\n", n_allow, allow[0], allow[n_allow - 1]);
  Serial.println("v1 understands pin numbers only -- try \"blink pin 4 every 500ms\"");
  Serial.print("\n> ");
}

// text -> emitted symbol ids. Mirrors host_verify/repl.c exactly; that file is
// what src/c_check.py holds against PyTorch, so any change here needs the same
// change there or the device stops being the thing that was verified.
static int generate(const char *text, int *out, int max_out) {
  static int ids[MAX_IDS];
  int n = tok_encode(&bpe, text, ids, MAX_IDS - 1);

  int budget = model.c.seq_len - MAX_NEW - 1;
  if (n > budget) { memmove(ids, ids + (n - budget), budget * sizeof(int)); n = budget; }
  ids[n++] = SYM_GO;

  llm_reset(run);
  for (int i = 0; i < n; i++) llm_forward(&model, run, ids[i]);

  int emitted = 0;
  while (emitted < max_out && emitted < MAX_NEW) {
    int next = llm_argmax(run->logits, model.c.vocab);
    out[emitted++] = next;
    if (next == SYM_END) break;
    llm_forward(&model, run, next);
  }
  return emitted;
}

static void handle(const char *line) {
  if (!line[0]) return;

  uint32_t t0 = millis();
  int out[MAX_NEW];
  int n = generate(line, out, MAX_NEW);
  uint32_t ms = millis() - t0;

  Command c;
  if (cmd_parse(out, n, &c) != 0) {
    // Refuse, do not guess. Reachable two ways: an utterance naming more pins
    // than CMD_MAX_PINS holds, or -- if it happens on ordinary input -- the
    // symbol ids and the model having drifted apart, in which case every parse
    // after it is suspect too. The symbol count separates the cases.
    Serial.printf("refused: I don't understand that   (%d symbols)\n", n);
    return;
  }

  char msg[128];
  Verdict v = gpio_execute(&c, msg, sizeof(msg));
  Serial.printf("%s%s   (%lums)\n", v == VERDICT_EXECUTE ? "" : "refused: ", msg,
                (unsigned long)ms);
}

void loop() {
  static char line[MAX_LINE];
  static int len = 0;

  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      line[len] = 0;
      Serial.println(line);
      handle(line);
      len = 0;
      Serial.print("\n> ");
    } else if (len < MAX_LINE - 1) {
      line[len++] = ch;
    }
  }
}
