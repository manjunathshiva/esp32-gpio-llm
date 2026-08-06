// espcontrol -- natural-language GPIO on an ESP32-S3, with no network.
//
// Type "blink pin 4 twice a second" over serial and pin 4 blinks. The whole
// path is on-chip: BPE encode (tokenizer.h) -> 312K-parameter transformer
// (llm.h) -> emitted symbols -> Command (command.h) -> GPIO (gpio_control.c).
//
// The model is *not* linked into the sketch. It lives in its own flash
// partition and is memory-mapped, so a firmware change does not mean reflashing
// 1219 KB and the build stays fast. See README.md for the two flash commands.
//
// v2 understands names as well as pin numbers. The model *copies* the name out
// of the utterance and alias.h resolves it; a name that is not in the table is
// refused by name, never swapped for the nearest one that is. Create them with
// "alias desk lamp = 4" -- see handle_alias, and README.md for why creating one
// by voice is deliberately still v2.1.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include <Preferences.h>

#include "llm.h"
#include "tokenizer.h"
#include "command.h"
#include "alias.h"
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

// The alias table, and the only thing that decides whether a name means
// anything. Persisted as a flat blob: it is a fixed-size POD struct, it is a
// few hundred bytes, and a reboot that forgets every name the user created
// would make the feature useless.
static AliasTable aliases;
static Preferences prefs;

static void alias_load() {
  alias_init(&aliases);
  prefs.begin("espcontrol", true);
  size_t n = prefs.getBytesLength("aliases");
  if (n == sizeof(AliasTable)) prefs.getBytes("aliases", &aliases, n);
  prefs.end();
  if (aliases.n < 0 || aliases.n > ALIAS_MAX) alias_init(&aliases);
}

static void alias_save() {
  prefs.begin("espcontrol", false);
  prefs.putBytes("aliases", &aliases, sizeof(AliasTable));
  prefs.end();
}

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

  alias_load();
  Serial.printf("aliases: %d stored\n", aliases.n);
  Serial.println("try \"blink pin 4 every 500ms\", or name something:");
  Serial.println("  alias desk lamp = 4   then   turn on the desk lamp");
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

// `alias` is typed, not spoken. Creating a name by natural language ("call pin
// 4 the desk lamp") is the <alias> action, whose symbol id is reserved but
// which nothing is trained to emit yet -- so until v2.1 the table is edited
// with an explicit command that never reaches the model.
//
//   alias                      list
//   alias desk lamp = 4        create or replace
//   alias porch lights = 5 6 7 8
//   alias del desk lamp        remove
//
// Returns true when the line was an alias command and has been dealt with.
static bool handle_alias(const char *line) {
  if (strncmp(line, "alias", 5) != 0 || (line[5] && line[5] != ' ')) return false;
  const char *arg = line + 5;
  while (*arg == ' ') arg++;

  if (!*arg) {
    if (!aliases.n) { Serial.println("no aliases yet -- try: alias desk lamp = 4"); return true; }
    for (int i = 0; i < aliases.n; i++) {
      Serial.printf("  %-24s ->", aliases.e[i].name);
      for (int k = 0; k < aliases.e[i].n_pins; k++)
        Serial.printf(" %d", aliases.e[i].pins[k]);
      Serial.println();
    }
    return true;
  }

  if (strncmp(arg, "del ", 4) == 0) {
    const char *nm = arg + 4;
    while (*nm == ' ') nm++;
    Serial.println(alias_remove(&aliases, nm) == 0 ? "removed" : "no such alias");
    alias_save();
    return true;
  }

  const char *eq = strchr(arg, '=');
  if (!eq) { Serial.println("usage: alias <name> = <pin> [pin ...]"); return true; }

  char nm[ALIAS_NAME_BUF];
  int len = (int)(eq - arg);
  while (len > 0 && arg[len - 1] == ' ') len--;
  if (len <= 0 || len >= (int)sizeof(nm)) { Serial.println("bad name"); return true; }
  memcpy(nm, arg, (size_t)len);
  nm[len] = 0;

  int pins[ALIAS_MAX_PINS], np = 0;
  for (const char *p = eq + 1; *p && np < ALIAS_MAX_PINS; ) {
    while (*p == ' ' || *p == ',') p++;
    if (!*p) break;
    if (*p < '0' || *p > '9') { Serial.println("pins must be numbers"); return true; }
    int v = 0;
    while (*p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    pins[np++] = v;
  }
  if (!np) { Serial.println("give at least one pin"); return true; }

  // Checked here so a name cannot be bound to a pin the board does not have.
  // Refusing at creation is better than refusing at every use, and it keeps
  // the table from being a second place where an illegal pin can hide.
  int n_allow;
  const int *allow = gpio_allowed_pins(&n_allow);
  for (int i = 0; i < np; i++) {
    bool ok = false;
    for (int j = 0; j < n_allow; j++) if (allow[j] == pins[i]) { ok = true; break; }
    if (!ok) { Serial.printf("pin %d is not a GPIO on this board\n", pins[i]); return true; }
  }

  if (alias_set(&aliases, nm, pins, np) != 0) {
    Serial.printf("table is full (%d entries)\n", ALIAS_MAX);
    return true;
  }
  alias_save();
  Serial.printf("\"%s\" ->", nm);
  for (int i = 0; i < np; i++) Serial.printf(" %d", pins[i]);
  Serial.println();
  return true;
}

static void handle(const char *line) {
  if (!line[0]) return;
  if (handle_alias(line)) return;

  uint32_t t0 = millis();
  int out[MAX_NEW];
  int n = generate(line, out, MAX_NEW);
  uint32_t ms = millis() - t0;

  Command c;
  if (cmd_parse(out, n, &bpe, &c) != 0) {
    // Refuse, do not guess. Reachable two ways: an utterance naming more pins
    // than CMD_MAX_PINS holds, or -- if it happens on ordinary input -- the
    // symbol ids and the model having drifted apart, in which case every parse
    // after it is suspect too. The symbol count separates the cases.
    Serial.printf("refused: I don't understand that   (%d symbols)\n", n);
    return;
  }

  char msg[128];
  // Resolution first: an unknown name is not a bad pin, and it is the one
  // refusal that has to name what it could not find. Only after every name has
  // become a real pin does the board get a say.
  Verdict v = alias_resolve(&c, &aliases, msg, sizeof(msg));
  if (v != VERDICT_EXECUTE) {
    Serial.printf("refused: %s   (%lums)\n", msg, (unsigned long)ms);
    return;
  }

  v = gpio_execute(&c, msg, sizeof(msg));
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
