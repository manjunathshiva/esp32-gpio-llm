// Emitted symbols -> a command struct. The C half of data/frames.py.
//
// The model never writes JSON. It emits a fixed sequence of reserved token ids
// and this file reassembles them, so a malformed command is not representable:
// there is no string to mis-quote and no field name to misspell. Anything that
// does not fit the grammar is rejected outright rather than half-interpreted.
//
//   ACTION  TARGET*  [LEVEL]  [<int> digits]  [<cnt> digits]  <end>
//
// `cmd_parse` mirrors frames.from_symbols + frames.validate; `cmd_range_check`
// mirrors frames.range_check. Change either side and you must change the other
// -- there is no test that catches the drift, only behaviour on hardware.
//
// **Parsing and legality are separate on purpose.** cmd_parse accepts pin 100
// and a 12000 ms interval; cmd_range_check is what refuses them. The first
// grammar gave every allowed pin its own symbol, which made an illegal pin
// unrepresentable -- and a model that cannot say "pin 100" does not refuse, it
// says the nearest thing it can. "switch off pin 100" came back as pin 10: a
// real pin, switched off, already valid by the time it reached the allowlist.
// Numbers are copied digit by digit now so this check has something to reject.
#ifndef COMMAND_H
#define COMMAND_H
#include <stdint.h>
#include <stdio.h>
// Flat include: the Arduino build copies every source into one sketch
// directory, so a "../generated/" path would only work on the host. Host
// builds pass -Ifirmware/generated instead.
#include "symbols.h"

// Mirrors of frames.py. MAX_SEQ_PINS is what the hardware runs; CMD_MAX_PINS is
// what the parser will hold, and it is deliberately larger so an over-long
// chase arrives intact and gets refused with a count instead of being silently
// truncated to something runnable.
#define CMD_INTERVAL_MIN 50
#define CMD_INTERVAL_MAX 10000
#define CMD_MAX_SEQ_PINS 6
#define CMD_MAX_PINS 16
#define CMD_MAX_DIGITS 6
#define CMD_UNSET (-1)

typedef enum {
  ACT_SET, ACT_READ, ACT_BLINK, ACT_SEQ, ACT_STOP, ACT_UNKNOWN
} CmdAction;

typedef enum { LVL_NONE, LVL_HIGH, LVL_LOW, LVL_TOGGLE } CmdLevel;

typedef enum {
  VERDICT_EXECUTE, VERDICT_BAD_PIN, VERDICT_BAD_INTERVAL,
  VERDICT_TOO_MANY_PINS, VERDICT_UNKNOWN
} Verdict;

typedef struct {
  CmdAction action;
  int pins[CMD_MAX_PINS];
  int n_pins;
  int all;                  // the <all> target was present
  CmdLevel level;
  int interval_ms;          // CMD_UNSET when the speaker gave none
  int count;                // CMD_UNSET when unspecified; 0 means forever
} Command;

static const char *cmd_action_name(CmdAction a) {
  switch (a) {
    case ACT_SET: return "set";
    case ACT_READ: return "read";
    case ACT_BLINK: return "blink";
    case ACT_SEQ: return "seq";
    case ACT_STOP: return "stop";
    default: return "unknown";
  }
}

// Parse `n` emitted token ids. Returns 0 on success, -1 if the sequence is not
// something the grammar can produce. On -1 the Command is not usable.
static int cmd_parse(const int *ids, int n, Command *c) {
  c->action = ACT_UNKNOWN;
  c->n_pins = 0; c->all = 0; c->level = LVL_NONE;
  c->interval_ms = CMD_UNSET; c->count = CMD_UNSET;

  int i = 0;
  if (n <= 0) return -1;

  switch (ids[0]) {
    case SYM_SET:     c->action = ACT_SET; break;
    case SYM_READ:    c->action = ACT_READ; break;
    case SYM_BLINK:   c->action = ACT_BLINK; break;
    case SYM_SEQ:     c->action = ACT_SEQ; break;
    case SYM_STOP:    c->action = ACT_STOP; break;
    case SYM_UNKNOWN: c->action = ACT_UNKNOWN; break;
    default: return -1;                       // no action leads the sequence
  }
  i = 1;

  if (c->action == ACT_UNKNOWN)
    return (i < n && ids[i] == SYM_END && i + 1 == n) ? 0 : -1;

  // A run of digits, ending at the first id that is not one. No terminator
  // symbol: with names gone the only thing one closed was a number, and a
  // number closes itself.
  #define READ_NUM(dst)                                    \
    do {                                                   \
      int v = 0, k = 0, d;                                 \
      while (i < n && (d = sym_digit_value(ids[i])) >= 0) { \
        if (++k > CMD_MAX_DIGITS) return -1;               \
        v = v * 10 + d; i++;                               \
      }                                                    \
      if (k == 0) return -1;                               \
      (dst) = v;                                           \
    } while (0)

  while (i < n) {
    if (ids[i] == SYM_ALL) { c->all = 1; i++; }
    else if (ids[i] == SYM_PIN) {
      i++;
      if (c->n_pins >= CMD_MAX_PINS) return -1;
      READ_NUM(c->pins[c->n_pins]);
      c->n_pins++;
    } else break;
  }

  if (i < n) {
    if (ids[i] == SYM_HIGH)        { c->level = LVL_HIGH; i++; }
    else if (ids[i] == SYM_LOW)    { c->level = LVL_LOW; i++; }
    else if (ids[i] == SYM_TOGGLE) { c->level = LVL_TOGGLE; i++; }
  }
  if (i < n && ids[i] == SYM_INT) { i++; READ_NUM(c->interval_ms); }
  if (i < n && ids[i] == SYM_CNT) { i++; READ_NUM(c->count); }
  #undef READ_NUM

  if (i >= n || ids[i] != SYM_END || i + 1 != n) return -1;

  // --- shape, mirroring frames.validate -------------------------------------
  int targets = c->n_pins + (c->all ? 1 : 0);
  int timed = (c->interval_ms != CMD_UNSET) + (c->count != CMD_UNSET);

  switch (c->action) {
    case ACT_SET:
      // Several pins at one level is legal; <all> mixed into a list is not,
      // because <all> already means the whole board. Mirrors frames.validate.
      if (targets < 1 || (c->all && c->n_pins) || c->level == LVL_NONE || timed)
        return -1;
      break;
    case ACT_READ:
      if (targets != 1 || c->all || c->level != LVL_NONE) return -1;
      break;
    case ACT_BLINK:
      // Timing is optional but not half-given: gpio_control has defaults, so an
      // omitted pair means "device default", while one slot alone is a parse
      // that lost information.
      if (targets != 1 || c->level != LVL_NONE || timed == 1) return -1;
      break;
    case ACT_SEQ:
      // Only the lower bound is structural. The upper bound is how many pins
      // the hardware will drive, and that is cmd_range_check's call.
      if (c->n_pins < 2 || c->all || c->level != LVL_NONE || timed == 1) return -1;
      break;
    case ACT_STOP:
      if (targets > 1 || c->level != LVL_NONE || c->interval_ms != CMD_UNSET)
        return -1;
      break;
    default: return -1;
  }
  if (c->count != CMD_UNSET && c->count < 0) return -1;
  return 0;
}

// Is this command runnable on this board? `allow` is the pin allowlist, owned
// by gpio_control. Writes a user-facing reason into `msg` when it is not.
static Verdict cmd_range_check(const Command *c, const int *allow, int n_allow,
                               char *msg, int msg_n) {
  if (msg && msg_n) msg[0] = 0;

  if (c->action == ACT_UNKNOWN) {
    if (msg) snprintf(msg, (size_t)msg_n, "I don't understand that");
    return VERDICT_UNKNOWN;
  }

  for (int i = 0; i < c->n_pins; i++) {
    int ok = 0;
    for (int j = 0; j < n_allow; j++) if (allow[j] == c->pins[i]) { ok = 1; break; }
    if (!ok) {
      if (msg) snprintf(msg, (size_t)msg_n,
                        "pin %d is not a GPIO on this board", c->pins[i]);
      return VERDICT_BAD_PIN;
    }
  }

  if (c->action == ACT_SEQ && c->n_pins > CMD_MAX_SEQ_PINS) {
    if (msg) snprintf(msg, (size_t)msg_n,
                      "a chase runs at most %d pins, got %d",
                      CMD_MAX_SEQ_PINS, c->n_pins);
    return VERDICT_TOO_MANY_PINS;
  }

  if (c->interval_ms != CMD_UNSET &&
      (c->interval_ms < CMD_INTERVAL_MIN || c->interval_ms > CMD_INTERVAL_MAX)) {
    if (msg) snprintf(msg, (size_t)msg_n,
                      "interval %dms is outside %d-%dms", c->interval_ms,
                      CMD_INTERVAL_MIN, CMD_INTERVAL_MAX);
    return VERDICT_BAD_INTERVAL;
  }
  return VERDICT_EXECUTE;
}

// Human-readable form of a parsed command, for the serial console.
static void cmd_describe(const Command *c, char *out, int out_n) {
  int k = snprintf(out, (size_t)out_n, "%s", cmd_action_name(c->action));
  if (c->all) k += snprintf(out + k, (size_t)(out_n - k), " all");
  for (int i = 0; i < c->n_pins && k < out_n; i++)
    k += snprintf(out + k, (size_t)(out_n - k), " pin%d", c->pins[i]);
  if (c->level != LVL_NONE && k < out_n)
    k += snprintf(out + k, (size_t)(out_n - k), " %s",
                  c->level == LVL_HIGH ? "high" :
                  c->level == LVL_LOW ? "low" : "toggle");
  if (c->interval_ms != CMD_UNSET && k < out_n)
    k += snprintf(out + k, (size_t)(out_n - k), " %dms", c->interval_ms);
  if (c->count != CMD_UNSET && k < out_n)
    snprintf(out + k, (size_t)(out_n - k), " x%d", c->count);
}

#endif
