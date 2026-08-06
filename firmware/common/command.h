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
// For tok_decode. A name is the only slot whose content is text, so this is
// the only reason the command parser needs the tokenizer at all.
#include "tokenizer.h"

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

// Named targets. Four is well past what anyone says in one breath ("the lamp,
// the fan and the buzzer" is three) and each costs SYM_MAX_NAME+1 bytes of a
// stack-allocated Command, so this is not free.
#define CMD_MAX_NAMES 4
#define CMD_NAME_BUF (SYM_MAX_NAME + 1)

typedef enum {
  ACT_SET, ACT_READ, ACT_BLINK, ACT_SEQ, ACT_STOP, ACT_UNKNOWN, ACT_ALIAS
} CmdAction;

typedef enum { LVL_NONE, LVL_HIGH, LVL_LOW, LVL_TOGGLE } CmdLevel;

typedef enum {
  VERDICT_EXECUTE, VERDICT_BAD_PIN, VERDICT_BAD_INTERVAL,
  VERDICT_TOO_MANY_PINS, VERDICT_UNKNOWN,
  // Only reachable once the alias table has been consulted. Refusing an
  // unresolvable name *by name* is the whole point of copying it rather than
  // classifying it -- "I don't know 'aquarium pump'" is a sentence the user can
  // act on, and it is what stops the model reaching for the nearest name it
  // does know.
  VERDICT_UNKNOWN_NAME, VERDICT_NAME_TOO_FEW_PINS
} Verdict;

typedef struct {
  CmdAction action;
  int pins[CMD_MAX_PINS];
  int n_pins;
  // Names as copied, before resolution. Kept apart from pins[] because at parse
  // time nothing here knows what a name resolves to -- alias_resolve() is what
  // turns these into pins, and it is the only thing that may.
  //
  // Known limitation: resolved pins are appended, so "chase pin 4 and the desk
  // lamp" runs the pin before the lamp regardless of the order spoken. Order is
  // cosmetic for set/read/stop, and a mixed pin+name chase is under half a
  // percent of the corpus. Recorded rather than fixed, so the next person knows
  // it is a choice and not an oversight.
  char names[CMD_MAX_NAMES][CMD_NAME_BUF];
  int n_names;
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
    case ACT_ALIAS: return "alias";
    default: return "unknown";
  }
}

// Parse `n` emitted token ids. Returns 0 on success, -1 if the sequence is not
// something the grammar can produce. On -1 the Command is not usable.
static int cmd_parse(const int *ids, int n, const Bpe *bpe, Command *c) {
  c->action = ACT_UNKNOWN;
  c->n_pins = 0; c->n_names = 0; c->all = 0; c->level = LVL_NONE;
  c->interval_ms = CMD_UNSET; c->count = CMD_UNSET;
  c->names[0][0] = 0;

  int i = 0;
  if (n <= 0) return -1;

  switch (ids[0]) {
    case SYM_SET:     c->action = ACT_SET; break;
    case SYM_READ:    c->action = ACT_READ; break;
    case SYM_BLINK:   c->action = ACT_BLINK; break;
    case SYM_SEQ:     c->action = ACT_SEQ; break;
    case SYM_STOP:    c->action = ACT_STOP; break;
    case SYM_ALIAS:   c->action = ACT_ALIAS; break;
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
    } else if (ids[i] == SYM_NAME) {
      i++;
      if (c->n_names >= CMD_MAX_NAMES) return -1;
      int start = i;
      while (i < n && ids[i] != SYM_NEND) {
        // A reserved id inside a span means the generation lost track of the
        // name boundary. Refuse the whole command: the alternative is decoding
        // a marker into text and resolving whatever that spells.
        if (ids[i] < SYM_RESERVED_N) return -1;
        i++;
      }
      if (i >= n) return -1;                    // <nend> never arrived
      if (i == start) return -1;                // empty name
      char *dst = c->names[c->n_names];
      if (tok_decode(bpe, ids + start, i - start, dst, CMD_NAME_BUF) < 0)
        return -1;
      // The copy carries the leading space ByteLevel folded into its first
      // token. Trim both ends here so the name reads correctly when it is
      // echoed back in a refusal; case and inner spacing are folded later, by
      // whoever compares it against the table.
      int a = 0, b = (int)strlen(dst);
      while (dst[a] == ' ') a++;
      while (b > a && dst[b - 1] == ' ') b--;
      if (b <= a) return -1;                    // whitespace only
      memmove(dst, dst + a, (size_t)(b - a));
      dst[b - a] = 0;
      i++;                                      // consume <nend>
      c->n_names++;
      if (c->n_names < CMD_MAX_NAMES) c->names[c->n_names][0] = 0;
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
  int targets = c->n_pins + c->n_names + (c->all ? 1 : 0);
  int timed = (c->interval_ms != CMD_UNSET) + (c->count != CMD_UNSET);
  int rate_no_count = (c->interval_ms != CMD_UNSET && c->count == CMD_UNSET);

  switch (c->action) {
    case ACT_SET:
      // Several pins at one level is legal; <all> mixed into a list is not,
      // because <all> already means the whole board. Mirrors frames.validate.
      if (targets < 1 || (c->all && targets > 1) || c->level == LVL_NONE || timed)
        return -1;
      break;
    case ACT_READ:
      if (targets < 1 || c->all || c->level != LVL_NONE) return -1;
      break;
    case ACT_BLINK:
      // A count with no rate is legal -- "blink pin 4 five times" means five
      // cycles at the device default. A rate with no count is not: it would
      // discard a number the speaker said.
      if (targets < 1 || (c->all && targets > 1) || c->level != LVL_NONE ||
          rate_no_count) return -1;
      break;
    case ACT_SEQ:
      // Only the lower bound is structural. The upper bound is how many pins
      // the hardware will drive, and that is cmd_range_check's call.
      //
      // A name suspends the lower bound: "chase the porch lights" is one target
      // that the alias table expands into several, and how many pins a name
      // covers is not knowable here. Demanding two targets would leave the
      // model no way to say it except by inventing a second one. The count is
      // checked after resolution, where it is real. Mirrors frames.validate.
      if ((targets < 2 && !c->n_names) || c->all || c->level != LVL_NONE ||
          rate_no_count)
        return -1;
      break;
    case ACT_ALIAS:
      // "call pin 4 the desk lamp": pins are what gets named, the one name is
      // what they get called. Exactly one name, because two would be
      // ambiguous about which is the new label. Mirrors frames.validate.
      if (c->n_names != 1 || c->n_pins < 1 || c->all ||
          c->level != LVL_NONE || timed)
        return -1;
      break;
    case ACT_STOP:
      // No targets means everything; any other number is a list of pins. One
      // target was the old limit, and "stop pins 4 and 5" came back as <seq>,
      // starting an animation instead of ending one.
      if (c->all || c->level != LVL_NONE || c->interval_ms != CMD_UNSET)
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
  // Names as well as pins: after alias_resolve there are none left, but before
  // it -- and always, for <alias> -- the name is the whole point of the line.
  for (int i = 0; i < c->n_names && k < out_n; i++)
    k += snprintf(out + k, (size_t)(out_n - k), " \"%s\"", c->names[i]);
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
