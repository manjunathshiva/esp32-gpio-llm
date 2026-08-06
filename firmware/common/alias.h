// The alias table: name -> pins, and the resolution the model is not allowed
// to do.
//
// This file is the other half of the v1 pin-number fix. There, the model was
// made able to *say* "pin 100" so that something with the board in front of it
// could refuse the number by value; here it is made able to say a name that
// does not exist, so that something with the user's table in front of it can
// refuse the name by name. Both follow the same rule: the model transcribes,
// and the layer that owns the facts judges.
//
// The tempting design is the other one -- teach the model which names exist,
// by putting them in the grammar or the training corpus. It reproduces the
// pin-100 bug exactly. A model that can only emit names it was trained on
// answers "turn on the aquarium pump" with the nearest name it knows, and a
// real device turns on. Nothing in the pipeline can detect that, because the
// output is a perfectly valid command for a device that exists.
//
// So the table lives here, is edited at runtime, and is the *only* thing that
// decides whether a name means anything.
#ifndef ALIAS_H
#define ALIAS_H
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "command.h"

#define ALIAS_MAX 16                    // entries
#define ALIAS_MAX_PINS 6                // pins one name may cover
#define ALIAS_NAME_BUF CMD_NAME_BUF

typedef struct {
  char name[ALIAS_NAME_BUF];
  int pins[ALIAS_MAX_PINS];
  int n_pins;
} AliasEntry;

typedef struct {
  AliasEntry e[ALIAS_MAX];
  int n;
} AliasTable;

// Case and inner spacing are not meaningful in a name someone typed, so both
// are folded before comparison. Mirrors frames.Name.norm(); if the two ever
// disagree, a name that scores as correct on host data stops resolving on the
// device, which is the kind of divergence that looks like a model regression.
static void alias_norm(const char *s, char *out, int max_out) {
  int j = 0, sp = 0;
  for (; *s && j < max_out - 1; s++) {
    char c = *s;
    if (c == ' ' || c == '\t' || c == '_' || c == '-') { sp = 1; continue; }
    if (sp && j) out[j++] = ' ';
    sp = 0;
    if (j < max_out - 1)
      out[j++] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
  }
  out[j] = 0;
}

static void alias_init(AliasTable *t) { t->n = 0; }

// Find by folded name. Returns the index, or -1.
static int alias_find(const AliasTable *t, const char *name) {
  char want[ALIAS_NAME_BUF], have[ALIAS_NAME_BUF];
  alias_norm(name, want, sizeof want);
  for (int i = 0; i < t->n; i++) {
    alias_norm(t->e[i].name, have, sizeof have);
    if (strcmp(want, have) == 0) return i;
  }
  return -1;
}

// Create or replace an entry. Returns 0, or -1 if the table is full or the
// request is malformed. Replacing rather than appending keeps "call pin 5 the
// desk lamp" from leaving two entries that differ only in case.
static int alias_set(AliasTable *t, const char *name, const int *pins,
                     int n_pins) {
  if (!name || !*name || n_pins < 1 || n_pins > ALIAS_MAX_PINS) return -1;
  int ix = alias_find(t, name);
  if (ix < 0) {
    if (t->n >= ALIAS_MAX) return -1;
    ix = t->n++;
  }
  snprintf(t->e[ix].name, sizeof t->e[ix].name, "%s", name);
  for (int i = 0; i < n_pins; i++) t->e[ix].pins[i] = pins[i];
  t->e[ix].n_pins = n_pins;
  return 0;
}

static int alias_remove(AliasTable *t, const char *name) {
  int ix = alias_find(t, name);
  if (ix < 0) return -1;
  t->e[ix] = t->e[--t->n];
  return 0;
}

// Expand every name in `c` into pins, in place.
//
// Appends, which is why a mixed "pin 4 and the desk lamp" chase runs the pin
// first -- see the note on Command.names. On any failure the Command is left
// unusable rather than partly resolved: a half-resolved command still looks
// runnable, and would drive whichever pins happened to resolve.
static Verdict alias_resolve(Command *c, const AliasTable *t,
                             char *msg, int msg_n) {
  if (msg && msg_n) msg[0] = 0;

  // An <alias> command's name is being *defined*, not looked up. Resolving it
  // would refuse every new name for not existing yet, which is the one moment
  // a name legitimately does not.
  if (c->action == ACT_ALIAS) return VERDICT_EXECUTE;

  for (int i = 0; i < c->n_names; i++) {
    int ix = alias_find(t, c->names[i]);
    if (ix < 0) {
      // By name, and never by substitution. The user can act on this: they
      // either mistyped it or have not created it yet.
      if (msg) snprintf(msg, (size_t)msg_n, "I don't know \"%s\"", c->names[i]);
      c->n_pins = 0;
      return VERDICT_UNKNOWN_NAME;
    }
    const AliasEntry *e = &t->e[ix];
    if (c->n_pins + e->n_pins > CMD_MAX_PINS) {
      if (msg) snprintf(msg, (size_t)msg_n,
                        "\"%s\" covers more pins than one command holds",
                        c->names[i]);
      c->n_pins = 0;
      return VERDICT_TOO_MANY_PINS;
    }
    for (int k = 0; k < e->n_pins; k++) c->pins[c->n_pins++] = e->pins[k];
  }

  // Deferred from cmd_parse, which could not know how many pins a name covers.
  // "chase the desk lamp" parses cleanly and is refused here, with the count.
  if (c->action == ACT_SEQ && c->n_pins < 2) {
    if (msg) snprintf(msg, (size_t)msg_n,
                      "a chase needs at least 2 pins, \"%s\" is %d",
                      c->n_names ? c->names[0] : "that", c->n_pins);
    c->n_pins = 0;
    return VERDICT_NAME_TOO_FEW_PINS;
  }

  c->n_names = 0;               // consumed; the command is now pins only
  return VERDICT_EXECUTE;
}

// Apply an <alias> command: bind its name to its pins. Returns 0, or -1 with a
// reason in `msg`.
//
// Range-checking happens before this, in cmd_range_check, so a name can never
// be bound to a pin the board does not have. Refusing at creation is better
// than refusing at every use, and it keeps the table from becoming a second
// place an illegal pin can hide.
static int alias_apply(const Command *c, AliasTable *t, char *msg, int msg_n) {
  if (c->action != ACT_ALIAS || c->n_names != 1 || c->n_pins < 1) {
    if (msg) snprintf(msg, (size_t)msg_n, "not a naming command");
    return -1;
  }
  if (c->n_pins > ALIAS_MAX_PINS) {
    if (msg) snprintf(msg, (size_t)msg_n,
                      "a name covers at most %d pins, got %d",
                      ALIAS_MAX_PINS, c->n_pins);
    return -1;
  }
  if (alias_set(t, c->names[0], c->pins, c->n_pins) != 0) {
    if (msg) snprintf(msg, (size_t)msg_n, "the alias table is full (%d)",
                      ALIAS_MAX);
    return -1;
  }
  int k = snprintf(msg, (size_t)msg_n, "\"%s\" ->", c->names[0]);
  for (int i = 0; i < c->n_pins && k < msg_n; i++)
    k += snprintf(msg + k, (size_t)(msg_n - k), " %d", c->pins[i]);
  return 0;
}

#endif
