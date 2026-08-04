#pragma once

#include <stddef.h>
#include "command.h"

// The sketch is compiled as C++ and this file as C, so without this the linker
// looks for a mangled name and reports the functions as undefined.
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Drive GPIO from a parsed Command: set/read level, blink a pin, or run a
 * chase. Adapted from femtoclaw's tool_gpio, with the cJSON layer removed --
 * the model emits symbols, command.h reassembles them, and nothing in this
 * path ever builds or parses a string the model wrote.
 *
 * This file owns the board. `gpio_allowed_pins()` is the allowlist that
 * cmd_range_check() tests against, and it is the only description of which pins
 * exist -- the model carries none. Porting to another ESP32 is a change here
 * plus a matching change to frames.PINS_S3, not a retrain.
 */

/** The safe-pin allowlist. Sets *n to its length. */
const int *gpio_allowed_pins(int *n);

/**
 * Range-check and run. `output` receives a human-readable result -- what was
 * done, or why it was refused -- suitable for printing straight to serial.
 * Returns the verdict so the caller can tell refusal from success without
 * parsing the message.
 */
Verdict gpio_execute(const Command *c, char *output, size_t output_size);

/** Stop every running animation. */
void gpio_stop_all(void);

#ifdef __cplusplus
}
#endif
