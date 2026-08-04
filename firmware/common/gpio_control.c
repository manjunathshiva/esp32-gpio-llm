/* GPIO control, driven by a parsed Command.
 *
 * From femtoclaw's tools/tool_gpio.c, which is itself a fork of MimiClaw by
 * Ziboyan Wang -- see Acknowledgments in the top-level README. The eight-slot
 * esp_timer animation engine
 * (slot_stop / find_slot_by_pin / anim_timer_cb and friends) is that file's,
 * kept intact -- it is the part worth copying. What changed is the entry point:
 * femtoclaw was called by an LLM emitting JSON, so it parsed cJSON and
 * validated fields that might be missing, malformed or the wrong type. Here the
 * model emits symbols and command.h has already reassembled them, so a Command
 * arrives structurally valid or not at all, and this file only has to answer
 * whether the board can run it.
 *
 * Range checking lives in cmd_range_check (command.h) against the allowlist
 * below. It is not defence in depth against the model -- it is the only thing
 * that knows the board, and it is why "turn on pin 100" produces a sentence
 * instead of an action on pin 10.
 */

#include "gpio_control.h"

#include <stdio.h>
#include <string.h>
#include "driver/gpio.h"
#include "esp_timer.h"

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32S3_BETA
static const int s_safe_pins[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    21, 38, 39, 40, 41, 42, 48
};
#else
/* ESP32-WROOM-32 */
static const int s_safe_pins[] = {
    2, 4, 5, 12, 13, 14, 15, 18, 19, 22, 23, 25, 26, 27, 32, 33
};
#endif

#define SAFE_PIN_COUNT (int)(sizeof(s_safe_pins) / sizeof(s_safe_pins[0]))
#define MAX_SLOT_PINS  CMD_MAX_PINS
#define MAX_ANIM_SLOTS 8

const int *gpio_allowed_pins(int *n)
{
    if (n) *n = SAFE_PIN_COUNT;
    return s_safe_pins;
}

/* --- Multi-slot animation state (femtoclaw) --- */

typedef enum { ANIM_NONE, ANIM_BLINK, ANIM_SEQUENCE } anim_mode_t;

typedef struct {
    esp_timer_handle_t timer;
    anim_mode_t mode;
    int pins[MAX_SLOT_PINS];
    int pin_count;
    int current_idx;
    int level;
    int cycles_done;
    int cycles_total;   /* 0 = infinite */
} anim_slot_t;

static anim_slot_t s_slots[MAX_ANIM_SLOTS];

static void slot_stop(anim_slot_t *slot)
{
    if (slot->timer) esp_timer_stop(slot->timer);
    for (int i = 0; i < slot->pin_count; i++)
        gpio_set_level((gpio_num_t)slot->pins[i], 0);
    slot->mode = ANIM_NONE;
    slot->pin_count = 0;
}

static anim_slot_t *find_slot_by_pin(int pin)
{
    for (int s = 0; s < MAX_ANIM_SLOTS; s++) {
        if (s_slots[s].mode == ANIM_NONE) continue;
        for (int p = 0; p < s_slots[s].pin_count; p++)
            if (s_slots[s].pins[p] == pin) return &s_slots[s];
    }
    return NULL;
}

static anim_slot_t *find_free_slot(void)
{
    for (int s = 0; s < MAX_ANIM_SLOTS; s++)
        if (s_slots[s].mode == ANIM_NONE) return &s_slots[s];
    return NULL;
}

static void anim_timer_cb(void *arg)
{
    anim_slot_t *slot = (anim_slot_t *)arg;

    if (slot->mode == ANIM_BLINK) {
        slot->level = !slot->level;
        gpio_set_level((gpio_num_t)slot->pins[0], slot->level);
        if (!slot->level) {
            slot->cycles_done++;
            if (slot->cycles_total > 0 && slot->cycles_done >= slot->cycles_total)
                slot_stop(slot);
        }
    } else if (slot->mode == ANIM_SEQUENCE) {
        gpio_set_level((gpio_num_t)slot->pins[slot->current_idx], 0);
        slot->current_idx++;
        if (slot->current_idx >= slot->pin_count) {
            slot->current_idx = 0;
            slot->cycles_done++;
            if (slot->cycles_total > 0 && slot->cycles_done >= slot->cycles_total) {
                slot_stop(slot);
                return;
            }
        }
        gpio_set_level((gpio_num_t)slot->pins[slot->current_idx], 1);
    }
}

static void ensure_slot_timer(anim_slot_t *slot)
{
    if (slot->timer) return;
    esp_timer_create_args_t args = {
        .callback = anim_timer_cb, .arg = slot, .name = "gpio_anim",
    };
    esp_timer_create(&args, &slot->timer);
}

void gpio_stop_all(void)
{
    for (int s = 0; s < MAX_ANIM_SLOTS; s++)
        if (s_slots[s].mode != ANIM_NONE) slot_stop(&s_slots[s]);
}

static void pin_out(int pin)
{
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << pin,
        .mode = GPIO_MODE_INPUT_OUTPUT,   /* readable while driven */
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io);
}

/* Device defaults for an omitted timing pair. An unspecified slot means "you
 * choose", never zero -- the grammar leaves both out together precisely so the
 * device can answer this, and inventing numbers in the parser would make the
 * model hallucinate values the speaker never said. */
#define DEFAULT_BLINK_MS 500
#define DEFAULT_SEQ_MS   200

Verdict gpio_execute(const Command *c, char *output, size_t out_n)
{
    char why[128];
    Verdict v = cmd_range_check(c, s_safe_pins, SAFE_PIN_COUNT, why, sizeof(why));
    if (v != VERDICT_EXECUTE) {
        snprintf(output, out_n, "%s", why);
        return v;
    }

    int interval = c->interval_ms;
    int count = (c->count == CMD_UNSET) ? 0 : c->count;

    switch (c->action) {
    case ACT_SET: {
        int level = (c->level == LVL_HIGH) ? 1 : 0;
        if (c->all) {
            if (c->level == LVL_TOGGLE) {
                for (int i = 0; i < SAFE_PIN_COUNT; i++) {
                    pin_out(s_safe_pins[i]);
                    gpio_set_level((gpio_num_t)s_safe_pins[i],
                                   !gpio_get_level((gpio_num_t)s_safe_pins[i]));
                }
                snprintf(output, out_n, "toggled all %d pins", SAFE_PIN_COUNT);
            } else {
                gpio_stop_all();
                for (int i = 0; i < SAFE_PIN_COUNT; i++) {
                    pin_out(s_safe_pins[i]);
                    gpio_set_level((gpio_num_t)s_safe_pins[i], level);
                }
                snprintf(output, out_n, "all %d pins %s", SAFE_PIN_COUNT,
                         level ? "high" : "low");
            }
            return VERDICT_EXECUTE;
        }
        /* One level across however many pins were named. Each is toggled
         * against its own current state, so "toggle pins 4 and 5" inverts both
         * independently rather than forcing them to agree. */
        int k = 0;
        for (int i = 0; i < c->n_pins; i++) {
            int pin = c->pins[i];
            anim_slot_t *s = find_slot_by_pin(pin);
            if (s) slot_stop(s);      /* a manual set wins over an animation */
            pin_out(pin);
            int lv = (c->level == LVL_TOGGLE)
                         ? !gpio_get_level((gpio_num_t)pin) : level;
            gpio_set_level((gpio_num_t)pin, lv);
            k += snprintf(output + k, out_n - k, k ? ", %d" : "pin %d", pin);
            if (k >= (int)out_n) { k = (int)out_n - 1; break; }
        }
        if (c->level == LVL_TOGGLE) snprintf(output + k, out_n - k, " toggled");
        else snprintf(output + k, out_n - k, " %s", level ? "high" : "low");
        return VERDICT_EXECUTE;
    }

    case ACT_READ: {
        int pin = c->pins[0];
        pin_out(pin);
        snprintf(output, out_n, "pin %d is %s", pin,
                 gpio_get_level((gpio_num_t)pin) ? "high" : "low");
        return VERDICT_EXECUTE;
    }

    case ACT_BLINK: {
        if (c->all) { snprintf(output, out_n, "cannot blink every pin at once");
                      return VERDICT_UNKNOWN; }
        int pin = c->pins[0];
        int ms = (interval == CMD_UNSET) ? DEFAULT_BLINK_MS : interval;
        anim_slot_t *s = find_slot_by_pin(pin);
        if (!s) s = find_free_slot();
        if (!s) { snprintf(output, out_n, "all %d animation slots are busy",
                           MAX_ANIM_SLOTS); return VERDICT_UNKNOWN; }
        slot_stop(s);
        pin_out(pin);
        s->mode = ANIM_BLINK; s->pins[0] = pin; s->pin_count = 1;
        s->level = 0; s->cycles_done = 0; s->cycles_total = count;
        gpio_set_level((gpio_num_t)pin, 0);
        ensure_slot_timer(s);
        esp_timer_start_periodic(s->timer, (uint64_t)ms * 1000);
        if (count) snprintf(output, out_n, "blinking pin %d every %dms, %d times",
                            pin, ms, count);
        else       snprintf(output, out_n, "blinking pin %d every %dms", pin, ms);
        return VERDICT_EXECUTE;
    }

    case ACT_SEQ: {
        int ms = (interval == CMD_UNSET) ? DEFAULT_SEQ_MS : interval;
        anim_slot_t *s = NULL;
        for (int i = 0; i < c->n_pins; i++) {
            anim_slot_t *own = find_slot_by_pin(c->pins[i]);
            if (own) slot_stop(own);
        }
        s = find_free_slot();
        if (!s) { snprintf(output, out_n, "all %d animation slots are busy",
                           MAX_ANIM_SLOTS); return VERDICT_UNKNOWN; }
        slot_stop(s);
        for (int i = 0; i < c->n_pins; i++) {
            pin_out(c->pins[i]);
            gpio_set_level((gpio_num_t)c->pins[i], 0);
            s->pins[i] = c->pins[i];
        }
        s->pin_count = c->n_pins;
        s->mode = ANIM_SEQUENCE; s->current_idx = 0;
        s->cycles_done = 0; s->cycles_total = count;
        gpio_set_level((gpio_num_t)s->pins[0], 1);
        ensure_slot_timer(s);
        esp_timer_start_periodic(s->timer, (uint64_t)ms * 1000);
        snprintf(output, out_n, "chasing %d pins every %dms", c->n_pins, ms);
        return VERDICT_EXECUTE;
    }

    case ACT_STOP: {
        if (c->n_pins == 0) {
            gpio_stop_all();
            snprintf(output, out_n, "stopped");
        } else {
            anim_slot_t *s = find_slot_by_pin(c->pins[0]);
            if (s) slot_stop(s);
            snprintf(output, out_n, "stopped pin %d", c->pins[0]);
        }
        return VERDICT_EXECUTE;
    }

    default:
        snprintf(output, out_n, "I don't understand that");
        return VERDICT_UNKNOWN;
    }
}
