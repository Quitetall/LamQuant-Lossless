#pragma once
#ifndef POWER_STATES_H
#define POWER_STATES_H

#include <stdbool.h>

// Safe mode recovery FSM states (visible to other modules for status checks)
typedef enum {
    SAFE_MODE_ENTRY,      // Immediate packet drop, data integrity priority
    SAFE_MODE_RECOVERY,   // Stack reset, pipeline teardown
    SAFE_MODE_RESUME      // Normal operation restored
} SafeModeState;

// --- Power state transitions (implemented in power_states.c) ---
void enter_safe_mode(void);
bool in_safe_mode(void);
void enter_dormant_state(void);

// --- Cross-module externs that power_states.c calls ---
// These must be implemented by the respective modules.
// If a module is not linked, provide a weak stub.
extern void ble_spi_emergency_flush(void);
extern void ble_enter_standby(void);
extern void scheduler_abort_inference(void);
extern void dsp_reset_pipeline(void);
extern void stack_reset_canaries(void);
extern void scheduler_reinit(void);

#endif // POWER_STATES_H
