#include <stdbool.h>
#include <stdint.h>
#include "power_states.h"

// External state handling references protecting safe mode entry
extern void ble_spi_emergency_flush(void);
extern void ble_enter_standby(void);
extern void scheduler_abort_inference(void);
extern void dsp_reset_pipeline(void);
extern void stack_reset_canaries(void);
extern void scheduler_reinit(void);

static volatile SafeModeState safe_state = SAFE_MODE_RESUME;

// Set dynamically allowing soft resets vs strict physical Watchdog trap locks
#define SAFE_MODE_WATCHDOG_RESET 1

void enter_safe_mode(void) {
    // 1. Immediate: Drop BLE frames physically preventing TX propagation
    ble_spi_emergency_flush();
    ble_enter_standby();  
    
    // 2. Clear volatile lifting/FocalNet structures 
    scheduler_abort_inference();
    dsp_reset_pipeline();  
    
    // 3. Recovery sequence transitions
    safe_state = SAFE_MODE_RECOVERY;
    
    // 4. Trap bounds natively vs softly mapping resets
    #if SAFE_MODE_WATCHDOG_RESET
    // Forces hardware watchdog mapping organically executing out of lock bounds
    while(1) { __asm__ volatile("wfi"); }
    #else
    stack_reset_canaries();
    scheduler_reinit();
    safe_state = SAFE_MODE_RESUME;
    #endif
}

bool in_safe_mode(void) {
    return safe_state != SAFE_MODE_RESUME;
}

// Low level sleep loops for nominal states
void enter_dormant_state(void) {
    __asm__ volatile ("wfi");
}
