#include <stdint.h>
#include <stdbool.h>
#include "power_states.h"
#include "stack_guard.h"

/**
 * Hardware Stack Enforcer - LamQuant Gen 6
 * RP2350 Physical Memory Protection (PMP) Core Bindings
 */

#define PMP_CFG_L         (1 << 7)    // Lock bit protecting state registers
#define PMP_CFG_A_NAPOT   (3 << 3)    // Naturally Aligned Power-Of-Two

// Injected directly via `main()` early boot initializing strict trap registers
void stack_setup_hardware_trap(void) {
    // Use NAPOT mode to protect exactly 2KB at 0x20007800
    // Address encoding: (0x20007800 >> 2) | ((2048 / 8) - 1)
    uint32_t pmpaddr0 = (0x20007800 >> 2) | 0xFF;

    // Config: Lock (L) + NAPOT (A_NAPOT). No R/W/X permissions = Poisonous range.
    uint32_t pmpcfg = PMP_CFG_L | PMP_CFG_A_NAPOT;

    __asm__ volatile (
        "csrw pmpaddr0, %0\n"
        "csrw pmpcfg0, %1\n"
        :
        : "r"(pmpaddr0), "r"(pmpcfg)
    );
}

// Exception hook implicitly branched when the pointer violates limits
void __attribute__((naked)) stack_overflow_exception(void) {
    // Immediate safe mode routing dropping allocated execution frames organically
    __asm__ volatile (
        "li   sp, 0x20008000\n"  // Reset SP to absolute TCM top
        "j    enter_safe_mode\n"
    );
}

// PMP trap is hardware-locked (L bit = 1), cannot be modified after boot.
// This is intentionally a no-op. The PMP-based stack guard at 0x20007800
// is permanently active once stack_setup_hardware_trap() runs at boot.
void stack_reset_canaries(void) {
    // No-op: PMP entries with the Lock bit are immutable until reset.
}
