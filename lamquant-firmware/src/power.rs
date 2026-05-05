//! Power state coordination — safe mode, dormant (WFI).
//!
//! Port of `firmware/core/power_states.c`. RISC-V `wfi` replaces ARM `wfe`.

/// Enter safe mode: halt scheduler, flush BLE, spin until watchdog reset.
///
/// Called on:
///   - CRC integrity check failure
///   - Stack guard PMP trap
///   - Unrecoverable scheduler fault
///
/// The watchdog (500 ms timeout) resets the device. SRAM contents are
/// preserved across this reset so TNN weights survive.
#[inline(never)]
pub fn enter_safe_mode() -> ! {
    // Phase 1: minimal — just halt. Phase 5 wires in:
    //   - ble_emergency_flush()
    //   - scheduler_abort_inference()
    //   - dsp_reset_pipeline()
    //   - safety_log_event(EVENT_SAFE_MODE)
    loop {
        wait_for_interrupt();
    }
}

/// Sleep until next interrupt (DMA complete, timer, etc.).
///
/// On Hazard3 RISC-V the `wfi` instruction halts the core until any
/// enabled interrupt fires. Lower-power than spinning. The DMA-complete
/// ISR is the primary wake source during normal operation.
#[inline(always)]
pub fn wait_for_interrupt() {
    riscv::asm::wfi();
}
