//! Hardware stack overflow trap via RISC-V Physical Memory Protection (PMP).
//!
//! Port of `firmware/core/stack_guard.c`. The Hazard3 core implements PMP
//! per the RISC-V Privileged Architecture spec. We configure PMP entry 0
//! as a NAPOT region covering 2 KB at the stack-top guard zone. Locked
//! at boot (`L = 1`) so it cannot be modified afterward — any access to
//! that region traps and we branch to safe mode.

const STACK_GUARD_ADDR: u32 = 0x2000_7800;
const STACK_GUARD_SIZE: u32 = 2048; // 2 KB

// pmpcfg encoding:
const PMP_CFG_L: u32 = 1 << 7; // Lock bit — entry immutable until reset
const PMP_CFG_A_NAPOT: u32 = 3 << 3; // Naturally-Aligned Power-Of-Two
                                     // No R/W/X bits set = poisonous region

/// Install the PMP-locked stack guard. Call exactly once, before any
/// task could underflow the stack into the guard region.
///
/// After this returns, the 2 KB region at `STACK_GUARD_ADDR` is hardware-
/// protected. Any load/store there raises a precise access fault trap.
#[inline(never)]
pub fn install() {
    // pmpaddr0 = (base >> 2) | ((size / 8) - 1)
    // For NAPOT: low bits encode the region size as a string of 1s.
    // 2 KB / 8 - 1 = 0xFF (8 trailing 1-bits).
    let pmpaddr0: u32 = (STACK_GUARD_ADDR >> 2) | ((STACK_GUARD_SIZE / 8) - 1);
    let pmpcfg0: u32 = PMP_CFG_L | PMP_CFG_A_NAPOT;

    // SAFETY: csrw to pmpaddr0/pmpcfg0 is safe at boot. The locked PMP
    // entry only restricts access to the guard zone; it cannot make
    // existing valid memory inaccessible.
    unsafe {
        core::arch::asm!(
            "csrw pmpaddr0, {addr}",
            "csrw pmpcfg0, {cfg}",
            addr = in(reg) pmpaddr0,
            cfg = in(reg) pmpcfg0,
            options(nostack, preserves_flags),
        );
    }
}
