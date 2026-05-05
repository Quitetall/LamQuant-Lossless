//! LamQuant v7.7 firmware — library surface.
//!
//! Exposes the pure-Rust DSP, neural, codec, and transport modules that
//! make up the firmware pipeline. The bare-metal binary (`main.rs`)
//! consumes this library on the embedded target; host tests consume it
//! via `cargo test -p lamquant-firmware`.
//!
//! `no_std` on every target. The bin provides the `#[global_allocator]`
//! and `#[panic_handler]`. The lib does not, so it can be unit-tested on
//! the host where std supplies them.
//!
//! Modules (Phase 1-3 complete, 4-6 in progress):
//!   dsp     — biquad prefilter, LPC analysis/synthesis, lifting DWT, WHT
//!   neural  — ternary MAC, FSQ, focal modulation, SNN
//!   codec   — rANS context, Golomb-Rice, hybrid entropy, FSQ adaptive
//!   transport — BLE, USB, mailbox (riscv32-only — gated by cfg)
//!   afe     — ADS1299 driver (riscv32-only — gated by cfg)
//!   safety  — patient-safety subsystems (BLE retry, seizure log, ...)

#![cfg_attr(not(test), no_std)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;

pub mod dsp;
pub mod neural;
// `safety` is pure data structures (no HAL) so it builds on host too —
// useful for unit tests of the audit-trail logic.
pub mod safety;

// Hardware-bound modules: peripheral drivers, scheduler, transport.
// Compiled only when targeting the bare-metal RP2350. Host tests skip them.
#[cfg(target_arch = "riscv32")]
pub mod afe;
#[cfg(target_arch = "riscv32")]
pub mod codec;
#[cfg(target_arch = "riscv32")]
pub mod scheduler;
#[cfg(target_arch = "riscv32")]
pub mod transport;

// Boot-time integrity check + power state coordination + RISC-V CSR
// stack guard. RISC-V-only because they reference CSR instructions or
// hardware registers.
#[cfg(target_arch = "riscv32")]
pub mod integrity;
#[cfg(target_arch = "riscv32")]
pub mod power;
#[cfg(target_arch = "riscv32")]
pub mod stack_guard;
