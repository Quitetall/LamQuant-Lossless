//! LamQuant v7.7 firmware — bare-metal Rust on RP2350 (Hazard3 RISC-V).
//!
//! Boot sequence:
//!   1. RP2350 boot ROM loads firmware from QSPI flash (XIP)
//!   2. riscv-rt _start sets up stack, .data, .bss, calls main()
//!   3. main() initializes:
//!        a. Bump allocator (64 KB scratch heap for codec Vec usage)
//!        b. Peripherals + clocks (12 MHz XTAL → 150 MHz sys clock)
//!        c. PMP stack guard (NAPOT region at stack top, hardware-locked)
//!        d. CRC32 firmware integrity check (cold boot only — Phase 3)
//!        e. KAT (ternary MAC parity) — Phase 3
//!        f. ADS1299 + BLE init — Phase 5
//!        g. Scheduler launch — Phase 5
//!   4. Main loop: scheduler tick + watchdog pet + serial command poll

#![no_std]
#![no_main]

extern crate alloc;

mod integrity;
mod power;
mod stack_guard;

use core::mem::MaybeUninit;
use core::ptr::addr_of_mut;
use embedded_alloc::LlffHeap as Heap;
use panic_halt as _;
use rp235x_hal as hal;
use rp235x_hal::pac;

// ─── Boot ROM image definition ────────────────────────────────────
//
// RP2350 requires an embedded image header so the boot ROM knows what
// kind of binary this is.

#[link_section = ".start_block"]
#[used]
pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

// ─── Global allocator ─────────────────────────────────────────────
//
// Bump allocator over a fixed-size heap. No fragmentation, no GC. Codec
// path uses Vec for variable-length subbands; allocations are
// deterministic per window (no surprise OOM).

#[global_allocator]
static HEAP: Heap = Heap::empty();

const HEAP_SIZE: usize = 64 * 1024; // 64 KB scratch for codec allocations

#[link_section = ".bss"]
static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];

// ─── Crystal frequency for clock setup ────────────────────────────

const XTAL_HZ: u32 = 12_000_000;

// ─── Entry point ──────────────────────────────────────────────────

#[hal::entry]
fn main() -> ! {
    // Step 1: Initialize the heap *before* anything that allocates.
    // SAFETY: Single-threaded, called exactly once at boot, prior to any
    // code that allocates.
    unsafe {
        HEAP.init(addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE);
    }

    // Step 2: Take ownership of peripherals.
    let mut pac = pac::Peripherals::take().unwrap();

    // Step 3: Reset + initialize clocks (12 MHz XTAL → 150 MHz sys clock).
    let mut watchdog = hal::watchdog::Watchdog::new(pac.WATCHDOG);
    let _clocks = hal::clocks::init_clocks_and_plls(
        XTAL_HZ,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .ok()
    .expect("clock init failed");

    // Step 4: PMP stack guard (RISC-V CSR, NAPOT region at stack top).
    // Locked at boot; any stack overflow → trap → safe mode.
    stack_guard::install();

    // Step 5: Firmware integrity check (CRC32 over weight tables).
    // Phase 3 wires actual weight buffers; for Phase 1 this is a stub
    // that exercises the CRC engine.
    if !integrity::check_firmware_crc() {
        power::enter_safe_mode();
    }

    // Step 6: Start watchdog (500 ms timeout — must pet every loop iter).
    watchdog.start(fugit::ExtU32::millis(500));

    // Step 7: Phase 1 main loop — pet watchdog, sleep until interrupt.
    // Phase 2-5 add: scheduler tick, command poll, ADC handling.
    loop {
        watchdog.feed();
        power::wait_for_interrupt();
    }
}
