//! LamQuant v7.7 firmware — bare-metal binary entry point.
//!
//! Boot sequence (riscv32 target only):
//!   1. RP2350 boot ROM loads firmware from QSPI flash (XIP)
//!   2. riscv-rt _start sets up stack, .data, .bss, calls main()
//!   3. main() initializes:
//!        a. Bump allocator (64 KB scratch heap for codec Vec usage)
//!        b. Peripherals + clocks (12 MHz XTAL → 150 MHz sys clock)
//!        c. PMP stack guard (NAPOT region at stack top, hardware-locked)
//!        d. CRC32 firmware integrity check (cold boot only — Phase 3)
//!        e. KAT (ternary MAC parity) — boot-time invariant check
//!        f. ADS1299 + BLE init — Phase 5
//!        g. Scheduler launch — Phase 5
//!   4. Main loop: scheduler tick + watchdog pet + serial command poll
//!
//! On host (`cargo test`), the binary collapses to a no-op so `cargo
//! build` doesn't fail. All testable logic lives in the library
//! (`src/lib.rs`) and runs under the host test harness.

#![cfg_attr(target_arch = "riscv32", no_std)]
#![cfg_attr(target_arch = "riscv32", no_main)]

// ───────────── Embedded path (RP2350 / riscv32) ─────────────

#[cfg(target_arch = "riscv32")]
mod embedded {
    extern crate alloc;

    use core::mem::MaybeUninit;
    use core::ptr::addr_of_mut;
    use embedded_alloc::LlffHeap as Heap;
    use lamquant_firmware::{integrity, neural, power, stack_guard};
    use panic_halt as _;
    use rp235x_hal as hal;
    use rp235x_hal::pac;

    // ─── Boot ROM image definition ────────────────────────────────────
    #[link_section = ".start_block"]
    #[used]
    pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

    // ─── Global allocator ─────────────────────────────────────────────
    #[global_allocator]
    static HEAP: Heap = Heap::empty();
    const HEAP_SIZE: usize = 64 * 1024;

    #[link_section = ".bss"]
    static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];

    const XTAL_HZ: u32 = 12_000_000;

    #[hal::entry]
    fn entry() -> ! {
        // SAFETY: Single-threaded, called exactly once at boot.
        unsafe {
            HEAP.init(addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE);
        }

        let mut pac = pac::Peripherals::take().unwrap();
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

        // PMP stack guard (RISC-V CSR, NAPOT region at stack top).
        stack_guard::install();

        // Firmware integrity check (CRC32 over weight tables).
        if !integrity::check_firmware_crc() {
            power::enter_safe_mode();
        }

        // Ternary MAC parity KAT — mandatory before processing patient data.
        // Catches compiler bitfield rotation, struct-packing surprises,
        // codegen regressions.
        if neural::ternary_mac::boot_parity_kat().is_err() {
            power::enter_safe_mode();
        }

        watchdog.start(fugit::ExtU32::millis(500));

        loop {
            watchdog.feed();
            power::wait_for_interrupt();
        }
    }
}

// ───────────── Host stub (cargo test on x86_64) ─────────────

#[cfg(not(target_arch = "riscv32"))]
fn main() {
    // No-op: binary exists only to satisfy `[[bin]]`; all testable logic
    // is in the library and runs via `cargo test`.
}
