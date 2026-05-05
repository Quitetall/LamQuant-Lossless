//! LamQuant v7.7 firmware — bare-metal binary entry point.
//!
//! Boot sequence (riscv32 target only):
//!   1. RP2350 boot ROM loads firmware from QSPI flash (XIP)
//!   2. riscv-rt _start sets up stack, .data, .bss, calls main()
//!   3. main() initializes:
//!        a. Bump allocator (64 KB scratch heap for codec Vec usage)
//!        b. Peripherals + clocks (12 MHz XTAL → 150 MHz sys clock)
//!        c. PMP stack guard (NAPOT region at stack top, hardware-locked)
//!        d. CRC32 firmware integrity check (Phase 3 wired)
//!        e. KAT (ternary MAC parity) — boot-time invariant check
//!        f. PipelineScheduler init + safety subsystems init (Phase 5)
//!        g. Watchdog start (500 ms timeout)
//!   4. Main loop: pet watchdog, when ADC window ready → encode + tx
//!
//! Phase 6 wiring: PipelineScheduler is now linked. SPI0 (ADS1299), SPI1
//! (BLE), USB CDC peripherals are stubbed via mock buffers — full
//! peripheral wiring is the final follow-up before flashing real silicon.
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
    use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use embedded_alloc::LlffHeap as Heap;
    use lamquant_firmware::dsp::biquad::{NUM_CHANNELS, WINDOW_SAMPLES};
    use lamquant_firmware::safety::SafetyState;
    use lamquant_firmware::scheduler::{CodecMode, OutputMode, PipelineScheduler};
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

    // ─── Pipeline state — singleton, BSS-allocated ────────────────────
    //
    // The C firmware put these in dedicated SRAM banks via section
    // attributes (`.sram2_sram3` for the ADC buffer, `.sram4_tnn` for the
    // weights, etc.). Phase 6 leaves them in default `.bss`; per-bank
    // placement attributes are added in the linker-script polish pass
    // before flashing real silicon.

    /// Raw ADC sample buffer. DMA target. 21 × 2500 × 4 = 210 KB.
    #[link_section = ".bss"]
    static mut RAW_ADC_BUFFER: [[i32; WINDOW_SAMPLES]; NUM_CHANNELS] =
        [[0; WINDOW_SAMPLES]; NUM_CHANNELS];

    /// SNN per-group activity classification, set by SNN inference each
    /// window. Latent shape (8 groups × 79 timesteps).
    #[link_section = ".bss"]
    static mut SNN_ACTIVITY_MAP: [[u8; 79]; 8] = [[0; 79]; 8];

    /// PipelineScheduler — owns biquad bank, LPC scratch, subbands, FSQ
    /// adaptive, rANS encoder, LPC delta encoder, output buf (~110 KB).
    #[link_section = ".bss"]
    static mut PIPELINE: MaybeUninit<PipelineScheduler> = MaybeUninit::uninit();

    /// Patient-safety subsystems (~59 KB).
    #[link_section = ".bss"]
    static mut SAFETY: MaybeUninit<SafetyState> = MaybeUninit::uninit();

    /// Set by the ADC DMA-complete ISR. Cleared by the scheduler.
    static ADC_BUFFER_READY: AtomicBool = AtomicBool::new(false);

    /// Phase 6 stub clock. Real time comes from `hal::timer::Timer::now()`
    /// once the timer peripheral is plumbed through.
    static CLOCK_MS: AtomicU32 = AtomicU32::new(0);

    fn now_ms() -> u32 {
        // Monotonically advance by 10 s per call (matches window cadence
        // so safety event timestamps look sane in the audit log).
        CLOCK_MS.fetch_add(10_000, Ordering::Relaxed)
    }

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

        // Initialize singletons. SAFETY: BSS storage is zero-initialized
        // by riscv-rt; we replace with a fully-constructed instance and
        // never alias the MaybeUninit slot afterward.
        unsafe {
            PIPELINE.write(PipelineScheduler::new());
            SAFETY.write(SafetyState::default());
            (*SAFETY.as_mut_ptr()).init(now_ms());
        }

        // Configure default session: clinical neural, BLE-only output.
        let pipeline: &mut PipelineScheduler = unsafe { &mut *PIPELINE.as_mut_ptr() };
        pipeline.set_codec_mode(CodecMode::Neural);
        pipeline.output_mode = OutputMode::BleOnly;

        // Start watchdog.
        watchdog.start(fugit::ExtU32::millis(500));

        // ── Main loop ──────────────────────────────────────────────
        loop {
            watchdog.feed();

            if ADC_BUFFER_READY.swap(false, Ordering::AcqRel) {
                run_one_window();
            } else {
                // Sleep until the next interrupt (DMA complete, host serial,
                // etc.). Saves ~3 mA vs polling on Hazard3.
                power::wait_for_interrupt();
            }
        }
    }

    /// Process one ADC window: HP biquad → LPC → lifting → SNN → encode → TX.
    fn run_one_window() {
        let pipeline: &mut PipelineScheduler = unsafe { &mut *PIPELINE.as_mut_ptr() };
        let safety: &mut SafetyState = unsafe { &mut *SAFETY.as_mut_ptr() };
        let signal: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS] =
            unsafe { &mut *addr_of_mut!(RAW_ADC_BUFFER) };
        let activity_map: &[[u8; 79]; 8] =
            unsafe { &*core::ptr::addr_of!(SNN_ACTIVITY_MAP) };

        let snn_activity_sum = activity_map
            .iter()
            .map(|row| row.iter().map(|&v| v as u32).sum::<u32>())
            .sum::<u32>()
            .min(255) as u8;

        let result = pipeline.encode_window(
            signal,
            activity_map,
            snn_activity_sum,
            safety,
            now_ms(),
        );

        // Phase 6 stub: real path hands `result.bytes` to BLE DMA TX
        // (transport::ble) or USB CDC writer (transport::usb). For now
        // the bytes live in the scheduler buffer and the safety subsystem
        // has already pushed them onto the BLE retry queue.
        let _ = result;
    }

    /// Public hook — ADC DMA-complete ISR (real wiring lands with SPI0).
    /// Calling convention preserved so the future ISR can `extern "C"` it.
    #[no_mangle]
    pub extern "C" fn on_adc_dma_complete() {
        ADC_BUFFER_READY.store(true, Ordering::Release);
    }
}

// ───────────── Host stub (cargo test on x86_64) ─────────────

#[cfg(not(target_arch = "riscv32"))]
fn main() {
    // No-op: binary exists only to satisfy `[[bin]]`; all testable logic
    // is in the library and runs via `cargo test`.
}
