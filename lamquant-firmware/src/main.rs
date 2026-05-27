//! LamQuant v7.7 firmware — bare-metal binary entry point.
//!
//! Boot sequence (riscv32 target only):
//!   1. RP2350 boot ROM loads firmware from QSPI flash (XIP)
//!   2. riscv-rt _start sets up stack, .data, .bss, calls main()
//!   3. main() initializes:
//!        a. Bump allocator (16 KB scratch heap for codec Vec usage)
//!        b. Peripherals + clocks (12 MHz XTAL → 150 MHz sys clock)
//!        c. PMP stack guard (NAPOT region at stack top, hardware-locked)
//!        d. CRC32 firmware integrity check (Phase 3 wired)
//!        e. KAT (ternary MAC parity) — boot-time invariant check
//!        f. PipelineScheduler init + safety subsystems init (Phase 5)
//!        g. Watchdog start (500 ms timeout)
//!   4. Main loop: pet watchdog, when ADC window ready → encode + tx
//!
//! Phase 6 wiring: PipelineScheduler is now linked. ADC fills directly
//! into `pipeline.lpc.residual` via aliased raw pointer (matches the C
//! `#define lpc_residual raw_adc_buffer` aliasing trick) so we save 210 KB
//! of duplicate buffer space.
//!
//! On host (`cargo test`), the binary collapses to a no-op so `cargo
//! build` doesn't fail. All testable logic lives in the library
//! (`src/lib.rs`) and runs under the host test harness.

#![cfg_attr(target_arch = "riscv32", no_std)]
#![cfg_attr(target_arch = "riscv32", no_main)]

// ───────────── Embedded path (RP2350 / riscv32) ─────────────
// Gated on the `firmware-bin` feature (default-on). Library-only
// consumers (tools/hazard3_bench etc.) build with `default-features =
// false` so the rp235x-hal peripheral driver + its critical-section
// impl are not linked in.

#[cfg(all(target_arch = "riscv32", feature = "firmware-bin"))]
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
    // Hazard3 RISC-V panic handler. `panic-probe` is Cortex-M-only
    // upstream, so the RISC-V build uses `panic-halt` directly. defmt
    // text is still available via defmt-RTT for non-panic logging.
    use panic_halt as _;
    use rp235x_hal as hal;
    use rp235x_hal::pac;

    // ─── Boot ROM image definition ────────────────────────────────────
    #[link_section = ".start_block"]
    #[used]
    pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

    // ─── Global allocator ─────────────────────────────────────────────
    //
    // DSP path is alloc-free (lifting + LPC inlined, no Vec).
    //
    // **Known issue (Phase 7):** `neural::focal::forward` still uses
    // transient `Vec<Vec<i16>>` activation buffers with a peak heap demand
    // of ~80 KB (stage 3, focal1→focal2). With the 6 KB heap below, Mode 1
    // (Neural) encoding will OOM at runtime on RP2350. This is **pre-
    // existing** — the focal path was scaffolded with a TODO to swap
    // `ActBuf` for a passed-in static scratch (see focal.rs:140). RP2350
    // RAM cannot accommodate both the current 205 KB residual + 209 KB
    // subbands + 80 KB focal heap; the proper fix overlays focal scratch
    // on the post-lifting `residual` buffer (which is dead after lifting).
    //
    // 6 KB suffices for the rANS table builder + error fmt. Mode 2
    // (lossless) is fully alloc-free post-DSP.
    #[global_allocator]
    static HEAP: Heap = Heap::empty();
    const HEAP_SIZE: usize = 6 * 1024;

    #[link_section = ".bss"]
    static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];

    // ─── Pipeline state — singleton, BSS-allocated ────────────────────

    /// PipelineScheduler — owns biquad bank, LPC scratch (== ADC buffer),
    /// subbands, FSQ adaptive, rANS encoder, LPC delta encoder.
    /// `lpc.residual` doubles as the raw ADC fill target.
    #[link_section = ".bss"]
    static mut PIPELINE: MaybeUninit<PipelineScheduler> = MaybeUninit::uninit();

    /// Patient-safety subsystems.
    #[link_section = ".bss"]
    static mut SAFETY: MaybeUninit<SafetyState> = MaybeUninit::uninit();

    /// SNN per-group activity classification, set by SNN inference each
    /// window. Latent shape (8 groups × 79 timesteps).
    #[link_section = ".bss"]
    static mut SNN_ACTIVITY_MAP: [[u8; 79]; 8] = [[0; 79]; 8];

    /// Set by the ADC DMA-complete ISR. Cleared by the scheduler.
    static ADC_BUFFER_READY: AtomicBool = AtomicBool::new(false);

    /// Phase 6 stub clock. Real time comes from `hal::timer::Timer::now()`
    /// once the timer peripheral is plumbed through.
    static CLOCK_MS: AtomicU32 = AtomicU32::new(0);

    fn now_ms() -> u32 {
        // Monotonically advance by 10 s per call (matches window cadence).
        CLOCK_MS.fetch_add(10_000, Ordering::Relaxed)
    }

    const XTAL_HZ: u32 = 12_000_000;

    /// Aliased pointer to the ADC fill region (== pipeline.lpc.residual).
    /// Used by the DRDY ISR to deposit one sample per channel without
    /// taking out a `&mut PipelineScheduler` borrow. SAFETY contract:
    /// caller MUST stop continuous ADC mode before encoding starts.
    fn adc_buffer_ptr() -> *mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS] {
        unsafe { addr_of_mut!((*PIPELINE.as_mut_ptr()).lpc.residual) }
    }

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

        // PMP stack guard.
        stack_guard::install();

        // Firmware integrity check (CRC32 over weight tables).
        if !integrity::check_firmware_crc() {
            power::enter_safe_mode();
        }

        // Ternary MAC parity KAT — mandatory before any patient data.
        if neural::ternary_mac::boot_parity_kat().is_err() {
            power::enter_safe_mode();
        }

        // Initialize singletons. SAFETY: BSS slots zero-initialized by
        // riscv-rt; we replace with a constructed instance and never
        // alias the MaybeUninit slot afterward.
        unsafe {
            PIPELINE.write(PipelineScheduler::new());
            SAFETY.write(SafetyState::default());
            (*SAFETY.as_mut_ptr()).init(now_ms());
        }

        // Configure default session: clinical neural, BLE-only output.
        let pipeline: &mut PipelineScheduler = unsafe { &mut *PIPELINE.as_mut_ptr() };
        pipeline.set_codec_mode(CodecMode::Neural);
        pipeline.output_mode = OutputMode::BleOnly;

        // Suppress unused-pointer warning until Phase 7 wires the ADS1299
        // ISR which calls adc_buffer_ptr() to deposit samples.
        let _ = adc_buffer_ptr;

        watchdog.start(fugit::ExtU32::millis(500));

        // ── Main loop ──────────────────────────────────────────────
        loop {
            watchdog.feed();

            if ADC_BUFFER_READY.swap(false, Ordering::AcqRel) {
                run_one_window();
            } else {
                power::wait_for_interrupt();
            }
        }
    }

    /// Process one ADC window: HP biquad → LPC → lifting → SNN → encode → TX.
    ///
    /// SAFETY: ADC must have stopped continuous mode before this runs (see
    /// `adc_buffer_ptr` contract). Uses raw pointer aliasing to read the
    /// signal from `pipeline.lpc.residual` while `pipeline` is also borrowed
    /// mutably; the per-channel LPC analyze copies to a Vec<i64> snapshot
    /// before writing back, so no interleaved RW.
    fn run_one_window() {
        let pipeline: &mut PipelineScheduler = unsafe { &mut *PIPELINE.as_mut_ptr() };
        let safety: &mut SafetyState = unsafe { &mut *SAFETY.as_mut_ptr() };

        // SAFETY: signal aliases pipeline.lpc.residual. encode_window first
        // runs HP biquad in-place (writes to signal == lpc.residual), then
        // analyze_all_channels takes a Vec<i64> snapshot per channel before
        // writing back to lpc.residual — no interleaved RW.
        let signal: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS] =
            unsafe { &mut *adc_buffer_ptr() };

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

        // Phase 7 stub: real path hands `result.bytes` to BLE DMA TX
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

#[cfg(any(not(target_arch = "riscv32"), not(feature = "firmware-bin")))]
fn main() {
    // No-op: binary exists only to satisfy `[[bin]]`; all testable logic
    // is in the library and runs via `cargo test`.
}
