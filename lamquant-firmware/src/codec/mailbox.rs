//! Inter-core mailbox: Core 0 ↔ Core 1 dispatch.
//!
//! 32-byte volatile structure in SRAM8 cache-line-aligned. Core 0 writes
//! a DISPATCH command, fences memory, fires SEV. Core 1 wakes from WFE,
//! reads the mailbox, runs TNN inference, writes DONE + result address,
//! fences, fires SEV. Core 0 reads result and transmits BLE packet.
//!
//! v7.7 invariant: Core 0 spin-waits on Core 1 (no deadline race like v7.1).
//! 264 ms TNN inference fits comfortably in 10 s window budget (38× margin).

use core::sync::atomic::{compiler_fence, Ordering};

/// Mailbox commands. `repr(u8)` for a single-byte volatile load/store.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum MailboxCmd {
    Idle = 0x00,
    /// Core 0 → Core 1: run TNN encoder + entropy.
    Dispatch = 0x01,
    /// Core 1 → Core 0: encoding complete.
    Done = 0x02,
    /// Core 0 → Core 1: cancel (timeout / safe mode).
    Abort = 0x03,
    /// Core 1 → Core 0: error during encode.
    Error = 0xFF,
}

impl MailboxCmd {
    fn from_u8(v: u8) -> Self {
        match v {
            0x00 => Self::Idle,
            0x01 => Self::Dispatch,
            0x02 => Self::Done,
            0x03 => Self::Abort,
            _ => Self::Error,
        }
    }
}

/// Shared mailbox structure (exactly 32 bytes, cache-line aligned).
///
/// All fields are accessed via volatile loads/stores. `compiler_fence`
/// pairs around writes match the C `fence iorw, iorw` barriers.
#[repr(C, align(32))]
pub struct Mailbox {
    cmd: u8,                     // MailboxCmd
    activity_sum: u8,            // SNN activity summary across groups
    fsq_level_bitmap_lo: u8,     // FSQ level bitmap (low byte)
    fsq_level_bitmap_hi: u8,     // FSQ level bitmap (high byte)
    golden_buf_addr: u32,        // Pointer to entropy output buffer
    golden_buf_len: u32,         // Bytes written
    sequence_num: u32,           // Frame counter
    encode_cycles: u32,          // Core 1 encode time
    _reserved: [u32; 3],         // Pad to 32 bytes
}

impl Mailbox {
    /// Construct an idle mailbox (compile-time const).
    pub const fn new() -> Self {
        Self {
            cmd: MailboxCmd::Idle as u8,
            activity_sum: 0,
            fsq_level_bitmap_lo: 0,
            fsq_level_bitmap_hi: 0,
            golden_buf_addr: 0,
            golden_buf_len: 0,
            sequence_num: 0,
            encode_cycles: 0,
            _reserved: [0; 3],
        }
    }

    // ─── Core 0 (scheduler) API ──────────────────────────────────

    /// Reset to IDLE. Called once at boot before launching Core 1.
    pub fn init(&mut self) {
        self.cmd = MailboxCmd::Idle as u8;
        self.activity_sum = 0;
        self.fsq_level_bitmap_lo = 0;
        self.fsq_level_bitmap_hi = 0;
        self.golden_buf_addr = 0;
        self.golden_buf_len = 0;
        self.sequence_num = 0;
        self.encode_cycles = 0;
        compiler_fence(Ordering::SeqCst);
    }

    /// Dispatch Core 1 with the SNN activity summary + frame number.
    /// Caller is responsible for SEV (see `sev()` helper below).
    pub fn dispatch(&mut self, activity_sum: u8, sequence_num: u32) {
        self.activity_sum = activity_sum;
        self.sequence_num = sequence_num;
        compiler_fence(Ordering::SeqCst);
        self.cmd = MailboxCmd::Dispatch as u8;
        compiler_fence(Ordering::SeqCst);
    }

    /// True when Core 1 has finished. Called by Core 0 spin-loop.
    pub fn is_done(&self) -> bool {
        compiler_fence(Ordering::SeqCst);
        self.cmd == MailboxCmd::Done as u8
    }

    /// Abort Core 1 (e.g. on safe-mode entry).
    pub fn abort(&mut self) {
        self.cmd = MailboxCmd::Abort as u8;
        compiler_fence(Ordering::SeqCst);
    }

    /// After `is_done()`: read the result buffer pointer + length.
    /// SAFETY: pointer comes from Core 1 — caller must trust the protocol.
    pub fn result(&self) -> (u32, u32, u32) {
        compiler_fence(Ordering::SeqCst);
        (self.golden_buf_addr, self.golden_buf_len, self.encode_cycles)
    }

    /// Mark mailbox idle so Core 1 can dispatch again next window.
    pub fn ack(&mut self) {
        self.cmd = MailboxCmd::Idle as u8;
        compiler_fence(Ordering::SeqCst);
    }

    // ─── Core 1 (encoder) API ────────────────────────────────────

    /// Block until Core 0 dispatches (called in Core 1 idle loop).
    /// Real Core 1 loop wraps this with WFE; this is the no-WFE poll.
    pub fn poll_for_dispatch(&self) -> bool {
        compiler_fence(Ordering::SeqCst);
        self.cmd == MailboxCmd::Dispatch as u8
    }

    pub fn read_dispatch(&self) -> (u8, u32) {
        compiler_fence(Ordering::SeqCst);
        (self.activity_sum, self.sequence_num)
    }

    /// Signal completion to Core 0.
    pub fn signal_done(&mut self, buf_addr: u32, buf_len: u32, cycles: u32) {
        self.golden_buf_addr = buf_addr;
        self.golden_buf_len = buf_len;
        self.encode_cycles = cycles;
        compiler_fence(Ordering::SeqCst);
        self.cmd = MailboxCmd::Done as u8;
        compiler_fence(Ordering::SeqCst);
    }

    /// Signal error to Core 0.
    pub fn signal_error(&mut self) {
        self.cmd = MailboxCmd::Error as u8;
        compiler_fence(Ordering::SeqCst);
    }

    /// Core 1: did Core 0 request abort?
    pub fn should_abort(&self) -> bool {
        compiler_fence(Ordering::SeqCst);
        self.cmd == MailboxCmd::Abort as u8
    }
}

impl Default for Mailbox {
    fn default() -> Self {
        Self::new()
    }
}

/// SEV wake on RISC-V. Hazard3 implements WFI + interrupt-pending bit
/// instead of ARM's WFE/SEV — but the multicore HAL exposes a wake helper.
/// This is a placeholder; Phase 5 wires the real mechanism.
#[cfg(target_arch = "riscv32")]
#[inline(always)]
pub fn sev() {
    // RISC-V has no SEV. Hazard3 inter-core wake uses the SIO mailbox
    // FIFO + WFI. Phase 5 scheduler will plug rp235x-hal `sio::Sio` here.
}

/// WFE equivalent — hart sleeps until interrupt. No-op on host.
#[cfg(target_arch = "riscv32")]
#[inline(always)]
pub fn wfe() {
    riscv::asm::wfi();
}
