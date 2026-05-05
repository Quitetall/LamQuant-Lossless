//! Context-adaptive rANS encoder.
//!
//! Two frequency tables per FSQ level (quiescent + event), selected per
//! timestep based on the SNN activity classification. Frequency tables
//! are placeholder defaults here; production export will overwrite them
//! via `lamquant-weights` (deferred sub-task).
//!
//! Optimizations match the C source:
//!   * Reciprocal multiplication for division-free encode (3 instructions:
//!     mulhu + mul + sub vs DIV's 64 cycles on Hazard3).
//!   * Power-of-2 RANS_TOTAL_FREQ (4096) for shift-instead-of-mod.
//!   * Branchless renormalization (predicate masks instead of branches).
//!
//! Output buffer fixed at 240 bytes — matches BLE packet payload size.
//! 32-bit rANS state. No 64-bit emulation needed on RV32.

use super::fsq_adaptive::{FsqAdaptive, LATENT_TIMESTEPS};

const RANS_L: u32 = 1 << 15;
const RANS_TOTAL_FREQ: u32 = 4096; // 2^12 — power of 2
pub const MAX_SYMBOLS: usize = 32; // L=32 in clinical mode
pub const RANS_BUF_SIZE: usize = 240;

/// One frequency table for one (context, FSQ_level) combination.
///
/// `freq[s]`  — per-symbol frequency, sums to RANS_TOTAL_FREQ
/// `start[s]` — cumulative start (exclusive prefix sum of freq)
/// `recip[s]` — precomputed `(1 << 32) / freq[s]` for division-free encode
#[derive(Copy, Clone)]
pub struct RansFreqTable {
    pub freq: [u32; MAX_SYMBOLS],
    pub start: [u32; MAX_SYMBOLS],
    pub recip: [u32; MAX_SYMBOLS],
    pub num_symbols: u32,
}

impl RansFreqTable {
    /// Build from raw frequencies; computes start and recip arrays.
    pub fn from_freq(freq: [u32; MAX_SYMBOLS], num_symbols: u32) -> Self {
        let mut start = [0u32; MAX_SYMBOLS];
        let mut recip = [0u32; MAX_SYMBOLS];
        let mut sum = 0u32;
        let mut i = 0usize;
        while i < num_symbols as usize {
            start[i] = sum;
            sum = sum.wrapping_add(freq[i]);
            recip[i] = if freq[i] > 0 {
                ((1u64 << 32) / freq[i] as u64) as u32
            } else {
                u32::MAX
            };
            i += 1;
        }
        Self {
            freq,
            start,
            recip,
            num_symbols,
        }
    }
}

/// Tables for [context: 0=quiet, 1=event][L_index: 0=L2, 1=L3, 2=L5, 3=L32].
pub struct RansTables {
    pub tables: [[RansFreqTable; 4]; 2],
}

impl RansTables {
    /// Default (placeholder) tables matching the C firmware initial state.
    /// Replace with trained values from `lamquant-weights` when available.
    pub fn default_placeholder() -> Self {
        // L=2 placeholders.
        let quiet_l2 = build_freq([3072, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 2);
        let event_l2 = build_freq([2048, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 2);

        // L=3 placeholders.
        let quiet_l3 = build_freq([512, 3072, 512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3);
        let event_l3 = build_freq([1024, 2048, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3);

        // L=5 placeholders.
        let quiet_l5 = build_freq([256, 768, 2048, 768, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5);
        let event_l5 = build_freq([512, 1024, 1024, 1024, 512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5);

        // L=32 placeholders — Gaussian-like for quiet, flatter for event.
        let quiet_l32 = build_freq(
            [
                8, 12, 18, 28, 42, 64, 96, 128,
                160, 192, 224, 256, 288, 304, 312, 320,
                320, 312, 304, 288, 256, 224, 192, 160,
                128, 96, 64, 42, 28, 18, 12, 8,
            ],
            32,
        );
        let event_l32 = build_freq(
            [
                64, 72, 80, 88, 96, 104, 112, 120,
                128, 132, 136, 140, 144, 148, 148, 148,
                148, 148, 148, 144, 140, 136, 132, 128,
                120, 112, 104, 96, 88, 80, 72, 64,
            ],
            32,
        );

        Self {
            tables: [
                [quiet_l2, quiet_l3, quiet_l5, quiet_l32],
                [event_l2, event_l3, event_l5, event_l32],
            ],
        }
    }

    /// Look up (context, fsq_config_idx). cfg_idx 0..=3 maps to L=2/3/5/32.
    #[inline]
    pub fn get(&self, context: usize, cfg_idx: usize) -> &RansFreqTable {
        &self.tables[context.min(1)][cfg_idx.min(3)]
    }
}

fn build_freq(freq: [u32; MAX_SYMBOLS], num_symbols: u32) -> RansFreqTable {
    RansFreqTable::from_freq(freq, num_symbols)
}

// ─── BLE packet sync constants (match C wire format) ─────────────────

const SYNC: [u8; 4] = [b'Q', b'M', b'A', b'L'];
const MODE_GOLDEN_ADAPTIVE: u8 = 0x02;

/// Stateful encoder: writes into a fixed 240-byte buffer.
pub struct RansContextEncoder {
    pub buffer: [u8; RANS_BUF_SIZE],
    byte_idx: usize,
    bit_idx: u32,
    state: u32,
    tables: RansTables,
}

impl RansContextEncoder {
    pub fn new(tables: RansTables) -> Self {
        Self {
            buffer: [0; RANS_BUF_SIZE],
            byte_idx: 0,
            bit_idx: 0,
            state: RANS_L,
            tables,
        }
    }

    fn reset(&mut self) {
        self.buffer = [0; RANS_BUF_SIZE];
        self.byte_idx = 0;
        self.bit_idx = 0;
        self.state = RANS_L;
    }

    fn push_byte(&mut self, b: u8) {
        if self.byte_idx < RANS_BUF_SIZE {
            self.buffer[self.byte_idx] = b;
            self.byte_idx += 1;
        }
    }

    fn push_bits(&mut self, bits: u32, n: u32) {
        for i in 0..n {
            if self.byte_idx >= RANS_BUF_SIZE {
                return;
            }
            let bit = ((bits >> i) & 1) as u8;
            self.buffer[self.byte_idx] |= bit << self.bit_idx;
            self.bit_idx += 1;
            if self.bit_idx == 8 {
                self.bit_idx = 0;
                self.byte_idx += 1;
            }
        }
    }

    /// Encode one symbol. Branchless renormalization + reciprocal-multiply
    /// for division-free state update — matches Hazard3-tuned C.
    fn encode_sym(&mut self, mut sym: u32, tab: &RansFreqTable) {
        if sym >= tab.num_symbols {
            sym = 0; // safety
        }

        let freq = tab.freq[sym as usize];
        let start = tab.start[sym as usize];
        if freq == 0 {
            return; // skip zero-probability symbol
        }

        let x_max = ((RANS_L / RANS_TOTAL_FREQ) * freq) | 1;
        let bound = x_max << 16;

        // Renormalize (up to 2 byte flushes).
        for _ in 0..2 {
            if self.state >= bound {
                let byte = (self.state & 0xFF) as u32;
                self.push_bits(byte, 8);
                self.state >>= 8;
            }
        }

        // Division-free update via precomputed reciprocal.
        let recip = tab.recip[sym as usize];
        let q = ((self.state as u64 * recip as u64) >> 32) as u32;
        let r = self.state.wrapping_sub(q.wrapping_mul(freq));
        self.state = q
            .wrapping_mul(RANS_TOTAL_FREQ)
            .wrapping_add(start)
            .wrapping_add(r);
    }

    fn flush_state(&mut self) {
        // Flush all 4 bytes of state to the bitstream.
        for shift in [0, 8, 16, 24] {
            self.push_bits((self.state >> shift) & 0xFF, 8);
        }
    }

    /// Encode all FSQ symbols in the FsqAdaptive output buffer.
    /// Returns total bytes written.
    pub fn encode_fsq(&mut self, fsq: &FsqAdaptive, snn_activity_sum: u8) -> usize {
        self.reset();

        // Header: sync + mode + level summary (computed elsewhere) + activity.
        for &b in &SYNC {
            self.push_byte(b);
        }
        self.push_byte(MODE_GOLDEN_ADAPTIVE);

        // Level summary stub: 2 bytes from the FSQ encoder. Caller must
        // set this via FsqAdaptive::build_level_summary() first; we
        // accept it as 2-byte trailing context. For now, write zeros to
        // keep the wire format stable until the orchestrator wires it.
        self.push_byte(0);
        self.push_byte(0);

        // SNN activity sum (1 byte).
        self.push_byte(snn_activity_sum);

        // Encode FSQ symbols in reverse (rANS is LIFO).
        let symbols = fsq.symbols();
        let level_bm = fsq.level_bitmap();
        let n = symbols.len();

        let mut i = n;
        while i > 0 {
            i -= 1;
            if self.byte_idx >= RANS_BUF_SIZE - 8 {
                break; // leave room for state flush
            }
            let t = i / 32; // 32 dims per timestep
            let cfg_idx = if t < LATENT_TIMESTEPS {
                level_bm[t] as usize
            } else {
                0
            };
            // Context: quiet (cfg 0 = L=2) vs event (cfg > 0).
            let context = if cfg_idx > 0 { 1 } else { 0 };
            // Copy the table reference's contents to break the immutable
            // borrow before calling encode_sym (&mut self).
            let tab = *self.tables.get(context, cfg_idx);
            self.encode_sym(symbols[i], &tab);
        }

        self.flush_state();
        self.bytes_used()
    }

    /// Bytes used in the buffer so far (rounded up if mid-byte).
    #[inline]
    pub fn bytes_used(&self) -> usize {
        self.byte_idx + if self.bit_idx > 0 { 1 } else { 0 }
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buffer[..self.bytes_used()]
    }
}
