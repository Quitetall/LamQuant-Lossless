//! Hybrid entropy orchestrator. Dispatches to one of two paths:
//!
//!   Mode 1 (Neural):    FSQ symbols → context-adaptive rANS (`rans_context`)
//!                       Output: 240-byte BLE packet.
//!
//!   Mode 2 (Lossless):  LPC delta side-info + run-length Golomb-Rice on
//!                       thresholded detail subbands.
//!                       Output: variable-size BLE packet, bit-exact roundtrip.
//!
//! v7.7 simplification: legacy v7.1 had a "golden/lightning" deadline race
//! where lightning ran in parallel as a fallback. v7.7 selects one mode at
//! session start (host serial command) and runs deterministically — 264 ms
//! TNN inference fits 38× within the 10 s window budget.
//!
//! Output buffer fixed at 240 bytes — matches BLE packet payload size.

use core::cmp::min;

use super::detail_threshold::subband_mask;
use super::fsq_adaptive::FsqAdaptive;
use super::lpc_delta::LpcDelta;
use super::quality::QualityMode;
use super::rans_context::{RansContextEncoder, RANS_BUF_SIZE};

use crate::dsp::lifting::{Subbands, L1_DETAIL_LEN, L2_DETAIL_LEN, L3_DETAIL_LEN};

const NUM_CHANNELS: usize = 21;
const LPC_ORDER: usize = 8;

/// Wire-format mode byte. Stays in the BLE packet header so the host
/// decoder picks the right path.
#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum WirePacketMode {
    Neural = 0x02,   // matches MODE_GOLDEN_ADAPTIVE in C wire format
    Lossless = 0x10, // new in v7.7
}

// ─── Bit-level packer for the lossless path ──────────────────────

struct BitPacker {
    buf: [u8; RANS_BUF_SIZE],
    byte_idx: usize,
    bit_idx: u32,
}

impl BitPacker {
    const fn new() -> Self {
        Self {
            buf: [0; RANS_BUF_SIZE],
            byte_idx: 0,
            bit_idx: 0,
        }
    }

    fn push_byte(&mut self, b: u8) {
        if self.byte_idx < RANS_BUF_SIZE {
            // Round up to next byte boundary first.
            if self.bit_idx > 0 {
                self.bit_idx = 0;
                self.byte_idx += 1;
                if self.byte_idx >= RANS_BUF_SIZE {
                    return;
                }
            }
            self.buf[self.byte_idx] = b;
            self.byte_idx += 1;
        }
    }

    fn push_bits(&mut self, bits: u32, n: u32) {
        for i in 0..n {
            if self.byte_idx >= RANS_BUF_SIZE {
                return;
            }
            let bit = ((bits >> i) & 1) as u8;
            self.buf[self.byte_idx] |= bit << self.bit_idx;
            self.bit_idx += 1;
            if self.bit_idx == 8 {
                self.bit_idx = 0;
                self.byte_idx += 1;
            }
        }
    }

    fn bytes_used(&self) -> usize {
        self.byte_idx + if self.bit_idx > 0 { 1 } else { 0 }
    }
}

// ─── Golomb-Rice helpers ─────────────────────────────────────────

/// Zigzag-encode a signed residual to unsigned, then split into unary
/// quotient + binary remainder of `k` bits.
fn rice_encode(p: &mut BitPacker, residual: i32, k: u32) {
    let mapped: u32 = if residual >= 0 {
        (residual as u32) << 1
    } else {
        ((-(residual as i64)) as u32) << 1 | 1
    };
    let q = mapped >> k;
    let rem = mapped & ((1u32 << k) - 1);

    let mut written = 0;
    while written < q && p.byte_idx < RANS_BUF_SIZE - 2 {
        p.push_bits(1, 1);
        written += 1;
    }
    p.push_bits(0, 1);
    p.push_bits(rem, k);
}

/// Encode a sparse (thresholded) subband: run-length zeros + Golomb-Rice
/// values with adaptive k. Format:
///   [4 bits: k_init]
///   For each coefficient:
///     if zero → continues run
///     else → emit run-length (Rice k=3), then the value (Rice k=adaptive)
///   trailing run flushed at end
fn rice_encode_sparse_subband(p: &mut BitPacker, coeffs: &[i32], k_init: u32) {
    p.push_bits(k_init & 0x0F, 4);

    let mut k = k_init;
    let mut run: u32 = 0;
    let mut abs_accum: u32 = 0;
    let mut sample_count: u32 = 0;

    for &c in coeffs.iter() {
        if p.byte_idx >= RANS_BUF_SIZE - 4 {
            break;
        }
        if c == 0 {
            run += 1;
            continue;
        }

        // Flush the run of zeros (Rice with k=3).
        let q = run >> 3;
        let rem = run & 0x07;
        let mut written = 0;
        while written < q && p.byte_idx < RANS_BUF_SIZE - 2 {
            p.push_bits(1, 1);
            written += 1;
        }
        p.push_bits(0, 1);
        p.push_bits(rem, 3);
        run = 0;

        // Emit the non-zero value.
        rice_encode(p, c, k);

        // Update adaptive k from running mean(|coeff|).
        abs_accum += c.unsigned_abs();
        sample_count += 1;
        if sample_count >= 16 {
            let mean_abs = abs_accum / sample_count;
            let mut new_k: u32 = 0;
            let mut m = mean_abs;
            while m > 1 {
                m >>= 1;
                new_k += 1;
            }
            k = new_k.clamp(1, 8);
            abs_accum = 0;
            sample_count = 0;
        }
    }

    // Trailing run.
    if run > 0 && p.byte_idx < RANS_BUF_SIZE - 2 {
        let q = run >> 3;
        let rem = run & 0x07;
        let mut written = 0;
        while written < q && p.byte_idx < RANS_BUF_SIZE - 2 {
            p.push_bits(1, 1);
            written += 1;
        }
        p.push_bits(0, 1);
        p.push_bits(rem, 3);
    }
}

// ─── Public orchestrator ─────────────────────────────────────────

/// Output of an entropy encode: pointer + length into a caller-owned buffer.
pub struct EntropyOutput<'a> {
    pub bytes: &'a [u8],
    pub mode: WirePacketMode,
}

/// Mode 1: encode the FSQ-quantized neural latent into the rANS packet.
/// Wraps `RansContextEncoder::encode_fsq` and stamps the mode byte.
pub fn encode_neural<'a>(
    rans: &'a mut RansContextEncoder,
    fsq: &FsqAdaptive,
    snn_activity_sum: u8,
) -> EntropyOutput<'a> {
    rans.encode_fsq(fsq, snn_activity_sum);
    EntropyOutput {
        bytes: rans.buffer(),
        mode: WirePacketMode::Neural,
    }
}

/// Mode 2: encode LPC coefficients + thresholded detail subbands.
///
/// Output buffer must be at least 240 bytes. Format on the wire:
///   [SYNC: 4B][mode: 1B][LPC delta payload: variable][subband mask: 1B]
///   [L3 detail: 21 channels × adaptive][L2: optional][L1: optional]
pub fn encode_lossless<'a>(
    out: &'a mut [u8; RANS_BUF_SIZE],
    lpc_delta: &mut LpcDelta,
    lpc_coeffs: &[[i32; LPC_ORDER]; NUM_CHANNELS],
    subbands: &Subbands,
    mode: QualityMode,
) -> EntropyOutput<'a> {
    let mut p = BitPacker::new();

    // Sync header.
    p.push_byte(b'Q');
    p.push_byte(b'M');
    p.push_byte(b'A');
    p.push_byte(b'L');
    p.push_byte(WirePacketMode::Lossless as u8);

    // LPC delta payload (variable: 169 / 337 / 673 bytes).
    // Use a temporary scratch since LpcDelta::encode wants a contiguous
    // slice. The lossless packet is bigger than the rANS slot, so cap.
    let mut lpc_buf = [0u8; 673];
    let lpc_n = lpc_delta.encode(lpc_coeffs, &mut lpc_buf);
    let lpc_to_copy = min(lpc_n, RANS_BUF_SIZE.saturating_sub(p.byte_idx));
    for &b in &lpc_buf[..lpc_to_copy] {
        p.push_byte(b);
    }

    // Subband mask + per-subband Rice payloads.
    let mask = subband_mask(mode);
    p.push_byte(mask);

    if mask & 0b001 != 0 {
        for ch in 0..NUM_CHANNELS {
            if p.byte_idx >= RANS_BUF_SIZE - 8 {
                break;
            }
            rice_encode_sparse_subband(
                &mut p,
                &subbands.l3_detail[ch][..L3_DETAIL_LEN],
                4,
            );
        }
    }
    if mask & 0b010 != 0 {
        for ch in 0..NUM_CHANNELS {
            if p.byte_idx >= RANS_BUF_SIZE - 8 {
                break;
            }
            rice_encode_sparse_subband(
                &mut p,
                &subbands.l2_detail[ch][..L2_DETAIL_LEN],
                3,
            );
        }
    }
    if mask & 0b100 != 0 {
        for ch in 0..NUM_CHANNELS {
            if p.byte_idx >= RANS_BUF_SIZE - 8 {
                break;
            }
            rice_encode_sparse_subband(
                &mut p,
                &subbands.l1_detail[ch][..L1_DETAIL_LEN],
                3,
            );
        }
    }

    let n = p.bytes_used();
    out[..n].copy_from_slice(&p.buf[..n]);
    EntropyOutput {
        bytes: &out[..n],
        mode: WirePacketMode::Lossless,
    }
}
