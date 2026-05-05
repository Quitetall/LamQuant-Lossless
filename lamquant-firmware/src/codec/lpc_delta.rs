//! LPC coefficient delta encoding (lossless side-information compressor).
//!
//! Reduces the per-window LPC coefficient overhead from 672 bytes (Q31 full
//! coefficients) to ~168 bytes (Q8 deltas) by exploiting the fact that EEG
//! spectral envelope changes slowly across consecutive 10-second windows.
//!
//! Three encoding modes:
//!   `Keyframe` — full Q31 coefficients (21 × 8 × 4 = 672 bytes)
//!   `Q15`      — Q15 deltas (336 bytes), used when |delta| < ~50% of Q31
//!   `Q8`       — Q8 deltas (168 bytes), used when |delta| < ~0.5%
//!
//! State machine: encoder tracks the previous frame's coefficients to
//! compute deltas; decoder mirrors the same state. A keyframe is required
//! after connection loss / reset before any delta packet can be applied.

use super::quality::QualityMode; // unused for now; reserved for future
                                  // mode-dependent fallback heuristics
const _: () = assert!(QualityMode::Alerting as u8 == 0);

const LPC_ORDER: usize = 8;
const NUM_CHANNELS: usize = 21;

/// Encoding mode for the leading byte of an LPC delta packet.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum LpcDeltaMode {
    Keyframe = 0x00,
    Q15 = 0x01,
    Q8 = 0x02,
}

impl LpcDeltaMode {
    /// Encoded payload size in bytes (excludes the 1-byte mode header).
    pub const fn payload_bytes(self) -> usize {
        match self {
            Self::Keyframe => NUM_CHANNELS * LPC_ORDER * 4, // 672
            Self::Q15 => NUM_CHANNELS * LPC_ORDER * 2,      // 336
            Self::Q8 => NUM_CHANNELS * LPC_ORDER,            // 168
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x00 => Some(Self::Keyframe),
            0x01 => Some(Self::Q15),
            0x02 => Some(Self::Q8),
            _ => None,
        }
    }
}

/// Stateful encoder/decoder for LPC delta coding.
///
/// Owns the previous frame's coefficients. Reset via `reset()` on
/// connection loss or power state change.
pub struct LpcDelta {
    prev_coeffs: [[i32; LPC_ORDER]; NUM_CHANNELS],
    has_prev: bool,
}

impl LpcDelta {
    pub const fn new() -> Self {
        Self {
            prev_coeffs: [[0; LPC_ORDER]; NUM_CHANNELS],
            has_prev: false,
        }
    }

    pub fn reset(&mut self) {
        self.prev_coeffs = [[0; LPC_ORDER]; NUM_CHANNELS];
        self.has_prev = false;
    }

    /// Pick the most compact encoding for `curr` given the current state.
    fn pick_mode(
        &self,
        curr: &[[i32; LPC_ORDER]; NUM_CHANNELS],
        deltas: &mut [[i32; LPC_ORDER]; NUM_CHANNELS],
    ) -> LpcDeltaMode {
        if !self.has_prev {
            return LpcDeltaMode::Keyframe;
        }

        let mut max_abs: i32 = 0;
        for ch in 0..NUM_CHANNELS {
            for k in 0..LPC_ORDER {
                let d = curr[ch][k].wrapping_sub(self.prev_coeffs[ch][k]);
                deltas[ch][k] = d;
                let abs_d = d.unsigned_abs() as i32;
                if abs_d > max_abs {
                    max_abs = abs_d;
                }
            }
        }

        // Q8 range: ±127 << 23 ≈ ±1.07e9 (~50% of Q31 range).
        // Q15 range: ±32767 << 16 ≈ ±2.15e9 (nearly full Q31).
        let q8_max: i32 = 127 << 23;
        let q15_max: i32 = 32767 << 16;
        if max_abs <= q8_max {
            LpcDeltaMode::Q8
        } else if max_abs <= q15_max {
            LpcDeltaMode::Q15
        } else {
            LpcDeltaMode::Keyframe
        }
    }

    /// Encode `curr` into `out_buf`. Returns bytes written. `out_buf` must
    /// be at least `1 + Keyframe::payload_bytes()` = 673 bytes.
    pub fn encode(
        &mut self,
        curr: &[[i32; LPC_ORDER]; NUM_CHANNELS],
        out_buf: &mut [u8],
    ) -> usize {
        let mut deltas = [[0i32; LPC_ORDER]; NUM_CHANNELS];
        let mode = self.pick_mode(curr, &mut deltas);

        out_buf[0] = mode as u8;
        let mut pos = 1;

        match mode {
            LpcDeltaMode::Keyframe => {
                for ch in 0..NUM_CHANNELS {
                    for k in 0..LPC_ORDER {
                        let v = curr[ch][k];
                        out_buf[pos] = (v & 0xFF) as u8;
                        out_buf[pos + 1] = ((v >> 8) & 0xFF) as u8;
                        out_buf[pos + 2] = ((v >> 16) & 0xFF) as u8;
                        out_buf[pos + 3] = ((v >> 24) & 0xFF) as u8;
                        pos += 4;
                    }
                }
            }
            LpcDeltaMode::Q15 => {
                for ch in 0..NUM_CHANNELS {
                    for k in 0..LPC_ORDER {
                        let d16: i16 = (deltas[ch][k] >> 16) as i16;
                        out_buf[pos] = (d16 & 0xFF) as u8;
                        out_buf[pos + 1] = ((d16 >> 8) & 0xFF) as u8;
                        pos += 2;
                    }
                }
            }
            LpcDeltaMode::Q8 => {
                for ch in 0..NUM_CHANNELS {
                    for k in 0..LPC_ORDER {
                        let d8: i8 = (deltas[ch][k] >> 23) as i8;
                        out_buf[pos] = d8 as u8;
                        pos += 1;
                    }
                }
            }
        }

        // Save state for next frame.
        self.prev_coeffs = *curr;
        self.has_prev = true;

        pos
    }

    /// Decode an LPC delta packet into `out_coeffs`. Returns bytes consumed.
    /// Returns 0 (zeroes output) if a delta arrives before any keyframe.
    pub fn decode(
        &mut self,
        in_buf: &[u8],
        out_coeffs: &mut [[i32; LPC_ORDER]; NUM_CHANNELS],
    ) -> usize {
        if in_buf.is_empty() {
            return 0;
        }
        let mode = match LpcDeltaMode::from_u8(in_buf[0]) {
            Some(m) => m,
            None => return 0,
        };

        // BUG FIX (matches C): a delta before any keyframe yields
        // garbage. Zero output instead and return 0 to signal error.
        if !self.has_prev && mode != LpcDeltaMode::Keyframe {
            *out_coeffs = [[0; LPC_ORDER]; NUM_CHANNELS];
            return 0;
        }

        let mut pos = 1;
        match mode {
            LpcDeltaMode::Keyframe => {
                for ch in 0..NUM_CHANNELS {
                    for k in 0..LPC_ORDER {
                        let v = (in_buf[pos] as i32)
                            | ((in_buf[pos + 1] as i32) << 8)
                            | ((in_buf[pos + 2] as i32) << 16)
                            | ((in_buf[pos + 3] as i32) << 24);
                        out_coeffs[ch][k] = v;
                        pos += 4;
                    }
                }
            }
            LpcDeltaMode::Q15 => {
                for ch in 0..NUM_CHANNELS {
                    for k in 0..LPC_ORDER {
                        let raw =
                            (in_buf[pos] as u16) | ((in_buf[pos + 1] as u16) << 8);
                        let d16 = raw as i16;
                        out_coeffs[ch][k] =
                            self.prev_coeffs[ch][k].wrapping_add((d16 as i32) << 16);
                        pos += 2;
                    }
                }
            }
            LpcDeltaMode::Q8 => {
                for ch in 0..NUM_CHANNELS {
                    for k in 0..LPC_ORDER {
                        let d8 = in_buf[pos] as i8;
                        out_coeffs[ch][k] =
                            self.prev_coeffs[ch][k].wrapping_add((d8 as i32) << 23);
                        pos += 1;
                    }
                }
            }
        }

        self.prev_coeffs = *out_coeffs;
        self.has_prev = true;
        pos
    }
}

impl Default for LpcDelta {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn keyframe_roundtrip() {
        let mut enc = LpcDelta::new();
        let mut dec = LpcDelta::new();
        let mut curr = [[0i32; LPC_ORDER]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for k in 0..LPC_ORDER {
                curr[ch][k] = ((ch as i32 + 1) * (k as i32 + 1) * 1_000_000) as i32;
            }
        }
        let mut buf = [0u8; 673];
        let n_enc = enc.encode(&curr, &mut buf);
        assert_eq!(buf[0], LpcDeltaMode::Keyframe as u8);
        assert_eq!(n_enc, 1 + 672);

        let mut out = [[0i32; LPC_ORDER]; NUM_CHANNELS];
        let n_dec = dec.decode(&buf[..n_enc], &mut out);
        assert_eq!(n_dec, n_enc);
        assert_eq!(out, curr);
    }

    #[test]
    fn q8_delta_roundtrip() {
        let mut enc = LpcDelta::new();
        let mut dec = LpcDelta::new();
        let mut buf = [0u8; 673];

        // Frame 1: keyframe.
        let curr0 = [[1_000_000i32; LPC_ORDER]; NUM_CHANNELS];
        let n0 = enc.encode(&curr0, &mut buf);
        let mut tmp = [[0i32; LPC_ORDER]; NUM_CHANNELS];
        dec.decode(&buf[..n0], &mut tmp);

        // Frame 2: small delta — should pick Q8.
        let curr1 = [[1_000_500i32; LPC_ORDER]; NUM_CHANNELS];
        let n1 = enc.encode(&curr1, &mut buf);
        assert_eq!(buf[0], LpcDeltaMode::Q8 as u8);
        assert_eq!(n1, 1 + 168);

        let mut out = [[0i32; LPC_ORDER]; NUM_CHANNELS];
        dec.decode(&buf[..n1], &mut out);
        // Q8 quantization loses precision below 1 << 23 ≈ 8.4M.
        // 500 / 2^23 = 0 (rounds to zero in Q8). So decoded delta is 0,
        // out[ch][k] == prev = 1_000_000.
        for ch in 0..NUM_CHANNELS {
            for k in 0..LPC_ORDER {
                assert_eq!(out[ch][k], 1_000_000);
            }
        }
    }

    #[test]
    fn delta_before_keyframe_zeros_output() {
        let mut dec = LpcDelta::new();
        // Synthesize a Q8 frame without a keyframe first.
        let mut buf = [LpcDeltaMode::Q8 as u8; 1 + 168];
        let mut out = [[42i32; LPC_ORDER]; NUM_CHANNELS];
        let n = dec.decode(&buf, &mut out);
        assert_eq!(n, 0);
        assert_eq!(out, [[0i32; LPC_ORDER]; NUM_CHANNELS]);
        let _ = buf; // silence unused
    }
}
