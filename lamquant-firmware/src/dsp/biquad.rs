//! Q30 biquad highpass prefilter (Stage 1 of pipeline).
//!
//! Single-stage HP per channel for DC removal (0.5 Hz default). Pure
//! integer arithmetic — no float, no libm, no soft-float.
//!
//! In Gen 7.7 the LP filtering is delegated to the 3-level lifting DWT
//! (L3 approx is bandlimited 0-31.25 Hz) and 60 Hz rejection to detail-
//! coefficient thresholding (60 Hz lives in L2 detail at 31-62 Hz).
//!
//! Coefficients are Q30 (range [-2.0, +2.0)) because the HP biquad's
//! `b1 ≈ -1.98` overflows Q31's [-1.0, +1.0) range. Generated offline by
//! Python (Butterworth order 2 via bilinear transform) and pasted below.

pub const NUM_CHANNELS: usize = 21;
pub const WINDOW_SAMPLES: usize = 2500;

/// Highpass cutoff selection (host commands `F` over serial choose this).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum HpFilter {
    Off = 0,
    Hz0_1 = 1,
    /// Production default. Matches Python golden vectors bit-for-bit.
    Hz0_5 = 2,
    Hz1_0 = 3,
}

impl HpFilter {
    pub const NUM_OPTIONS: usize = 4;
    pub const DEFAULT: Self = Self::Hz0_5;

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Hz0_1 => "0.1 Hz",
            Self::Hz0_5 => "0.5 Hz",
            Self::Hz1_0 => "1.0 Hz",
        }
    }
}

/// Q30 = 2^30. Coefficients in [-2.0, +2.0). Row order: { b0, b1, b2, a1, a2 }.
const HP_COEFFS: [[i32; 5]; HpFilter::NUM_OPTIONS] = [
    // HP_OFF — passthrough (b0 = 1.0)
    [1_073_741_824, 0, 0, 0, 0],
    // 0.1 Hz Butterworth order-2, fs=250
    [
        1_071_835_315,
        -2_143_670_630,
        1_071_835_315,
        -2_143_667_245,
        1_069_932_191,
    ],
    // 0.5 Hz Butterworth order-2, fs=250 — production default
    [
        1_064_243_069,
        -2_128_486_138,
        1_064_243_069,
        -2_128_402_106,
        1_054_828_345,
    ],
    // 1.0 Hz Butterworth order-2, fs=250
    [
        1_054_828_333,
        -2_109_656_665,
        1_054_828_333,
        -2_109_323_487,
        1_036_248_020,
    ],
];

/// Per-channel biquad state (Direct Form 1).
#[derive(Default, Copy, Clone)]
struct BiquadState {
    b0: i32,
    b1: i32,
    b2: i32,
    a1: i32,
    a2: i32,
    x1: i32,
    x2: i32,
    y1: i32,
    y2: i32,
}

impl BiquadState {
    fn init(&mut self, c: &[i32; 5]) {
        self.b0 = c[0];
        self.b1 = c[1];
        self.b2 = c[2];
        self.a1 = c[3];
        self.a2 = c[4];
        self.x1 = 0;
        self.x2 = 0;
        self.y1 = 0;
        self.y2 = 0;
    }

    /// Process one sample. Q30 multiplications, saturating Q31 accumulators.
    #[inline(always)]
    fn process(&mut self, x0: i32) -> i32 {
        let mut acc = mul_q30(self.b0, x0);
        acc = sat_add(acc, mul_q30(self.b1, self.x1));
        acc = sat_add(acc, mul_q30(self.b2, self.x2));
        acc = sat_sub(acc, mul_q30(self.a1, self.y1));
        acc = sat_sub(acc, mul_q30(self.a2, self.y2));
        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = acc;
        acc
    }
}

/// Q30 multiply: `(a * b) >> 30`, signed.
#[inline(always)]
fn mul_q30(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 30) as i32
}

/// Saturating Q31 add: clamp to [i32::MIN, i32::MAX].
#[inline(always)]
fn sat_add(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// Saturating Q31 sub: clamp to [i32::MIN, i32::MAX].
#[inline(always)]
fn sat_sub(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}

/// 21-channel HP biquad bank.
///
/// Owns the 21 per-channel filter states. Caller passes in the active
/// `HpFilter` choice and the 21×N sample buffer (mutated in-place). When
/// the cutoff changes between calls, delay lines reset cleanly.
pub struct HpFilterBank {
    states: [BiquadState; NUM_CHANNELS],
    bound: Option<HpFilter>,
}

impl HpFilterBank {
    pub const fn new() -> Self {
        Self {
            states: [BiquadState {
                b0: 0,
                b1: 0,
                b2: 0,
                a1: 0,
                a2: 0,
                x1: 0,
                x2: 0,
                y1: 0,
                y2: 0,
            }; NUM_CHANNELS],
            bound: None,
        }
    }

    /// Reset all delay lines. Called from safe-mode / mode-switch.
    pub fn reset(&mut self) {
        self.bound = None;
    }

    /// Run HP prefilter on `[NUM_CHANNELS][window_len]` Q31 samples in-place.
    ///
    /// `channel_mask` is a 21-bit field: bit `ch` set = enabled. Disabled
    /// channels are zeroed AFTER filtering so downstream stages still see
    /// a stable buffer.
    pub fn run(
        &mut self,
        signal: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
        window_len: usize,
        hp: HpFilter,
        channel_mask: u32,
    ) {
        debug_assert!(window_len <= WINDOW_SAMPLES);

        // Re-bind delay lines if the cutoff changed.
        if self.bound != Some(hp) {
            let coeffs = &HP_COEFFS[hp as usize];
            for st in &mut self.states {
                st.init(coeffs);
            }
            self.bound = Some(hp);
        }

        // Filter all channels in-place.
        for ch in 0..NUM_CHANNELS {
            let st = &mut self.states[ch];
            let row = &mut signal[ch];
            for i in 0..window_len {
                row[i] = st.process(row[i]);
            }
        }

        // Software channel gate: zero anything host has masked off.
        const FULL_MASK: u32 = (1 << NUM_CHANNELS) - 1;
        if channel_mask != FULL_MASK {
            for ch in 0..NUM_CHANNELS {
                if (channel_mask >> ch) & 1 == 0 {
                    signal[ch][..window_len].fill(0);
                }
            }
        }
    }
}

impl Default for HpFilterBank {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests (host only) ────────────────────────────────────────────

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn hp_off_is_passthrough() {
        let mut bank = HpFilterBank::new();
        let mut signal = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                signal[ch][i] = ((ch * 100 + i) as i32) << 8;
            }
        }
        let original = signal;
        bank.run(&mut signal, WINDOW_SAMPLES, HpFilter::Off, (1 << 21) - 1);
        // HP_OFF has b0=1.0, b1=b2=a1=a2=0 → output = b0 * x0 = x0.
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                assert_eq!(signal[ch][i], original[ch][i]);
            }
        }
    }

    #[test]
    fn channel_mask_zeros_disabled() {
        let mut bank = HpFilterBank::new();
        let mut signal = [[42i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        // Enable only channel 0 and 5.
        let mask = (1 << 0) | (1 << 5);
        bank.run(&mut signal, WINDOW_SAMPLES, HpFilter::Off, mask);
        for ch in 0..NUM_CHANNELS {
            let enabled = (mask >> ch) & 1 == 1;
            for i in 0..WINDOW_SAMPLES {
                if enabled {
                    assert_eq!(signal[ch][i], 42, "ch{ch} should be passthrough");
                } else {
                    assert_eq!(signal[ch][i], 0, "ch{ch} should be zeroed");
                }
            }
        }
    }

    #[test]
    fn dc_removed_by_hp_0_5() {
        let mut bank = HpFilterBank::new();
        let mut signal = [[1_000_000i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        bank.run(&mut signal, WINDOW_SAMPLES, HpFilter::Hz0_5, (1 << 21) - 1);
        // After HP filter, DC should approach zero (settling). Last 100
        // samples should be much smaller than the 1M input.
        for ch in 0..NUM_CHANNELS {
            let tail_avg: i64 = signal[ch][2400..]
                .iter()
                .map(|&v| v.unsigned_abs() as i64)
                .sum::<i64>()
                / 100;
            assert!(tail_avg < 10_000, "ch{ch} tail avg {tail_avg} too large");
        }
    }
}
