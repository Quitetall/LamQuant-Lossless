//! ADR 0069 S7b — the LMQ training normalization pipeline, hoisted from Python
//! (`lamquant_codec/training/lma_dataset.py::decode_lma_signal`) into Rust so
//! LML + LMQ draw from ONE definition. These are **Lossy** transforms (the LML
//! lossless backend must never run them; the pass framework enforces that) and
//! are **host-only** (`#[cfg(feature = "archive")]`) — `sosfiltfilt` is a
//! non-causal forward-backward filter (a T2/basestation pass), so there is no
//! MCU/no_std variant here (a causal streaming HP for T0/T1 is a deferred
//! follow-up, flagged in the pass metadata).
//!
//! Calibration (digital→µV) is upstream and already bit-exact in Rust
//! (`container_read_phys_f32`); this module is everything AFTER it: channel
//! select → resample→250 Hz → 0.5 Hz zero-phase highpass → Q31.
//!
//! Parity: bit-exactness vs scipy/LAPACK is impossible for the FP stages, so
//! `tests/normalize_parity.rs` asserts exact equality on the final int32 Q31
//! output where deterministic and a documented ±LSB tolerance on the
//! FP-divergent stages. See that gate before switching `lma_dataset.py` over.

/// Target sample rate the LMQ path normalizes to (Hz).
pub const TARGET_SR: f64 = 250.0;
/// Q31 headroom (`convert_lml` default; `lma_dataset.py::Q31_HEADROOM`).
pub const Q31_HEADROOM: f64 = 0.72;
/// Highpass cutoff (Hz) — `lma_dataset.py::HIGHPASS_HZ`.
pub const HIGHPASS_HZ: f64 = 0.5;

/// `scipy.signal.butter(2, 0.5, btype='high', fs=250.0, output='sos')` — a
/// single second-order section `[b0, b1, b2, a0, a1, a2]`, dumped verbatim from
/// scipy 1.18.0. Fixed operating point (250 Hz / 0.5 Hz HP = [`HIGHPASS_HZ`] /
/// [`TARGET_SR`]); regenerate (and re-dump the golden) if either changes.
/// `a0 == 1.0`. These are NOT derived from the symbolic consts at runtime (no
/// Rust `butter` impl); the `sosfiltfilt_matches_scipy_oracle_on_ramp20` test
/// is the guard — wrong coeffs make its output diverge from the scipy oracle.
const HP_B: [f64; 3] = [0.991153595101663, -1.982307190203326, 0.991153595101663];
const HP_A: [f64; 3] = [1.0, -1.9822289297925284, 0.9823854506141251];
/// `scipy.signal.sosfilt_zi(sos)` for the section above — the steady-state
/// initial condition filtfilt scales by the first (odd-extended) sample.
const HP_ZI: [f64; 2] = [-0.991153595101663, 0.991153595101663];
/// scipy `sosfiltfilt` default edge/pad length for a single section:
/// `3 * (2 * n_sections + 1)` = 9.
const HP_PADLEN: usize = 9;

/// Odd extension of `x` by `n` samples each end — `scipy.signal._arraytools.odd_ext`:
/// left = `2*x[0] - x[n..1]`, right = `2*x[-1] - x[-2..-(n+1)]`. Requires `x.len() > n`.
/// Private; the sole caller (`sosfiltfilt_hp`) enforces the precondition — the
/// `debug_assert` makes the contract explicit if that ever changes.
fn odd_ext(x: &[f64], n: usize) -> Vec<f64> {
    debug_assert!(x.len() > n, "odd_ext requires x.len() ({}) > n ({n})", x.len());
    let len = x.len();
    let mut out = Vec::with_capacity(len + 2 * n);
    // left: 2*x[0] - x[n], 2*x[0] - x[n-1], …, 2*x[0] - x[1]
    for j in (1..=n).rev() {
        out.push(2.0 * x[0] - x[j]);
    }
    out.extend_from_slice(x);
    // right: 2*x[-1] - x[len-2], 2*x[-1] - x[len-3], …, 2*x[-1] - x[len-1-n]
    for k in 0..n {
        out.push(2.0 * x[len - 1] - x[len - 2 - k]);
    }
    out
}

/// One second-order section, Direct-Form-II-Transposed (scipy `sosfilt`'s
/// recurrence), with initial state `zi`. Returns the filtered signal.
fn sosfilt_one(b: [f64; 3], a: [f64; 3], x: &[f64], zi: [f64; 2]) -> Vec<f64> {
    let (b0, b1, b2) = (b[0], b[1], b[2]);
    let (a1, a2) = (a[1], a[2]); // a0 == 1.0
    let mut z0 = zi[0];
    let mut z1 = zi[1];
    let mut y = Vec::with_capacity(x.len());
    for &xn in x {
        let yn = b0 * xn + z0;
        z0 = b1 * xn - a1 * yn + z1;
        z1 = b2 * xn - a2 * yn;
        y.push(yn);
    }
    y
}

/// Zero-phase 0.5 Hz Butterworth highpass — the Rust port of
/// `sosfiltfilt(butter(2, 0.5, fs=250, 'high'), x)`. Forward-backward with odd
/// padding + `sosfilt_zi`-scaled initial conditions, matching scipy's algorithm
/// (validated to <1e-9 against a scipy oracle in the unit tests). Input/output
/// f64; a channel shorter than `HP_PADLEN` is returned unfiltered (scipy would
/// raise — the caller guarantees window length ≫ 9).
pub fn sosfiltfilt_hp(x: &[f64]) -> Vec<f64> {
    let n = HP_PADLEN;
    if x.len() <= n {
        return x.to_vec();
    }
    let ext = odd_ext(x, n);
    let x0 = ext[0];
    let fwd = sosfilt_one(HP_B, HP_A, &ext, [HP_ZI[0] * x0, HP_ZI[1] * x0]);

    // reverse, filter again, reverse back
    let mut rev: Vec<f64> = fwd.into_iter().rev().collect();
    let y0 = rev[0];
    let bwd = sosfilt_one(HP_B, HP_A, &rev, [HP_ZI[0] * y0, HP_ZI[1] * y0]);
    rev = bwd.into_iter().rev().collect();

    // trim the odd-extension padding
    rev[n..rev.len() - n].to_vec()
}

/// Q31 quantization — `lma_dataset.py:429-430`:
/// `gain = 0.72 / max|data|; (data * gain * 2147483647.0).astype(int32)`.
/// The two multiplies are left-to-right f64 (matching numpy's evaluation
/// order), and `as i32` truncates toward zero — the values are in `[-0.72,
/// 0.72] * (2^31 - 1)`, safely inside the i32 range, so no saturation. Returns
/// `None` for an all-flat signal (`max|data| < 1e-12`), matching the Python.
pub fn q31_normalize(data: &[Vec<f64>]) -> Option<Vec<Vec<i32>>> {
    let max_abs = data
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    if max_abs < 1e-12 {
        return None;
    }
    let gain = Q31_HEADROOM / max_abs;
    Some(
        data.iter()
            .map(|row| row.iter().map(|&v| (v * gain * 2_147_483_647.0) as i32).collect())
            .collect(),
    )
}

/// EEG normalization for a signal ALREADY at 250 Hz with channels already in
/// the 21-target order (channel-select + resample are identity here — those
/// stages land in later S7b increments). Per-channel zero-phase HP → Q31.
/// Returns `None` on an all-flat signal.
pub fn normalize_eeg_250hz(data: &[Vec<f64>]) -> Option<Vec<Vec<i32>>> {
    let filtered: Vec<Vec<f64>> = data.iter().map(|ch| sosfiltfilt_hp(ch)).collect();
    q31_normalize(&filtered)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validate `sosfiltfilt_hp` against a scipy 1.18.0 oracle:
    /// `sosfiltfilt(butter(2,0.5,fs=250,'high'), arange(20.0))`.
    #[test]
    fn sosfiltfilt_matches_scipy_oracle_on_ramp20() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y = sosfiltfilt_hp(&x);
        assert_eq!(y.len(), 20);
        // scipy 1.18.0 reference (dumped): first 5 and last 3 samples.
        let head = [
            -14.1847655339, -13.6226265744, -13.0645789879, -12.5106270293, -11.9607746278,
        ];
        let tail = [-5.1870319854, -4.6948357005, -4.2067679559];
        for (i, &e) in head.iter().enumerate() {
            assert!(
                (y[i] - e).abs() < 1e-9,
                "head[{i}]: got {}, want {e} (Δ={:.3e})",
                y[i],
                (y[i] - e).abs()
            );
        }
        for (k, &e) in tail.iter().enumerate() {
            let i = 20 - 3 + k;
            assert!(
                (y[i] - e).abs() < 1e-9,
                "tail[{k}]: got {}, want {e} (Δ={:.3e})",
                y[i],
                (y[i] - e).abs()
            );
        }
    }

    /// A constant signal → zero-phase HP removes the DC → ~0 → q31 returns None
    /// (all-flat guard), matching `lma_dataset.py`'s `max_abs < 1e-12` branch.
    #[test]
    fn flat_signal_q31_returns_none() {
        let flat = vec![vec![5.0f64; 64]; 3];
        let filtered: Vec<Vec<f64>> = flat.iter().map(|c| sosfiltfilt_hp(c)).collect();
        assert!(q31_normalize(&filtered).is_none());
    }

    /// Q31 truncates toward zero (not round), left-to-right f64 multiply.
    #[test]
    fn q31_truncates_toward_zero() {
        // max_abs = 1.0 → gain = 0.72; 0.5 * 0.72 * 2147483647 = 773094113.0-ish
        let data = vec![vec![0.5f64, -0.5, 1.0, -1.0]];
        let q = q31_normalize(&data).unwrap();
        let expect = |v: f64| -> i32 { (v * (0.72 / 1.0) * 2_147_483_647.0) as i32 };
        assert_eq!(q[0][0], expect(0.5));
        assert_eq!(q[0][1], expect(-0.5));
        assert_eq!(q[0][2], expect(1.0));
        assert_eq!(q[0][3], expect(-1.0));
    }
}
