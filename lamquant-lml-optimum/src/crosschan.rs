//! Closed-loop **cross-channel linear prediction** (ADR 0054 Phase 3, lever 4).
//!
//! The lever-4 probe showed the dominant remaining gap to HHI is inter-channel
//! redundancy — ~1.5 bits/sample/channel realizable on CHB-MIT's bipolar montage
//! (the channels are electrode *differences* in telescoping chains ⇒ strongly
//! linearly dependent). HHI removes it with cross-channel LMS; this is the LMO
//! deterministic analogue: a **full causal least-squares predictor** — each
//! channel `i` predicted from a linear combination of the already-coded channels
//! `0..i`.
//!
//! **Closed-loop** is the load-bearing property: the predictor operates on the
//! *reconstructed* prior channels (the integer `x̂ⱼ` the decoder also has), so
//! lossy quantization of earlier channels causes no drift — encoder and decoder
//! form the bit-identical integer prediction `round(Σⱼ aᵢⱼ x̂ⱼ)`.
//!
//! Coefficients `aᵢⱼ` are fit once per window from the channel covariance
//! (sequential LS via a Cholesky solve of each leading block) and shipped in the
//! header; deterministic `f64`, so no neural model ⇒ LMO stays out of PCCP.
//!
//! `fit_predictor` is host-only (`encode`); [`predict_channel`] +
//! (de)serialization are `no_std` (the decoder needs them).
//!
//! ## STATUS: research module, NOT wired into the LMO container (negative result)
//!
//! The end-to-end A/B (`examples/crosschan_ab.rs`) found that the naive
//! *predict-then-wavelet* cascade is **worse** than the plain 9/7+arith baseline
//! at every operating point and every ridge (+11% PRD at best, @ 2.0 BPS): the
//! predictor *does* cut residual energy to ~40% (the inter-channel redundancy is
//! real, matching the lever-4 probe's ~1.5 bit/sample), but the decorrelated
//! residual is **harder to code per bit** — it flattens the intra-channel
//! spectrum the 9/7 wavelet exploits, and the per-channel pass loses the
//! baseline's joint PCRD allocation. Capturing the cross-channel gain needs an
//! *integrated* RDO design (HHI runs cross-channel LMS inside one DCT+TCQ+CABAC
//! loop), not this cascade. This module is retained for the headroom analysis
//! and as a basis for that future integrated approach; it is intentionally not
//! given an LMO `transform_id`.

use alloc::vec::Vec;

use crate::wavelet97::round_i64;

/// Fixed-point scale for shipped coefficients (Q16 — EEG montage coefficients are
/// `O(1)`, this resolves them to ~1.5e-5). Stored as `i32`.
const COEFF_Q: f64 = 65536.0;

/// Above this channel count we skip cross-channel fitting (the per-window
/// `O(n³)` solve + the `O(n²)` coefficient table stop being worth it; HHI groups
/// channels too). Comfortably covers clinical montages (≤256 → still fine, but
/// the solve cost grows; 64 keeps fit time negligible).
pub const MAX_FIT_CHANNELS: usize = 64;

/// A causal lower-triangular predictor: `coeffs[i]` has length `i` (the weights
/// on reconstructed channels `0..i`), quantized to Q16 `i32`.
#[derive(Debug, Clone, Default)]
pub struct CrossChanPredictor {
    pub coeffs: Vec<Vec<i32>>,
}

impl CrossChanPredictor {
    pub fn n_ch(&self) -> usize {
        self.coeffs.len()
    }

    /// The integer prediction of channel `i` from the reconstructed prior
    /// channels — the bit-identical value encoder and decoder both compute.
    /// `recon_prior[j]` is `x̂ⱼ` for `j < i` (length `t`).
    pub fn predict_channel(&self, i: usize, recon_prior: &[Vec<i64>], t: usize) -> Vec<i64> {
        let row = &self.coeffs[i];
        let mut out = Vec::with_capacity(t);
        #[allow(clippy::needless_range_loop)] // k indexes each prior channel at one time sample
        for k in 0..t {
            let mut acc = 0.0f64;
            for (j, &q) in row.iter().enumerate() {
                acc += (q as f64 / COEFF_Q) * recon_prior[j][k] as f64;
            }
            out.push(round_i64(acc));
        }
        out
    }

    /// Serialize to `[n_ch:u16][ per i: row i32 LE × i ]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.coeffs.len() as u16).to_le_bytes());
        for row in &self.coeffs {
            for &q in row {
                out.extend_from_slice(&q.to_le_bytes());
            }
        }
        out
    }

    /// Deserialize from `bytes` at `offset`; returns `(predictor, consumed)`.
    /// The row lengths are implicit (`row i` has length `i`), so no per-row count
    /// is stored.
    pub fn from_bytes(bytes: &[u8], offset: usize) -> Option<(Self, usize)> {
        if offset + 2 > bytes.len() {
            return None;
        }
        let n_ch = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as usize;
        let mut pos = offset + 2;
        let mut coeffs = Vec::with_capacity(n_ch);
        for i in 0..n_ch {
            let mut row = Vec::with_capacity(i);
            for _ in 0..i {
                if pos + 4 > bytes.len() {
                    return None;
                }
                row.push(i32::from_le_bytes([
                    bytes[pos],
                    bytes[pos + 1],
                    bytes[pos + 2],
                    bytes[pos + 3],
                ]));
                pos += 4;
            }
            coeffs.push(row);
        }
        Some((Self { coeffs }, pos - offset))
    }
}

/// Fit the causal cross-channel predictor from the window's channel covariance.
/// `coeffs[0]` is empty (channel 0 is coded as-is); `coeffs[i]` is the LS
/// solution of `Σ_{<i} a = Cov(i, <i)`. Returns an all-empty predictor (no
/// cross-channel prediction) when `n_ch > MAX_FIT_CHANNELS` or the signal is
/// degenerate.
#[cfg(feature = "encode")]
pub fn fit_predictor(signal: &[Vec<i64>]) -> CrossChanPredictor {
    fit_predictor_ridged(signal, 1e-6)
}

/// As [`fit_predictor`], with an explicit diagonal ridge as a fraction of the
/// mean channel variance. Under lossy coding the prior channels carry
/// quantization noise, so the clean-signal LS predictor over-amplifies it on a
/// near-singular montage; a ridge ≈ the prior noise variance is the Wiener
/// correction (regularized / shrunk coefficients). Larger ridge ⇒ smaller
/// coefficients ⇒ less noise amplification but weaker decorrelation.
#[cfg(feature = "encode")]
pub fn fit_predictor_ridged(signal: &[Vec<i64>], ridge_frac: f64) -> CrossChanPredictor {
    let n = signal.len();
    let empty = CrossChanPredictor {
        coeffs: (0..n).map(|_| Vec::new()).collect(),
    };
    if n == 0 || n > MAX_FIT_CHANNELS {
        return empty;
    }
    let t = signal[0].len();
    if t == 0 {
        return empty;
    }

    // Mean-removed channel covariance (population).
    let mus: Vec<f64> = signal
        .iter()
        .map(|c| c.iter().map(|&v| v as f64).sum::<f64>() / t as f64)
        .collect();
    let mut cov = alloc::vec![alloc::vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for (&left, &right) in signal[i].iter().zip(&signal[j]).take(t) {
                s += (left as f64 - mus[i]) * (right as f64 - mus[j]);
            }
            let c = s / t as f64;
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }

    let mut coeffs: Vec<Vec<i32>> = Vec::with_capacity(n);
    coeffs.push(Vec::new()); // channel 0: no predictor
    for i in 1..n {
        // Leading SPD block Σ_{<i} (ridge for the near-singular bipolar montage).
        let mut a = alloc::vec![alloc::vec![0.0f64; i]; i];
        let ridge = ridge_frac * (0..i).map(|d| cov[d][d]).sum::<f64>() / i as f64;
        for r in 0..i {
            for c in 0..i {
                a[r][c] = cov[r][c] + if r == c { ridge } else { 0.0 };
            }
        }
        let b: Vec<f64> = (0..i).map(|j| cov[i][j]).collect();
        let row_f = solve_spd_cholesky(&a, &b).unwrap_or_else(|| alloc::vec![0.0; i]);
        // Quantize to Q16 i32 (clamped — montage coeffs are O(1)).
        let row: Vec<i32> = row_f
            .iter()
            .map(|&v| {
                (v * COEFF_Q)
                    .round()
                    .clamp(i32::MIN as f64, i32::MAX as f64) as i32
            })
            .collect();
        coeffs.push(row);
    }
    CrossChanPredictor { coeffs }
}

/// Solve `A x = b` for symmetric positive-definite `A` (Cholesky). `None` if not
/// positive-definite.
#[cfg(feature = "encode")]
pub(crate) fn solve_spd_cholesky(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    let mut l = alloc::vec![alloc::vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for (&left, &right) in l[i][..j].iter().zip(&l[j][..j]) {
                s -= left * right;
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // Forward solve L y = b.
    let mut y = alloc::vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    // Back solve Lᵀ x = y.
    let mut x = alloc::vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;

    #[test]
    fn predictor_serde_roundtrip() {
        let p = CrossChanPredictor {
            coeffs: alloc::vec![Vec::new(), alloc::vec![123], alloc::vec![-7, 65536]],
        };
        let bytes = p.to_bytes();
        let (q, consumed) = CrossChanPredictor::from_bytes(&bytes, 0).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(q.coeffs, p.coeffs);
    }

    #[test]
    fn predicts_exact_linear_combo() {
        // ch2 = 2*ch0 - ch1 exactly ⇒ fitted predictor should reconstruct it and
        // the integer prediction should be near-exact.
        let t = 500;
        let ch0: Vec<i64> = (0..t).map(|i| ((i * 7) % 50) as i64 - 25).collect();
        let ch1: Vec<i64> = (0..t).map(|i| ((i * 13) % 40) as i64 - 20).collect();
        let ch2: Vec<i64> = (0..t).map(|i| 2 * ch0[i] - ch1[i]).collect();
        let sig = alloc::vec![ch0.clone(), ch1.clone(), ch2.clone()];
        let p = fit_predictor(&sig);
        let pred2 = p.predict_channel(2, &[ch0, ch1], t);
        let max_err = pred2
            .iter()
            .zip(&ch2)
            .map(|(a, b)| (a - b).abs())
            .max()
            .unwrap();
        assert!(
            max_err <= 1,
            "linear-combo channel should predict within 1 LSB, got {max_err}"
        );
    }

    #[test]
    fn independent_channels_small_coeffs() {
        // Uncorrelated channels ⇒ predictor coefficients near zero.
        let t = 2000;
        let ch0: Vec<i64> = (0..t).map(|i| ((i * 31) % 97) as i64 - 48).collect();
        let ch1: Vec<i64> = (0..t)
            .map(|i| (((i * 17 + 5) % 89) as i64 - 44) * 3)
            .collect();
        let sig = alloc::vec![ch0, ch1];
        let p = fit_predictor(&sig);
        // coeff is Q16; "near zero" = well under 0.25·Q.
        assert!(p.coeffs[1][0].unsigned_abs() < (0.25 * COEFF_Q) as u32);
    }

    #[test]
    fn over_cap_returns_empty() {
        let sig: Vec<Vec<i64>> = (0..(MAX_FIT_CHANNELS + 1))
            .map(|_| alloc::vec![1i64; 8])
            .collect();
        let p = fit_predictor(&sig);
        assert!(p.coeffs.iter().all(|r| r.is_empty()));
    }
}
