//! Track 2 P2 (ADR 0051): deadzone scalar quantization of wavelet subband
//! coefficients + per-subband synthesis-gain weighting for rate-controlled
//! lossy coding.
//!
//! The target-BPS encoder quantizes each lifting subband with its own step
//! `q_s`, LPC-codes the quantized indices, and entropy-codes the residual. To
//! spend bits where they matter, steps are weighted by each subband's
//! synthesis L2 gain `G_s` (how much an error injected in subband `s`
//! amplifies through the inverse transform into the sample domain): high-gain
//! subbands get finer steps. A single global `scale` then trades total rate
//! against distortion (binary-searched to the BPS ceiling in `lml.rs`).

use alloc::vec;
use alloc::vec::Vec;

use crate::lifting;

/// Nearest-multiple index of `q` (ties away from zero), integer-only.
/// `q` must be >= 1. `quant_index(v, 1) == v` (lossless passthrough).
#[inline]
pub fn quant_index(v: i64, q: i64) -> i64 {
    debug_assert!(q >= 1);
    let half = q / 2;
    if v >= 0 {
        (v + half) / q
    } else {
        -((-v + half) / q)
    }
}

/// Quantize a subband to indices with step `q`.
pub fn quantize(sub: &[i64], q: i64) -> Vec<i64> {
    sub.iter().map(|&v| quant_index(v, q)).collect()
}

/// Dequantize indices back to (lossy) coefficients: `idx * q`.
pub fn dequantize(idx: &[i64], q: i64) -> Vec<i64> {
    idx.iter().map(|&v| v * q).collect()
}

/// Inverse lifting for a given `n_levels` from its ordered subbands
/// (`[approx, detail_top, ..., detail_1]`). Mirror of the forward split used
/// by the encoder; `n_levels == 0` is the identity (one full-length subband).
pub fn inverse_for_levels(n_levels: u8, subs: &[Vec<i64>]) -> Vec<i64> {
    match n_levels {
        3 => lifting::inverse_3level(&subs[0], &subs[1], &subs[2], &subs[3]),
        2 => {
            let l2a = lifting::inverse(&subs[0], &subs[1]);
            lifting::inverse(&l2a, &subs[2])
        }
        1 => lifting::inverse(&subs[0], &subs[1]),
        _ => subs[0].clone(),
    }
}

/// Per-subband synthesis L2 gain `G_s` for the given `n_levels` and subband
/// lengths. Estimated by injecting a unit impulse at the centre of each
/// subband and running the inverse transform: `G_s = ||inverse(impulse)||² `.
/// A large impulse amplitude is used so the integer lifting rounding (`>>1`,
/// `>>2`) does not annihilate the response. Returns one gain per subband.
// Host-only: f64 R-D weighting, used only by the encode-side rate search.
// The firmware (no_std) build never runs target-BPS encoding — its decode
// path reads the per-subband quantizer steps from the wire and dequantizes
// with integer math — so excluding this keeps the firmware binary
// full-integer (no f64 / libm) and byte-identical-behaviour to pre-H.BWC.
#[cfg(feature = "archive")]
pub fn synthesis_gains(n_levels: u8, sub_lens: &[usize]) -> Vec<f64> {
    const IMP: i64 = 4096;
    let n_sub = sub_lens.len();
    let mut gains = vec![1.0f64; n_sub];
    for s in 0..n_sub {
        if sub_lens[s] == 0 {
            continue;
        }
        let mut subs: Vec<Vec<i64>> = sub_lens.iter().map(|&l| vec![0i64; l]).collect();
        subs[s][sub_lens[s] / 2] = IMP;
        let recon = inverse_for_levels(n_levels, &subs);
        let energy: f64 = recon.iter().map(|&v| (v as f64) * (v as f64)).sum();
        gains[s] = (energy / (IMP as f64 * IMP as f64)).max(1e-9);
    }
    gains
}

/// Per-subband quantizer steps for a global `scale`, weighted so high-gain
/// subbands quantize finer. `q_s = max(1, round(scale * sqrt(maxG / G_s)))`.
/// `scale == 0` ⇒ all steps 1 (near-lossless / max rate). Monotone: larger
/// `scale` ⇒ coarser steps ⇒ lower rate.
#[cfg(feature = "archive")]
pub fn steps_for_scale(scale: f64, gains: &[f64]) -> Vec<i64> {
    let max_g = gains.iter().cloned().fold(1e-9f64, f64::max);
    gains
        .iter()
        .map(|&g| {
            // libm::sqrt + manual round keep this no_std-clean (firmware build):
            // f64::sqrt/round require std. `scale * k >= 0` always (scale from
            // the rate search is >= 0, k >= 1), so `+ 0.5` truncating == round.
            let k = libm::sqrt(max_g / g);
            let q = (scale * k + 0.5) as i64;
            q.max(1)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_index_q1_is_identity() {
        for v in [-5000i64, -1, 0, 1, 4097] {
            assert_eq!(quant_index(v, 1), v);
        }
    }

    #[test]
    fn quant_dequant_bounded_by_half_step() {
        for q in [3i64, 7, 16, 101] {
            for v in [-9001i64, -100, 0, 37, 8002] {
                let idx = quant_index(v, q);
                let rec = idx * q;
                assert!((v - rec).abs() <= q / 2, "v={} q={} err>{}", v, q, q / 2);
            }
        }
    }

    #[cfg(feature = "archive")]
    #[test]
    fn synthesis_gains_positive_and_per_subband() {
        // 3-level: 4 subbands. All gains finite + positive.
        let g = synthesis_gains(3, &[320, 320, 640, 1280]);
        assert_eq!(g.len(), 4);
        assert!(g.iter().all(|&x| x > 0.0 && x.is_finite()));
    }

    #[cfg(feature = "archive")]
    #[test]
    fn steps_monotone_in_scale() {
        let g = synthesis_gains(3, &[320, 320, 640, 1280]);
        let q0 = steps_for_scale(0.0, &g);
        let q_big = steps_for_scale(50.0, &g);
        assert!(q0.iter().all(|&q| q == 1));
        assert!(
            q_big.iter().zip(&q0).all(|(b, s)| b >= s),
            "larger scale must not give finer steps"
        );
    }
}
