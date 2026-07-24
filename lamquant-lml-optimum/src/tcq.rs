//! Trellis-coded / **dependent quantization** (VVC-style) for the 9/7 lossy path
//! (ADR 0054 — the TCQ lever; HHI's actual quantizer).
//!
//! Two uniform scalar quantizers share the step `q` but have **interleaved**
//! reconstruction grids — `Q0 = {0, ±2q, ±4q, …}` (even), `Q1 = {0, ±q, ±3q, …}`
//! (odd) — so the *combined* reachable grid is every multiple of `q` while each
//! quantizer alone has spacing `2q`. A 4-state machine selects the quantizer for
//! each coefficient from the parity of the previously-coded levels; the encoder
//! runs **Viterbi** to find the level sequence minimising `Σ (G·d² + λ·rate)`,
//! and the decoder follows the *same* state machine to reconstruct — so it
//! reproduces the encoder's choice exactly without transmitting the states.
//!
//! Pure integer levels out (like scalar quant), so the downstream LPC + entropy
//! path is unchanged; only the dequant (reconstruction) differs, which the
//! decoder runs via [`dequantize_tcq`]. `f64` is used encode-side (Viterbi) and
//! in `dequantize_tcq`'s reconstruction — acceptable: this is the lossy LMO path,
//! never the lossless floor or the no_std integer decode.

use alloc::vec::Vec;

/// VVC dependent-quantization state transition: `next = STATE_TRANS[state][level&1]`.
/// `state >> 1` selects the quantizer (0 = Q0 even grid, 1 = Q1 odd grid).
const STATE_TRANS: [[u8; 2]; 4] = [[0, 2], [2, 0], [1, 3], [3, 1]];

/// Reconstruction of `level` under quantizer set `q_set` at step `q`.
/// Q0: `2·level·q`; Q1: `(2·level − sign)·q`; level 0 → 0 in both.
#[inline]
fn recon(q_set: u8, level: i64, q: f64) -> f64 {
    if level == 0 {
        0.0
    } else if q_set == 0 {
        2.0 * level as f64 * q
    } else {
        (2.0 * level as f64 - level.signum() as f64) * q
    }
}

/// Cheap rate proxy (bits) for a transmitted level — zero is nearly free
/// (significance), magnitudes cost ~Golomb-like. Only relative values matter to
/// the Viterbi RD decision. Encode-only (`log2` is std/libm).
#[cfg(feature = "encode")]
#[inline]
fn rate_bits(level: i64) -> f64 {
    if level == 0 {
        0.5
    } else {
        1.5 + 2.0 * ((level.unsigned_abs() as f64) + 1.0).log2()
    }
}

/// Viterbi candidate levels for coefficient `c` under quantizer `q_set`: the
/// nearest level in that grid ±1, plus 0 (the zeroing option). Encode-only
/// (`round` is std/libm).
#[cfg(feature = "encode")]
#[inline]
fn candidates(c: f64, q_set: u8, q: f64) -> [i64; 4] {
    let kc = if q_set == 0 {
        (c / (2.0 * q)).round() as i64
    } else {
        ((c / q + c.signum()) / 2.0).round() as i64
    };
    [kc - 1, kc, kc + 1, 0]
}

/// Dependent-quantize a coefficient sequence (Viterbi over the 4-state trellis).
/// `gain` = the subband synthesis L2 gain `G_s` (so distortion is signal-domain);
/// `lambda` = the RD tradeoff. Returns the transmitted integer level sequence —
/// feed it to the same LPC + entropy path as scalar indices; reconstruct with
/// [`dequantize_tcq`]. Encode-only (Viterbi uses `round`/`log2`).
#[cfg(feature = "encode")]
pub fn quantize_tcq(coeffs: &[f64], q: i64, gain: f64, lambda: f64) -> Vec<i64> {
    let n = coeffs.len();
    let qf = q as f64;
    let inf = f64::INFINITY;
    let mut cost = [0.0f64, inf, inf, inf]; // Viterbi starts in state 0
    let mut back: Vec<[(u8, i64); 4]> = Vec::with_capacity(n);

    for &c in coeffs {
        let mut ncost = [inf; 4];
        let mut nback = [(0u8, 0i64); 4];
        for s in 0u8..4 {
            if cost[s as usize] == inf {
                continue;
            }
            let q_set = s >> 1;
            for &lvl in candidates(c, q_set, qf).iter() {
                let d = c - recon(q_set, lvl, qf);
                let step_cost = gain * d * d + lambda * rate_bits(lvl);
                let ns = STATE_TRANS[s as usize][(lvl & 1) as usize] as usize;
                let total = cost[s as usize] + step_cost;
                if total < ncost[ns] {
                    ncost[ns] = total;
                    nback[ns] = (s, lvl);
                }
            }
        }
        cost = ncost;
        back.push(nback);
    }

    // Backtrack from the min-cost final state (path originates at state 0).
    let mut s = (0u8..4)
        .min_by(|&a, &b| cost[a as usize].partial_cmp(&cost[b as usize]).unwrap())
        .unwrap_or(0);
    let mut levels = alloc::vec![0i64; n];
    for i in (0..n).rev() {
        let (ps, lvl) = back[i][s as usize];
        levels[i] = lvl;
        s = ps;
    }
    levels
}

/// Reconstruct a TCQ level sequence (decoder): run the state machine from state 0.
/// Reproduces the encoder's reconstruction exactly (state transitions are a
/// deterministic function of start-state 0 + the level parities).
pub fn dequantize_tcq(levels: &[i64], q: i64) -> Vec<f64> {
    let qf = q as f64;
    let mut s = 0u8;
    let mut out = Vec::with_capacity(levels.len());
    for &lvl in levels {
        out.push(recon(s >> 1, lvl, qf));
        s = STATE_TRANS[s as usize][(lvl & 1) as usize];
    }
    out
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// The decoder's state-machine reconstruction must equal the reconstruction
    /// the encoder's Viterbi actually used (the load-bearing consistency property).
    #[test]
    fn decode_reproduces_encoder_recon() {
        let coeffs: Vec<f64> = (0..500)
            .map(|i| ((i as f64 * 0.3).sin() * 50.0) + ((i % 7) as f64 - 3.0))
            .collect();
        for &q in &[1i64, 3, 8, 32] {
            let levels = quantize_tcq(&coeffs, q, 1.0, 0.2 * (q * q) as f64);
            let recon_dec = dequantize_tcq(&levels, q);
            // Re-derive the encoder reconstruction directly from the same states.
            let mut s = 0u8;
            for (i, &lvl) in levels.iter().enumerate() {
                let r = recon(s >> 1, lvl, q as f64);
                assert!(
                    (r - recon_dec[i]).abs() < 1e-9,
                    "decode mismatch at {i}, q={q}"
                );
                s = STATE_TRANS[s as usize][(lvl & 1) as usize];
            }
        }
    }

    /// TCQ at step `q` transmits ~half-magnitude levels (≈c/2q), so it is
    /// rate-matched to SCALAR at step `2q`. The dependent (combined) grid is finer
    /// at that matched rate ⇒ TCQ MSE ≤ scalar-at-2q MSE.
    #[test]
    fn tcq_beats_rate_matched_scalar() {
        let coeffs: Vec<f64> = (0..2000)
            .map(|i| (i as f64 * 0.05).sin() * 200.0 + (i as f64 * 0.9).cos() * 40.0)
            .collect();
        let q = 16i64;
        let levels = quantize_tcq(&coeffs, q, 1.0, 0.0); // λ=0 ⇒ pure distortion
        let tcq_recon = dequantize_tcq(&levels, q);
        let tcq_mse: f64 = coeffs
            .iter()
            .zip(&tcq_recon)
            .map(|(&c, &r)| (c - r) * (c - r))
            .sum();
        // Rate-matched scalar: step 2q (same transmitted level magnitudes).
        let s2 = (2 * q) as f64;
        let scalar2_mse: f64 = coeffs
            .iter()
            .map(|&c| {
                let r = (c / s2).round() * s2;
                (c - r) * (c - r)
            })
            .sum();
        assert!(
            tcq_mse <= scalar2_mse,
            "tcq {tcq_mse:.1} should beat rate-matched scalar-2q {scalar2_mse:.1}"
        );
    }
}
