//! The Optimum-tier LOSSLESS codec (LMO `transform_id=2`, ADR 0054 Lever C).
//!
//! Gates 0a/0b showed the single-channel path (5/3 + LPC + Golomb) is already
//! near-optimal — arithmetic entropy and learned lifting gave ~0%. The one
//! untouched structural redundancy is **cross-channel** (gate 0c: **−10.5%** on
//! CHB-MIT). This codec captures it as a **best-of-prior integer spatial
//! prediction** wrapped around the proven lossless `lml` codec:
//!
//!   encode: each channel `i` is either coded raw or as the exact integer
//!   residual `ch[i] − round_q16(g · ch[ref])` against the lowest-cost earlier
//!   reference `ref < i` (LS gain `g`, quantized Q16, shipped). The residual
//!   signal is then compressed by `lml::compress` (5/3 + LPC + Golomb, byte-exact).
//!
//!   decode: `lml::decompress` recovers the exact residual signal, then channels
//!   are reconstructed in order — `ch[i] = residual[i] + round_q16(g · ch[ref])`,
//!   using the already-reconstructed (and, because everything is lossless, exact)
//!   reference. **Pure integer** (no `f64`) ⇒ `no_std`-decodable and host↔MCU
//!   bit-identical; the gain fit is the only float, and it lives encode-side.
//!
//! Per-channel keep-smaller (raw vs predicted) means a channel that wouldn't
//! benefit stays raw, so prediction can never enlarge a channel. The LMO
//! container additionally auto-picks this whole body vs the id=0 5/3 floor
//! (keep-smaller) — both lossless ⇒ never worse than the floor.
//!
//! ## Body layout
//! ```text
//!   [0]      kernel_version (u8)
//!   [1]      feature_bitmask (u8; bit0 = cross-channel)
//!   [2..4]   n_ch (u16 LE)
//!   [4..]    per-channel metadata, n_ch entries:
//!              [flag u8]  0 = raw
//!                         1 = predicted → [ref_idx u16 LE][gain_q i32 LE]
//!   [..]     lml stream  (lml::compress of the residual signal)
//! ```

use alloc::vec::Vec;

use lamquant_lml_mcu::error::{LmlError, LmlResult};
use lamquant_lml_mcu::lml;

/// Intra-id=2 kernel version (bump on any body-format / prediction-math change).
const KERNEL_VERSION: u8 = 1;
/// feature_bitmask bit 0: cross-channel spatial prediction present. (Written by
/// the encoder; decode is forward-compatible and does not hard-require the bit.)
#[cfg(feature = "encode")]
const FEATURE_CROSSCHAN: u8 = 0x01;

/// Q16 integer prediction — the bit-identical quantity encode and decode form.
/// `round_q16(g·x)` with `g` in Q16 fixed point, round-half-up via arithmetic
/// shift (negatives handled by the sign-extending `>>`). NO float.
#[inline]
fn predict_q16(gain_q: i32, x: i64) -> i64 {
    ((gain_q as i64 * x) + (1 << 15)) >> 16
}

/// Decode an id=2 body back to the signal. `no_std`-capable (integer only).
pub fn decode(body: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    if body.len() < 4 {
        return Err(LmlError::Truncated { expected: 4, actual: body.len(), context: "lmo_lossless header" });
    }
    let version = body[0];
    if version != KERNEL_VERSION {
        return Err(LmlError::UnsupportedVersion(version));
    }
    let _feature = body[1];
    let n_ch = u16::from_le_bytes([body[2], body[3]]) as usize;

    // Parse the n_ch metadata entries; the remainder is the lml stream.
    let mut pos = 4usize;
    let mut choices: Vec<(bool, usize, i32)> = Vec::with_capacity(n_ch); // (predicted, ref_idx, gain_q)
    for _ in 0..n_ch {
        if pos >= body.len() {
            return Err(LmlError::Truncated { expected: pos + 1, actual: body.len(), context: "lmo_lossless meta flag" });
        }
        let flag = body[pos];
        pos += 1;
        if flag == 0 {
            choices.push((false, 0, 0));
        } else {
            if pos + 6 > body.len() {
                return Err(LmlError::Truncated { expected: pos + 6, actual: body.len(), context: "lmo_lossless meta pred" });
            }
            let ref_idx = u16::from_le_bytes([body[pos], body[pos + 1]]) as usize;
            let gain_q = i32::from_le_bytes([body[pos + 2], body[pos + 3], body[pos + 4], body[pos + 5]]);
            pos += 6;
            choices.push((true, ref_idx, gain_q));
        }
    }

    let resid = lml::decompress(&body[pos..])?;
    if resid.len() != n_ch {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "lmo_lossless n_ch={n_ch} disagrees with lml stream {}",
            resid.len()
        )));
    }

    let mut recon: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for (i, &(predicted, ref_idx, gain_q)) in choices.iter().enumerate() {
        if !predicted {
            recon.push(resid[i].clone());
            continue;
        }
        if ref_idx >= i {
            return Err(LmlError::InvalidHeader(alloc::format!(
                "lmo_lossless ch {i} references non-prior channel {ref_idx}"
            )));
        }
        let r = &resid[i];
        let src = &recon[ref_idx];
        if src.len() != r.len() {
            return Err(LmlError::InvalidHeader(alloc::format!(
                "lmo_lossless ch {i} length {} != ref {ref_idx} length {}",
                r.len(),
                src.len()
            )));
        }
        let ch: Vec<i64> = r.iter().zip(src).map(|(&e, &x)| e + predict_q16(gain_q, x)).collect();
        recon.push(ch);
    }
    Ok(recon)
}

/// LS gain (Q16) predicting `target` from `refc` — minimises residual energy.
/// Float, encode-side only; the shipped `gain_q` makes decode integer-exact.
#[cfg(feature = "encode")]
fn fit_gain_q16(target: &[i64], refc: &[i64]) -> i32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&a, &b) in target.iter().zip(refc) {
        num += a as f64 * b as f64;
        den += b as f64 * b as f64;
    }
    if den == 0.0 {
        return 0;
    }
    let g = num / den;
    (g * 65536.0).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

#[cfg(feature = "encode")]
fn residual_q16(target: &[i64], refc: &[i64], gain_q: i32) -> Vec<i64> {
    target.iter().zip(refc).map(|(&a, &b)| a - predict_q16(gain_q, b)).collect()
}

/// Compressed size of a single channel through the floor codec (the per-channel
/// keep-smaller decision metric; same header overhead on both sides so the
/// raw-vs-residual comparison is fair).
#[cfg(feature = "encode")]
fn channel_cost(ch: &[i64]) -> usize {
    lml::compress(core::slice::from_ref(&ch.to_vec()), 0).map(|v| v.len()).unwrap_or(usize::MAX)
}

/// ref_idx(2) + gain_q(4) + flag(1) extra bytes a predicted channel costs.
#[cfg(feature = "encode")]
const PRED_OVERHEAD: usize = 7;

/// Encode `signal` losslessly with best-of-prior cross-channel spatial prediction.
/// Host-only (the gain fit + the per-channel cost search use `std`).
#[cfg(feature = "encode")]
pub fn encode(signal: &[Vec<i64>]) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(alloc::format!("n_ch={n_ch} out of range 1..=1024")));
    }
    let t = signal[0].len();
    for (c, ch) in signal.iter().enumerate() {
        if ch.len() != t {
            return Err(LmlError::InvalidHeader(alloc::format!(
                "ragged channels: ch {c} has {} samples, expected {t}",
                ch.len()
            )));
        }
    }

    // Per-channel: keep the smaller of raw vs the best-of-prior spatial residual.
    let mut choices: Vec<(bool, usize, i32)> = Vec::with_capacity(n_ch);
    let mut resid_signal: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for i in 0..n_ch {
        let mut best_cost = channel_cost(&signal[i]);
        let mut best: (bool, usize, i32, Vec<i64>) = (false, 0, 0, signal[i].clone());
        for j in 0..i {
            let gain_q = fit_gain_q16(&signal[i], &signal[j]);
            if gain_q == 0 {
                continue;
            }
            let r = residual_q16(&signal[i], &signal[j], gain_q);
            let cost = channel_cost(&r).saturating_add(PRED_OVERHEAD);
            if cost < best_cost {
                best_cost = cost;
                best = (true, j, gain_q, r);
            }
        }
        choices.push((best.0, best.1, best.2));
        resid_signal.push(best.3);
    }

    let lml_stream = lml::compress(&resid_signal, 0)?;

    let mut body = Vec::with_capacity(4 + n_ch + lml_stream.len());
    body.push(KERNEL_VERSION);
    body.push(FEATURE_CROSSCHAN);
    body.extend_from_slice(&(n_ch as u16).to_le_bytes());
    for &(predicted, ref_idx, gain_q) in &choices {
        if !predicted {
            body.push(0);
        } else {
            body.push(1);
            body.extend_from_slice(&(ref_idx as u16).to_le_bytes());
            body.extend_from_slice(&gain_q.to_le_bytes());
        }
    }
    body.extend_from_slice(&lml_stream);
    Ok(body)
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;
    use alloc::vec;

    /// Correlated multichannel signal: each channel is a gain·(shared base) +
    /// per-channel detail, so cross-channel prediction has real redundancy.
    fn make_corr_signal(n_ch: usize, t: usize) -> Vec<Vec<i64>> {
        let base: Vec<i64> = (0..t).map(|i| ((i as f64 * 0.05).sin() * 3000.0) as i64).collect();
        (0..n_ch)
            .map(|c| {
                let g = 0.6 + 0.1 * c as f64;
                (0..t)
                    .map(|i| {
                        let detail = (((i + c * 7) as f64 * 0.9).sin() * 120.0) as i64;
                        (g * base[i] as f64) as i64 + detail
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn roundtrip_bit_exact() {
        for (n_ch, t) in [(1usize, 500usize), (4, 1024), (8, 2049), (16, 777)] {
            let sig = make_corr_signal(n_ch, t);
            let body = encode(&sig).expect("encode");
            let back = decode(&body).expect("decode");
            assert_eq!(back, sig, "lmo_lossless must be bit-exact ({n_ch}x{t})");
        }
    }

    #[test]
    fn predicts_correlated_channels() {
        // Highly correlated channels ⇒ at least one channel should be predicted
        // (flag=1 present in the metadata).
        let sig = make_corr_signal(8, 2048);
        let body = encode(&sig).unwrap();
        let n_ch = u16::from_le_bytes([body[2], body[3]]) as usize;
        let mut pos = 4;
        let mut any_pred = false;
        for _ in 0..n_ch {
            if body[pos] == 1 {
                any_pred = true;
                pos += 7;
            } else {
                pos += 1;
            }
        }
        assert!(any_pred, "expected ≥1 predicted channel on correlated data");
    }

    #[test]
    fn smaller_than_floor_on_correlated() {
        let sig = make_corr_signal(8, 4096);
        let id2 = encode(&sig).unwrap().len();
        let floor = lml::compress(&sig, 0).unwrap().len();
        assert!(id2 < floor, "id=2 {id2} should beat floor {floor} on correlated channels");
    }

    #[test]
    fn raw_channel_zero_overhead_path() {
        // Single channel: nothing to predict from ⇒ all raw, still bit-exact.
        let sig = vec![(0..600).map(|i| ((i * 13) % 91) as i64 - 45).collect::<Vec<i64>>()];
        let body = encode(&sig).unwrap();
        assert_eq!(decode(&body).unwrap(), sig);
    }
}
