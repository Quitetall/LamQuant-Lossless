//! LMO-native per-subband **PCRD** rate allocation over the **float 9/7** CDF
//! wavelet (ADR 0054 Phase 3, lever 2).
//!
//! This mirrors the integer 5/3 `compress_target_bps_pcrd` in the `mcu` floor —
//! same greedy Lagrangian bit-allocation, the same quantize→LPC→entropy chain,
//! the same per-subband packet body — but the transform is the float 9/7 wavelet
//! ([`crate::wavelet97`]) instead of the integer 5/3 lifting. Only the two ends
//! change: the **forward/inverse transform** and the **quant boundary**
//! (float coefficient → i64 index). Everything between is the proven `mcu`
//! machinery, reused verbatim:
//!   * [`lpc::analyze_with_mode`] / [`lpc::synthesize`] (operate on i64 indices),
//!   * Golomb-Rice / zero-RLE entropy ([`golomb`] / [`zrle`]),
//!   * [`lml::scope_lpc_mode`] / [`lml::lpc_max_order`] / [`lml::BIAS_CTX`] —
//!     identical LPC policy, so a 9/7-vs-5/3 PRD comparison is *purely* the
//!     transform.
//!
//! ## Lossy-only
//!
//! 9/7 is float and not integer-reversible, so this path serves **only** the
//! `TargetBps` (WP1–WP8) rate-controlled mode. Lossless / bounded-MAE stay on the
//! integer 5/3 floor.
//!
//! ## Body wire format (wrapped by the LMO container)
//!
//! ```text
//!   [0..2]   n_ch        (u16 LE)
//!   [2..4]   t           (u16 LE)
//!   [4]      n_levels    (u8)
//!   [5]      n_sub       (u8)
//!   [6..10]  meta_len    (u32 LE)
//!   [10..14] payload_len (u32 LE)
//!   [14..14+4*n_sub]  per-subband quantizer steps q_s (u32 LE × n_sub)
//!   [meta]    per-channel × per-subband:  [order:u8][lpc coeffs: i32 LE × order]
//!   [payload] per-channel × per-subband:  [tag:u8][entropy bytes]   (tag 0=Golomb,1=zRLE)
//! ```

use alloc::vec;
use alloc::vec::Vec;

use lamquant_lml_mcu::error::{LmlError, LmlResult};
use lamquant_lml_mcu::{golomb, lml, lpc, zrle};

use crate::{tcq, wavelet97};

/// Fixed body header length (before the per-subband step table). v3 adds a
/// `quant_mode` byte (0 = scalar deadzone, 1 = TCQ dependent quantization).
const BODY_HEADER: usize = 15;

/// quant_mode values in the body header.
const QM_SCALAR: u8 = 0;
const QM_TCQ: u8 = 1;

/// TCQ Viterbi λ as a fraction of `gain·q²` (the high-rate RD λ for step q). A
/// single robust constant — TCQ's RD decision is not very λ-sensitive.
#[cfg(feature = "encode")]
const TCQ_LAMBDA_C: f64 = 0.12;

/// Quantize one channel-subband to (levels, reconstruction) under the chosen
/// quantizer. Scalar = deadzone `idx`, recon `idx·q`. TCQ = dependent-quantized
/// levels (Viterbi), recon via the state machine. Levels feed LPC+entropy as the
/// `idx` either way; the reconstruction drives the PCRD distortion + the decoder.
#[cfg(feature = "encode")]
fn quantize_subband(coeffs: &[f64], q: i64, gain: f64, use_tcq: bool, dz_offset: f64) -> (Vec<i64>, Vec<f64>) {
    if use_tcq {
        let lambda = gain * (q as f64) * (q as f64) * TCQ_LAMBDA_C;
        let levels = tcq::quantize_tcq(coeffs, q, gain, lambda);
        let recon = tcq::dequantize_tcq(&levels, q);
        (levels, recon)
    } else {
        let qf = q as f64;
        let idx: Vec<i64> = coeffs.iter().map(|&c| quant_f64_dz(c, q, dz_offset)).collect();
        let recon: Vec<f64> = idx.iter().map(|&i| i as f64 * qf).collect();
        (idx, recon)
    }
}

/// Candidate quantizer steps — geometric grid, dense at small `q`. Identical to
/// the 5/3 PCRD grid so the rate–distortion sweep is comparable.
const CANDIDATES: [i64; 46] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48, 56,
    64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
    12288, 16384,
];

/// Per-subband subband lengths for `t` samples and `n_levels` (mirrors the `mcu`
/// `subband_lengths`: ordered `[approx, detail_top, ..., detail_1]`).
fn subband_lengths(t: usize, n_levels: u8) -> Vec<usize> {
    if n_levels == 0 {
        return vec![t];
    }
    let mut details = Vec::with_capacity(n_levels as usize);
    let mut approx = t;
    for _ in 0..n_levels {
        let next_approx = approx.div_ceil(2);
        details.push(approx / 2);
        approx = next_approx;
    }
    let mut out = Vec::with_capacity(n_levels as usize + 1);
    out.push(approx);
    for d in details.iter().rev() {
        out.push(*d);
    }
    out
}

/// Per-subband synthesis L2 gain `G_s` for the 9/7 inverse — the 9/7 analogue of
/// `quant::synthesis_gains`. A unit impulse at each subband's centre is run back
/// through the inverse transform; `G_s = ||inverse(impulse)||²`.
fn synthesis_gains_97(sub_lens: &[usize], n_levels: u8) -> Vec<f64> {
    const IMP: f64 = 4096.0;
    let n_sub = sub_lens.len();
    let mut gains = vec![1.0f64; n_sub];
    for s in 0..n_sub {
        if sub_lens[s] == 0 {
            continue;
        }
        let mut subs: Vec<Vec<f64>> = sub_lens.iter().map(|&l| vec![0.0f64; l]).collect();
        subs[s][sub_lens[s] / 2] = IMP;
        let recon = wavelet97::inverse_97_levels(&subs, n_levels);
        let energy: f64 = recon.iter().map(|&v| (v as f64) * (v as f64)).sum();
        gains[s] = (energy / (IMP * IMP)).max(1e-9);
    }
    gains
}

/// Deadzone scalar quantizer (ADR 0054 lossy RDOQ-lite): `idx = sign(c)·⌊|c|/q +
/// δ⌋`. `δ = 0.5` is round-nearest (== [`quant_f64`]); `δ < 0.5` widens the
/// dead-zone around 0 (zeros more small coefficients, trading distortion for
/// rate) — the dominant practical RDOQ gain for Laplacian wavelet coefficients
/// (JPEG2000/VVC use ≈0.375). Reconstruction stays `idx·q`, so DECODE is
/// unchanged — `δ` is a pure encode-side RD choice, no wire-format change.
#[cfg(feature = "encode")]
#[inline]
fn quant_f64_dz(coeff: f64, q: i64, dz_offset: f64) -> i64 {
    let a = (coeff.abs() / q as f64 + dz_offset).floor() as i64;
    if coeff < 0.0 {
        -a
    } else {
        a
    }
}

/// Entropy-code one subband's residual (LPC residual, or — in the order-0
/// bypass — the bias-cancelled quantizer indices), keeping the smallest of the
/// available coders. Output `[tag][coded]`:
///   * `0` = Golomb-Rice, `1` = zero-RLE — always available (no_std), the same
///     tag values the `mcu` track-2 reader uses.
///   * `4` = integer empirical-categorical order-0, `5` = order-1 context
///     ([`crate::arith_int`]) — the PRODUCTION arithmetic coders (ADR 0054
///     lever-3 stage 3b). Integer-only / `no_std`, so they are always available
///     and the default LMO decode stays MCU-decodable.
///   * `2` = `arith_cat` order-0, `3` = order-1 — the std/`constriction`
///     measurement oracle, only under `experimental_arithmetic`. Kept so the
///     integer coder's compression can be A/B'd against the reference; never
///     selected in a default build.
fn encode_residual(values: &[i64]) -> LmlResult<Vec<u8>> {
    let mut tag = 0u8;
    let mut best = golomb::encode_dense(values)?;
    let z = zrle::encode_dense(values)?;
    if z.len() < best.len() {
        tag = 1;
        best = z;
    }
    // Production integer arithmetic (no_std). A coder may bail on a too-wide
    // alphabet — that just means "not a candidate", absorbed by keep-smallest.
    if let Ok(a4) = crate::arith_int::encode_dense(values) {
        if a4.len() < best.len() {
            tag = 4;
            best = a4;
        }
    }
    if let Ok(a5) = crate::arith_int::encode_dense_ctx(values) {
        if a5.len() < best.len() {
            tag = 5;
            best = a5;
        }
    }
    #[cfg(feature = "experimental_arithmetic")]
    {
        use lamquant_lml_mcu::arith_cat;
        if let Ok(a0) = arith_cat::encode_dense(values) {
            if a0.len() < best.len() {
                tag = 2;
                best = a0;
            }
        }
        if let Ok(a1) = arith_cat::encode_dense_ctx(values) {
            if a1.len() < best.len() {
                tag = 3;
                best = a1;
            }
        }
    }
    let mut out = Vec::with_capacity(1 + best.len());
    out.push(tag);
    out.extend_from_slice(&best);
    Ok(out)
}

/// Decode one residual written by [`encode_residual`], starting at `offset`.
/// Returns `(values, bytes_consumed_from_offset)`.
fn decode_residual(data: &[u8], offset: usize) -> LmlResult<(Vec<i64>, usize)> {
    if offset >= data.len() {
        return Err(LmlError::Truncated {
            expected: offset + 1,
            actual: data.len(),
            context: "9/7 residual coder tag",
        });
    }
    let tag = data[offset];
    let (vals, consumed) = match tag {
        0 => golomb::decode_dense(data, offset + 1)?,
        1 => zrle::decode_dense(data, offset + 1)?,
        4 => crate::arith_int::decode_dense(data, offset + 1)?,
        5 => crate::arith_int::decode_dense_ctx(data, offset + 1)?,
        #[cfg(feature = "experimental_arithmetic")]
        2 => lamquant_lml_mcu::arith_cat::decode_dense(data, offset + 1)?,
        #[cfg(feature = "experimental_arithmetic")]
        3 => lamquant_lml_mcu::arith_cat::decode_dense_ctx(data, offset + 1)?,
        other => {
            return Err(LmlError::InvalidHeader(alloc::format!(
                "unknown 9/7 residual coder tag 0x{:02X}",
                other
            )))
        }
    };
    Ok((vals, consumed + 1))
}

/// Encode one channel's quantized subband to its `(meta, payload)` packet,
/// choosing the smaller of two substrates (ADR 0054 lever-3 stage 3a):
///
///   * **LPC path** — `analyze_with_mode` picks an order; `meta = [order][coeffs…]`,
///     `payload = encode_residual(lpc_residual)`. The current (lever-2) behaviour.
///   * **Coefficient-domain bypass** — force `order = 0` and entropy-code the
///     bias-cancelled indices *directly* (`encode_residual(idx)`); `meta = [0]`.
///     This is the EBCOT-aligned substrate: no LPC, let the (arithmetic) entropy
///     coder model the coefficient distribution itself.
///
/// Keep-smallest over `meta.len()+payload.len()`, so the bypass can only help.
/// `decode_97` already reconstructs both: an `order == 0` packet round-trips
/// because `synthesize(order=0)` is `bias_restore`, the inverse of the
/// `bias_cancel` that `analyze(_, 0, _)` applies. (When `experimental_arithmetic`
/// is off this still adds the Golomb/zRLE-on-indices alternative — never worse.)
#[cfg(feature = "encode")]
fn encode_subband(idx: &[i64], sb_idx: usize, mode: lpc::LpcMode) -> LmlResult<(Vec<u8>, Vec<u8>)> {
    // LPC path.
    let scoped = lml::scope_lpc_mode(mode, lml::lpc_max_order(idx.len()));
    let (coeffs, residual, order) = lpc::analyze_with_mode(idx, sb_idx, scoped, lml::BIAS_CTX, None);
    let lpc_payload = encode_residual(&residual)?;
    let lpc_total = 1 + 4 * coeffs.len() + lpc_payload.len();

    // Coefficient-domain bypass: order 0, code the bias-cancelled indices direct.
    let (_c0, residual0) = lpc::analyze(idx, 0, lml::BIAS_CTX);
    let bypass_payload = encode_residual(&residual0)?;
    let bypass_total = 1 + bypass_payload.len();

    if bypass_total < lpc_total {
        Ok((vec![0u8], bypass_payload))
    } else {
        let mut meta = Vec::with_capacity(1 + 4 * coeffs.len());
        meta.push(order as u8);
        for &c in &coeffs {
            meta.extend_from_slice(&c.to_le_bytes());
        }
        Ok((meta, lpc_payload))
    }
}

/// Encode `signal` ([n_ch][T]) to a target bits-per-sample using the float 9/7
/// transform + per-subband PCRD allocation. Returns the LMO-native **body**
/// (the LMO container header is added by the caller). Lossy-only.
#[cfg(feature = "encode")]
pub fn encode_target_bps_97(
    signal: &[Vec<i64>],
    target_bps: f64,
    mode: lpc::LpcMode,
) -> LmlResult<Vec<u8>> {
    // RDOQ-lite: try a few deadzone offsets and keep the lowest-PRD body. Decode
    // is δ-agnostic (reconstruction is idx·q), so this is encode-only and never
    // worse than round-nearest (δ=0.5 is in the set). δ≈0.375–0.42 wins the
    // WP3–WP5 band; 0.5 wins at high BPS — keep-best covers both (ADR 0054).
    let mut best: Option<(f64, Vec<u8>)> = None;
    let consider = |body: Vec<u8>, best: &mut Option<(f64, Vec<u8>)>| -> LmlResult<()> {
        let p = prd_i64(signal, &decode_97(&body)?);
        if best.as_ref().map_or(true, |(bp, _)| p < *bp) {
            *best = Some((p, body));
        }
        Ok(())
    };
    for &dz in &[0.5f64, 0.42, 0.375] {
        consider(encode_target_bps_97_q(signal, target_bps, mode, dz, false)?, &mut best)?;
    }
    // TCQ dependent-quantization candidate (ADR 0054). Decode auto-detects via the
    // body quant_mode byte; keep-best ⇒ never worse than the scalar paths.
    consider(encode_target_bps_97_q(signal, target_bps, mode, 0.5, true)?, &mut best)?;
    Ok(best.expect("at least one candidate").1)
}

/// CfP §4 mean-removed PRD (%), encode-side keep-best metric.
#[cfg(feature = "encode")]
fn prd_i64(orig: &[Vec<i64>], recon: &[Vec<i64>]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (o, r) in orig.iter().zip(recon) {
        let m = o.iter().sum::<i64>() as f64 / o.len().max(1) as f64;
        for (a, b) in o.iter().zip(r) {
            let e = (*a - *b) as f64;
            num += e * e;
            den += (*a as f64 - m) * (*a as f64 - m);
        }
    }
    if den == 0.0 {
        0.0
    } else {
        100.0 * (num / den).sqrt()
    }
}

/// As [`encode_target_bps_97`], with an explicit deadzone offset `dz_offset`
/// (RDOQ-lite screen; `0.5` = round-nearest). Decode is identical regardless.
#[cfg(feature = "encode")]
pub fn encode_target_bps_97_dz(
    signal: &[Vec<i64>],
    target_bps: f64,
    mode: lpc::LpcMode,
    dz_offset: f64,
) -> LmlResult<Vec<u8>> {
    encode_target_bps_97_q(signal, target_bps, mode, dz_offset, false)
}

/// As [`encode_target_bps_97_dz`], with `use_tcq` selecting TCQ dependent
/// quantization (ADR 0054) instead of scalar deadzone. Decode auto-detects from
/// the body's `quant_mode` byte.
#[cfg(feature = "encode")]
pub fn encode_target_bps_97_q(
    signal: &[Vec<i64>],
    target_bps: f64,
    mode: lpc::LpcMode,
    dz_offset: f64,
    use_tcq: bool,
) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };
    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "n_ch={n_ch} out of range 1..=1024"
        )));
    }
    if t == 0 || t > u16::MAX as usize {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "T={t} out of range 1..={}",
            u16::MAX
        )));
    }
    for (c, ch) in signal.iter().enumerate() {
        if ch.len() != t {
            return Err(LmlError::InvalidHeader(alloc::format!(
                "ragged channels: ch {c} has {} samples, expected {t}",
                ch.len()
            )));
        }
    }
    if !(target_bps.is_finite() && target_bps > 0.0) {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "target_bps must be finite > 0, got {target_bps}"
        )));
    }

    let n_levels = lml::compute_n_levels(t);
    let sub_lens = subband_lengths(t, n_levels);
    let n_sub = sub_lens.len();
    let gains = synthesis_gains_97(&sub_lens, n_levels);
    let chan_subs: Vec<Vec<Vec<f64>>> = signal
        .iter()
        .map(|ch| wavelet97::forward_97_levels(ch, n_levels))
        .collect();

    let nm = (n_ch * t) as f64;
    let fixed_overhead = BODY_HEADER + 4 * n_sub;

    // Per (subband, candidate step): (rate_bits, distortion) summed over channels.
    let mut rd: Vec<Vec<(f64, f64)>> = vec![vec![(0.0, 0.0); CANDIDATES.len()]; n_sub];
    for sb in 0..n_sub {
        for (ci, &q) in CANDIDATES.iter().enumerate() {
            let mut rate_bytes = 0usize;
            let mut sse = 0.0f64;
            for subs in &chan_subs {
                let sub = &subs[sb];
                let (idx, recon) = quantize_subband(sub, q, gains[sb], use_tcq, dz_offset);
                for (&c, &r) in sub.iter().zip(recon.iter()) {
                    let e = c - r;
                    sse += e * e;
                }
                let (meta, payload) = encode_subband(&idx, sb, mode)?;
                rate_bytes += meta.len() + payload.len();
            }
            rd[sb][ci] = (rate_bytes as f64 * 8.0, gains[sb] * sse);
        }
    }

    // Greedy R-D allocation (identical to the 5/3 PCRD): start coarsest, repeatedly
    // apply the per-subband refinement with the best ΔD/ΔR that still fits budget.
    let budget = (target_bps * nm - fixed_overhead as f64 * 8.0).max(0.0);
    let coarsest = CANDIDATES.len() - 1;
    let mut chosen = vec![coarsest; n_sub];
    let mut cur_rate: f64 = (0..n_sub).map(|sb| rd[sb][coarsest].0).sum();
    loop {
        let mut best: Option<(usize, usize, f64)> = None;
        let mut best_ratio = 0.0f64;
        for sb in 0..n_sub {
            for ci in 0..chosen[sb] {
                let dr = rd[sb][ci].0 - rd[sb][chosen[sb]].0;
                let dd = rd[sb][chosen[sb]].1 - rd[sb][ci].1;
                if dr > 1e-9 && dd > 0.0 && cur_rate + dr <= budget {
                    let ratio = dd / dr;
                    if ratio > best_ratio {
                        best_ratio = ratio;
                        best = Some((sb, ci, dr));
                    }
                }
            }
        }
        match best {
            Some((sb, ci, dr)) => {
                chosen[sb] = ci;
                cur_rate += dr;
            }
            None => break,
        }
    }
    let qs: Vec<i64> = (0..n_sub).map(|sb| CANDIDATES[chosen[sb]]).collect();

    // Final encode at the chosen per-subband steps.
    let mut meta_body = Vec::new();
    let mut payload = Vec::new();
    for subs in &chan_subs {
        for (sb_idx, sub) in subs.iter().enumerate() {
            let (idx, _recon) = quantize_subband(sub, qs[sb_idx], gains[sb_idx], use_tcq, dz_offset);
            let (meta, pay) = encode_subband(&idx, sb_idx, mode)?;
            meta_body.extend_from_slice(&meta);
            payload.extend_from_slice(&pay);
        }
    }

    // Assemble body.
    let mut body = Vec::with_capacity(fixed_overhead + meta_body.len() + payload.len());
    body.extend_from_slice(&(n_ch as u16).to_le_bytes());
    body.extend_from_slice(&(t as u16).to_le_bytes());
    body.push(n_levels);
    body.push(n_sub as u8);
    body.push(if use_tcq { QM_TCQ } else { QM_SCALAR });
    body.extend_from_slice(&(meta_body.len() as u32).to_le_bytes());
    body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    for &q in &qs {
        body.extend_from_slice(&(q as u32).to_le_bytes());
    }
    body.extend_from_slice(&meta_body);
    body.extend_from_slice(&payload);
    Ok(body)
}

/// Decode a 9/7 PCRD body written by [`encode_target_bps_97`] back to the signal.
/// no_std-capable (float inverse needs no std/libm).
pub fn decode_97(body: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    if body.len() < BODY_HEADER {
        return Err(LmlError::Truncated {
            expected: BODY_HEADER,
            actual: body.len(),
            context: "9/7 body header",
        });
    }
    let n_ch = u16::from_le_bytes([body[0], body[1]]) as usize;
    let t = u16::from_le_bytes([body[2], body[3]]) as usize;
    let n_levels = body[4];
    let n_sub = body[5] as usize;
    let quant_mode = body[6];
    let meta_len = u32::from_le_bytes([body[7], body[8], body[9], body[10]]) as usize;
    let payload_len = u32::from_le_bytes([body[11], body[12], body[13], body[14]]) as usize;

    let sub_lens = subband_lengths(t, n_levels);
    if sub_lens.len() != n_sub {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "9/7 n_sub={n_sub} disagrees with derived {} (t={t}, levels={n_levels})",
            sub_lens.len()
        )));
    }

    let steps_off = BODY_HEADER;
    let meta_off = steps_off + 4 * n_sub;
    let pay_off = meta_off + meta_len;
    if body.len() < pay_off + payload_len {
        return Err(LmlError::Truncated {
            expected: pay_off + payload_len,
            actual: body.len(),
            context: "9/7 body meta+payload",
        });
    }
    let mut qs = Vec::with_capacity(n_sub);
    for s in 0..n_sub {
        let o = steps_off + 4 * s;
        qs.push(u32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]) as i64);
    }
    let meta = &body[meta_off..meta_off + meta_len];
    let payload = &body[pay_off..pay_off + payload_len];

    let mut meta_pos = 0usize;
    let mut pay_pos = 0usize;
    let mut out: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for _ch in 0..n_ch {
        let mut subs_f: Vec<Vec<f64>> = Vec::with_capacity(n_sub);
        for &q in &qs {
            if meta_pos >= meta.len() {
                return Err(LmlError::Truncated {
                    expected: meta_pos + 1,
                    actual: meta.len(),
                    context: "9/7 lpc order",
                });
            }
            let order = meta[meta_pos] as usize;
            meta_pos += 1;
            if meta_pos + 4 * order > meta.len() {
                return Err(LmlError::Truncated {
                    expected: meta_pos + 4 * order,
                    actual: meta.len(),
                    context: "9/7 lpc coeffs",
                });
            }
            let mut coeffs = Vec::with_capacity(order);
            for _ in 0..order {
                coeffs.push(i32::from_le_bytes([
                    meta[meta_pos],
                    meta[meta_pos + 1],
                    meta[meta_pos + 2],
                    meta[meta_pos + 3],
                ]));
                meta_pos += 4;
            }
            let (residual, consumed) = decode_residual(payload, pay_pos)?;
            pay_pos += consumed;
            let idx = lpc::synthesize(&residual, &coeffs, order, lml::BIAS_CTX);
            if quant_mode == QM_TCQ {
                subs_f.push(tcq::dequantize_tcq(&idx, q));
            } else {
                let qf = q as f64;
                subs_f.push(idx.iter().map(|&i| i as f64 * qf).collect());
            }
        }
        out.push(wavelet97::inverse_97_levels(&subs_f, n_levels));
    }
    Ok(out)
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;

    fn make_signal(n_ch: usize, t: usize, seed: i64) -> Vec<Vec<i64>> {
        (0..n_ch)
            .map(|c| {
                let ph = seed.wrapping_add(c as i64 * 911);
                (0..t)
                    .map(|i| {
                        let x = (i as i64 + ph) as f64;
                        let lo = (x * 0.05).sin() * 3000.0;
                        let hi = (x * 0.9).sin() * 250.0;
                        let spike = if (i as i64 + ph) % 101 == 0 { 1200.0 } else { 0.0 };
                        (lo + hi + spike) as i64
                    })
                    .collect()
            })
            .collect()
    }

    fn prd(orig: &[Vec<i64>], recon: &[Vec<i64>]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (o, r) in orig.iter().zip(recon.iter()) {
            let m = o.iter().sum::<i64>() as f64 / o.len().max(1) as f64;
            for (a, b) in o.iter().zip(r.iter()) {
                let e = (*a - *b) as f64;
                num += e * e;
                den += (*a as f64 - m) * (*a as f64 - m);
            }
        }
        if den == 0.0 {
            0.0
        } else {
            100.0 * (num / den).sqrt()
        }
    }

    #[test]
    fn roundtrip_shape_and_rate_ceiling() {
        let signal = make_signal(8, 2560, 99);
        let nm = (8 * 2560) as f64;
        for &target in &[3.0f64, 2.0, 1.5] {
            let body = encode_target_bps_97(&signal, target, lpc::LpcMode::default()).unwrap();
            let bps = body.len() as f64 * 8.0 / nm;
            assert!(
                bps <= target * 1.10,
                "9/7 target {target}: BPS {bps:.3} exceeds ceiling+10%"
            );
            let recon = decode_97(&body).unwrap();
            assert_eq!(recon.len(), 8);
            assert_eq!(recon[0].len(), 2560);
        }
    }

    #[test]
    fn higher_budget_lower_distortion() {
        let signal = make_signal(8, 2560, 7);
        let lo = decode_97(&encode_target_bps_97(&signal, 1.0, lpc::LpcMode::default()).unwrap()).unwrap();
        let hi = decode_97(&encode_target_bps_97(&signal, 4.0, lpc::LpcMode::default()).unwrap()).unwrap();
        assert!(
            prd(&signal, &hi) <= prd(&signal, &lo),
            "more bits must not raise PRD: 4.0→{:.2} vs 1.0→{:.2}",
            prd(&signal, &hi),
            prd(&signal, &lo)
        );
    }
}
