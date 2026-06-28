//! The Optimum-tier LOSSLESS codec (LMO `transform_id=2`, ADR 0054 Lever C).
//!
//! Gates 0a/0b showed the single-channel path (5/3 + LPC + Golomb) is already
//! near-optimal — arithmetic entropy and learned lifting gave ~0%. The one
//! untouched structural redundancy is **cross-channel** (gate 0c: **−10.5%** on
//! CHB-MIT, up to **−22%** on 64ch referential). This codec captures it as a
//! **multi-reference integer spatial prediction** wrapped around the proven
//! lossless `lml` codec:
//!
//!   encode: each channel `i` is coded raw or as the exact integer residual
//!   `ch[i] − round_q16(Σ_r g_r · ch[ref_r])` against up to `MAX_REFS` earlier
//!   references `ref_r < i` (joint LS gains, quantized Q16, shipped). References
//!   are chosen byte-greedily for the first (so it never regresses the single-best
//!   codec) then energy-greedily for the rest, keeping more refs only while they
//!   shrink the channel. The residual signal is then `lml::compress`'d.
//!
//!   decode: `lml::decompress` recovers the exact residual signal, then channels
//!   are reconstructed in order — `ch[i] = residual[i] + round_q16(Σ_r g_r ·
//!   ch[ref_r])` — using the already-reconstructed (exact, lossless) references.
//!   **Pure integer** (no `f64`) ⇒ `no_std`-decodable, host↔MCU bit-identical;
//!   the gain fit is the only float, encode-side.
//!
//! Per-channel keep-smaller (raw vs predicted) + container auto-pick vs the id=0
//! floor ⇒ never worse than the floor.
//!
//! ## Body layout
//! ```text
//!   [0]      kernel_version (u8)
//!   [1]      feature_bitmask (u8; bit0 = cross-channel)
//!   [2..4]   n_ch (u16 LE)
//!   [4..]    per-channel metadata, n_ch entries:
//!              [n_refs u8]  0 = raw
//!                           k = predicted → k × ([ref_idx u16 LE][gain_q i32 LE])
//!   [..]     lml stream  (lml::compress of the residual signal)
//! ```

use alloc::vec::Vec;

use lamquant_lml_mcu::error::{LmlError, LmlResult};
use lamquant_lml_mcu::lml;

/// Intra-id=2 kernel version (bump on any body-format / prediction-math change).
/// v2 = multi-reference prediction (v1 was single-reference).
const KERNEL_VERSION: u8 = 3;
/// Residual-coder mode byte (after the per-channel metadata): 0 = the lml codec
/// (5/3+LPC+Golomb), 1 = per-channel RLS prediction + Golomb (ADR 0054 — wins on
/// non-stationary signals where the static path collapses). Keep-best per body.
const CODER_LML: u8 = 0;
const CODER_RLS: u8 = 1;
/// Multivariate cross-channel RLS on the raw signal (all channels coded "raw").
const CODER_MV_RLS: u8 = 2;
/// Per-channel RLS with change-point segmentation (ADR 0054 Lever C) — resets the
/// RLS at causal signal-derived regime boundaries in addition to the fixed period.
/// A SEPARATE mode so the plain `CODER_RLS` wire stays byte-identical. Keep-best.
const CODER_RLS_SEG: u8 = 3;
/// LSB bit-plane split layer. When the signal's low bit-plane(s) are heavily biased
/// (e.g. linked-ear-montage LSB ≈ 0.3 bits — prediction would scramble that structure
/// into a full-entropy residual LSB), strip `s` low bits, code the upper signal via the
/// normal pipeline and the LSB plane(s) separately. The body starts with `SPLIT_MAGIC`
/// (≠ `KERNEL_VERSION`) so `decode` dispatches BEFORE the version check; `s=0` (no split)
/// returns the unchanged base body ⇒ byte-identical, never-worse, goldens unaffected.
const SPLIT_MAGIC: u8 = 0xFE;
/// Max low bit-planes the encoder will try to strip.
#[cfg(feature = "encode")]
const MAX_SPLIT: usize = 2;
/// feature_bitmask bit 0: cross-channel spatial prediction present. (Written by
/// the encoder; decode is forward-compatible and does not hard-require the bit.)
#[cfg(feature = "encode")]
const FEATURE_CROSSCHAN: u8 = 0x01;
/// Max references per predicted channel (diminishing returns past ~3; bounds the
/// header + the encoder search).
#[cfg(feature = "encode")]
const MAX_REFS: usize = 3;

/// Integer multi-reference prediction — the bit-identical quantity encode and
/// decode form: `round_q16(Σ_r gain_q[r]·chans[ref[r]][k])`. Round-half-up via
/// arithmetic shift. NO float. (Q16 gains, ≤3 refs ⇒ the i64 accumulator cannot
/// overflow for realistic O(1) gains and ≤24-bit samples.)
#[inline]
fn predict_multi(refs: &[usize], gains_q: &[i32], chans: &[Vec<i64>], k: usize) -> i64 {
    let mut acc: i64 = 0;
    for (g, &r) in gains_q.iter().zip(refs) {
        acc += *g as i64 * chans[r][k];
    }
    (acc + (1 << 15)) >> 16
}

/// Decode an id=2 body back to the signal. `no_std`-capable (integer only).
pub fn decode(body: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    if body.first() == Some(&SPLIT_MAGIC) {
        return decode_lsb_split(body);
    }
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
    let mut metas: Vec<(Vec<usize>, Vec<i32>)> = Vec::with_capacity(n_ch);
    for _ in 0..n_ch {
        if pos >= body.len() {
            return Err(LmlError::Truncated { expected: pos + 1, actual: body.len(), context: "lmo_lossless meta n_refs" });
        }
        let n_refs = body[pos] as usize;
        pos += 1;
        let mut refs = Vec::with_capacity(n_refs);
        let mut gains = Vec::with_capacity(n_refs);
        for _ in 0..n_refs {
            if pos + 6 > body.len() {
                return Err(LmlError::Truncated { expected: pos + 6, actual: body.len(), context: "lmo_lossless meta ref" });
            }
            refs.push(u16::from_le_bytes([body[pos], body[pos + 1]]) as usize);
            gains.push(i32::from_le_bytes([body[pos + 2], body[pos + 3], body[pos + 4], body[pos + 5]]));
            pos += 6;
        }
        metas.push((refs, gains));
    }

    if pos >= body.len() {
        return Err(LmlError::Truncated { expected: pos + 1, actual: body.len(), context: "lmo_lossless coder_mode" });
    }
    let coder_mode = body[pos];
    pos += 1;
    let resid = match coder_mode {
        CODER_LML => lml::decompress(&body[pos..])?,
        CODER_RLS => crate::rls::decode(&body[pos..])?,
        CODER_RLS_SEG => crate::rls::decode_seg(&body[pos..])?,
        CODER_MV_RLS => crate::mv_rls::decode(&body[pos..])?,
        other => {
            return Err(LmlError::InvalidHeader(alloc::format!(
                "lmo_lossless unknown coder_mode 0x{other:02X}"
            )))
        }
    };
    if resid.len() != n_ch {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "lmo_lossless n_ch={n_ch} disagrees with lml stream {}",
            resid.len()
        )));
    }

    let mut recon: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for (i, (refs, gains)) in metas.iter().enumerate() {
        if refs.is_empty() {
            recon.push(resid[i].clone());
            continue;
        }
        let r = &resid[i];
        for &j in refs {
            if j >= i {
                return Err(LmlError::InvalidHeader(alloc::format!(
                    "lmo_lossless ch {i} references non-prior channel {j}"
                )));
            }
            if recon[j].len() != r.len() {
                return Err(LmlError::InvalidHeader(alloc::format!(
                    "lmo_lossless ch {i} length {} != ref {j} length {}",
                    r.len(),
                    recon[j].len()
                )));
            }
        }
        let ch: Vec<i64> = (0..r.len()).map(|k| r[k] + predict_multi(refs, gains, &recon, k)).collect();
        recon.push(ch);
    }
    Ok(recon)
}

/// Decode an LSB-split body: `[SPLIT_MAGIC][s u8][n_ch u16][t u32][upper_len u32]
/// [upper_body][ (lsb_len u32)(lsb_body) × n_ch ]`. `no_std`-capable. Reconstructs
/// `x = (upper << s) | lsb`. Recurses into `decode` for the upper body (which is a
/// normal body, so no nesting).
fn decode_lsb_split(body: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    if body.len() < 12 {
        return Err(LmlError::Truncated { expected: 12, actual: body.len(), context: "lmo_lossless lsb-split header" });
    }
    let s = body[1] as usize;
    if s > 16 {
        return Err(LmlError::InvalidHeader(alloc::format!("lsb-split shift {s} out of range")));
    }
    let n_ch = u16::from_le_bytes([body[2], body[3]]) as usize;
    let t = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    let upper_len = u32::from_le_bytes([body[8], body[9], body[10], body[11]]) as usize;
    let upend = 12usize.checked_add(upper_len).filter(|&e| e <= body.len())
        .ok_or(LmlError::Truncated { expected: 12 + upper_len, actual: body.len(), context: "lsb-split upper body" })?;
    let upper = decode(&body[12..upend])?;
    if upper.len() != n_ch {
        return Err(LmlError::InvalidHeader(alloc::format!("lsb-split n_ch {n_ch} != upper {}", upper.len())));
    }
    let mut pos = upend;
    let mut recon: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        if pos + 4 > body.len() {
            return Err(LmlError::Truncated { expected: pos + 4, actual: body.len(), context: "lsb-split lsb len" });
        }
        let lsb_len = u32::from_le_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]) as usize;
        pos += 4;
        let end = pos.checked_add(lsb_len).filter(|&e| e <= body.len())
            .ok_or(LmlError::Truncated { expected: pos + lsb_len, actual: body.len(), context: "lsb-split lsb body" })?;
        let lsb = crate::entropy::decode(&body[pos..end])?;
        pos = end;
        if lsb.len() != t || upper[c].len() != t {
            return Err(LmlError::InvalidHeader(alloc::format!("lsb-split ch {c} length mismatch")));
        }
        let ch: Vec<i64> = (0..t).map(|n| (upper[c][n] << s) | lsb[n]).collect();
        recon.push(ch);
    }
    Ok(recon)
}

// ─── encode (host-only) ───────────────────────────────────────────────────────

#[cfg(feature = "encode")]
fn dot(a: &[i64], b: &[i64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// Joint LS float gains predicting `target` from `refs` (Gaussian elimination on
/// the small normal equations `X'X g = X'y`, ridge for safety).
#[cfg(feature = "encode")]
#[allow(clippy::needless_range_loop)] // index-based Gaussian elimination
fn joint_ls(target: &[i64], refs: &[usize], chans: &[Vec<i64>]) -> Vec<f64> {
    let k = refs.len();
    let mut a = alloc::vec![alloc::vec![0.0f64; k]; k];
    let mut b = alloc::vec![0.0f64; k];
    for r in 0..k {
        for c in 0..k {
            a[r][c] = dot(&chans[refs[r]], &chans[refs[c]]);
        }
        a[r][r] += 1e-6 * a[r][r].max(1.0);
        b[r] = dot(&chans[refs[r]], target);
    }
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        if a[col][col].abs() < 1e-12 {
            continue;
        }
        for r in (col + 1)..k {
            let f = a[r][col] / a[col][col];
            for c in col..k {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut g = alloc::vec![0.0f64; k];
    for r in (0..k).rev() {
        let mut s = b[r];
        for c in (r + 1)..k {
            s -= a[r][c] * g[c];
        }
        g[r] = if a[r][r].abs() < 1e-12 { 0.0 } else { s / a[r][r] };
    }
    g
}

/// Quantize float gains to Q16 i32 (clamped). Returns `None` if every gain
/// rounds to 0 (no useful prediction).
#[cfg(feature = "encode")]
fn quantize_gains(g: &[f64]) -> Option<Vec<i32>> {
    let q: Vec<i32> = g
        .iter()
        .map(|&v| (v * 65536.0).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
        .collect();
    if q.iter().all(|&x| x == 0) {
        None
    } else {
        Some(q)
    }
}

/// Residual `target − round_q16(Σ gq·chans[ref])` using the QUANTIZED gains, so
/// decode (same `predict_multi`) reconstructs exactly.
#[cfg(feature = "encode")]
fn residual_multi(target: &[i64], refs: &[usize], gains_q: &[i32], chans: &[Vec<i64>]) -> Vec<i64> {
    (0..target.len()).map(|k| target[k] - predict_multi(refs, gains_q, chans, k)).collect()
}

/// Single-channel floor cost (the per-channel keep-smaller decision metric).
#[cfg(feature = "encode")]
fn channel_cost(ch: &[i64]) -> usize {
    lml::compress(core::slice::from_ref(&ch.to_vec()), 0).map(|v| v.len()).unwrap_or(usize::MAX)
}

/// Per-ref overhead: ref_idx(2) + gain(4). Plus 1 n_refs byte per channel.
#[cfg(feature = "encode")]
const PER_REF_OVERHEAD: usize = 6;

/// Energy-best prior `j ∉ chosen` for predicting `residual` (single-tap LS
/// energy reduction `<resid,ch_j>²/<ch_j,ch_j>`).
#[cfg(feature = "encode")]
#[allow(clippy::needless_range_loop)] // j indexes chans and is matched against `chosen`
fn best_energy_ref(residual: &[i64], chans: &[Vec<i64>], n_prior: usize, chosen: &[usize]) -> Option<usize> {
    let mut best = (usize::MAX, 0.0f64);
    for j in 0..n_prior {
        if chosen.contains(&j) {
            continue;
        }
        let den = dot(&chans[j], &chans[j]);
        if den <= 0.0 {
            continue;
        }
        let num = dot(residual, &chans[j]);
        let red = num * num / den;
        if red > best.1 {
            best = (j, red);
        }
    }
    if best.0 == usize::MAX {
        None
    } else {
        Some(best.0)
    }
}

/// Encode `signal` losslessly with multi-reference cross-channel spatial
/// prediction. Host-only.
#[cfg(feature = "encode")]
pub fn encode(signal: &[Vec<i64>]) -> LmlResult<Vec<u8>> {
    let base = encode_with_geometry(signal, None)?;
    // Keep-best LSB bit-plane split: only when a low bit-plane is biased, and only if
    // it shrinks the body. s=0 (no biased low bits) returns `base` unchanged ⇒ never-worse.
    let s = lsb_bias_count(signal);
    if s == 0 {
        return Ok(base);
    }
    match encode_lsb_split(signal, s) {
        Ok(split) if split.len() < base.len() => Ok(split),
        _ => Ok(base),
    }
}

/// Number of consecutive low bit-planes biased enough to be worth splitting off
/// (per-bit entropy < 0.92). 0 ⇒ no split. Cheap pre-check so the (expensive) split
/// encode is only attempted when a free win is plausible.
#[cfg(feature = "encode")]
fn lsb_bias_count(signal: &[Vec<i64>]) -> usize {
    let mut s = 0usize;
    while s < MAX_SPLIT {
        let (mut n0, mut n1) = (0u64, 0u64);
        for ch in signal {
            for &x in ch {
                if (x >> s) & 1 == 0 { n0 += 1; } else { n1 += 1; }
            }
        }
        let n = n0 + n1;
        if n == 0 { break; }
        let p1 = n1 as f64 / n as f64;
        let h = if p1 <= 0.0 || p1 >= 1.0 { 0.0 } else { -(p1 * p1.log2() + (1.0 - p1) * (1.0 - p1).log2()) };
        if h > 0.92 { break; }
        s += 1;
    }
    s
}

/// Encode with the low `s` bit-planes split off and coded separately. The upper signal
/// (`x >> s`) goes through the normal cross-channel + keep-best pipeline; each channel's
/// LSB plane (`x & ((1<<s)-1)`) is entropy-coded on its own (per-channel so each stream
/// stays under golomb's u16 length cap). Mirrors `decode_lsb_split`'s framing exactly.
#[cfg(feature = "encode")]
fn encode_lsb_split(signal: &[Vec<i64>], s: usize) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    let t = signal[0].len();
    let mask = (1i64 << s) - 1;
    let upper: Vec<Vec<i64>> = signal.iter().map(|ch| ch.iter().map(|&x| x >> s).collect()).collect();
    let upper_body = encode_with_geometry(&upper, None)?;
    let mut out = Vec::with_capacity(12 + upper_body.len());
    out.push(SPLIT_MAGIC);
    out.push(s as u8);
    out.extend_from_slice(&(n_ch as u16).to_le_bytes());
    out.extend_from_slice(&(t as u32).to_le_bytes());
    out.extend_from_slice(&(upper_body.len() as u32).to_le_bytes());
    out.extend_from_slice(&upper_body);
    for ch in signal {
        let lsb_ch: Vec<i64> = ch.iter().map(|&x| x & mask).collect();
        let b = crate::entropy::encode(&lsb_ch)?;
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(&b);
    }
    Ok(out)
}

/// Like [`encode`], but a montage `geom` (per-channel electrode coords, encode-only,
/// NEVER serialized) seeds the per-channel cross-channel reference search with the
/// geometrically nearest prior electrodes as one more keep-smaller candidate. The
/// chosen `(ref, gain)` pairs are written exactly as today, so the wire format and the
/// `no_std` decoder are unchanged; geometry-free (`geom = None`) is byte-identical to
/// [`encode`]. (Phase A1, `eeg-codec-design-from-port` §2 Stage 1.)
#[cfg(feature = "encode")]
pub fn encode_with_geometry(
    signal: &[Vec<i64>],
    geom: Option<&crate::montage::MontageGeometry>,
) -> LmlResult<Vec<u8>> {
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

    let mut metas: Vec<(Vec<usize>, Vec<i32>)> = Vec::with_capacity(n_ch);
    let mut resid_signal: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for i in 0..n_ch {
        let raw_cost = channel_cost(&signal[i]);
        let mut best_cost = raw_cost;
        let mut best_refs: Vec<usize> = Vec::new();
        let mut best_gains: Vec<i32> = Vec::new();
        let mut best_resid = signal[i].clone();

        // (1) byte-greedy SINGLE reference (never regresses the single-best codec).
        for j in 0..i {
            let g = joint_ls(&signal[i], &[j], signal);
            let Some(gq) = quantize_gains(&g) else { continue };
            let r = residual_multi(&signal[i], &[j], &gq, signal);
            let cost = channel_cost(&r).saturating_add(PER_REF_OVERHEAD + 1);
            if cost < best_cost {
                best_cost = cost;
                best_refs = alloc::vec![j];
                best_gains = gq;
                best_resid = r;
            }
        }

        // (2) energy-greedy ADD references while they shrink the channel.
        while best_refs.len() < MAX_REFS.min(i) && !best_refs.is_empty() {
            let Some(j) = best_energy_ref(&best_resid, signal, i, &best_refs) else { break };
            let mut refs = best_refs.clone();
            refs.push(j);
            let g = joint_ls(&signal[i], &refs, signal);
            let Some(gq) = quantize_gains(&g) else { break };
            let r = residual_multi(&signal[i], &refs, &gq, signal);
            let cost = channel_cost(&r).saturating_add(refs.len() * PER_REF_OVERHEAD + 1);
            if cost < best_cost {
                best_cost = cost;
                best_refs = refs;
                best_gains = gq;
                best_resid = r;
            } else {
                break;
            }
        }

        // (3) GEOMETRY candidate (encode-only search prior; never serialized): the
        // montage-nearest prior electrodes as one keep-smaller reference set. Geometry
        // captures cross-channel coupling the energy-greedy heuristic can miss on the
        // multi-reference tail; keep it only if it shrinks the channel.
        if let Some(g) = geom {
            let geo_refs = g.nearest_prior(i, MAX_REFS.min(i));
            if !geo_refs.is_empty() && geo_refs != best_refs {
                let gv = joint_ls(&signal[i], &geo_refs, signal);
                if let Some(gq) = quantize_gains(&gv) {
                    let r = residual_multi(&signal[i], &geo_refs, &gq, signal);
                    let cost = channel_cost(&r).saturating_add(geo_refs.len() * PER_REF_OVERHEAD + 1);
                    if cost < best_cost {
                        best_refs = geo_refs;
                        best_gains = gq;
                        best_resid = r;
                    }
                }
            }
        }

        metas.push((best_refs, best_gains));
        resid_signal.push(best_resid);
    }

    let assemble = |metas: &[(Vec<usize>, Vec<i32>)], coder_mode: u8, stream: &[u8]| -> Vec<u8> {
        let mut body = Vec::with_capacity(5 + n_ch + stream.len());
        body.push(KERNEL_VERSION);
        body.push(FEATURE_CROSSCHAN);
        body.extend_from_slice(&(n_ch as u16).to_le_bytes());
        for (refs, gains) in metas {
            body.push(refs.len() as u8);
            for (&r, &g) in refs.iter().zip(gains) {
                body.extend_from_slice(&(r as u16).to_le_bytes());
                body.extend_from_slice(&g.to_le_bytes());
            }
        }
        body.push(coder_mode);
        body.extend_from_slice(stream);
        body
    };

    // Candidate A: cross-channel prediction + keep-best{lml, RLS} on the residual.
    let lml_stream = lml::compress(&resid_signal, 0)?;
    let (cc_coder, cc_stream) = match crate::rls::encode(&resid_signal) {
        Ok(r) if r.len() < lml_stream.len() => (CODER_RLS, r),
        _ => (CODER_LML, lml_stream),
    };
    let body_cc = assemble(&metas, cc_coder, &cc_stream);

    // Candidate B: RLS directly on the ORIGINAL signal (NO cross-channel). On
    // non-stationary signals temporal RLS adaptation beats spatial decorrelation
    // (the ma 21ch case: cross-channel gets selected but sabotages RLS). All
    // channels are coded "raw" (no refs), so decode reconstructs them directly.
    let raw_metas: Vec<(Vec<usize>, Vec<i32>)> = (0..n_ch).map(|_| (Vec::new(), Vec::new())).collect();
    let mut body = body_cc;
    if let Ok(raw_rls) = crate::rls::encode(signal) {
        let cand = assemble(&raw_metas, CODER_RLS, &raw_rls);
        if cand.len() < body.len() {
            body = cand;
        }
    }
    // Candidate C: multivariate cross-channel RLS on the raw signal — wins on hard
    // non-stationary high-amplitude EEG where the static best-of-prior collapses.
    if let Ok(mv) = crate::mv_rls::encode(signal) {
        let cand = assemble(&raw_metas, CODER_MV_RLS, &mv);
        if cand.len() < body.len() {
            body = cand;
        }
    }
    // Candidate D: per-channel RLS with change-point segmentation (Lever C) — resets
    // the predictor at signal-derived regime boundaries (seizure onset, artefact),
    // defeating HHI's fixed IntraPeriod. Never-worse keep-best (seg-off is the plain
    // CODER_RLS candidate above; this only wins when boundary-resets help).
    if let Ok(seg) = crate::rls::encode_seg(signal) {
        let cand = assemble(&raw_metas, CODER_RLS_SEG, &seg);
        if cand.len() < body.len() {
            body = cand;
        }
    }
    Ok(body)
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn lsb_split_triggers_shrinks_and_roundtrips() {
        // bit0 always 0 (all even) + structured upper ⇒ scale_cond would code the
        // dead LSB raw; the split must strip it, shrink, and round-trip bit-exact.
        let mut signal = vec![vec![0i64; 3000]; 5];
        for c in 0..5 {
            for n in 0..3000 {
                signal[c][n] = (((n as i64 * 37 + c as i64 * 401) % 400) - 200) * 2;
            }
        }
        assert_eq!(lsb_bias_count(&signal), 1, "all-even ⇒ 1 biased low bit");
        let base = encode_with_geometry(&signal, None).unwrap();
        let best = encode(&signal).unwrap();
        assert_eq!(best[0], SPLIT_MAGIC, "split should be selected");
        assert!(best.len() < base.len(), "split {} must beat base {}", best.len(), base.len());
        assert_eq!(decode(&best).unwrap(), signal, "split must round-trip bit-exact");
    }

    #[test]
    fn lsb_split_absent_when_lsb_clean() {
        // full-entropy low bits ⇒ no split ⇒ byte-identical to the base (never-worse).
        let mut signal = vec![vec![0i64; 2000]; 4];
        let mut st = 0x1234_5678u64;
        for ch in signal.iter_mut() {
            for x in ch.iter_mut() {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                *x = (st as i64 % 4000) - 2000;
            }
        }
        assert_eq!(lsb_bias_count(&signal), 0, "random low bits ⇒ no split");
        assert_eq!(encode(&signal).unwrap(), encode_with_geometry(&signal, None).unwrap());
    }

    /// Correlated multichannel signal: each channel is a gain·(shared base) +
    /// per-channel detail, so cross-channel prediction has real redundancy.
    fn make_corr_signal(n_ch: usize, t: usize) -> Vec<Vec<i64>> {
        let base: Vec<i64> = (0..t).map(|i| ((i as f64 * 0.05).sin() * 3000.0) as i64).collect();
        let base2: Vec<i64> = (0..t).map(|i| ((i as f64 * 0.013).cos() * 1500.0) as i64).collect();
        (0..n_ch)
            .map(|c| {
                let g = 0.6 + 0.1 * c as f64;
                let g2 = 0.3 - 0.02 * c as f64;
                (0..t)
                    .map(|i| {
                        let detail = (((i + c * 7) as f64 * 0.9).sin() * 120.0) as i64;
                        (g * base[i] as f64) as i64 + (g2 * base2[i] as f64) as i64 + detail
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
    fn two_basis_signal_beats_floor_and_round_trips() {
        // Channels mix TWO shared bases. This test previously asserted that the
        // FINAL serialized body carried ≥2 cross-channel refs. That invariant no
        // longer holds: `entropy::encode` gained the scale-conditioned adaptive
        // coder (ADR 0054 Phase A), and on these smooth signals the *temporal*
        // RLS candidates (B/C) now code strictly smaller than the cross-channel
        // candidate, so the never-worse keep-best legitimately serializes 0 refs
        // (the body SHRINKS 8291→7555 bytes — a strict win, not a regression).
        // The cross-channel multi-ref path is still computed as a keep-best
        // candidate; what stays externally observable is that the codec (a) beats
        // the per-channel lml floor by exploiting the shared structure and (b)
        // round-trips bit-exact whichever candidate wins.
        let sig = make_corr_signal(10, 3000);
        let body = encode(&sig).unwrap();
        let floor = lml::compress(&sig, 0).unwrap().len();
        assert!(body.len() < floor, "should beat lml floor {floor}, got {}", body.len());
        assert_eq!(decode(&body).unwrap(), sig, "encode/decode must be bit-exact");
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
        let sig = vec![(0..600).map(|i| ((i * 13) % 91) as i64 - 45).collect::<Vec<i64>>()];
        let body = encode(&sig).unwrap();
        assert_eq!(decode(&body).unwrap(), sig);
    }

    /// A1 guarantee: geometry-free `encode_with_geometry` is BYTE-IDENTICAL to `encode`
    /// (no regression / no wire change), and a geometry-seeded body still roundtrips
    /// bit-exact through the unchanged decoder (geometry is an encoder-only search prior).
    #[test]
    fn geometry_free_byte_identical_and_geometry_roundtrips() {
        use crate::montage::MontageGeometry;
        for (n_ch, t) in [(8usize, 2048usize), (16, 1500)] {
            let sig = make_corr_signal(n_ch, t);
            assert_eq!(
                encode(&sig).unwrap(),
                encode_with_geometry(&sig, None).unwrap(),
                "geometry-free encode must be byte-identical to encode()"
            );
            // a synthetic line montage (channel c at x=c) ⇒ nearest = adjacent channels
            let coords: Vec<Option<[f64; 3]>> = (0..n_ch).map(|c| Some([c as f64, 0.0, 0.0])).collect();
            let geom = MontageGeometry::new(coords);
            let body = encode_with_geometry(&sig, Some(&geom)).unwrap();
            assert_eq!(decode(&body).unwrap(), sig, "geometry body must roundtrip bit-exact");
        }
    }
}
