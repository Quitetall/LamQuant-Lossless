//! Peer-stack step 1 — mine H.BWC's affine cross-channel Linear Model (empirically +3.45% referential
//! in the H.BWC ablation) into our TIME-domain codec.
//!
//! H.BWC's LM is a per-block backward-adaptive least-squares `cur ≈ scale·ref + OFFSET` against a
//! signaled best-reference channel, applied in the sample domain. Our mv_rls cross-channel blends
//! index-adjacent channels with a gain-like predictor but NO explicit DC-OFFSET term, and every prior
//! crosschan probe (M1/M2/M3) used pure gain. This probe isolates the one thing we never tried: the
//! OFFSET. Second stage on the best-config mv_rls residual, per-block backward-adaptive (fit on the
//! PREVIOUS block ⇒ zero side-info for the coefficients; only the ref index is signaled), keep-best per
//! channel over {raw, gain-only, affine(gain+offset)}. The gain-only arm reproduces M1 (per-block);
//! the affine arm adds the offset. Decisive question: does the OFFSET beat gain-only on referential,
//! and does either beat the shipped container?
//!
//! Baselines: CONT = shipped LmoCodec Lossless container (the bar); B0 = best-config mv_rls residual.
//! Reversibility is structural (integer combo of already-decoded earlier-channel exact residuals +
//! the current channel's already-reconstructed previous block ⇒ decoder re-derives (g,off) identically).
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example affine_xchan_probe -- [BLK] <bin>...

use std::fs;
use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{entropy, mv_rls, LmoCodec};

const WIN: usize = 32768;
const N_CFG: usize = 7;
const REF_SIDEINFO: usize = 2; // bytes/channel: ref index + scheme selector

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut s = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64);
            off += 4;
        }
        s.push(ch);
    }
    s
}

#[inline]
fn enc_len(v: &[i64]) -> usize {
    entropy::encode(v).map(|g| g.len()).expect("entropy encode")
}

fn enc_windowed(v: &[i64]) -> usize {
    let mut tot = 0;
    let mut s = 0;
    while s < v.len() {
        let e = (s + WIN).min(v.len());
        tot += enc_len(&v[s..e]);
        s = e;
    }
    tot
}

fn container_bytes(sig: &[Vec<i64>]) -> usize {
    let t = sig[0].len();
    let (mut tot, mut start) = (0, 0);
    while start < t {
        let end = (start + WIN).min(t);
        let win: Vec<Vec<i64>> = sig.iter().map(|c| c[start..end].to_vec()).collect();
        tot += LmoCodec.encode(&win, Mode::Lossless).map(|x| x.len()).unwrap_or(0);
        start = end;
    }
    tot
}

fn best_config_residual(sig: &[Vec<i64>]) -> (Vec<Vec<i64>>, usize) {
    let (mut best_res, mut best_bytes) = (Vec::new(), usize::MAX);
    for cfg in 0..N_CFG {
        let res = mv_rls::residuals(sig, cfg, 0);
        let bytes: usize = res.iter().map(|c| enc_windowed(c)).sum();
        if bytes < best_bytes {
            best_bytes = bytes;
            best_res = res;
        }
    }
    (best_res, best_bytes)
}

fn abscorr(a: &[i64], b: &[i64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<i64>() as f64 / n, b.iter().sum::<i64>() as f64 / n);
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 { 0.0 } else { (sab / (saa * sbb).sqrt()).abs() }
}

/// Per-block backward-adaptive cross-channel prediction: fit (g[,off]) on the PREVIOUS block, apply to
/// the current block. `use_offset=false` ⇒ gain-only (M1-style); true ⇒ affine (H.BWC LM-style).
fn xchan_residual(rc: &[i64], rj: &[i64], blk: usize, use_offset: bool) -> Vec<i64> {
    let n = rc.len();
    let mut out = rc.to_vec();
    let mut m = 1;
    while m * blk < n {
        let (ps, pe) = ((m - 1) * blk, m * blk); // previous block: fit
        let (cs, ce) = (m * blk, ((m + 1) * blk).min(n)); // current block: apply
        let cnt = (pe - ps) as f64;
        let (mut sc, mut sj, mut scj, mut sjj) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in ps..pe {
            let (x, y) = (rc[i] as f64, rj[i] as f64);
            sc += x;
            sj += y;
            scj += x * y;
            sjj += y * y;
        }
        let (mc, mj) = (sc / cnt, sj / cnt);
        let varj = sjj / cnt - mj * mj;
        let g = if varj > 1e-9 { (scj / cnt - mc * mj) / varj } else { 0.0 };
        let off = if use_offset { mc - g * mj } else { 0.0 };
        for i in cs..ce {
            let pred = (g * rj[i] as f64 + off).round() as i64;
            out[i] = rc[i] - pred;
        }
        m += 1;
    }
    out
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cli_b = args.first().and_then(|a| a.parse::<usize>().ok());
    if cli_b.is_some() {
        args.remove(0);
    }
    let blocks: Vec<usize> = cli_b.map(|b| vec![b]).unwrap_or_else(|| vec![16, 32, 64, 128]);

    println!("# Peer-stack step 1: affine cross-channel LM (mine H.BWC's +3.45% referential lever).");
    println!("# 2nd stage on best-config mv_rls residual, per-block backward-adaptive, keep-best/channel");
    println!("# over {{raw, gain-only, affine(+offset)}}. CONT=shipped container; B0=raw mv_rls residual.");
    println!("# GAIN = keep-best raw-vs-gain (≈M1); AFFINE = keep-best raw-vs-affine; ORA = raw/gain/affine.\n");
    println!("{:>12} {:>4} {:>8} {:>8} | {:>9} {:>9} {:>9} | {:>9}",
             "recording", "BLK", "CONT bps", "B0 bps", "GAIN/CONT", "AFFN/CONT", "ORA/CONT", "AFFN/B0");

    for path in &args {
        let sig = read_bin(path);
        let (c, t) = (sig.len(), sig[0].len());
        let nm = (c * t) as f64;
        let cont = container_bytes(&sig);
        let (res, b0) = best_config_residual(&sig);

        // best-correlated earlier reference per channel (channel-major ⇒ j<c decodable)
        let bestref: Vec<Option<usize>> = (0..c)
            .map(|ci| {
                (0..ci)
                    .map(|j| (j, abscorr(&res[ci], &res[j])))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(j, _)| j)
            })
            .collect();

        for &blk in &blocks {
            let (mut gain, mut affine, mut ora) = (0usize, 0usize, 0usize);
            for ci in 0..c {
                let base = enc_windowed(&res[ci]);
                let (mut cg, mut ca) = (base, base);
                if let Some(j) = bestref[ci] {
                    cg = base.min(enc_windowed(&xchan_residual(&res[ci], &res[j], blk, false)) + REF_SIDEINFO);
                    ca = base.min(enc_windowed(&xchan_residual(&res[ci], &res[j], blk, true)) + REF_SIDEINFO);
                }
                gain += cg;
                affine += ca;
                ora += cg.min(ca);
            }
            let pct = |x: usize| 100.0 * (x as f64 - cont as f64) / cont as f64;
            let affn_vs_b0 = 100.0 * (affine as f64 - b0 as f64) / b0 as f64;
            let name = path.rsplit('/').next().unwrap_or(path);
            println!("{name:>12} {blk:>4} {:>8.4} {:>8.4} | {:>+8.3}% {:>+8.3}% {:>+8.3}% | {:>+8.3}%",
                     cont as f64 * 8.0 / nm, b0 as f64 * 8.0 / nm, pct(gain), pct(affine), pct(ora), affn_vs_b0);
        }
    }
    println!("\n# AFFN/CONT ≤ 0 on referential (siena/eegmmidb/tuar) ⇒ the affine-offset cross-channel LM is");
    println!("# a real deterministic win over production ⇒ integrate as a keep-best candidate (peer-stack).");
    println!("# AFFN vs GAIN isolates the OFFSET's value; AFFN/B0 = the pure 2nd-stage gain over raw mv_rls.");
}
