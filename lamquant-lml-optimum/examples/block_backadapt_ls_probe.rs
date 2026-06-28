//! SHOT 1 (ADR 0064) — per-block BACKWARD-ADAPTIVE cross-channel LS.
//!
//! H.BWC's referential edge is a per-block (16-sample template) backward-adaptive
//! least-squares cross-channel predictor with 1-2 refs and ZERO side-info: the
//! decoder re-derives the coefficients from already-reconstructed context
//! (`bwc-crosschan/src/lib.rs:43,70-73`). It sits in the UNTRIED MIDDLE of our
//! adaptation-rate axis — between candidate A (per-WINDOW static LS, too rigid)
//! and mv_rls (per-SAMPLE RLS, the current best). This probe settles, directly,
//! whether that middle captures anything mv_rls misses.
//!
//! Mechanism: process channels in causal order; for channel i pick the R most-
//! correlated PRIOR channels (global covariance, fixed per recording); per block
//! of B samples, fit LS coefficients on the PREVIOUS block (the backward template,
//! reconstructed == original in lossless ⇒ zero side-info), apply to the current
//! block, code the integer residual. Bit-exact: decoder re-runs the same fit on
//! its reconstructed history. Round-trip gated.
//!
//! Decisive comparison: per-block-LS residual bytes vs (a) no-spatial floor
//! (LmlCodec — does it decorrelate?), (b) mv_rls container base (does it beat the
//! adaptive predictor?), (c) HHI oracle. Strong prior: mv_rls (per-sample) ≥
//! per-block; this proves it on a number.
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example block_backadapt_ls_probe -- <bin>...

use std::fs;
use lamquant_lml_mcu::codec::{Codec, LmlCodec, Mode};
use lamquant_lml_optimum::{entropy, LmoCodec};

const W: usize = 32768; // entropy-coding window (golomb u16 cap)

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut s = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t { ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64); off += 4; }
        s.push(ch);
    }
    s
}

#[inline]
fn rnd(x: f64) -> i64 { x.round() as i64 }

/// The R prior channels (< i) most correlated with channel i by |covariance|.
fn pick_refs(sig: &[Vec<i64>], i: usize, r: usize) -> Vec<usize> {
    let t = sig[i].len();
    let mu_i = sig[i].iter().map(|&v| v as f64).sum::<f64>() / t as f64;
    let mut scored: Vec<(f64, usize)> = (0..i).map(|j| {
        let mu_j = sig[j].iter().map(|&v| v as f64).sum::<f64>() / t as f64;
        let mut cov = 0.0;
        for k in 0..t { cov += (sig[i][k] as f64 - mu_i) * (sig[j][k] as f64 - mu_j); }
        (cov.abs(), j)
    }).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.into_iter().take(r).map(|(_, j)| j).collect()
}

/// Solve the R×R normal equations (LS) on a template window for predicting
/// `tgt` from `refs` (small R, plain Gaussian elimination). Returns coeffs or None.
fn fit_ls(tgt: &[i64], refs: &[&[i64]], lo: usize, hi: usize) -> Option<Vec<f64>> {
    let r = refs.len();
    if r == 0 || hi <= lo { return None; }
    let mut ata = vec![vec![0.0f64; r]; r];
    let mut atb = vec![0.0f64; r];
    for k in lo..hi {
        for a in 0..r {
            let xa = refs[a][k] as f64;
            atb[a] += xa * tgt[k] as f64;
            for b in 0..r { ata[a][b] += xa * refs[b][k] as f64; }
        }
    }
    // tiny ridge for conditioning
    let tr: f64 = (0..r).map(|d| ata[d][d]).sum::<f64>() / r as f64;
    for d in 0..r { ata[d][d] += 1e-6 * tr.max(1.0); }
    // Gaussian elimination
    for c in 0..r {
        let mut piv = c;
        for rr in (c + 1)..r { if ata[rr][c].abs() > ata[piv][c].abs() { piv = rr; } }
        if ata[piv][c].abs() < 1e-12 { return None; }
        ata.swap(c, piv); atb.swap(c, piv);
        for rr in (c + 1)..r {
            let f = ata[rr][c] / ata[c][c];
            for cc in c..r { ata[rr][cc] -= f * ata[c][cc]; }
            atb[rr] -= f * atb[c];
        }
    }
    let mut x = vec![0.0f64; r];
    for c in (0..r).rev() {
        let mut s = atb[c];
        for cc in (c + 1)..r { s -= ata[c][cc] * x[cc]; }
        x[c] = s / ata[c][c];
    }
    Some(x)
}

/// Per-block backward-adaptive LS residual for one channel. Coefficients fit on the
/// PREVIOUS block [s-B, s); applied to the current block [s, e). First block: raw.
/// Returns (residual, round_trip_ok).
fn block_backadapt_residual(tgt: &[i64], refs: &[&[i64]], b: usize) -> (Vec<i64>, bool) {
    let t = tgt.len();
    let mut res = vec![0i64; t];
    let mut s = 0;
    while s < t {
        let e = (s + b).min(t);
        if s < b {
            // no full backward template yet → code raw
            for k in s..e { res[k] = tgt[k]; }
        } else {
            let coeffs = fit_ls(tgt, refs, s - b, s);
            match coeffs {
                Some(c) => {
                    for k in s..e {
                        let mut pred = 0.0;
                        for (a, rf) in refs.iter().enumerate() { pred += c[a] * rf[k] as f64; }
                        res[k] = tgt[k] - rnd(pred);
                    }
                }
                None => { for k in s..e { res[k] = tgt[k]; } }
            }
        }
        s = e;
    }
    // decode-side reconstruction (refs are already-decoded == original in lossless)
    let mut rec = vec![0i64; t];
    let mut ok = true;
    let mut s = 0;
    while s < t {
        let e = (s + b).min(t);
        if s < b {
            for k in s..e { rec[k] = res[k]; }
        } else {
            let coeffs = fit_ls(&rec, refs, s - b, s); // rec[..s] == tgt[..s] if ok so far
            match coeffs {
                Some(c) => for k in s..e {
                    let mut pred = 0.0;
                    for (a, rf) in refs.iter().enumerate() { pred += c[a] * rf[k] as f64; }
                    rec[k] = res[k] + rnd(pred);
                },
                None => for k in s..e { rec[k] = res[k]; },
            }
        }
        s = e;
    }
    if rec != tgt { ok = false; }
    (res, ok)
}

fn coded_bytes<C: Codec>(codec: &C, sig: &[Vec<i64>]) -> usize {
    let t = sig[0].len();
    let mut tot = 0; let mut s = 0;
    while s < t {
        let e = (s + W).min(t);
        let win: Vec<Vec<i64>> = sig.iter().map(|ch| ch[s..e].to_vec()).collect();
        tot += codec.encode(&win, Mode::Lossless).map(|x| x.len()).unwrap_or(0);
        s = e;
    }
    tot
}

/// entropy::encode the residual per 32768-window (golomb u16 cap), summed.
fn entropy_bytes(res: &[i64]) -> usize {
    let mut tot = 0; let mut s = 0;
    while s < res.len() {
        let e = (s + W).min(res.len());
        tot += entropy::encode(&res[s..e]).map(|g| g.len()).unwrap_or(1 << 30);
        s = e;
    }
    tot
}

fn main() {
    println!("# SHOT 1 — per-block backward-adaptive cross-channel LS (HHI's referential mechanism)\n");
    let blocks = [8usize, 16, 32, 64, 128, 256];
    let refcnts = [1usize, 2];
    for path in std::env::args().skip(1) {
        let sig = read_bin(&path);
        let (nch, t) = (sig.len(), sig[0].len());
        let name = path.rsplit('/').next().unwrap_or(&path);
        if nch < 2 { println!("# {} ({}ch) skip", name, nch); continue; }
        let nm = (nch * t) as f64;
        let base = coded_bytes(&LmoCodec, &sig);   // mv_rls container (current best)
        let floor = coded_bytes(&LmlCodec, &sig);  // no-spatial floor
        println!("## {} ({}ch x {})", name, nch, t);
        println!("  LmoCodec base (mv_rls) = {:>10} ({:.4} bps)   |   LmlCodec floor = {:>10} ({:.4} bps)",
                 base, base as f64 * 8.0 / nm, floor, floor as f64 * 8.0 / nm);
        // precompute refs per channel for max R
        let refs_all: Vec<Vec<usize>> = (0..nch).map(|i| pick_refs(&sig, i, 2)).collect();
        println!("  {:<8} {:<5} {:>12} {:>9} {:>11} {:>11} {:>6}", "block", "refs", "bytes", "bps", "vs floor", "vs base", "rt");
        // Two coders of the spatial residual: ENTROPY-only (spatial decorrelation
        // alone) and FLOOR (LmlCodec = 5/3+LPC+entropy ⇒ spatial-THEN-temporal,
        // the fair cascade vs HHI's spatial+temporal). entropy-only exposes how
        // much pure spatial prediction removes; floor is the real comparison.
        for &r in &refcnts {
            for &b in &blocks {
                let mut ent_tot = 0usize;
                let mut allok = true;
                let mut resid_sig: Vec<Vec<i64>> = Vec::with_capacity(nch);
                for i in 0..nch {
                    let refs_idx: Vec<usize> = refs_all[i].iter().cloned().take(r).collect();
                    if refs_idx.is_empty() {
                        ent_tot += entropy_bytes(&sig[i]);
                        resid_sig.push(sig[i].clone()); // ch0: raw
                        continue;
                    }
                    let refs: Vec<&[i64]> = refs_idx.iter().map(|&j| sig[j].as_slice()).collect();
                    let (res, ok) = block_backadapt_residual(&sig[i], &refs, b);
                    allok &= ok;
                    ent_tot += entropy_bytes(&res);
                    resid_sig.push(res);
                }
                let cascade = coded_bytes(&LmlCodec, &resid_sig); // spatial-then-temporal
                let vfloor = 100.0 * (cascade as f64 - floor as f64) / floor as f64;
                let vbase = 100.0 * (cascade as f64 - base as f64) / base as f64;
                println!("  B={:<6} R={:<3}  cascade={:>11} ({:.4} bps) {:>+9.2}% vs floor {:>+9.2}% vs base  (ent-only {:>11}) rt={}",
                         b, r, cascade, cascade as f64 * 8.0 / nm, vfloor, vbase, ent_tot, if allok {"ok"} else {"FAIL"});
            }
        }
        println!();
    }
    println!("# GATE: a per-block-LS config beats mv_rls base (vs base < 0) WITHOUT win-set regression ⇒ build SHOT 1.");
    println!("# If it only beats the floor (decorrelates) but never the mv_rls base ⇒ dominated by per-sample RLS, mechanism proven.");
}
