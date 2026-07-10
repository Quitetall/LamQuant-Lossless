//! Peer-codec (no transform) — the decisive isolation: JOINT vs SEPARATED cross-channel prediction.
//!
//! Diagnosis (ADR 0085 follow-up): H.BWC's referential edge is cross-channel PREDICTION (the transform
//! never wins for EEG). Our mv_rls does it JOINTLY — one RLS over 8 own taps + up to 32 INDEX-ADJACENT
//! cross taps (40 coefficients estimated online, high variance on referential common-mode). H.BWC
//! SEPARATES it — a temporal intra predictor + a parsimonious per-block backward-adaptive cross-channel
//! predictor against 1-2 SELECTED best-correlated references. Every prior graft tested H.BWC's cross-
//! channel as a 2nd stage ON TOP of mv_rls (redundant). This tests it as the PRIMARY predictor.
//!
//! Same intra RLS in both arms (residuals_params m=…) ⇒ the ONLY variable is the cross-channel structure:
//!   JOINT     = residuals_params(sig, λ, reset, m=32, 0)        — our mv_rls (index-adjacent, joint).
//!   SEPARATED = per-block-B backward-adaptive gain vs the best-correlated EARLIER channel applied to the
//!               RAW signal first, THEN residuals_params(cross_resid, λ, reset, m=0, 0) intra-only RLS.
//! Both coded with our real coder; SEPARATED pays 1 ref-index byte/channel. Baselines: shipped container
//! (CONT) + JOINT. Sweep the cross-channel block size B (H.BWC uses a 16-sample template).
//!
//! GATE: SEPARATED < JOINT and → CONT on referential (siena/eegmmidb/tuar) ⇒ the parsimonious selected-
//! ref per-block cross-channel is the missing lever ⇒ a no-transform peer codec is viable. SEPARATED ≥
//! JOINT ⇒ our joint RLS already dominates ⇒ the referential gap is not this structure either.
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example separated_predict_probe -- [B] <bin>...

use std::fs;
use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{entropy, mv_rls, LmoCodec};

const WIN: usize = 32768;
// mv_rls CONFIGS[0] = (λ=0.999, reset=8192, m=32) — the joint default.
const LAMBDA: f64 = 0.999;
const RESET: usize = 8192;

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
    let (mut tot, mut s) = (0, 0);
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

/// Per-block-B backward-adaptive cross-channel gain vs `rj`: fit g on the PREVIOUS block, apply to the
/// current block (zero side-info for g; decoder re-derives). out = rc - round(g·rj).
fn cross_predict(rc: &[i64], rj: &[i64], blk: usize) -> Vec<i64> {
    let n = rc.len();
    let mut out = rc.to_vec();
    let mut m = 1;
    while m * blk < n {
        let (ps, pe) = ((m - 1) * blk, m * blk);
        let (cs, ce) = (m * blk, ((m + 1) * blk).min(n));
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in ps..pe {
            num += rc[i] as f64 * rj[i] as f64;
            den += (rj[i] as f64).powi(2);
        }
        let g = if den > 1e-9 { num / den } else { 0.0 };
        for i in cs..ce {
            out[i] = rc[i] - (g * rj[i] as f64).round() as i64;
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
    let blocks: Vec<usize> = cli_b.map(|b| vec![b]).unwrap_or_else(|| vec![16, 64, 256]);

    println!("# JOINT vs SEPARATED cross-channel (no transform). Same intra RLS both arms. CONT=container.");
    println!("# JOINT = mv_rls (index-adjacent, m=32, joint). SEP = per-block-B best-ref cross gain on RAW");
    println!("# THEN intra-only RLS (m=0). Δ vs CONT + SEP vs JOINT. B = cross-channel block (H.BWC ~16).\n");
    println!("{:>12} {:>4} {:>8} {:>8} {:>8} | {:>9} {:>9} {:>9}",
             "recording", "B", "CONT bps", "JOINT", "SEP", "JNT/CONT", "SEP/CONT", "SEP/JNT");

    for path in &args {
        let sig = read_bin(path);
        let (c, t) = (sig.len(), sig[0].len());
        let nm = (c * t) as f64;
        let cont = container_bytes(&sig);
        // JOINT: our mv_rls (m=32), coded.
        let joint = mv_rls::residuals_params(&sig, LAMBDA, RESET, 32, 0);
        let joint_bytes: usize = joint.iter().map(|ch| enc_windowed(ch)).sum();

        // best-correlated earlier reference per channel (on raw), channel-major ⇒ j<c
        let bestref: Vec<Option<usize>> = (0..c)
            .map(|ci| {
                (0..ci)
                    .map(|j| (j, abscorr(&sig[ci], &sig[j])))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(j, _)| j)
            })
            .collect();

        let cont_bps = cont as f64 * 8.0 / nm;
        let jnt_bps = joint_bytes as f64 * 8.0 / nm;

        for &b in &blocks {
            // SEPARATED: cross-predict raw per channel, then intra-only RLS on the cross-residual matrix.
            let cross: Vec<Vec<i64>> = (0..c)
                .map(|ci| match bestref[ci] {
                    Some(j) => cross_predict(&sig[ci], &sig[j], b),
                    None => sig[ci].clone(),
                })
                .collect();
            let sep_res = mv_rls::residuals_params(&cross, LAMBDA, RESET, 0, 0); // m=0 intra-only
            let sep_bytes: usize = sep_res.iter().map(|ch| enc_windowed(ch)).sum::<usize>() + c; // +1 refidx byte/ch

            let sep_bps = sep_bytes as f64 * 8.0 / nm;
            let jc = 100.0 * (joint_bytes as f64 - cont as f64) / cont as f64;
            let sc = 100.0 * (sep_bytes as f64 - cont as f64) / cont as f64;
            let sj = 100.0 * (sep_bytes as f64 - joint_bytes as f64) / joint_bytes as f64;
            let name = path.rsplit('/').next().unwrap_or(path);
            println!("{name:>12} {b:>4} {cont_bps:>8.4} {jnt_bps:>8.4} {sep_bps:>8.4} | {jc:>+8.3}% {sc:>+8.3}% {sj:>+8.3}%");
        }
    }
    println!("\n# SEP/JNT < 0 on referential ⇒ separated selected-ref per-block cross-channel BEATS our joint");
    println!("# RLS ⇒ the missing lever is prediction STRUCTURE (parsimony+selection), a no-transform peer");
    println!("# codec is viable. SEP/JNT ≥ 0 ⇒ our joint RLS already dominates ⇒ the gap is not this either.");
}
