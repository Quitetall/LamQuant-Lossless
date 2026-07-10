//! Phase-1b DE-RISK (Option A) — the OVERLAPPED transform: does H.BWC's *actual* transform close the
//! referential gap the non-overlapping DCT skeleton (dct_domain_predict_probe) could not?
//!
//! H.BWC's IntDCT is Geiger–Schuller "Improved integer transforms for lossless AUDIO coding" — the
//! audio lineage means an OVERLAPPED (MDCT-style) transform. Overlap is the one component the prior
//! probe proxied away, and the single most likely reason a non-overlapping clone underperformed real
//! H.BWC by 10-45%: 50% frame overlap preserves the cross-frame sample continuity that non-overlapping
//! block-DCT destroys, so cross-frame coefficient prediction should get much stronger.
//!
//! This probe: windowed 50%-overlap MDCT (sine / Princen–Bradley window, critically sampled: 2N-sample
//! frames, N-sample hop, N coefficients per frame ⇒ #coeffs ≈ #samples), float-rounded (optimistic
//! entropy proxy for the reversible IntMDCT, which costs ~1-3% more) → per-frequency mv_rls prediction
//! (own past FRAMES = intra, cross-channel present = inter) → real entropy coder + IDEAL pooled coarse-
//! context entropy (CABAC-class ceiling). Baselines: shipped container (CONT, the bar) + best-config
//! TIME-domain mv_rls (B0). NB MDCT has NO pure-DC bin (half-integer freqs k+0.5) — EEG drift energy
//! spreads across low-freq coeffs; that is a real property this measures, not a bug.
//!
//! GATE: MDCT ≤ CONT on the referential lose-set (siena/eegmmidb/tuar) ⇒ the deterministic transform-
//! domain equal-codec is VIABLE (build the real IntMDCT). Still positive under this optimistic proxy ⇒
//! the deterministic transform path is retired; route to the learned-adaptive layer.
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example mdct_domain_predict_probe -- [N] <bin>...

use std::collections::HashMap;
use std::fs;

use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{entropy, mv_rls, LmoCodec};

const WIN: usize = 32768;
const N_CFG: usize = 7;

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    assert!(b.len() >= 8, "truncated file {path}");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    assert!(b.len() >= 8 + nch * t * 4, "truncated file {path}");
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

/// MDCT analysis basis with the sine (Princen–Bradley) window folded in: cbasis[k][n], k∈[0,N),
/// n∈[0,2N). X[k] = Σ_n cbasis[k][n] · x[frame_start + n]. Orthonormal scale sqrt(2/N).
/// X[k] = sqrt(2/N) Σ_n w[n]·x[n]·cos( (π/N)(n + 0.5 + N/2)(k + 0.5) ),  w[n] = sin( (π/2N)(n+0.5) ).
fn mdct_basis(n: usize) -> Vec<Vec<f64>> {
    let nn = n as f64;
    let scale = (2.0 / nn).sqrt();
    let n0 = 0.5 + nn / 2.0;
    let mut m = vec![vec![0.0f64; 2 * n]; n];
    for k in 0..n {
        let kf = k as f64 + 0.5;
        for i in 0..(2 * n) {
            let w = ((core::f64::consts::PI / (2.0 * nn)) * (i as f64 + 0.5)).sin();
            m[k][i] = scale * w * ((core::f64::consts::PI / nn) * (i as f64 + n0) * kf).cos();
        }
    }
    m
}

fn container_bytes(sig: &[Vec<i64>]) -> usize {
    let t = sig[0].len();
    let mut tot = 0;
    let mut start = 0;
    while start < t {
        let end = (start + WIN).min(t);
        let win: Vec<Vec<i64>> = sig.iter().map(|c| c[start..end].to_vec()).collect();
        tot += LmoCodec.encode(&win, Mode::Lossless).map(|x| x.len()).unwrap_or(0);
        start = end;
    }
    tot
}

fn best_config_bytes(sig: &[Vec<i64>]) -> usize {
    let mut best = usize::MAX;
    for cfg in 0..N_CFG {
        let res = mv_rls::residuals(sig, cfg, 0);
        let bytes: usize = res.iter().map(|ch| enc_windowed(ch)).sum();
        best = best.min(bytes);
    }
    best
}

#[inline]
fn magbucket(x: i64) -> u64 {
    (64 - x.unsigned_abs().leading_zeros()).min(15) as u64
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cli_n = args.first().and_then(|a| a.parse::<usize>().ok());
    if cli_n.is_some() {
        args.remove(0);
    }
    let blocks: Vec<usize> = cli_n.map(|n| vec![n]).unwrap_or_else(|| vec![256, 512, 1024]);

    println!("# Phase-1b: OVERLAPPED-MDCT (H.BWC's actual transform) + DCT-domain prediction vs container.");
    println!("# N = MDCT coeffs/frame (2N-sample frame, N hop, 50% overlap). CONT = shipped container bps;");
    println!("# B0 = best-config TIME-domain mv_rls. MDCTpred = real coder; MDCTctx = ideal pooled coarse-");
    println!("# context ceiling. Δ vs CONT, per-sample; negative = beats production. Proxy is OPTIMISTIC.\n");
    println!("{:>12} {:>5} {:>8} {:>8} | {:>9} {:>9} | {:>9} {:>9}",
             "recording", "N", "CONT bps", "B0 bps", "MDCTpred", "Δ/CONT", "MDCTctx", "Δ/CONT");

    for path in &args {
        let sig = read_bin(path);
        let (c, t) = (sig.len(), sig[0].len());
        let cont = container_bytes(&sig);
        let b0 = best_config_bytes(&sig);
        let cont_bps = cont as f64 * 8.0 / (c * t) as f64;
        let b0_bps = b0 as f64 * 8.0 / (c * t) as f64;

        for &n in &blocks {
            if t < 3 * n {
                continue;
            }
            let basis = mdct_basis(n);
            let frames = t / n - 1; // frame f reads [f*n, f*n+2n); need (frames-1)*n+2n <= t
            let coeff_nm = (c * frames * n) as f64; // critically sampled ⇒ #coeffs ≈ #samples

            // MDCT: coef[c][f][k]
            let mut coef = vec![vec![vec![0i64; n]; frames]; c];
            for ci in 0..c {
                for f in 0..frames {
                    let base = f * n;
                    for k in 0..n {
                        let row = &basis[k];
                        let mut acc = 0.0f64;
                        for i in 0..(2 * n) {
                            acc += row[i] * sig[ci][base + i] as f64;
                        }
                        coef[ci][f][k] = acc.round() as i64;
                    }
                }
            }
            // per frequency k: mv_rls over C×frames (own past frames = intra + cross-channel present =
            // inter), gather frequency-major per channel.
            let mut resid: Vec<Vec<i64>> = vec![Vec::with_capacity(n * frames); c];
            for k in 0..n {
                let coefk: Vec<Vec<i64>> =
                    (0..c).map(|ci| (0..frames).map(|f| coef[ci][f][k]).collect()).collect();
                let resk = mv_rls::residuals(&coefk, 0, 0);
                for ci in 0..c {
                    resid[ci].extend_from_slice(&resk[ci]);
                }
            }
            // real coder, frequency-major, windowed
            let mdct_enc: usize = (0..c).map(|ci| enc_windowed(&resid[ci])).sum();
            // ideal pooled coarse-context (log-freq band × prev-frame-same-freq magnitude), across channels
            let mut pooled: HashMap<u32, HashMap<i64, u32>> = HashMap::new();
            let mut pcnt: HashMap<u32, u32> = HashMap::new();
            let fr = frames;
            for ci in 0..c {
                let r = &resid[ci];
                for i in 0..r.len() {
                    let k = (i / fr) as u32;
                    let band = 32 - (k + 1).leading_zeros();
                    let prevmag = if i % fr == 0 { 0 } else { magbucket(r[i - 1]) as u32 };
                    let ctx = band * 16 + prevmag;
                    *pooled.entry(ctx).or_default().entry(r[i]).or_insert(0) += 1;
                    *pcnt.entry(ctx).or_insert(0) += 1;
                }
            }
            let mut ctx_bits = 0.0f64;
            for (ctx, hist) in &pooled {
                let nn = pcnt[ctx] as f64;
                let mut h = 0.0f64;
                for &cnt in hist.values() {
                    let p = cnt as f64 / nn;
                    h -= p * p.log2();
                }
                ctx_bits += nn * h;
            }
            let mdct_ctx = (ctx_bits / 8.0).ceil() as usize;

            let mdct_enc_bps = mdct_enc as f64 * 8.0 / coeff_nm;
            let mdct_ctx_bps = mdct_ctx as f64 * 8.0 / coeff_nm;
            // Δ vs container are per-sample rates (coeff_nm ≈ c*t, critically sampled).
            let d_enc = 100.0 * (mdct_enc_bps - cont_bps) / cont_bps;
            let d_ctx = 100.0 * (mdct_ctx_bps - cont_bps) / cont_bps;
            let name = path.rsplit('/').next().unwrap_or(path);
            println!("{name:>12} {n:>5} {cont_bps:>8.4} {b0_bps:>8.4} | {mdct_enc_bps:>9.4} {d_enc:>+8.3}% | {mdct_ctx_bps:>9.4} {d_ctx:>+8.3}%");
        }
    }
    println!("\n# GATE: MDCTctx Δ/CONT ≤ 0 on siena/eegmmidb/tuar ⇒ overlap closes the gap ⇒ the deterministic");
    println!("#   transform-domain equal-codec is VIABLE (build the real reversible IntMDCT). Positive even");
    println!("#   under this optimistic proxy ⇒ transform-domain deterministic path retired ⇒ learned layer.");
}
