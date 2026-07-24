//! Phase-1 DE-RISK PROBE (approved equal-codec plan) — does H.BWC's *architecture* close the
//! referential gap our TIME-domain codec leaves?
//!
//! Primary-source (H.BWC spec Draft 5) lossless pipeline = IntDCT → **DCT-domain** backward-adaptive
//! NLMS prediction (inter- + intra-channel) → CABAC. NO TCQ/RDOQ/wavelet. We are architecturally
//! orthogonal: time-domain mv_rls → per-channel range code. We have NEVER built transform → predict-
//! in-the-transform-domain → context-code. This probe measures whether that architecture reaches the
//! shipped container (the real bar) on the referential lose-set, BEFORE building the IntDCT + binary
//! context coder for real.
//!
//! Architecture under test (faithful to H.BWC, using proxies for the de-risk):
//!   1. Per channel, block into B and apply DCT-II (float, rounded — the standard entropy proxy for a
//!      reversible IntDCT, which costs ~1-3% MORE; so this probe is OPTIMISTIC for the DCT path).
//!   2. DCT-domain prediction: for each frequency bin k, treat the coefficient sequence across blocks
//!      × channels as a signal and run our mv_rls on it — which IS exactly H.BWC's structure: predict
//!      X[c][m][k] from [own K past BLOCKS at freq k = intra] + [M prior channels at the SAME block =
//!      inter, present co-temporal]. Backward-adaptive, zero side-info (decoder replays on recon).
//!   3. Score the coefficient residual two ways: our real `entropy::encode` (scale_cond keep-best) and
//!      an IDEAL per-(freq-band × magnitude) context entropy (the CABAC-class ceiling a binary context
//!      coder would approach). Emit frequency-major so each band is a coder-lockable run.
//!
//! Baselines: CONT = shipped LmoCodec Lossless container (the bar); B0 = best-config TIME-domain
//! mv_rls residual (isolates the domain change). GATE: DCT-domain ≤ CONT on siena/eegmmidb/tuar.
//!
//! Proxy caveat: float-DCT-rounded is NOT lossless (rounding loss); it estimates the IntDCT
//! coefficient entropy within ~1-3% (spectral_probe convention) and is OPTIMISTIC — if it does NOT
//! beat CONT here, the real IntDCT path won't either.
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example dct_domain_predict_probe -- [B] <bin>...

#![allow(clippy::needless_range_loop)] // explicit matrix indices mirror the DCT equations

use std::collections::HashMap;
use std::fs;

use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{entropy, mv_rls, LmoCodec};

const WIN: usize = 32768;
const N_CFG: usize = 7;

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

/// Ortho-normalized DCT-II basis[k][n] for block size b (from spectral_probe).
fn dct_basis(b: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0f64; b]; b];
    let s0 = (1.0 / b as f64).sqrt();
    let s = (2.0 / b as f64).sqrt();
    for k in 0..b {
        let sc = if k == 0 { s0 } else { s };
        for n in 0..b {
            m[k][n] = sc * (core::f64::consts::PI * (n as f64 + 0.5) * k as f64 / b as f64).cos();
        }
    }
    m
}

/// Shipped container bytes (the real bar).
fn container_bytes(sig: &[Vec<i64>]) -> usize {
    let t = sig[0].len();
    let mut tot = 0;
    let mut start = 0;
    while start < t {
        let end = (start + WIN).min(t);
        let win: Vec<Vec<i64>> = sig.iter().map(|c| c[start..end].to_vec()).collect();
        tot += LmoCodec
            .encode(&win, Mode::Lossless)
            .map(|x| x.len())
            .unwrap_or(0);
        start = end;
    }
    tot
}

/// Best-config TIME-domain mv_rls residual bytes (isolates the domain change).
fn best_config_bytes(sig: &[Vec<i64>]) -> usize {
    let mut best = usize::MAX;
    for cfg in 0..N_CFG {
        let res = mv_rls::residuals(sig, cfg, 0);
        let mut bytes = 0usize;
        for ch in &res {
            bytes += enc_windowed(ch);
        }
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
    let cli_b = args.first().and_then(|a| a.parse::<usize>().ok());
    if cli_b.is_some() {
        args.remove(0);
    }
    let blocks: Vec<usize> = cli_b
        .map(|b| vec![b])
        .unwrap_or_else(|| vec![256, 512, 1024]);

    println!("# Phase-1 de-risk: DCT-domain prediction (H.BWC architecture) vs shipped container.");
    println!("# CONT = shipped container bps (bar). B0 = best-config TIME-domain mv_rls bps.");
    println!(
        "# DCTpred = DCT → per-freq mv_rls (intra=own past blocks + inter=cross-channel present) →"
    );
    println!(
        "#   real entropy coder. DCTctx = same residual under an IDEAL freq-band×magnitude context"
    );
    println!("#   (the CABAC-class ceiling). Δ vs CONT; negative = beats production. DCT proxy is");
    println!("#   OPTIMISTIC (IntDCT costs ~1-3% more) — a positive Δ here means the real path won't win.\n");
    println!(
        "{:>12} {:>5} {:>8} {:>8} | {:>9} {:>9} | {:>9} {:>9}",
        "recording", "B", "CONT bps", "B0 bps", "DCTpred", "Δ/CONT", "DCTctx", "Δ/CONT"
    );

    for path in &args {
        let sig = read_bin(path);
        let (c, t) = (sig.len(), sig[0].len());
        let nm = (c * t) as f64;
        let cont = container_bytes(&sig);
        let b0 = best_config_bytes(&sig);
        let cont_bps = cont as f64 * 8.0 / nm;
        let b0_bps = b0 as f64 * 8.0 / nm;

        for &b in &blocks {
            let basis = dct_basis(b);
            let m_blocks = t / b; // full blocks only; tail (< b) coded raw per channel, added to both
            if m_blocks == 0 {
                continue;
            }
            // 1. DCT each channel per block → coef[c][m][k]
            let mut coef = vec![vec![vec![0i64; b]; m_blocks]; c]; // [c][m][k]
            for ci in 0..c {
                for m in 0..m_blocks {
                    let base = m * b;
                    for k in 0..b {
                        let row = &basis[k];
                        let mut acc = 0.0f64;
                        for n in 0..b {
                            acc += row[n] * sig[ci][base + n] as f64;
                        }
                        coef[ci][m][k] = acc.round() as i64;
                    }
                }
            }
            // 2. per frequency k: run mv_rls over the C×M coefficient matrix (intra past-blocks +
            //    inter cross-channel present) → residual, gather frequency-major per channel.
            let mut resid: Vec<Vec<i64>> = vec![Vec::with_capacity(b * m_blocks); c];
            for k in 0..b {
                let coefk: Vec<Vec<i64>> = (0..c)
                    .map(|ci| (0..m_blocks).map(|m| coef[ci][m][k]).collect())
                    .collect();
                let resk = mv_rls::residuals(&coefk, 0, 0);
                for ci in 0..c {
                    resid[ci].extend_from_slice(&resk[ci]);
                }
            }
            // tail samples (raw) appended so both DCTpred and CONT cover the full signal
            let tail = t - m_blocks * b;
            // 3a. real coder, frequency-major, windowed; + raw tail bytes
            let mut dct_enc = 0usize;
            for ci in 0..c {
                dct_enc += enc_windowed(&resid[ci]);
                if tail > 0 {
                    dct_enc += enc_windowed(&sig[ci][m_blocks * b..]);
                }
            }
            // 3b. IDEAL context entropy — the CABAC-class ceiling. Context = COARSE log-frequency
            //     band × prev-block-same-freq magnitude bucket, POOLED ACROSS CHANNELS (a realistic
            //     shared-context coder). Coarse bands + pooling ⇒ tens of thousands of samples per
            //     context ⇒ NO overfitting (the fine per-freq×mag version collapses to garbage).
            let mut pooled: HashMap<u32, HashMap<i64, u32>> = HashMap::new();
            let mut pcnt: HashMap<u32, u32> = HashMap::new();
            let mb = m_blocks;
            for ci in 0..c {
                let r = &resid[ci];
                for i in 0..r.len() {
                    let k = (i / mb) as u32; // frequency bin (freq-major layout)
                    let band = 32 - (k + 1).leading_zeros(); // floor(log2(k+1))+1 ∈ [1,~11]
                    let prevmag = if i % mb == 0 {
                        0
                    } else {
                        magbucket(r[i - 1]) as u32
                    };
                    let ctx = band * 16 + prevmag;
                    *pooled.entry(ctx).or_default().entry(r[i]).or_insert(0) += 1;
                    *pcnt.entry(ctx).or_insert(0) += 1;
                }
            }
            let mut ctx_bits_tot = 0.0f64;
            for (ctx, hist) in &pooled {
                let n = pcnt[ctx] as f64;
                let mut h = 0.0f64;
                for &cnt in hist.values() {
                    let p = cnt as f64 / n;
                    h -= p * p.log2();
                }
                ctx_bits_tot += n * h;
            }
            let dct_ctx = (ctx_bits_tot / 8.0).ceil() as usize
                + (0..c)
                    .map(|ci| {
                        if tail > 0 {
                            enc_windowed(&sig[ci][m_blocks * b..])
                        } else {
                            0
                        }
                    })
                    .sum::<usize>();

            let dct_enc_bps = dct_enc as f64 * 8.0 / nm;
            let dct_ctx_bps = dct_ctx as f64 * 8.0 / nm;
            let d_enc = 100.0 * (dct_enc as f64 - cont as f64) / cont as f64;
            let d_ctx = 100.0 * (dct_ctx as f64 - cont as f64) / cont as f64;
            let name = path.rsplit('/').next().unwrap_or(path);
            println!("{name:>12} {b:>5} {cont_bps:>8.4} {b0_bps:>8.4} | {dct_enc_bps:>9.4} {d_enc:>+8.3}% | {dct_ctx_bps:>9.4} {d_ctx:>+8.3}%");
        }
    }
    println!(
        "\n# GATE: DCTctx Δ/CONT ≤ 0 on the referential lose-set (siena/eegmmidb/tuar) ⇒ the H.BWC"
    );
    println!("#   architecture closes the gap deterministically ⇒ build the real IntDCT + binary context");
    println!("#   coder (Phase 2). If Δ stays positive even with the optimistic DCT proxy ⇒ the");
    println!(
        "#   deterministic clone can't reach parity ⇒ route to the learned layer (ADR 0084 D4)."
    );
}
