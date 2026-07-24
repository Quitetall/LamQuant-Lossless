//! A.5 — MV-RLS codelength-regret measurement harness (Theorem A → a plotted result).
//!
//! Spec: `docs/proposals/lossless-frontier-krls-regret-2026-07.md` §A.5. For a recording
//! truncated to a sweep of lengths `T`, it measures — in REALIZED codelength bits
//! (`entropy::encode`), the honest currency — the excess of two ONLINE predictors over the
//! BEST FIXED linear predictor in HINDSIGHT (`residuals_hindsight`, ridge-matched):
//!
//!   • **A1 static** — `dL_static = L(growing-window LS) − L(hindsight)`. The online-vs-batch
//!     redundancy; Theorem A1 bounds it by `(d/2)log2 T`, `d = Σ_c (K+min(c,m))`. We use the
//!     numerically-STABLE block-refit forecaster (`residuals_growing_ls`), not the `λ=1` RLS
//!     recursion, which degrades over long `T`. `ratio = dL_static / (d/2 log2 T)` ⇒ ~O(1).
//!   • **A3 tracking win** — `dL_adapt = L(shipped mv_rls, λ=0.997+reset) − L(hindsight)`. On
//!     non-stationary EEG this goes NEGATIVE (the adaptive predictor beats the static-optimal),
//!     and |it| grows with `T × drift` — the mechanism behind beating a frozen codec (A2/A3).
//!
//! A per-recording `drift` index (std of per-window log2-energy) lets `dL_adapt` be plotted
//! against non-stationarity across recordings (the measurable core of the A.5-iii H.BWC
//! contrast — the static hindsight is the frozen-predictor proxy the theorem contrasts).
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release --example a5_redundancy_probe -- --synth
//! cargo run -p lamquant-lml-optimum --features encode --release --example a5_redundancy_probe -- <signal.bin>...
//! ```
//! `<signal.bin>` = `[n_ch u32 LE][t u32 LE][i32 LE samples…]` (the standard probe dump).

use std::env;
use std::fs;

use lamquant_lml_optimum::{entropy, mv_rls};

const K: usize = 8; // mirrors mv_rls::K (own temporal taps)
const M: usize = 32; // cross-channel cap (CONFIGS default)
                     // Matched ridge for BOTH the VAW forecaster (P₀=1/ridge) and the batch hindsight. ridge=1 ⇒
                     // P₀=1 (the shipped, numerically-stable RLS init) and is negligible vs EEG's Σφφᵀ~1e6·T after
                     // one sample, so the VAW bound's constant is unaffected but the recursion doesn't diverge.
const RIDGE: f64 = 1.0;
const BLOCK: usize = 256; // block-refit period for the stable online LS

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read signal.bin");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut sig = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64);
            off += 4;
        }
        sig.push(ch);
    }
    sig
}

/// Deterministic 1-channel piecewise-AR(4) synthetic (LCG noise). `switch=true` flips the
/// AR coeffs at T/2 ⇒ the adaptive predictor tracks and beats the single fixed hindsight.
fn synth(t: usize, switch: bool) -> Vec<Vec<i64>> {
    let a0 = [0.50f64, -0.25, 0.10, -0.05];
    let a1 = [0.20f64, 0.30, -0.15, 0.10];
    let sat = 1i64 << 20;
    let mut x = vec![0i64; t];
    let mut hist = [0.0f64; 4];
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    for n in 0..t {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = (((s >> 33) % 1001) as i64 - 500) as f64;
        let a = if switch && n >= t / 2 { &a1 } else { &a0 };
        let mut v = noise;
        for k in 0..4 {
            v += a[k] * hist[k];
        }
        let xi = (v.round() as i64).clamp(-sat, sat);
        x[n] = xi;
        for q in (1..4).rev() {
            hist[q] = hist[q - 1];
        }
        hist[0] = xi as f64;
    }
    vec![x]
}

fn sse(res: &[Vec<i64>]) -> f64 {
    res.iter()
        .flat_map(|r| r.iter())
        .map(|&v| {
            let f = v as f64;
            f * f
        })
        .sum()
}
fn n_samples(res: &[Vec<i64>]) -> f64 {
    res.iter().map(|r| r.len()).sum::<usize>() as f64
}
fn codelen_bits(res: &[Vec<i64>]) -> f64 {
    let mut bits = 0usize;
    for r in res {
        match entropy::encode(r) {
            Ok(v) => bits += 8 * v.len(),
            Err(_) => {
                bits += r
                    .iter()
                    .map(|&v| (64 - (2 * v.unsigned_abs() + 1).leading_zeros()) as usize)
                    .sum::<usize>()
            }
        }
    }
    bits as f64
}
fn total_d(n_ch: usize) -> usize {
    (0..n_ch).map(|c| K + c.min(M)).sum()
}
fn truncate(sig: &[Vec<i64>], t: usize) -> Vec<Vec<i64>> {
    sig.iter()
        .map(|ch| ch[..t.min(ch.len())].to_vec())
        .collect()
}

/// Non-stationarity proxy: std of per-`w`-window mean log2-energy. High ⇒ the signal's scale
/// drifts across the recording ⇒ a fixed predictor/coder mismatches ⇒ adaptivity pays.
fn drift_index(sig: &[Vec<i64>], w: usize) -> f64 {
    let t = sig[0].len();
    if t < 2 * w {
        return 0.0;
    }
    let nb = t / w;
    let mut logs = Vec::with_capacity(nb);
    for b in 0..nb {
        let mut e = 1.0f64;
        for ch in sig {
            for &v in &ch[b * w..(b + 1) * w] {
                e += (v as f64) * (v as f64);
            }
        }
        logs.push((e / (w as f64 * sig.len() as f64)).log2());
    }
    let mean = logs.iter().sum::<f64>() / logs.len() as f64;
    let var = logs.iter().map(|&l| (l - mean) * (l - mean)).sum::<f64>() / logs.len() as f64;
    var.sqrt()
}

fn measure(label: &str, sig: &[Vec<i64>]) {
    let n_ch = sig.len();
    let full_t = sig[0].len();
    let d = total_d(n_ch) as f64;
    let drift = drift_index(sig, 1024);
    println!("\n== {label}   ({n_ch}ch × {full_t}, d={d:.0}, drift={drift:.2}) ==");
    // A1 is an IDEAL-codelength law, so we measure the static regret in the ideal Gaussian
    // currency: R_ideal = (SSE_online − SSE_batch) / (2·σ²·ln2), σ² = batch per-sample MSE.
    // Theorem A1 ⇒ R_ideal ≈ (d/2)log2 T ⇒ ratioI ~ O(1). (The realized-coder dL_static is
    // shown too, but it carries coder + scale-mismatch terms above the parametric bound.)
    println!(
        "{:>8}  {:>6}  {:>12}  {:>11}  {:>7}  {:>13}",
        "T", "log2T", "(d/2)log2T", "R_ideal", "ratioI", "dL_adapt"
    );
    let mut t = 2048usize;
    let mut last_adapt = 0.0;
    loop {
        let tt = t.min(full_t);
        let s = truncate(sig, tt);
        let r_stat = mv_rls::residuals_vaw(&s, M, RIDGE); // exact VAW/online-ridge (A1)
        let r_adap = mv_rls::residuals_params(&s, 0.997, 4096, M, 0);
        let r_hind = mv_rls::residuals_hindsight(&s, M, RIDGE);
        let bound = 0.5 * d * (tt as f64).log2();
        let lh = codelen_bits(&r_hind);
        let dl_a = codelen_bits(&r_adap) - lh;
        last_adapt = dl_a;
        let n = n_samples(&r_hind);
        let sse_h = sse(&r_hind);
        let mse_h = sse_h / n;
        let r_ideal = if mse_h > 0.0 {
            (sse(&r_stat) - sse_h) / (2.0 * mse_h * std::f64::consts::LN_2)
        } else {
            0.0
        };
        println!(
            "{:>8}  {:>6.2}  {:>12.1}  {:>11.1}  {:>7.2}  {:>13.0}",
            tt,
            (tt as f64).log2(),
            bound,
            r_ideal,
            r_ideal / bound,
            dl_a
        );
        if tt >= full_t {
            break;
        }
        t *= 2;
    }
    println!(
        "   A1: R_ideal ≈ (d/2)log2T ⇒ ratioI ~O(1).  A3 win: dL_adapt={last_adapt:.0} bits @ T={full_t} (drift={drift:.2})"
    );
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let bins: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if bins.is_empty() {
        measure("SYNTH stationary AR(4)", &synth(1 << 16, false));
        measure("SYNTH switching AR(4) @ T/2", &synth(1 << 16, true));
    } else {
        // sorted by drift at the end for the A3 correlation view
        // (name, amp-drift, coeff-drift, bits/sample win)
        let mut summ: Vec<(String, f64, f64, f64)> = Vec::new();
        for p in &bins {
            let sig = read_bin(p);
            let name = p.rsplit('/').next().unwrap_or(p).to_string();
            measure(&name, &sig);
            let amp = drift_index(&sig, 1024);
            let cdrift = mv_rls::coeff_drift(&sig, M, 2048, RIDGE); // A.5-iii hypothesis (falsified)
            let full = sig[0].len();
            let s = truncate(&sig, full);
            let dl_a = codelen_bits(&mv_rls::residuals_params(&s, 0.997, 4096, M, 0))
                - codelen_bits(&mv_rls::residuals_hindsight(&s, M, RIDGE));
            summ.push((name, amp, cdrift, dl_a / (sig.len() * full) as f64));
        }
        // sort by COEFFICIENT drift — the hypothesis is bits/sample tracks THIS, not amp-drift
        summ.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        println!("\n== A.5-iii — tracking win vs non-stationarity (coeff-drift-sorted) ==");
        println!(
            "{:>26}  {:>9}  {:>10}  {:>12}",
            "recording", "amp-drift", "coeff-drift", "bits/samp"
        );
        for (n, amp, cd, bps) in &summ {
            println!("{n:>26}  {amp:>9.2}  {cd:>10.3}  {bps:>12.3}");
        }
        println!("   [A.5-iii: does bits/samp (the tracking win, normalized) rise monotonically with COEFF-drift? amp-drift was falsified.]");
    }
}
