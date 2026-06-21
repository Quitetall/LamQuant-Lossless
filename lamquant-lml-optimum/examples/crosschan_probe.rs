//! Lever-4 headroom probe (ADR 0054 Phase 3): how much is cross-channel
//! decorrelation worth on real multichannel EEG?
//!
//! The BWC method analysis showed HHI exploits inter-channel correlation
//! (cross-channel LMS order-32) while LMO codes every channel independently.
//! Before building the wire format, measure the ceiling cheaply (no eigen-solver,
//! no deps) two ways, in the high-rate / Gaussian approximation where rate ≈
//! `0.5·log2(variance) + const` per sample:
//!
//!   1. **Ideal full-KLT gain** — the best any cross-channel transform can do:
//!      `0.5/N · log2( ∏σ²ᵢ / det Σ )` bits/sample, where `Σ` is the `N×N`
//!      channel covariance. `det Σ = ∏λᵢ` (eigenvalues), so this needs only a
//!      Cholesky log-det, not the eigenvectors.
//!
//!   2. **Causal 1-tap proxy** — what a *deterministic, buildable* cross-channel
//!      predictor gets: code channel 0 as-is, then predict each channel from the
//!      previous one by a least-squares scalar gain and code the residual. Report
//!      the summed `0.5·log2` variance drop. A lower bound on the ideal; an upper
//!      bound on a 1-tap CC-LMS.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example crosschan_probe -- /tmp/chb01_01_60s.bin
//! ```

use std::fs;

fn read_window(path: &str) -> Vec<Vec<f64>> {
    let bytes = fs::read(path).expect("read window dump");
    let n_ch = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut sig = Vec::with_capacity(n_ch);
    for _ in 0..n_ch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            let v = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            ch.push(v as f64);
            off += 4;
        }
        sig.push(ch);
    }
    sig
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}

/// `N×N` channel covariance (population, mean-removed).
fn covariance(sig: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = sig.len();
    let t = sig[0].len();
    let mus: Vec<f64> = sig.iter().map(|c| mean(c)).collect();
    let mut cov = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for k in 0..t {
                s += (sig[i][k] - mus[i]) * (sig[j][k] - mus[j]);
            }
            let c = s / t as f64;
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }
    cov
}

/// LDLᵀ pivots `dᵢ` = the conditional variance of channel `i` given channels
/// `0..i` — i.e. the residual variance of the **full causal least-squares
/// predictor** that a deterministic cross-channel LMS would target. `∏ dᵢ =
/// det Σ`, so `Σ 0.5 log2 dᵢ` is also the ideal-KLT cost. A near-singular channel
/// set drives some `dᵢ → 0` (below the ADC noise floor); pivots are clamped to a
/// tiny epsilon to keep the factorization going, and the caller noise-floors them
/// for the *realizable* gain (you can't save bits coding below the noise floor).
fn ldl_pivots(cov: &[Vec<f64>]) -> Vec<f64> {
    let n = cov.len();
    let mut l = vec![vec![0.0f64; n]; n]; // unit lower-triangular
    let mut d = vec![0.0f64; n];
    for i in 0..n {
        let mut s = cov[i][i];
        for k in 0..i {
            s -= l[i][k] * l[i][k] * d[k];
        }
        d[i] = s.max(1e-6); // clamp near-singular pivots to stay positive
        l[i][i] = 1.0;
        for j in (i + 1)..n {
            let mut s = cov[j][i];
            for k in 0..i {
                s -= l[j][k] * l[i][k] * d[k];
            }
            l[j][i] = s / d[i];
        }
    }
    d
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/chb01_01_60s.bin".to_string());
    let sig = read_window(&path);
    let n = sig.len();
    let t = sig[0].len();
    let cov = covariance(&sig);

    // ADC noise floor: ~1 LSB² (you can't save bits coding below it).
    const NF: f64 = 1.0;
    let diag: Vec<f64> = (0..n).map(|i| cov[i][i]).collect();

    // Independent cost ∝ Σ 0.5 log2(σ²ᵢ). Full causal-LS / ideal-KLT cost ∝
    // Σ 0.5 log2(dᵢ) (LDL pivots = conditional variances).
    let pivots = ldl_pivots(&cov);
    let sum_log2_var: f64 = diag.iter().map(|&v| v.max(1e-12).log2()).sum();
    let sum_log2_piv: f64 = pivots.iter().map(|&v| v.max(1e-12).log2()).sum();
    let ideal_gain_bps = 0.5 * (sum_log2_var - sum_log2_piv) / n as f64;

    // Realizable: noise-floor both sides so near-zero KLT components (and the
    // high-rate approximation) don't claim gains below the ADC floor.
    let sum_log2_var_nf: f64 = diag.iter().map(|&v| v.max(NF).log2()).sum();
    let sum_log2_piv_nf: f64 = pivots.iter().map(|&v| v.max(NF).log2()).sum();
    let realizable_gain_bps = 0.5 * (sum_log2_var_nf - sum_log2_piv_nf) / n as f64;

    // Naive causal 1-tap (predict ch i from ch i-1 only) — shows why a FULL
    // multichannel predictor is needed: one neighbour captures almost nothing.
    let mut indep_cost = 0.5 * diag[0].max(NF).log2();
    let mut proxy_cost = indep_cost;
    for i in 1..n {
        indep_cost += 0.5 * diag[i].max(NF).log2();
        let denom = diag[i - 1].max(1e-12);
        let a = cov[i][i - 1] / denom;
        let res_var = (cov[i][i] - 2.0 * a * cov[i][i - 1] + a * a * cov[i - 1][i - 1]).max(NF);
        proxy_cost += 0.5 * res_var.log2();
    }
    let proxy_gain_bps = (indep_cost - proxy_cost) / n as f64;

    // Average pairwise |correlation| (descriptive).
    let mut sum_abs_corr = 0.0;
    let mut pairs = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = cov[i][j] / (cov[i][i].max(1e-12) * cov[j][j].max(1e-12)).sqrt();
            sum_abs_corr += r.abs();
            pairs += 1;
        }
    }
    let mean_abs_corr = sum_abs_corr / pairs.max(1) as f64;

    println!("# window: {n} ch x {t} samples ({path})");
    println!("# mean pairwise |corr| across channels      : {mean_abs_corr:.3}");
    println!(
        "# ideal full-KLT gain (high-rate, no floor) : {ideal_gain_bps:.3} bits/sample/ch  (inflated by near-zero components)"
    );
    println!(
        "# REALIZABLE full causal-LS gain (NF=1 LSB²): {realizable_gain_bps:.3} bits/sample/ch"
    );
    println!(
        "# naive causal 1-tap (prev channel only)    : {proxy_gain_bps:.3} bits/sample/ch  (why 1 neighbour is not enough)"
    );
    println!(
        "#   → at 2.0 BPS, a full cross-channel predictor realistically cuts ~{:.0}% of the rate.",
        100.0 * realizable_gain_bps / 2.0
    );
}
