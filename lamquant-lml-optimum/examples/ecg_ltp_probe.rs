//! ECG long-term-prediction probe (ADR 0054): ECG is a DIFFERENT codec from EEG —
//! it has strong QRS quasi-periodicity (consecutive beats are near-identical),
//! which HHI exploits (we're +34% behind on ECG lossless) and our EEG-tuned
//! 5/3+LPC+Golomb does not. The lever is a long-term / pitch predictor: estimate
//! the beat period P (autocorrelation peak), predict x[n] from x[n−P] by an LS
//! gain, code the exact integer residual.
//!
//! Measures four lossless structures per channel (end-to-end Golomb bytes):
//!   A  current:            5/3 → LPC → Golomb (the EEG codec on ECG)
//!   B  LTP + EEG codec:    LTP residual → 5/3 → LPC → Golomb
//!   C  LTP + LPC only:     LTP residual → LPC → Golomb (no wavelet, ECG-shaped)
//!   D  LPC only (no LTP):  LPC → Golomb (control — is the wavelet even helping?)
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example ecg_ltp_probe -- /tmp/ecg_full.bin [window]
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lifting, lml, lpc};

fn read_window(path: &str) -> Vec<Vec<i64>> {
    let bytes = fs::read(path).expect("read window dump");
    let n_ch = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut sig = Vec::with_capacity(n_ch);
    for _ in 0..n_ch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            ch.push(i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as i64);
            off += 4;
        }
        sig.push(ch);
    }
    sig
}

/// Golomb bytes of LPC(residual) for a single sequence (one "subband").
fn lpc_golomb(x: &[i64]) -> usize {
    let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(x.len()));
    let (coeffs, residual, _o) = lpc::analyze_with_mode(x, 0, scoped, lml::BIAS_CTX, None);
    1 + 4 * coeffs.len() + golomb::encode_dense(&residual).expect("golomb").len()
}

/// Block-adaptive LPC → Golomb (refit per block; right for non-stationary ECG).
fn block_lpc_golomb(x: &[i64], block: usize) -> usize {
    let mut bytes = 0;
    let mut off = 0;
    while off < x.len() {
        let end = (off + block).min(x.len());
        bytes += lpc_golomb(&x[off..end]);
        off = end;
    }
    bytes
}

/// 5/3 (3-level) → per-subband LPC → Golomb.
fn wavelet_lpc_golomb(x: &[i64]) -> usize {
    let (a3, d3, d2, d1) = lifting::forward_3level(x);
    let mut bytes = 0;
    for (sb, sub) in [a3, d3, d2, d1].iter().enumerate() {
        let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(sub.len()));
        let (coeffs, residual, _o) = lpc::analyze_with_mode(sub, sb, scoped, lml::BIAS_CTX, None);
        bytes += 1 + 4 * coeffs.len() + golomb::encode_dense(&residual).expect("golomb").len();
    }
    bytes
}

/// Beat period: lag in [min_p, max_p] maximising normalised autocorrelation
/// (mean-removed).
fn estimate_period(x: &[i64], min_p: usize, max_p: usize) -> usize {
    let n = x.len();
    let mean = x.iter().sum::<i64>() as f64 / n as f64;
    let xc: Vec<f64> = x.iter().map(|&v| v as f64 - mean).collect();
    let energy: f64 = xc.iter().map(|v| v * v).sum();
    let (mut best_p, mut best_r) = (min_p, f64::MIN);
    for p in min_p..=max_p.min(n / 2) {
        let mut s = 0.0;
        for n_i in p..n {
            s += xc[n_i] * xc[n_i - p];
        }
        let r = s / energy.max(1.0);
        if r > best_r {
            best_r = r;
            best_p = p;
        }
    }
    best_p
}

/// LS gain g predicting x[n] from x[n−P]; integer residual x[n]−round(g·x[n−P]).
fn ltp_residual(x: &[i64], p: usize) -> (Vec<i64>, f64) {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for n in p..x.len() {
        num += x[n] as f64 * x[n - p] as f64;
        den += x[n - p] as f64 * x[n - p] as f64;
    }
    let g = if den == 0.0 { 0.0 } else { num / den };
    let mut r = x.to_vec();
    for n in p..x.len() {
        r[n] = x[n] - (g * x[n - p] as f64).round() as i64;
    }
    (r, g)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/ecg_full.bin".to_string());
    let w: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30000);
    let sig = read_window(&path);
    let t = sig[0].len().min(w);
    println!(
        "# ECG LTP probe: {} ({}ch, window {}). Golomb bytes per channel.",
        path,
        sig.len(),
        t
    );
    println!(
        "# {:>3} {:>5} {:>6} | {:>9} {:>9} {:>9} {:>9} {:>9}",
        "ch", "P", "gain", "A(wv+lpc)", "D(lpc)", "E(blk256)", "F(blk512)", "G(ltp+blk)"
    );

    let nm = (sig.len() * t) as f64;
    let (mut ta, mut td, mut te, mut tf, mut tg) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for (ci, full) in sig.iter().enumerate() {
        let x = &full[..t];
        let p = estimate_period(x, 144, 450); // ~0.4–1.25 s at 360 Hz
        let (resid, g) = ltp_residual(x, p);
        let a = wavelet_lpc_golomb(x);
        let d = lpc_golomb(x);
        let e = block_lpc_golomb(x, 256);
        let f = block_lpc_golomb(x, 512);
        let gg = block_lpc_golomb(&resid, 256) + 6;
        ta += a;
        td += d;
        te += e;
        tf += f;
        tg += gg;
        println!(
            "  {:>3} {:>5} {:>6.3} | {:>9} {:>9} {:>9} {:>9} {:>9}",
            ci, p, g, a, d, e, f, gg
        );
    }
    let pct = |v: usize| -100.0 * (ta as f64 - v as f64) / ta as f64;
    let bps = |v: usize| v as f64 * 8.0 / nm;
    println!("\n# TOTAL  A={ta} ({:.3}bps)  D={td} ({:+.1}%)  E={te} ({:+.1}%, {:.3}bps)  F={tf} ({:+.1}%)  G={tg} ({:+.1}%)",
        bps(ta), pct(td), pct(te), bps(te), pct(tf), pct(tg));
    println!("# %Δ vs A (current 5/3+LPC). E=block-256 LPC, F=block-512, G=LTP+block-256. HHI ECG ≈ 3.31 bps.");
}
