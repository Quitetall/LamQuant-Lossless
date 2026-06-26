//! Phase A2 STEP-0: per-recording NON-STATIONARITY index, to find what predicts a win
//! vs H.BWC. Hypothesis: we win the recordings with high amplitude/scale non-stationarity
//! (our RLS + scale-cond adapt; HHI's stationary block-DCT+CABAC doesn't).
//!
//! Per recording (averaged over channels): windowed-RMS coefficient of variation (raw +
//! RLS residual), scale-context transition rate, and per-window residual-entropy CV.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example nonstationarity_probe -- <label> <window>.bin [<label2> <bin2> ...]
//! ```

use std::collections::HashMap;
use std::fs;

use lamquant_lml_optimum::rls;

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let bytes = fs::read(path).expect("read");
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

const W: usize = 2048; // ~analysis window

/// Coefficient of variation (std/mean) of windowed RMS — amplitude non-stationarity.
fn windowed_rms_cv(ch: &[i64]) -> f64 {
    let rms: Vec<f64> = ch
        .chunks(W)
        .filter(|w| w.len() >= W / 2)
        .map(|w| {
            let s: f64 = w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / w.len() as f64;
            s.sqrt()
        })
        .collect();
    if rms.len() < 2 {
        return 0.0;
    }
    let mean = rms.iter().sum::<f64>() / rms.len() as f64;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = rms.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / rms.len() as f64;
    var.sqrt() / mean
}

/// log2-scale context transition rate of |x| (EMA-tracked) — how often the local scale
/// regime changes. High = non-stationary scale (what scale-cond exploits).
fn scale_transition_rate(ch: &[i64]) -> f64 {
    let mut ema = 1.0f64;
    let mut prev_ctx = -1i32;
    let mut transitions = 0u64;
    for &x in ch {
        let ctx = ema.max(1.0).log2().floor() as i32;
        if prev_ctx >= 0 && ctx != prev_ctx {
            transitions += 1;
        }
        prev_ctx = ctx;
        ema = 0.95 * ema + 0.05 * (x.unsigned_abs() as f64);
    }
    transitions as f64 / ch.len().max(1) as f64
}

/// Per-window order-0 entropy CV of the RLS residual — distribution-shape drift.
fn resid_entropy_cv(ch: &[i64]) -> f64 {
    let res = rls::residual(ch);
    let h: Vec<f64> = res
        .chunks(W)
        .filter(|w| w.len() >= W / 2)
        .map(|w| {
            let mut hist: HashMap<i64, u64> = HashMap::new();
            for &v in w {
                *hist.entry(v).or_insert(0) += 1;
            }
            let n = w.len() as f64;
            -hist.values().map(|&c| { let p = c as f64 / n; p * p.log2() }).sum::<f64>()
        })
        .collect();
    if h.len() < 2 {
        return 0.0;
    }
    let mean = h.iter().sum::<f64>() / h.len() as f64;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = h.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / h.len() as f64;
    var.sqrt() / mean
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!(
        "  {:>26} | {:>7} {:>10} {:>9}",
        "recording", "ampCV", "scaleTrans", "entCV"
    );
    let mut i = 0;
    while i + 1 < args.len() {
        let label = &args[i];
        let path = &args[i + 1];
        i += 2;
        let sig = read_bin(path);
        let nc = sig.len() as f64;
        let amp = sig.iter().map(|c| windowed_rms_cv(c)).sum::<f64>() / nc;
        let st = sig.iter().map(|c| scale_transition_rate(c)).sum::<f64>() / nc;
        let ent = sig.iter().map(|c| resid_entropy_cv(c)).sum::<f64>() / nc;
        println!("  {label:>26} | {amp:>7.4} {st:>10.5} {ent:>9.4}");
    }
    println!("\n# Pair against the known H.BWC gaps: do the WIN recordings (ma Sub11/13) show higher");
    println!("# ampCV / scaleTrans / entCV than the LOSE recordings (eegmmidb, ma Sub10/14-17)?");
}
