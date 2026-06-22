//! Learned conditional entropy coder — the REAL number (causal + generalization).
//!
//! The earlier `entropy_backend_probe` "learned cond" row was a ceiling: non-causal
//! context, model fit on the eval data. This probe builds the deployable thing and
//! measures whether the win survives the two costs a real frozen-table coder pays:
//!   1. CAUSAL context — the local-scale context is computed from already-decoded
//!      samples only (an EWMA of past |residual|), so encoder and decoder agree.
//!   2. GENERALIZATION — the per-context bucket model is FIT on one split and used
//!      to code a DIFFERENT split (held-out windows, or a different corpus).
//!
//! Residual coding (bounded alphabet, CABAC/Exp-Golomb style): zigzag the residual,
//! split into a magnitude BUCKET b = #significant-bits (the learned/modeled part)
//! plus `b-1` raw uniform mantissa bits. The frozen table models p(bucket | causal
//! scale context); Golomb instead assumes a geometric bucket law with a per-block k.
//! coded bits = −log2 p_table[ctx][b] + mantissa_bits. Reconstruction is exact.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example learned_entropy_probe -- /tmp/ma_full.bin            # 60/40 window split
//! cargo run ... --example learned_entropy_probe -- /tmp/ma_full.bin /tmp/eeg64.bin  # cross-corpus
//! ```

use std::fs;

use lamquant_lml_optimum::{entropy, rls};

const NCTX: usize = 16; // causal-scale context buckets
const NBKT: usize = 48; // magnitude buckets (#significant bits); >max for i32 residuals
const W: usize = 16384; // coding window (decode-independent), matches production

fn read_window(path: &str) -> Vec<Vec<i64>> {
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

#[inline]
fn zigzag(e: i64) -> u64 {
    ((e << 1) ^ (e >> 63)) as u64
}

/// Magnitude bucket = number of significant bits of the zigzagged residual.
#[inline]
fn bucket(m: u64) -> usize {
    if m == 0 { 0 } else { (64 - m.leading_zeros()) as usize }
}

/// Raw uniform mantissa bits below the leading 1 of a bucket.
#[inline]
fn mantissa_bits(b: usize) -> usize {
    b.saturating_sub(1)
}

/// Causal local-scale context from an EWMA of past |residual| (reset per window).
#[inline]
fn scale_ctx(ewma: f64) -> usize {
    let s = (ewma + 1.0).log2().floor() as i64;
    s.clamp(0, NCTX as i64 - 1) as usize
}

#[inline]
fn update_ewma(ewma: f64, m: u64) -> f64 {
    ewma * 0.96875 + m as f64 * 0.03125 // ~32-sample EWMA
}

/// Accumulate per-(causal-context, bucket) counts over a set of windows.
fn fit(counts: &mut [[u64; NBKT]; NCTX], windows: &[&[i64]]) {
    for win in windows {
        let mut ewma = 4.0f64;
        for &e in *win {
            let m = zigzag(e);
            let ctx = scale_ctx(ewma);
            counts[ctx][bucket(m)] += 1;
            ewma = update_ewma(ewma, m);
        }
    }
}

/// Frozen log2-probability table with Laplace smoothing (the "shipped constants").
fn build_logp(counts: &[[u64; NBKT]; NCTX]) -> [[f64; NBKT]; NCTX] {
    let mut logp = [[0.0f64; NBKT]; NCTX];
    for c in 0..NCTX {
        let total: u64 = counts[c].iter().sum();
        let denom = total as f64 + NBKT as f64;
        for b in 0..NBKT {
            logp[c][b] = ((counts[c][b] as f64 + 1.0) / denom).log2();
        }
    }
    logp
}

/// Exact coded bits for a window under an ONLINE-ADAPTIVE causal conditional model
/// (no shipped table ⇒ no generalization gap; deterministic both sides). Per causal-
/// scale context, an online Laplace bucket model adapts within the window; mantissa
/// raw. This is the deployable, corpus-agnostic version of the learned coder.
fn adaptive_causal_bits(win: &[i64]) -> f64 {
    let mut counts = [[1u64; NBKT]; NCTX]; // Laplace prior
    let mut totals = [NBKT as u64; NCTX];
    let mut ewma = 4.0f64;
    let mut bits = 0.0;
    for &e in win {
        let m = zigzag(e);
        let ctx = scale_ctx(ewma);
        let b = bucket(m);
        bits += -((counts[ctx][b] as f64 / totals[ctx] as f64).log2()) + mantissa_bits(b) as f64;
        counts[ctx][b] += 1;
        totals[ctx] += 1;
        ewma = update_ewma(ewma, m);
    }
    bits
}

/// Exact coded bits for a window under the frozen causal model.
fn coded_bits(logp: &[[f64; NBKT]; NCTX], win: &[i64]) -> f64 {
    let mut bits = 0.0;
    let mut ewma = 4.0f64;
    for &e in win {
        let m = zigzag(e);
        let ctx = scale_ctx(ewma);
        let b = bucket(m);
        bits += -logp[ctx][b] + mantissa_bits(b) as f64;
        ewma = update_ewma(ewma, m);
    }
    bits
}

fn windows_of(resids: &[Vec<i64>]) -> Vec<&[i64]> {
    resids.iter().flat_map(|r| r.chunks(W)).collect()
}

fn main() {
    let fit_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/ma_full.bin".into());
    let eval_path = std::env::args().nth(2);

    let fit_resids: Vec<Vec<i64>> =
        read_window(&fit_path).iter().map(|ch| rls::residual(ch)).collect();

    // FIT / EVAL split: cross-corpus if a 2nd file is given, else 60/40 by window.
    let (fit_wins, eval_wins, eval_label): (Vec<&[i64]>, Vec<&[i64]>, String);
    let eval_resids: Vec<Vec<i64>>;
    if let Some(ep) = &eval_path {
        eval_resids = read_window(ep).iter().map(|ch| rls::residual(ch)).collect();
        fit_wins = windows_of(&fit_resids);
        eval_wins = windows_of(&eval_resids);
        eval_label = format!("cross-corpus eval={ep}");
    } else {
        let all = windows_of(&fit_resids);
        let split = all.len() * 3 / 5;
        fit_wins = all[..split].to_vec();
        eval_wins = all[split..].to_vec();
        eval_resids = Vec::new();
        eval_label = "held-out 40% of windows".into();
    }
    let _ = &eval_resids;

    let mut counts = [[0u64; NBKT]; NCTX];
    fit(&mut counts, &fit_wins);
    let logp = build_logp(&counts);

    // EVAL: learned causal frozen-table vs production block-Golomb, on the SAME windows.
    let mut learned_bits = 0.0f64;
    let mut adaptive_bits = 0.0f64;
    let mut golomb_bytes = 0usize;
    let mut n_samples = 0usize;
    for win in &eval_wins {
        learned_bits += coded_bits(&logp, win);
        adaptive_bits += adaptive_causal_bits(win);
        golomb_bytes += entropy::encode(win).unwrap().len();
        n_samples += win.len();
    }
    let learned_bytes = (learned_bits / 8.0).ceil() as usize;
    let adaptive_bytes = (adaptive_bits / 8.0).ceil() as usize;
    let g_bps = golomb_bytes as f64 * 8.0 / n_samples as f64;
    let l_bps = learned_bytes as f64 * 8.0 / n_samples as f64;
    let a_bps = adaptive_bytes as f64 * 8.0 / n_samples as f64;

    println!("# learned-entropy probe (CAUSAL context, generalized): fit={fit_path}");
    println!("# {eval_label}  ({} fit / {} eval windows, {n_samples} eval samples)",
        fit_wins.len(), eval_wins.len());
    println!("  {:>22} | {:>12} {:>9}", "coder", "bytes", "bps");
    println!("  {:>22} | {:>12} {:>9.4}", "block-Golomb (PROD)", golomb_bytes, g_bps);
    let pct = |b: usize| 100.0 * (b as f64 - golomb_bytes as f64) / golomb_bytes as f64;
    println!("  {:>22} | {:>12} {:>9.4}  {:+.1}%", "learned frozen (fit)", learned_bytes, l_bps, pct(learned_bytes));
    println!("  {:>22} | {:>12} {:>9.4}  {:+.1}%", "adaptive causal online", adaptive_bytes, a_bps, pct(adaptive_bytes));
    println!("# frozen = shipped table (generalization-fragile). adaptive = online, no table,");
    println!("# corpus-agnostic, deterministic/no_std — the deployable candidate.");
}
