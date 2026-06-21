//! Lever-C probe (ADR 0054 Optimum-lossless, gate 0c): does cross-channel
//! spatial prediction reduce **lossless** compressed size?
//!
//! For each channel, compares the floor pipeline (5/3 → LPC → Golomb) on the
//! original channel vs on the integer SPATIAL RESIDUAL `ch[i] - round(g·ch[ref])`
//! against a reference channel (g = LS scalar gain, fit per window, shipped). The
//! residual is exact ⇒ lossless reconstruction `ch[i] = residual + round(g·ch[ref])`
//! from the (losslessly available) reference; no quant-noise amplification (the
//! lossy lever-4 failure mode is absent). Keep-smaller per channel with a ~5-byte
//! gain+flag overhead charged, so it can never lose.
//!
//! Gated on END-TO-END Golomb bytes (NOT residual energy — energy fell 60% while
//! bytes rose in the lossy experiment). Tries two reference policies: previous
//! channel (adjacent montage pair) and best-of-all-prior channels.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example crosschan_lossless_ab -- /tmp/chb01_01_60s.bin [more.bin ...]
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

fn forward_subbands(ch: &[i64], n_levels: u8) -> Vec<Vec<i64>> {
    match n_levels {
        3 => {
            let (a3, d3, d2, d1) = lifting::forward_3level(ch);
            vec![a3, d3, d2, d1]
        }
        _ => vec![ch.to_vec()],
    }
}

/// Floor pipeline byte cost for one channel: 5/3 → per-subband LPC → Golomb.
fn pipeline_bytes(ch: &[i64], n_levels: u8) -> usize {
    let mut bytes = 0usize;
    for (sb_idx, sub) in forward_subbands(ch, n_levels).iter().enumerate() {
        let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(sub.len()));
        let (coeffs, residual, _o) = lpc::analyze_with_mode(sub, sb_idx, scoped, lml::BIAS_CTX, None);
        bytes += 1 + 4 * coeffs.len() + golomb::encode_dense(&residual).expect("golomb").len();
    }
    bytes
}

/// LS scalar gain predicting `target` from `refc` (minimises residual energy).
fn ls_gain(target: &[i64], refc: &[i64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&a, &b) in target.iter().zip(refc) {
        num += a as f64 * b as f64;
        den += b as f64 * b as f64;
    }
    if den == 0.0 { 0.0 } else { num / den }
}

fn spatial_residual(target: &[i64], refc: &[i64], g: f64) -> Vec<i64> {
    target.iter().zip(refc).map(|(&a, &b)| a - (g * b as f64).round() as i64).collect()
}

const GAIN_OVERHEAD: usize = 5; // 1 flag byte + 4-byte Q-format gain per predicted channel

fn main() {
    let paths: Vec<String> = {
        let a: Vec<String> = std::env::args().skip(1).collect();
        if a.is_empty() {
            vec!["/tmp/chb01_01_60s.bin".into(), "/tmp/chb01_01_mid.bin".into(), "/tmp/chb01_01_end.bin".into()]
        } else { a }
    };
    println!("# Lever-C probe: cross-channel LOSSLESS spatial prediction (end-to-end Golomb bytes)");
    println!("# {:<26} {:>10} {:>10} {:>8} {:>10} {:>8}", "window", "base B", "prev B", "prevΔ", "best B", "bestΔ");

    let (mut tb, mut tprev, mut tbest) = (0usize, 0usize, 0usize);
    let (mut sel_prev, mut sel_best, mut npred) = (0usize, 0usize, 0usize);
    for path in &paths {
        let sig = read_window(path);
        let n_ch = sig.len();
        let t = sig[0].len();
        let n_levels = lml::compute_n_levels(t);

        let base: Vec<usize> = sig.iter().map(|c| pipeline_bytes(c, n_levels)).collect();
        let base_sum: usize = base.iter().sum();

        let (mut prev_sum, mut best_sum) = (base[0], base[0]); // ch0 always coded raw
        for i in 1..n_ch {
            // Policy 1: predict from previous channel (adjacent montage pair).
            let g = ls_gain(&sig[i], &sig[i - 1]);
            let r = spatial_residual(&sig[i], &sig[i - 1], g);
            let prev_b = pipeline_bytes(&r, n_levels) + GAIN_OVERHEAD;
            let chosen_prev = prev_b.min(base[i]);
            prev_sum += chosen_prev;
            if prev_b < base[i] { sel_prev += 1; }

            // Policy 2: best-of-all-prior channels as reference.
            let mut best_b = base[i];
            for j in 0..i {
                let g = ls_gain(&sig[i], &sig[j]);
                let r = spatial_residual(&sig[i], &sig[j], g);
                let cand = pipeline_bytes(&r, n_levels) + GAIN_OVERHEAD;
                if cand < best_b { best_b = cand; }
            }
            best_sum += best_b;
            if best_b < base[i] { sel_best += 1; }
            npred += 1;
        }
        tb += base_sum; tprev += prev_sum; tbest += best_sum;
        let name = path.rsplit('/').next().unwrap_or(path);
        println!(
            "  {:<26} {:>10} {:>10} {:>+7.2}% {:>10} {:>+7.2}%",
            format!("{name} ({n_ch}x{t})"), base_sum,
            prev_sum, -100.0 * (base_sum - prev_sum) as f64 / base_sum as f64,
            best_sum, -100.0 * (base_sum - best_sum) as f64 / base_sum as f64
        );
    }
    println!(
        "  {:<26} {:>10} {:>10} {:>+7.2}% {:>10} {:>+7.2}%",
        "TOTAL", tb,
        tprev, -100.0 * (tb - tprev) as f64 / tb as f64,
        tbest, -100.0 * (tb - tbest) as f64 / tb as f64
    );
    println!(
        "\n# prev policy selected {}/{} predicted channels; best-ref policy {}/{}.",
        sel_prev, npred, sel_best, npred
    );
    println!("# Δ = size reduction vs all-original floor (negative = smaller = headroom). keep-smaller per channel.");
}
