//! Block-adaptive Golomb/Rice probe (ADR 0054 / research report): our
//! `golomb::encode_dense` picks ONE Rice parameter k per block. On non-stationary
//! residuals (ECG QRS bursts, EEG transients) the optimal k varies locally, so a
//! per-sub-block k should win. JPEG-LS / block-adaptive-Rice lineage; cheap,
//! no_std. Measure-first: does it actually help our LPC residuals?
//!
//! Compares analytic Rice bit-cost: single-k (current) vs per-block best-k (+5
//! bits/block for the shipped k), on the real LPC residuals of ECG and EEG.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example block_golomb_probe -- /tmp/ecg_100.bin ecg
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{lifting, lml, lpc};

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
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn rice_bits_k(res: &[i64], k: u32) -> u64 {
    res.iter().map(|&v| (zigzag(v) >> k) + 1 + k as u64).sum()
}

fn single_k_bits(res: &[i64]) -> u64 {
    (0..24).map(|k| rice_bits_k(res, k)).min().unwrap_or(0)
}

fn block_k_bits(res: &[i64], block: usize) -> u64 {
    let mut bits = 0u64;
    let mut off = 0;
    while off < res.len() {
        let end = (off + block).min(res.len());
        let b = &res[off..end];
        bits += 5 + (0..24).map(|k| rice_bits_k(b, k)).min().unwrap_or(0); // +5b shipped k
        off = end;
    }
    bits
}

fn lpc_residual(x: &[i64], sb: usize) -> Vec<i64> {
    let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(x.len()));
    lpc::analyze_with_mode(x, sb, scoped, lml::BIAS_CTX, None).1
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/ecg_100.bin".to_string());
    let kind = std::env::args().nth(2).unwrap_or_else(|| "ecg".to_string());
    let sig = read_window(&path);
    let t = sig[0].len().min(30000);

    // Collect the real LPC residuals: ECG = LPC on raw channel; EEG = per-subband.
    let mut residuals: Vec<Vec<i64>> = Vec::new();
    for full in &sig {
        let x = &full[..t];
        if kind == "eeg" {
            let (a3, d3, d2, d1) = lifting::forward_3level(x);
            for (sb, sub) in [a3, d3, d2, d1].iter().enumerate() {
                residuals.push(lpc_residual(sub, sb));
            }
        } else {
            residuals.push(lpc_residual(x, 0));
        }
    }

    println!(
        "# block-Golomb probe: {} ({}), {} residual blocks",
        path,
        kind,
        residuals.len()
    );
    println!(
        "# {:>8} | {:>12} {:>12} {:>8}",
        "block", "single-k", "block-k", "Δ"
    );
    let single: u64 = residuals.iter().map(|r| single_k_bits(r)).sum();
    for &b in &[64usize, 128, 256, 512] {
        let blk: u64 = residuals.iter().map(|r| block_k_bits(r, b)).sum();
        println!(
            "  {:>8} | {:>12} {:>12} {:>+7.2}%",
            b,
            single,
            blk,
            -100.0 * (single as f64 - blk as f64) / single as f64
        );
    }
    println!("# Δ = block-adaptive-k vs single-k Rice on the LPC residual. negative = smaller.");
}
