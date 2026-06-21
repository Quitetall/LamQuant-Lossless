//! Lever-3 stage 3a substrate diagnostic (ADR 0054 Phase 3).
//!
//! The lever-2 A/B proved 9/7 closes only ~8% of the HHI gap; the dominant gap
//! is the **coding backend**. This harness isolates *where* the entropy headroom
//! is by coding the real 9/7 subbands at a **matched quantizer step** (so the
//! distortion is identical across configs) and reporting total bytes for each
//! coder × domain:
//!
//!   * domain  — `LPC` (current: code the LPC residual) vs `coeff` (EBCOT-aligned:
//!     order-0 bypass, code the bias-cancelled indices directly);
//!   * coder   — Golomb/zRLE (no_std floor) vs `arith0` (empirical-categorical
//!     order-0) vs `arith1` (order-1 context).
//!
//! Lower bytes at the same step = the better backend at the same distortion.
//! The decision gate reads off which column wins (run with the feature):
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features experimental_arithmetic \
//!   --release --example coding_ab -- /tmp/chb01_01_60s.bin
//! ```
//!
//! Without `experimental_arithmetic` only the Golomb/zRLE columns print (the
//! no_std floor), which still shows the LPC-vs-coeff-domain split.

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lml, lpc, zrle};
use lamquant_lml_optimum::wavelet97;

fn read_window(path: &str) -> Vec<Vec<i64>> {
    let bytes = fs::read(path).expect("read window dump");
    let n_ch = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut sig = Vec::with_capacity(n_ch);
    for _ in 0..n_ch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            let v = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            ch.push(v as i64);
            off += 4;
        }
        sig.push(ch);
    }
    sig
}

#[inline]
fn round_i64(v: f64) -> i64 {
    wavelet97::round_i64(v)
}

/// Smallest of Golomb-Rice / zero-RLE (the no_std floor) for one value block.
fn floor_bytes(values: &[i64]) -> usize {
    let g = golomb::encode_dense(values).map(|v| v.len()).unwrap_or(usize::MAX);
    let z = zrle::encode_dense(values).map(|v| v.len()).unwrap_or(usize::MAX);
    g.min(z)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/chb01_01_60s.bin".to_string());
    let sig = read_window(&path);
    let n_ch = sig.len();
    let t = sig[0].len();
    let nm = (n_ch * t) as f64;
    let n_levels = lml::compute_n_levels(t);

    // Forward 9/7 once per channel; reuse across quant steps.
    let chan_subs: Vec<Vec<Vec<f64>>> = sig
        .iter()
        .map(|ch| wavelet97::forward_97_levels(ch, n_levels))
        .collect();

    println!("# window: {n_ch} ch x {t} samples ({path}), 9/7 L={n_levels}");
    println!(
        "# bytes (and bits/sample) per config, MATCHED quant step q → identical distortion.\n\
         #   LPC-* = code LPC residual; coeff-* = order-0 bypass (code indices direct).\n"
    );
    #[cfg(feature = "experimental_arithmetic")]
    println!(
        "{:>5} | {:>14} {:>14} | {:>14} {:>14} {:>14} {:>14}",
        "q", "LPC-floor", "coeff-floor", "LPC-arith0", "LPC-arith1", "coeff-arith0", "coeff-arith1"
    );
    #[cfg(not(feature = "experimental_arithmetic"))]
    println!("{:>5} | {:>14} {:>14}   (build with --features experimental_arithmetic for the arith columns)",
        "q", "LPC-floor", "coeff-floor");

    // Quant steps spanning the WP3–WP8 operating band (coarser = lower BPS).
    for &q in &[8i64, 16, 32, 64, 128, 256] {
        let qf = q as f64;
        let mut lpc_floor = 0usize;
        let mut coeff_floor = 0usize;
        #[cfg(feature = "experimental_arithmetic")]
        let (mut lpc_a0, mut lpc_a1, mut coeff_a0, mut coeff_a1) = (0usize, 0usize, 0usize, 0usize);

        for subs in &chan_subs {
            for (sb_idx, sub) in subs.iter().enumerate() {
                let idx: Vec<i64> = sub.iter().map(|&c| round_i64(c / qf)).collect();

                // LPC residual substrate.
                let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(idx.len()));
                let (coeffs, residual, _o) =
                    lpc::analyze_with_mode(&idx, sb_idx, scoped, lml::BIAS_CTX, None);
                let lpc_hdr = 1 + 4 * coeffs.len();
                lpc_floor += lpc_hdr + floor_bytes(&residual);

                // Coefficient-domain substrate: order-0 bias-cancelled indices.
                let (_c0, residual0) = lpc::analyze(&idx, 0, lml::BIAS_CTX);
                coeff_floor += 1 + floor_bytes(&residual0);

                #[cfg(feature = "experimental_arithmetic")]
                {
                    use lamquant_lml_mcu::arith_cat;
                    let a0 = |v: &[i64]| arith_cat::encode_dense(v).map(|x| x.len()).unwrap_or(usize::MAX);
                    let a1 = |v: &[i64]| arith_cat::encode_dense_ctx(v).map(|x| x.len()).unwrap_or(usize::MAX);
                    lpc_a0 += lpc_hdr + a0(&residual);
                    lpc_a1 += lpc_hdr + a1(&residual);
                    coeff_a0 += 1 + a0(&residual0);
                    coeff_a1 += 1 + a1(&residual0);
                }
            }
        }

        let bps = |b: usize| b as f64 * 8.0 / nm;
        #[cfg(feature = "experimental_arithmetic")]
        println!(
            "{:>5} | {:>8} {:>5.3} {:>8} {:>5.3} | {:>8} {:>5.3} {:>8} {:>5.3} {:>8} {:>5.3} {:>8} {:>5.3}",
            q,
            lpc_floor, bps(lpc_floor), coeff_floor, bps(coeff_floor),
            lpc_a0, bps(lpc_a0), lpc_a1, bps(lpc_a1), coeff_a0, bps(coeff_a0), coeff_a1, bps(coeff_a1),
        );
        #[cfg(not(feature = "experimental_arithmetic"))]
        println!(
            "{:>5} | {:>8} {:>5.3} {:>8} {:>5.3}",
            q, lpc_floor, bps(lpc_floor), coeff_floor, bps(coeff_floor)
        );
    }
}
