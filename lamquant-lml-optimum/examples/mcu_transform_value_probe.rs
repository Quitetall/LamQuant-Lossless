//! MCU codec question: does the 5/3 integer wavelet buy anything serious?
//!
//! The MCU lossless codec (`lml::compress`) is per-channel: **5/3 wavelet → per-subband LPC → Golomb**.
//! This session showed predictors beat transforms on EEG; the 5/3 lifting is also a real MCU compute
//! cost (bench: lifting ~19% off C). So: is the 5/3 STEP earning its keep, or would LPC-directly-on-raw
//! + Golomb (same predictor + coder, NO wavelet, cheaper) match it?
//!
//! Isolation, per channel:
//!   LML     = lml::compress (5/3 + LPC + Golomb) — the shipped MCU codec.
//!   LPC-raw = lpc::analyze(raw channel, order) → golomb::encode_dense(residual) + coeff side-info —
//!             the SAME LPC + Golomb applied to the raw samples with NO 5/3 transform. Sweep order.
//!   DIFF2   = second difference (x-2x₋₁+x₋₂) → Golomb — the cheapest integer predictor, floor.
//! Δ = LPC-raw vs LML isolates the 5/3's contribution. LPC-raw ≤ LML ⇒ 5/3 buys nothing ⇒ droppable on MCU.
//!
//! cargo run -p lamquant-lml-optimum --features encode --release --example mcu_transform_value_probe -- <bin>...

use std::fs;
use lamquant_lml_mcu::{golomb, lml, lpc};

const W: usize = 32768; // window ≤ golomb/lml u16 sample cap

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

/// LPC-on-raw + Golomb bytes for one channel at a given order (+ coeff side-info), windowed.
fn lpc_raw_bytes(ch: &[i64], order: usize) -> usize {
    let mut tot = 0;
    let mut s = 0;
    while s < ch.len() {
        let e = (s + W).min(ch.len());
        let (coeffs, resid) = lpc::analyze(&ch[s..e], order, order);
        tot += golomb::encode_dense(&resid).map(|g| g.len()).unwrap_or(1 << 30) + coeffs.len() * 2;
        s = e;
    }
    tot
}

/// Second-difference + Golomb bytes for one channel (cheapest integer predictor), windowed.
fn diff2_bytes(ch: &[i64]) -> usize {
    let n = ch.len();
    let mut r = Vec::with_capacity(n);
    for i in 0..n {
        let p = if i >= 2 { 2 * ch[i - 1] - ch[i - 2] } else if i == 1 { ch[0] } else { 0 };
        r.push(ch[i] - p);
    }
    let mut tot = 0;
    let mut s = 0;
    while s < r.len() {
        let e = (s + W).min(r.len());
        tot += golomb::encode_dense(&r[s..e]).map(|g| g.len()).unwrap_or(1 << 30);
        s = e;
    }
    tot
}

fn main() {
    let orders = [2usize, 4, 8, 16];
    println!("# Does the MCU 5/3 wavelet buy anything? LML = shipped 5/3+LPC+Golomb; LPC-raw = same LPC+");
    println!("# Golomb, NO wavelet (best order); DIFF2 = cheapest predictor. Δ = LPC-raw vs LML (>0 ⇒ 5/3 helps).\n");
    println!("{:>12} {:>8} {:>8} {:>4} {:>8} {:>9} {:>8}",
             "recording", "LML bps", "LPCraw", "ord", "DIFF2", "LPCraw/LML", "DIFF2/LML");

    for path in &std::env::args().skip(1).collect::<Vec<_>>() {
        let sig = read_bin(path);
        let (c, t) = (sig.len(), sig[0].len());
        let nm = (c * t) as f64;
        // LML windowed (the shipped MCU codec, per-channel 5/3+LPC+Golomb), summed over W-windows.
        let mut lml_bytes = 0usize;
        let mut start = 0;
        while start < t {
            let end = (start + W).min(t);
            let win: Vec<Vec<i64>> = sig.iter().map(|ch| ch[start..end].to_vec()).collect();
            lml_bytes += lml::compress(&win, 0).map(|v| v.len()).unwrap_or(1 << 30);
            start = end;
        }

        // best LPC-raw order (sum over channels)
        let (mut best_lpc, mut best_ord) = (usize::MAX, 0);
        for &o in &orders {
            let s: usize = sig.iter().map(|ch| lpc_raw_bytes(ch, o)).sum();
            if s < best_lpc {
                best_lpc = s;
                best_ord = o;
            }
        }
        let diff2: usize = sig.iter().map(|ch| diff2_bytes(ch)).sum();

        let bps = |x: usize| x as f64 * 8.0 / nm;
        let name = path.rsplit('/').next().unwrap_or(path);
        println!("{name:>12} {:>8.4} {:>8.4} {best_ord:>4} {:>8.4} {:>+8.3}% {:>+7.3}%",
                 bps(lml_bytes), bps(best_lpc), bps(diff2),
                 100.0 * (best_lpc as f64 - lml_bytes as f64) / lml_bytes as f64,
                 100.0 * (diff2 as f64 - lml_bytes as f64) / lml_bytes as f64);
    }
    println!("\n# LPCraw/LML ≈ 0 or <0 ⇒ the 5/3 wavelet buys ~nothing on the MCU (LPC alone matches/beats)");
    println!("# ⇒ droppable for a simpler/cheaper integer-predictor codec. Large + ⇒ 5/3 earns its cycles.");
}
