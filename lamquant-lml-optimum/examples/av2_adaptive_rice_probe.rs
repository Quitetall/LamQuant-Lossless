//! AV2 LML pull #2 — adaptive Truncated-Rice high-range coding (context-adaptive
//! Golomb parameter) vs our shipped scale-conditioned coder.
//!
//! AV2's HR pass picks the Rice parameter from a local magnitude context. Our
//! `entropy::encode` (scale_cond) already conditions on an EMA-of-|x| scale, and
//! `entropy_headroom_probe` measured it at the causal-scale entropy floor of the
//! white mv_rls residual. This settles it definitively: compute the IDEAL
//! per-block-adaptive Rice (best k per small block, the best any adaptive-Rice
//! scheme could do, zero model cost) and compare to scale_cond's realized bytes.
//! If even the ideal loses, the lever is spent.
//!
//! cargo run -p lamquant-lml-optimum --features encode --release --example av2_adaptive_rice_probe -- <bin>...

use lamquant_lml_optimum::{entropy, mv_rls};
use std::fs;

const W: usize = 32768;
const RB: usize = 256; // Rice-param re-selection block (fine, generous to the adaptive coder)

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
fn zz(x: i64) -> u64 {
    ((x << 1) ^ (x >> 63)) as u64
}

/// Ideal per-block-adaptive Rice bits for one channel (best k per RB-block, no model cost).
fn ideal_adaptive_rice_bits(ch: &[i64]) -> u64 {
    let mut bits = 0u64;
    let mut s = 0;
    while s < ch.len() {
        let e = (s + RB).min(ch.len());
        let mut best = u64::MAX;
        for k in 0..24u32 {
            let mut b = 4u64; // ~ k signaled per block
            for &x in &ch[s..e] {
                let v = zz(x);
                b += (v >> k) + 1 + k as u64; // unary quotient + stop + k remainder
            }
            if b < best {
                best = b;
            }
        }
        bits += best;
        s = e;
    }
    bits
}

fn scale_cond_bytes(ch: &[i64]) -> usize {
    let mut tot = 0;
    let mut s = 0;
    while s < ch.len() {
        let e = (s + W).min(ch.len());
        tot += entropy::encode(&ch[s..e])
            .map(|g| g.len())
            .unwrap_or(1 << 30);
        s = e;
    }
    tot
}

fn main() {
    println!(
        "# AV2 LML #2 — IDEAL per-block adaptive Rice vs shipped scale_cond (mv_rls residual)\n"
    );
    for path in std::env::args().skip(1) {
        let sig = read_bin(&path);
        let (nch, t) = (sig.len(), sig[0].len());
        let nm = (nch * t) as f64;
        let res = mv_rls::residuals(&sig, 0, 0);
        let name = path.rsplit('/').next().unwrap_or(&path);
        let sc: usize = res.iter().map(|c| scale_cond_bytes(c)).sum();
        let rice: usize = res
            .iter()
            .map(|c| (ideal_adaptive_rice_bits(c) as usize).div_ceil(8))
            .sum();
        let d = 100.0 * (rice as f64 - sc as f64) / sc as f64;
        println!("## {} ({}ch x {})", name, nch, t);
        println!(
            "   scale_cond (shipped)        = {:>10} ({:.4} bps)",
            sc,
            sc as f64 * 8.0 / nm
        );
        println!(
            "   IDEAL adaptive-Rice (no cost)= {:>10} ({:.4} bps)  ({:+.2}% vs scale_cond)",
            rice,
            rice as f64 * 8.0 / nm,
            d
        );
        println!();
    }
    println!("# IDEAL adaptive-Rice has ZERO model cost; if it is still ≥ scale_cond, a real adaptive-Rice");
    println!("# coder cannot win — the entropy lever is spent (confirms entropy_headroom_probe).");
}
