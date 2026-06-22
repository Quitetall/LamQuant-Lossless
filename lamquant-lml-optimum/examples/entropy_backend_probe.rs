//! Probe — entropy backend on the production RLS residual (research-direction
//! §1.1/§4-Phase-1/§6.4: tANS as default coder; §2.5/§2.6: EBCOT and rANS are
//! refuted dead-ends). The reconciled lever is: does an order-0 entropy coder
//! (tANS-class) beat block-adaptive Golomb on the residual we actually ship?
//!
//! Earlier this session "arithmetic = null vs Golomb" was measured on the
//! near-geometric LPC-*subband* residual (Golomb already optimal there). The
//! production lossless path now codes the **RLS residual**, which is bursty /
//! heavy-tailed (adaptation lag) — exactly where Golomb (geometric-optimal)
//! should leak and a shape-modeling coder should win. This probe asks that.
//!
//! `arith_int::encode_dense` is an order-0 integer range coder ⇒ it achieves
//! ~order-0 entropy, the SAME ceiling tANS reaches. So its bytes are a faithful
//! stand-in for "what tANS would achieve" (deploy tANS later for no_std; measure
//! achievable size now). The H0 floors below are coder-agnostic truth: if even
//! the floor doesn't beat block-Golomb, no order-0 coder (tANS included) can.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example entropy_backend_probe -- /tmp/ma_full.bin
//! ```

use std::fs;

use lamquant_lml_mcu::golomb;
use lamquant_lml_optimum::{arith_int, entropy, rls};

const BLOCK: usize = 256;

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

/// Order-0 empirical entropy of a residual slice, in bits/symbol.
fn h0_bits(res: &[i64]) -> f64 {
    if res.is_empty() {
        return 0.0;
    }
    let mut hist: std::collections::HashMap<i64, u64> = std::collections::HashMap::new();
    for &v in res {
        *hist.entry(v).or_insert(0) += 1;
    }
    let n = res.len() as f64;
    -hist.values().map(|&c| {
        let p = c as f64 / n;
        p * p.log2()
    }).sum::<f64>()
}

/// Coder-agnostic order-0 floor (bytes) — the best ANY order-0 coder (tANS,
/// arithmetic, Huffman→) can reach, ignoring model-transmission cost.
fn h0_floor_bytes(res: &[i64]) -> usize {
    ((res.len() as f64 * h0_bits(res)) / 8.0).ceil() as usize
}

/// Block-adaptive order-0 floor: re-fit the histogram per BLOCK. This is the
/// tANS-with-block-adaptation ceiling (still ignores per-block model cost, so
/// optimistic — a real win needs margin over block-Golomb to survive overhead).
fn h0_block_floor_bytes(res: &[i64]) -> usize {
    res.chunks(BLOCK).map(h0_floor_bytes).sum()
}

/// Exact codelength of an ADAPTIVE order-0 arithmetic coder (Laplace/KT online
/// model, NO transmitted histogram). bits = Σ −log2 p_t where p_t is the model's
/// estimate BEFORE seeing x_t, updated after. This is the real achievable size of
/// an adaptive range coder — it pays ~zero model cost and tracks non-stationarity,
/// so it's the candidate that could capture the H0-floor headroom block-Golomb
/// leaves. Computed directly (no coder needed): the arithmetic coder reaches this
/// to within ~1 byte total.
fn adaptive_order0_bytes(res: &[i64]) -> usize {
    use std::collections::HashMap;
    let mut count: HashMap<i64, u64> = HashMap::new();
    let mut total: u64 = 0;
    let mut bits = 0.0f64;
    for &x in res {
        let d = count.len() as u64; // distinct symbols seen so far
        let c = *count.get(&x).unwrap_or(&0);
        // Laplace over (seen symbols + one novel slot): never zero probability.
        let p = (c as f64 + 1.0) / (total as f64 + d as f64 + 1.0);
        bits += -p.log2();
        *count.entry(x).or_insert(0) += 1;
        total += 1;
    }
    (bits / 8.0).ceil() as usize
}

/// Real order-0 arithmetic bytes, block-adaptive, with Golomb fallback per block
/// when the alphabet is too wide (the keep-best behaviour the codec uses).
fn arith_block_bytes(res: &[i64]) -> usize {
    res.chunks(BLOCK)
        .map(|c| match arith_int::encode_dense(c) {
            Ok(b) => b.len().min(golomb::encode_dense(c).unwrap().len()),
            Err(_) => golomb::encode_dense(c).unwrap().len(),
        })
        .sum()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/ma_full.bin".to_string());
    let sig = read_window(&path);
    let n_ch = sig.len();
    let t = sig[0].len();
    let n_samples = (n_ch * t) as f64;

    // Production residual: per-channel RLS (the thing we actually ship).
    let resids: Vec<Vec<i64>> = sig.iter().map(|ch| rls::residual(ch)).collect();

    let mut block_golomb = 0usize; // production baseline (entropy::encode keep-best)
    let mut arith_block = 0usize;
    let mut adaptive = 0usize;
    let mut h0_floor = 0usize;
    let mut h0_block = 0usize;

    // Window the residual like production: each coding window decodes independently,
    // and `entropy::encode`'s single-k Golomb header caps at ~65535 symbols.
    const W: usize = 16384;
    for r in &resids {
        for win in r.chunks(W) {
            block_golomb += entropy::encode(win).unwrap().len();
            arith_block += arith_block_bytes(win);
            adaptive += adaptive_order0_bytes(win);
            h0_floor += h0_floor_bytes(win);
            h0_block += h0_block_floor_bytes(win);
        }
    }

    let base = block_golomb as f64;
    let row = |label: &str, bytes: usize| {
        let bps = bytes as f64 * 8.0 / n_samples;
        let vs = format!("{:+.1}%", 100.0 * (bytes as f64 - base) / base);
        println!("  {:>22} | {:>12} {:>9.4} {:>9}", label, bytes, bps, vs);
    };

    println!(
        "# entropy-backend probe on RLS residual: {} ({}ch × {})",
        path, n_ch, t
    );
    println!("# baseline = block-adaptive Golomb (production `entropy::encode`).");
    println!("  {:>22} | {:>12} {:>9} {:>9}", "coder", "bytes", "bps", "vs base");
    row("block-Golomb (PROD)", block_golomb);
    row("arith order-0 /block", arith_block); // static, transmits per-block model
    row("adaptive order-0", adaptive); // online, no transmitted model — the candidate
    row("H0 floor global", h0_floor);
    row("H0 floor /block", h0_block);
    println!("# adaptive order-0 = exact size of an online range coder (no transmitted model).");
    println!("# If it beats block-Golomb consistently, it's a real never-worse keep-best lever.");
}
