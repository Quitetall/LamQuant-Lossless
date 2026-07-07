//! Phase A — learned lossless entropy probe: can an ONLINE **scale-conditioned**
//! adaptive coder capture the −2 to −5% "shape | local-scale" headroom (STEP-0
//! `learned_conditional` ceiling) WITHOUT the cross-corpus generalization risk that
//! killed the frozen learned table (ADR 0076)?
//!
//! Idea: condition an online adaptive order-0 model on the LOCAL SCALE of the residual
//! (causal EMA of |x|, log2-bucketed). Scale is a universal signal property, so the
//! model adapts per-signal (zero transmitted/frozen model ⇒ generalizes by construction)
//! while still modeling shape-per-scale. The decisive test: plain adaptive order-0 was
//! −1.2% on eegmmidb but **+5.8% WORSE on ma** (its global model can't track ma's
//! non-stationary scale). Does scale-conditioning win on BOTH (a real never-worse lever)?
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example scale_cond_entropy_probe -- <window>.bin [...]
//! ```

use std::collections::HashMap;
use std::fs;

use lamquant_lml_mcu::golomb;
use lamquant_lml_optimum::{entropy, rls};

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

/// Plain online adaptive order-0 (one global Laplace model). The STEP-0 `adaptive`.
fn adaptive_order0_bytes(res: &[i64]) -> usize {
    let mut count: HashMap<i64, u64> = HashMap::new();
    let mut total: u64 = 0;
    let mut bits = 0.0f64;
    for &x in res {
        let d = count.len() as u64;
        let c = *count.get(&x).unwrap_or(&0);
        let p = (c as f64 + 1.0) / (total as f64 + d as f64 + 1.0);
        bits += -p.log2();
        *count.entry(x).or_insert(0) += 1;
        total += 1;
    }
    (bits / 8.0).ceil() as usize
}

/// ONLINE scale-conditioned adaptive order-0: a separate Laplace model per local-scale
/// context. ctx = floor(log2(EMA|x|)), computed CAUSALLY (from the running estimate
/// BEFORE coding x, updated after) so the decoder reproduces it. No transmitted/frozen
/// model ⇒ adapts per-signal ⇒ generalizes by construction.
fn scale_cond_adaptive_bytes(res: &[i64], alpha: f64) -> usize {
    let mut models: HashMap<i32, (HashMap<i64, u64>, u64)> = HashMap::new();
    let mut ema: f64 = 1.0; // running mean |x|
    let mut bits = 0.0f64;
    for &x in res {
        let ctx = ema.max(1.0).log2().floor() as i32;
        let (counts, total) = models.entry(ctx).or_insert_with(|| (HashMap::new(), 0));
        let d = counts.len() as u64;
        let c = *counts.get(&x).unwrap_or(&0);
        let p = (c as f64 + 1.0) / (*total as f64 + d as f64 + 1.0);
        bits += -p.log2();
        *counts.entry(x).or_insert(0) += 1;
        *total += 1;
        ema = alpha * ema + (1.0 - alpha) * (x.unsigned_abs() as f64);
    }
    (bits / 8.0).ceil() as usize
}

/// PRODUCTION-SHAPED variant: bounded-alphabet, no_std-implementable. Code the
/// magnitude BUCKET (bit-length of |x|, ~20 symbols) with an online adaptive model
/// conditioned on the EMA-scale context; sign + (bucket−1) mantissa bits are raw.
/// This is exactly what an integer range coder (extend `arith_int`) can ship. If it
/// captures ~the raw scale-cond win, the production coder will deliver it.
fn scale_cond_bucketed_bytes(res: &[i64], alpha: f64) -> usize {
    // bucket(m): 0 for m==0, else floor(log2(m))+1  (m in [2^(b-1), 2^b-1] ⇒ bucket b)
    fn bucket(m: u64) -> usize {
        if m == 0 { 0 } else { 64 - m.leading_zeros() as usize }
    }
    let mut models: HashMap<i32, (HashMap<usize, u64>, u64)> = HashMap::new();
    let mut ema: f64 = 1.0;
    let mut bits = 0.0f64;
    for &x in res {
        let ctx = ema.max(1.0).log2().floor() as i32;
        let m = x.unsigned_abs();
        let b = bucket(m);
        let (counts, total) = models.entry(ctx).or_insert_with(|| (HashMap::new(), 0));
        let d = counts.len() as u64;
        let c = *counts.get(&b).unwrap_or(&0);
        let p = (c as f64 + 1.0) / (*total as f64 + d as f64 + 1.0);
        bits += -p.log2(); // adaptive bucket symbol
        if b >= 1 {
            bits += (b - 1) as f64; // mantissa bits within the bucket
            bits += 1.0; // sign
        }
        *counts.entry(b).or_insert(0) += 1;
        *total += 1;
        ema = alpha * ema + (1.0 - alpha) * (m as f64);
    }
    (bits / 8.0).ceil() as usize
}

/// PRODUCTION design B: code the zigzag value DIRECTLY with a bounded per-context
/// alphabet [0, cap) + one escape symbol (residuals concentrate near 0, so the common
/// case is direct); escaped values code the (zz−cap) remainder bucketed. This is what
/// `arith_int` (MAX_ALPHABET_CTX=2048) can ship and should recover most of the raw win.
fn scale_cond_capped_bytes(res: &[i64], alpha: f64, cap: usize) -> usize {
    fn zigzag(x: i64) -> u64 {
        ((x << 1) ^ (x >> 63)) as u64
    }
    fn bucket(m: u64) -> usize {
        if m == 0 { 0 } else { 64 - m.leading_zeros() as usize }
    }
    // alphabet: symbols 0..cap = direct zigzag; symbol `cap` = escape.
    let esc = cap;
    let mut models: HashMap<i32, (HashMap<usize, u64>, u64)> = HashMap::new();
    let mut ema: f64 = 1.0;
    let mut bits = 0.0f64;
    for &x in res {
        let ctx = ema.max(1.0).log2().floor() as i32;
        let zz = zigzag(x);
        let sym = if (zz as usize) < cap { zz as usize } else { esc };
        let (counts, total) = models.entry(ctx).or_insert_with(|| (HashMap::new(), 0));
        let d = counts.len() as u64;
        let c = *counts.get(&sym).unwrap_or(&0);
        let p = (c as f64 + 1.0) / (*total as f64 + d as f64 + 1.0);
        bits += -p.log2();
        if sym == esc {
            // remainder coded bucketed (bucket symbol raw-ish + mantissa)
            let rem = zz - cap as u64;
            let b = bucket(rem);
            bits += 5.0; // ~bucket index (≤~20 buckets) — generous flat estimate
            if b >= 1 {
                bits += (b - 1) as f64;
            }
        }
        *counts.entry(sym).or_insert(0) += 1;
        *total += 1;
        ema = alpha * ema + (1.0 - alpha) * (x.unsigned_abs() as f64);
    }
    (bits / 8.0).ceil() as usize
}

/// STEP-0 optimistic ceiling: pool blocks by scale, order-0 floor per pool (no
/// transmission cost, distribution fit on the data — the upper bound on the lever).
fn learned_conditional_ceiling(res: &[i64]) -> usize {
    const BLK: usize = 256;
    let mut buckets: HashMap<i32, Vec<i64>> = HashMap::new();
    for block in res.chunks(BLK) {
        let mean = block.iter().map(|&x| x.unsigned_abs() as f64).sum::<f64>() / block.len().max(1) as f64;
        let ctx = mean.max(1.0).log2().floor() as i32;
        buckets.entry(ctx).or_default().extend_from_slice(block);
    }
    buckets
        .values()
        .map(|v| {
            let mut h: HashMap<i64, u64> = HashMap::new();
            for &x in v {
                *h.entry(x).or_insert(0) += 1;
            }
            let n = v.len() as f64;
            let bits: f64 = -h.values().map(|&c| { let p = c as f64 / n; p * p.log2() }).sum::<f64>();
            ((v.len() as f64 * bits) / 8.0).ceil() as usize
        })
        .sum()
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    const W: usize = 16384; // production coding-window
    println!(
        "  {:>30} | {:>9} {:>9} {:>9} {:>9} | {:>8} {:>8}",
        "window", "blkGolomb", "adapt0", "scRaw", "scBucket", "bkt vs G", "bkt<G?"
    );
    let (mut tg, mut ta, mut ts, mut tc, mut tk) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for path in &paths {
        let sig = read_bin(path);
        let resids: Vec<Vec<i64>> = sig.iter().map(|ch| rls::residual(ch)).collect();
        let (mut g, mut a, mut s, mut c, mut k) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for r in &resids {
            for win in r.chunks(W) {
                g += entropy::encode(win).map(|v| v.len()).unwrap_or_else(|_| golomb::encode_dense(win).unwrap().len());
                a += adaptive_order0_bytes(win);
                s += scale_cond_adaptive_bytes(win, 0.95);
                c += learned_conditional_ceiling(win);
                k += scale_cond_capped_bytes(win, 0.95, 512).min(scale_cond_bucketed_bytes(win, 0.95));
            }
        }
        let vs = 100.0 * (k as f64 - g as f64) / g as f64;
        let win_lt_g = if k < g { "YES" } else { "no" };
        let short: String = std::path::Path::new(path).file_name().unwrap().to_string_lossy().into_owned();
        let short = short.chars().rev().take(30).collect::<String>().chars().rev().collect::<String>();
        println!("  {:>30} | {:>9} {:>9} {:>9} {:>9} | {:>7.2}% {:>8}", short, g, a, s, k, vs, win_lt_g);
        tg += g; ta += a; ts += s; tc += c; tk += k;
    }
    let row = |lbl: &str, b: usize| println!("  {:>30} | {:>9} {:+.2}% vs block-Golomb", lbl, b, 100.0 * (b as f64 - tg as f64) / tg as f64);
    println!("  {:->30}-+-{:->39}", "", "");
    row("block-Golomb (PROD)", tg);
    row("adaptive order-0", ta);
    row("scale-cond adaptive (raw)", ts);
    row("scale-cond BUCKETED (ship)", tk);
    row("learned-cond CEILING", tc);
    println!("\n# scale-cond adaptive must beat block-Golomb on BOTH corpora (esp. ma, where plain");
    println!("# adaptive order-0 LOST +5.8%) to be a real generalizing never-worse keep-best lever.");
}
