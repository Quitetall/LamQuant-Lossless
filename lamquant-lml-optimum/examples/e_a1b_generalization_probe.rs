//! Research E-A1b (Front A, the decisive refinement): is the nonlinear structure in the
//! shipped MV-RLS residual (i) headroom BEYOND what `scale_cond` already captures, and
//! (ii) does it GENERALIZE (held-out + cross-corpus)? E-A1 found 2–6% conditional-entropy
//! reduction vs ORDER-0, but the container ships `scale_cond` (already scale-conditioned),
//! and the frozen-table approach died cross-corpus (+5.9%). So the honest test:
//!
//!   baseline  = scale-conditioned bucket model (mimics the shipped scale_cond)
//!   +prev1    = scale + own previous-residual magnitude
//!   +prev2    = scale + prev1 + prev2 magnitude (the nonlinear 2nd-order)
//!   +crossch  = scale + same-instant prior-channel magnitude (volume-conduction nonlinearity)
//!
//! Each models P(bucket | ctx) (bucket = bit_len(zigzag(residual)); mantissa bits cancel,
//! exactly scale_cond's structure). We TRAIN counts on one split and evaluate CROSS-ENTROPY
//! (bits/sample on the bucket) on a DISJOINT split — so overfit shows as inflation, not gain:
//!   regime 1: in-corpus held-out (train first 50% of each file, eval last 50%)
//!   regime 2: cross-corpus (train all of corpus A, eval all of corpus B)
//!
//! GATE: if +prev2/+crossch beat the scale baseline by a clear margin on BOTH held-out AND
//! cross-corpus, a nonlinear residual coder is a real, generalizing lever ⇒ Front A lives,
//! build it. If the gain vanishes (≈0) or inverts cross-corpus, the structure is either
//! already-captured-by-scale_cond or non-generalizing ⇒ Front A dead ⇒ pivot to Front B.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example e_a1b_generalization_probe -- <cmd> ...
//!   heldout <label> <bin> [<label> <bin> ...]      # regime 1, per file
//!   cross   <Atag> <Abin>... -- <Btag> <Bbin>...   # regime 2, A→B and B→A
//! ```

use std::collections::HashMap;
use std::fs;

use lamquant_lml_optimum::mv_rls;

const ALPHA: usize = 34;         // bucket alphabet size (bit_len of zigzag fits under this)
const LAP: f64 = 0.5;            // Laplace smoothing
const MIN_CTX: u64 = 48;         // contexts with fewer train counts back off to the scale model

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

fn bit_len(m: u64) -> u32 { if m == 0 { 0 } else { 64 - m.leading_zeros() } }
fn zigzag(x: i64) -> u64 { ((x << 1) ^ (x >> 63)) as u64 }
fn bucket(x: i64) -> u32 { bit_len(zigzag(x)).min(ALPHA as u32 - 1) }
const ALPHA_F: f64 = ALPHA as f64;
fn magc(x: i64) -> u64 { bit_len(x.unsigned_abs()).min(15) as u64 }

/// One sample: the four context keys + the target bucket.
#[derive(Clone, Copy)]
struct Sample { scaleb: u64, m1: u64, m2: u64, ccm: u64, bkt: u32 }

/// Extract per-file MV-RLS residual samples (scale = EMA of |residual|, bit-len bucket).
fn samples(path: &str) -> Vec<Sample> {
    let sig = read_bin(path);
    let res = mv_rls::residuals(&sig, 1, 0); // shipped config (faster, ma-winning), seg off
    let t = res[0].len();
    let mut out = Vec::new();
    for c in 0..res.len() {
        let mut ema = 0.0f64; // EMA scale, a=1/16 (matches scale_cond's smoothing intent)
        for n in 0..t {
            let scaleb = bit_len(ema.round() as u64) as u64;
            let m1 = if n >= 1 { magc(res[c][n - 1]) } else { 0 };
            let m2 = if n >= 2 { magc(res[c][n - 2]) } else { 0 };
            let ccm = if c >= 1 { magc(res[c - 1][n]) } else { 0 };
            out.push(Sample { scaleb, m1, m2, ccm, bkt: bucket(res[c][n]) });
            let a = res[c][n].unsigned_abs() as f64;
            ema += (a - ema) / 16.0;
        }
    }
    out
}

/// A context→bucket-histogram model with backoff to a context-less global histogram.
struct Model { ctx: HashMap<u64, [u64; ALPHA]>, global: [u64; ALPHA], gtot: u64 }
impl Model {
    fn train<F: Fn(&Sample) -> u64>(data: &[Sample], key: F) -> Model {
        let mut ctx: HashMap<u64, [u64; ALPHA]> = HashMap::new();
        let mut global = [0u64; ALPHA];
        let mut gtot = 0u64;
        for s in data {
            ctx.entry(key(s)).or_insert([0; ALPHA])[s.bkt as usize] += 1;
            global[s.bkt as usize] += 1;
            gtot += 1;
        }
        Model { ctx, global, gtot }
    }
    /// cross-entropy (bits/sample) of `data` under this model with the SAME key fn.
    fn xent<F: Fn(&Sample) -> u64>(&self, data: &[Sample], key: F) -> f64 {
        let mut bits = 0.0;
        for s in data {
            let (hist, tot) = match self.ctx.get(&key(s)) {
                Some(h) => { let t: u64 = h.iter().sum(); if t >= MIN_CTX { (h, t) } else { (&self.global, self.gtot) } }
                None => (&self.global, self.gtot),
            };
            let p = (hist[s.bkt as usize] as f64 + LAP) / (tot as f64 + LAP * ALPHA_F);
            bits += -p.log2();
        }
        bits / data.len() as f64
    }
}

fn evaluate(tag: &str, train: &[Sample], test: &[Sample]) {
    let base = Model::train(train, |s| s.scaleb);
    let p1 = Model::train(train, |s| s.scaleb | (s.m1 << 8));
    let p2 = Model::train(train, |s| s.scaleb | (s.m1 << 8) | (s.m2 << 16));
    let cc = Model::train(train, |s| s.scaleb | (s.ccm << 8));
    let b = base.xent(test, |s| s.scaleb);
    let e1 = p1.xent(test, |s| s.scaleb | (s.m1 << 8));
    let e2 = p2.xent(test, |s| s.scaleb | (s.m1 << 8) | (s.m2 << 16));
    let ec = cc.xent(test, |s| s.scaleb | (s.ccm << 8));
    let red = |x: f64| 100.0 * (b - x) / b;
    println!(
        "  {:>22} | scale {:>5.3} | +prev1 {:>+5.2}% | +prev2 {:>+5.2}% | +crossch {:>+5.2}%",
        tag, b, red(e1), red(e2), red(ec)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() { eprintln!("need cmd"); return; }
    println!("# cross-entropy bits/sample on the residual BUCKET under each context, on HELD-OUT data.");
    println!("# +prevN/+crossch reductions are vs the scale-conditioned baseline (= what scale_cond ships).");
    println!();
    match args[0].as_str() {
        "heldout" => {
            // regime 1: per file, train first 50%, eval last 50%
            println!("## Regime 1 — in-corpus held-out (train first 50% of file, eval last 50%)");
            let mut i = 1;
            while i + 1 < args.len() {
                let label = &args[i];
                let s = samples(&args[i + 1]);
                i += 2;
                let mid = s.len() / 2;
                evaluate(label, &s[..mid], &s[mid..]);
            }
        }
        "cross" => {
            // regime 2: cross-corpus. args: cross <Atag> <Abin>... -- <Btag> <Bbin>...
            let sep = args.iter().position(|a| a == "--").expect("need -- between corpora");
            let atag = &args[1];
            let mut a = Vec::new();
            let mut j = 2; // A bins: args[2..sep]
            while j < sep { a.extend(samples(&args[j])); j += 1; }
            let btag = &args[sep + 1];
            let mut b = Vec::new();
            let mut k = sep + 2; // B bins: args[sep+2..]
            while k < args.len() { b.extend(samples(&args[k])); k += 1; }
            println!("## Regime 2 — cross-corpus generalization (the killer test)");
            evaluate(&format!("{}→{}", atag, btag), &a, &b);
            evaluate(&format!("{}→{}", btag, atag), &b, &a);
        }
        _ => eprintln!("cmd must be heldout|cross"),
    }
    println!();
    println!("# GATE: a clear (+>~3%) reduction on BOTH held-out AND cross-corpus ⇒ Front A lives (build");
    println!("# a generalizing nonlinear residual coder). Gain ≈0 or inverting cross-corpus ⇒ already");
    println!("# captured by scale_cond / non-generalizing ⇒ Front A dead ⇒ pivot to Front B (neural decoder).");
}
