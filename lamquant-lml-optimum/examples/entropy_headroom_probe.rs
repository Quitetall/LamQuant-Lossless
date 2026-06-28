//! Entropy lever (ADR 0064 choice #3) — is there ANY headroom left in the entropy
//! coding of the SHIPPED MV-RLS residual, beyond scale_cond?
//!
//! The order-1 context probe already returned NULL on the container (mv_rls)
//! residual. This probe is the definitive, coder-independent test: measure the
//! ORACLE CONDITIONAL ENTROPY of the actual `mv_rls::residuals` under successively
//! richer contexts (the information-theoretic floor a *perfect* adaptive context
//! coder could reach, with ZERO model-transmission cost). If the best oracle floor
//! is not meaningfully below scale_cond's *realized* rate, the entropy lever is
//! spent — which also bounds what the joint single-loop optimizer (#2) could gain
//! from a richer coder, since the predictor (mv_rls) is already joint.
//!
//! Contexts (all oracle / non-causal where noted ⇒ optimistic lower bounds):
//!   H0                      order-0 entropy (clean rANS/arith floor, no context)
//!   H|scale_causal          EMA-of-|prev| log2 bucket  (what scale_cond models)
//!   H|scale_oracle          non-causal local-RMS bucket (best a scale ctx could do)
//!   H|prev                  bucket(previous residual value)        — order-1
//!   H|sign,scale            sign(prev) × scale_causal
//!   H|xchan,scale           same-time other-channel magnitude bucket × scale (E-A1)
//!   H|prev,scale_oracle     joint order-1 + oracle scale (the richest tested)
//! vs scale_cond REALIZED (entropy::encode, includes adaptation/model cost).
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example entropy_headroom_probe -- <bin>...

use std::collections::HashMap;
use std::fs;
use lamquant_lml_optimum::{entropy, mv_rls};

const W: usize = 32768;

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut s = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t { ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64); off += 4; }
        s.push(ch);
    }
    s
}

#[inline]
fn log2bucket(v: i64) -> u32 { 64 - (v.unsigned_abs().max(1)).leading_zeros() }

/// Conditional entropy H(X|ctx) in bits/symbol from a context→value-histogram map.
fn cond_entropy(ctx: &HashMap<u64, HashMap<i64, u64>>) -> f64 {
    let mut bits = 0.0f64;
    let mut n = 0u64;
    for hist in ctx.values() {
        let cn: u64 = hist.values().sum();
        n += cn;
        for &c in hist.values() {
            let p = c as f64 / cn as f64;
            bits -= c as f64 * p.log2();
        }
    }
    if n == 0 { 0.0 } else { bits / n as f64 }
}

fn entropy_realized(res: &[Vec<i64>]) -> usize {
    let mut tot = 0usize;
    for ch in res {
        let mut s = 0;
        while s < ch.len() {
            let e = (s + W).min(ch.len());
            tot += entropy::encode(&ch[s..e]).map(|g| g.len()).unwrap_or(1 << 30);
            s = e;
        }
    }
    tot
}

fn main() {
    println!("# Entropy headroom on the SHIPPED MV-RLS residual (oracle conditional entropy)\n");
    for path in std::env::args().skip(1) {
        let sig = read_bin(&path);
        let name = path.rsplit('/').next().unwrap_or(&path);
        let (nch, t) = (sig.len(), sig[0].len());
        let res = mv_rls::residuals(&sig, 0, 0); // shipped default config
        let nm = (nch * t) as f64;

        let realized = entropy_realized(&res);
        let realized_bps = realized as f64 * 8.0 / nm;

        // histograms
        let mut h0: HashMap<u64, HashMap<i64, u64>> = HashMap::new();        // single ctx=0
        let mut h_sc: HashMap<u64, HashMap<i64, u64>> = HashMap::new();      // causal scale (EMA|prev|)
        let mut h_so: HashMap<u64, HashMap<i64, u64>> = HashMap::new();      // oracle local-RMS scale
        let mut h_prev: HashMap<u64, HashMap<i64, u64>> = HashMap::new();    // order-1 value
        let mut h_ss: HashMap<u64, HashMap<i64, u64>> = HashMap::new();      // sign(prev)×scale
        let mut h_xc: HashMap<u64, HashMap<i64, u64>> = HashMap::new();      // xchan mag × scale
        let mut h_ps: HashMap<u64, HashMap<i64, u64>> = HashMap::new();      // order-1 × oracle scale

        for (ci, ch) in res.iter().enumerate() {
            // causal EMA of |x|
            let mut ema = 0.0f64;
            for i in 0..ch.len() {
                let x = ch[i];
                let prev = if i > 0 { ch[i - 1] } else { 0 };
                let sc = log2bucket(ema.round() as i64) as u64;
                // oracle local RMS over ±16
                let lo = i.saturating_sub(16);
                let hi = (i + 16).min(ch.len());
                let mut acc = 0u64;
                for &v in &ch[lo..hi] { acc += (v * v) as u64; }
                let rms = ((acc / (hi - lo) as u64) as f64).sqrt() as i64;
                let so = log2bucket(rms) as u64;
                let pv = log2bucket(prev) as u64;
                let sign = if prev < 0 { 1u64 } else { 0 };
                // cross-channel: magnitude bucket of same-time residual in the prior channel
                let xc = if ci > 0 { log2bucket(res[ci - 1][i]) as u64 } else { 0 };

                *h0.entry(0).or_default().entry(x).or_insert(0) += 1;
                *h_sc.entry(sc).or_default().entry(x).or_insert(0) += 1;
                *h_so.entry(so).or_default().entry(x).or_insert(0) += 1;
                *h_prev.entry(pv).or_default().entry(x).or_insert(0) += 1;
                *h_ss.entry((sign << 8) | sc).or_default().entry(x).or_insert(0) += 1;
                *h_xc.entry((xc << 8) | sc).or_default().entry(x).or_insert(0) += 1;
                *h_ps.entry((pv << 8) | so).or_default().entry(x).or_insert(0) += 1;

                ema = 0.95 * ema + 0.05 * (x.unsigned_abs() as f64);
            }
        }

        let rows = [
            ("H0 (order-0 floor)", cond_entropy(&h0)),
            ("H|scale_causal (≈scale_cond model)", cond_entropy(&h_sc)),
            ("H|scale_oracle (best scale ctx)", cond_entropy(&h_so)),
            ("H|prev (order-1)", cond_entropy(&h_prev)),
            ("H|sign,scale", cond_entropy(&h_ss)),
            ("H|xchan_mag,scale (E-A1)", cond_entropy(&h_xc)),
            ("H|prev,scale_oracle (richest)", cond_entropy(&h_ps)),
        ];
        println!("## {} ({}ch x {})  — scale_cond REALIZED = {:.4} bps", name, nch, t, realized_bps);
        for (lbl, bps) in rows {
            let gap = 100.0 * (bps - realized_bps) / realized_bps;
            println!("   {:<38} {:.4} bps   ({:+.2}% vs realized{})", lbl, bps, gap,
                     if bps < realized_bps { "  ← headroom" } else { "" });
        }
        println!();
    }
    println!("# Oracle bps are LOWER BOUNDS (no model cost). If the richest oracle floor is");
    println!("# not meaningfully below scale_cond REALIZED, the entropy lever is spent and a");
    println!("# joint loop cannot win via a richer coder (mv_rls already whitens; predictor is joint).");
}
