//! Stage-0A de-risk: does HIGH-ORDER RLS adaptive whitening close the tusz spectral
//! gap? `rls_probe.rs` swept orders {8,16,32} and found "order-8 sweet spot" — but only
//! on the non-peaky win-set (ma/eegmmidb), never tusz. Here we sweep orders {8,16,32,64}
//! × λ on the LOSE-set (tusz/tuar/tuev/eegmmidb) + a win-set regression check (ma/chb),
//! measured with the REAL `entropy::encode` coder, PER WINDOW (W=32768, like production,
//! and under golomb's u16 cap), vs the shipped container (LmoCodec) and vs HHI (known
//! from the sweep). Round-trip verified (bit-exact gate).
//!
//! Thesis: a sharp spectral peak = an AR pole near the unit circle; a high-order adaptive
//! whitener removes it (LPC = AR spectral modeling). RLS (true Kalman, condition-independent
//! convergence) makes high order work where H.BWC's LMS can't. If order helps tusz toward
//! HHI's 5.04 bps → build; if it plateaus above HHI → linear prediction can't capture it.
//!
//! cargo run -p lamquant-lml-optimum --features encode --release --example rls_spectral_probe -- <bin>...

use std::collections::HashMap;
use std::fs;
use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{entropy, LmoCodec};

const W: usize = 32768; // production window

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

/// True RLS (order, λ) — inverse-correlation P + Kalman gain (transcribed from rls_probe.rs).
struct Rls { order: usize, lambda: f64, w: Vec<f64>, p: Vec<Vec<f64>> }
impl Rls {
    fn new(order: usize, lambda: f64) -> Self {
        let mut p = vec![vec![0.0f64; order]; order];
        for (i, row) in p.iter_mut().enumerate() { row[i] = 1.0; } // δ=1 → P=I
        Self { order, lambda, w: vec![0.0; order], p }
    }
    fn predict(&self, hist: &[f64]) -> f64 {
        (0..self.order).map(|k| self.w[k] * hist[k]).sum()
    }
    fn adapt(&mut self, hist: &[f64], x: f64, pred: f64) {
        let n = self.order;
        let mut px = vec![0.0f64; n];
        for i in 0..n { px[i] = (0..n).map(|j| self.p[i][j] * hist[j]).sum(); }
        let denom = self.lambda + (0..n).map(|j| hist[j] * px[j]).sum::<f64>();
        let inv = 1.0 / denom;
        let e = x - pred;
        for i in 0..n { self.w[i] += px[i] * inv * e; }
        let ilam = 1.0 / self.lambda;
        for i in 0..n {
            let ki = px[i] * inv;
            for j in 0..n { self.p[i][j] = (self.p[i][j] - ki * px[j]) * ilam; }
        }
    }
}

/// Predictions beyond this magnitude ⇒ the matrix-RLS diverged (P inflated by the
/// 1/λ growth between resets). Abort BEFORE the residual is built — a real EEG
/// prediction is ≪ 2²⁰ (signal is ±2¹⁵); anything past this would make golomb write
/// a multi-GB unary prefix → OOM. Low bound = clean abort, never a huge alloc.
const DIVERGE: f64 = (1u64 << 20) as f64;

/// Per-window RLS residual with PERIODIC RESET (every `reset`, like production
/// mv_rls — keeps the matrix RLS numerically stable at high order). Returns None
/// if the predictor diverges (unstable config). Round-trip verified.
#[allow(clippy::needless_range_loop)]
fn rls_residual(x: &[i64], order: usize, lambda: f64, reset: usize) -> Option<(Vec<i64>, bool)> {
    let mut enc = Rls::new(order, lambda);
    let mut hist = vec![0.0f64; order];
    let mut res = Vec::with_capacity(x.len());
    for (n, &xi) in x.iter().enumerate() {
        if n != 0 && reset != 0 && n % reset == 0 {
            enc = Rls::new(order, lambda);
            hist.iter_mut().for_each(|h| *h = 0.0);
        }
        let pred = enc.predict(&hist);
        if !pred.is_finite() || pred.abs() > DIVERGE { return None; } // diverged
        res.push(xi - pred.round() as i64);
        enc.adapt(&hist, xi as f64, pred);
        for k in (1..order).rev() { hist[k] = hist[k - 1]; }
        if order > 0 { hist[0] = xi as f64; }
    }
    // decode-side reconstruction from the residual (bit-exact gate)
    let mut dec = Rls::new(order, lambda);
    let mut dh = vec![0.0f64; order];
    let mut ok = true;
    for (i, &e) in res.iter().enumerate() {
        if i != 0 && reset != 0 && i % reset == 0 {
            dec = Rls::new(order, lambda);
            dh.iter_mut().for_each(|h| *h = 0.0);
        }
        let pred = dec.predict(&dh);
        let xr = e + pred.round() as i64;
        if xr != x[i] { ok = false; break; }
        dec.adapt(&dh, xr as f64, pred);
        for k in (1..order).rev() { dh[k] = dh[k - 1]; }
        if order > 0 { dh[0] = xr as f64; }
    }
    Some((res, ok))
}

fn container_bytes(sig: &[Vec<i64>]) -> usize {
    let t = sig[0].len();
    let mut tot = 0; let mut s = 0;
    while s < t {
        let e = (s + W).min(t);
        let win: Vec<Vec<i64>> = sig.iter().map(|ch| ch[s..e].to_vec()).collect();
        tot += LmoCodec.encode(&win, Mode::Lossless).map(|x| x.len()).unwrap_or(0);
        s = e;
    }
    tot
}

// (order, λ, reset) — faster forgetting (lower λ) is paired with a SHORTER reset so the
// 1/λ growth of P can't inflate between resets (matches production mv_rls CONFIGS).
const CONFIGS: &[(usize, f64, usize)] = &[
    (8, 0.999, 8192), (16, 0.999, 8192), (16, 0.99, 1024),
    (32, 0.999, 8192), (32, 0.99, 1024), (32, 0.98, 512),
    (64, 0.999, 8192), (64, 0.99, 1024), (64, 0.98, 512),
];

fn main() {
    for path in std::env::args().skip(1) {
        let sig = read_bin(&path);
        let (c, t) = (sig.len(), sig[0].len());
        let nm = (c * t) as f64;
        let cont = container_bytes(&sig);
        let name = path.rsplit('/').next().unwrap_or(&path);
        println!("\n# {} ({}ch x {})  container = {} ({:.3} bps)", name, c, t, cont, cont as f64 * 8.0 / nm);
        println!("  {:<16} {:>10} {:>8} {:>6}", "RLS(order,λ)", "bytes", "bps", "rt");
        for &(order, lambda, reset) in CONFIGS {
            let mut tot = 0usize;
            let mut allrt = true;
            let mut diverged = false;
            'cfg: for ch in &sig {
                let mut s = 0;
                while s < t {
                    let e = (s + W).min(t);
                    match rls_residual(&ch[s..e], order, lambda, reset) {
                        Some((res, rt)) => {
                            allrt &= rt;
                            tot += entropy::encode(&res).map(|g| g.len()).unwrap_or(1 << 30) + 4;
                        }
                        None => { diverged = true; break 'cfg; }
                    }
                    s = e;
                }
            }
            if diverged {
                println!("  o={:<3} λ={:<5}    {:>10} {:>8} {:>6}", order, lambda, "—", "—", "DIVERGED");
                continue;
            }
            let d = 100.0 * (tot as f64 - cont as f64) / cont as f64;
            println!("  o={:<3} λ={:<5}    {:>10} {:>8.3} {:>+6.1}%{}", order, lambda, tot,
                     tot as f64 * 8.0 / nm, d, if allrt { "  ok" } else { "  FAIL" });
        }
        stage0b(&sig);
    }
    println!("\n# bps below container ⇒ higher order helps; compare to HHI (tusz 5.04, tuar ~5.51 bps).");
    println!("# GATE: high-order RLS closes the tusz gap toward HHI without regressing the win-set ⇒ build Stage 1.");
}

/// Stage 0B: on the best-whitened residual (order 32, λ 0.999), is `scale_cond` already at
/// the entropy floor, or is there headroom an RLS-adaptive context could reach? Compare to
/// (i) order-0 entropy (clean rANS ceiling) and (ii) an ORACLE-scale conditional entropy
/// (non-causal local-scale context — the best a perfect scale predictor could do).
fn h(counts: &HashMap<i64, u64>) -> f64 {
    let n: f64 = counts.values().map(|&c| c as f64).sum();
    if n == 0.0 { return 0.0; }
    -counts.values().map(|&c| { let p = c as f64 / n; p * p.log2() }).sum::<f64>()
}
fn stage0b(sig: &[Vec<i64>]) {
    let (order, lambda, reset) = (32usize, 0.999f64, 8192usize);
    let t = sig[0].len();
    let mut sc_bytes = 0usize;            // scale_cond/golomb keep-best (entropy::encode)
    let mut h0 = HashMap::new();          // order-0 residual histogram
    let mut hcond: HashMap<u32, HashMap<i64, u64>> = HashMap::new(); // residual | oracle scale
    let mut nsym = 0u64;
    for ch in sig {
        let mut s = 0;
        while s < t {
            let e = (s + W).min(t);
            let Some((res, _)) = rls_residual(&ch[s..e], order, lambda, reset) else { return; };
            sc_bytes += entropy::encode(&res).map(|g| g.len()).unwrap_or(1 << 30) + 4;
            // oracle non-causal local scale = bit_len of centered RMS over ±16 samples
            for i in 0..res.len() {
                let lo = i.saturating_sub(16);
                let hi = (i + 16).min(res.len());
                let mut acc = 0u64;
                for &v in &res[lo..hi] { acc += (v * v) as u64; }
                let rms = ((acc / (hi - lo) as u64) as f64).sqrt() as u64;
                let scale = 64 - rms.max(1).leading_zeros();
                *h0.entry(res[i]).or_insert(0) += 1;
                *hcond.entry(scale).or_default().entry(res[i]).or_insert(0) += 1;
                nsym += 1;
            }
            s = e;
        }
    }
    let nm = (sig.len() * t) as f64;
    let h0_bps = h(&h0);
    let cond_bits: f64 = hcond.values().map(|m| { let n: f64 = m.values().map(|&c| c as f64).sum(); h(m) * n }).sum();
    let cond_bps = cond_bits / nsym as f64;
    let sc_bps = sc_bytes as f64 * 8.0 / nm;
    println!("\n## Stage 0B — entropy back-end headroom on the order-32 whitened residual (tusz)");
    println!("  scale_cond (shipped)        : {:.3} bps", sc_bps);
    println!("  order-0 entropy (rANS limit): {:.3} bps", h0_bps);
    println!("  oracle-scale cond. entropy  : {:.3} bps  (the ceiling a perfect scale ctx reaches)", cond_bps);
    println!("  → scale_cond within {:+.1}% of the oracle-scale bound; HHI = 5.04 bps.", 100.0 * (sc_bps - cond_bps) / cond_bps);
    println!("  GATE: small gap ⇒ entropy back-end is a wash (the residual itself is the floor, far above HHI).");
}
