//! RLS adaptive-predictor probe (ADR 0054): recursive least squares — the SOTA
//! lossless predictor (MPEG-4 ALS, the 2024 multichannel-EEG SOTA). HHI uses
//! LMS-16; RLS converges far faster and tracks non-stationarity, which is exactly
//! where our static LPC collapses (the ma 21ch case). Our NLMS was the weak
//! cousin; this is the strong one.
//!
//! Lossless reconstruction: the prediction is rounded to an integer and the
//! integer residual is coded; both encoder and decoder run the *identical* f64
//! RLS on the (losslessly exact) reconstructed history ⇒ identical f64 pred ⇒
//! exact reconstruction. (Float-deterministic on one machine — the Optimum tier
//! is host-only; a cross-platform build would fix-point it like ALS.)
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example rls_probe -- /tmp/ma_full.bin 30000
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lifting, lml, lpc};

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

/// Recursive least squares predictor (order K, forgetting λ). State: weights `w`
/// and inverse-correlation matrix `p`. `step` predicts from `hist` (most-recent
/// first), then adapts using the true sample `x`.
struct Rls {
    order: usize,
    lambda: f64,
    w: Vec<f64>,
    p: Vec<Vec<f64>>,
}

impl Rls {
    fn new(order: usize, lambda: f64, delta: f64) -> Self {
        let mut p = vec![vec![0.0f64; order]; order];
        for (i, row) in p.iter_mut().enumerate() {
            row[i] = 1.0 / delta; // large initial inverse-correlation
        }
        Self { order, lambda, w: vec![0.0; order], p }
    }

    fn predict(&self, hist: &[f64]) -> f64 {
        let mut s = 0.0;
        for k in 0..self.order {
            s += self.w[k] * hist[k];
        }
        s
    }

    /// Adapt with the exact sample `x` and its prediction (both sides identical).
    fn adapt(&mut self, hist: &[f64], x: f64, pred: f64) {
        let n = self.order;
        // px = P · hist
        let mut px = vec![0.0f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += self.p[i][j] * hist[j];
            }
            px[i] = s;
        }
        let denom = self.lambda + (0..n).map(|j| hist[j] * px[j]).sum::<f64>();
        let inv = 1.0 / denom;
        let e = x - pred;
        // w += (px/denom) · e ;  P = (P − (px/denom)·pxᵀ)/λ
        for i in 0..n {
            self.w[i] += px[i] * inv * e;
        }
        let ilam = 1.0 / self.lambda;
        for i in 0..n {
            let ki = px[i] * inv;
            for j in 0..n {
                self.p[i][j] = (self.p[i][j] - ki * px[j]) * ilam;
            }
        }
    }
}

/// RLS residual + round-trip verification (decoder reconstructs from residual).
fn rls_residual(x: &[i64], order: usize, lambda: f64) -> (Vec<i64>, bool) {
    let delta = 1.0;
    let mut enc = Rls::new(order, lambda, delta);
    let mut hist = vec![0.0f64; order];
    let mut res = Vec::with_capacity(x.len());
    for &xi in x {
        let pred = enc.predict(&hist);
        res.push(xi - pred.round() as i64);
        enc.adapt(&hist, xi as f64, pred);
        for k in (1..order).rev() {
            hist[k] = hist[k - 1];
        }
        hist[0] = xi as f64;
    }
    // decode
    let mut dec = Rls::new(order, lambda, delta);
    let mut dh = vec![0.0f64; order];
    let mut ok = true;
    for (i, &e) in res.iter().enumerate() {
        let pred = dec.predict(&dh);
        let xr = e + pred.round() as i64;
        if xr != x[i] {
            ok = false;
            break;
        }
        dec.adapt(&dh, xr as f64, pred);
        for k in (1..order).rev() {
            dh[k] = dh[k - 1];
        }
        dh[0] = xr as f64;
    }
    (res, ok)
}

fn current_bytes(x: &[i64]) -> usize {
    let (a3, d3, d2, d1) = lifting::forward_3level(x);
    let mut b = 0;
    for (sb, sub) in [a3, d3, d2, d1].iter().enumerate() {
        let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(sub.len()));
        let (c, r, _) = lpc::analyze_with_mode(sub, sb, scoped, lml::BIAS_CTX, None);
        b += 1 + 4 * c.len() + golomb::encode_dense(&r).expect("g").len();
    }
    b
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/ma_full.bin".to_string());
    let w: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(30000);
    let sig = read_window(&path);
    let t = sig[0].len().min(w);
    let nm = (sig.len() * t) as f64;
    let cur: usize = sig.iter().map(|f| current_bytes(&f[..t])).sum();
    println!("# RLS probe: {} ({}ch, {}). current 5/3+LPC = {} ({:.3} bps)", path, sig.len(), t, cur, cur as f64 * 8.0 / nm);
    println!("  {:<22} {:>9} {:>8} {:>9} {:>5}", "config", "bytes", "Δvs cur", "bps", "rt");
    for &(order, lambda) in &[(8usize, 0.999f64), (16, 0.999), (16, 0.995), (16, 0.99), (32, 0.999), (32, 0.995)] {
        let mut tot = 0usize;
        let mut allrt = true;
        for f in &sig {
            let (res, rt) = rls_residual(&f[..t], order, lambda);
            allrt &= rt;
            tot += golomb::encode_dense(&res).expect("g").len();
        }
        let d = -100.0 * (cur as f64 - tot as f64) / cur as f64;
        println!("  RLS o={:<2} λ={:<5}        {:>9} {:>+7.1}% {:>9.3} {:>5}", order, lambda, tot, d, tot as f64 * 8.0 / nm, if allrt { "ok" } else { "FAIL" });
    }
    println!("# negative Δ = RLS beats current 5/3+LPC. (golomb on RLS residual, no wavelet.)");
}
