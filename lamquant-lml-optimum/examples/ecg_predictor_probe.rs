//! ECG predictor probe — close the +21% lossless gap vs HHI (4.01 vs 3.31 bps).
//! ECG is 2-ch (no cross-channel help); the gap is HHI's NLMS-16 + DCT/TCQ/CABAC on
//! the quasi-periodic QRS vs our RLS-8. First question (cheapest): is order-8 too
//! short for ECG morphology? Sweep variable-order RLS + reset period, code the
//! residual with the production block-Golomb (`entropy::encode`), report bps.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example ecg_predictor_probe -- /tmp/ecg_100.bin
//! ```

use std::fs;

use lamquant_lml_optimum::{entropy, rls};

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
fn round_i64(v: f64) -> i64 {
    if v >= 0.0 {
        (v + 0.5) as i64
    } else {
        (v - 0.5) as i64
    }
}

/// Variable-order RLS (same recursion as `rls.rs`, order parameterized).
struct Rls {
    n: usize,
    lambda: f64,
    w: Vec<f64>,
    p: Vec<f64>, // n×n row-major inverse-correlation
    hist: Vec<f64>,
}

impl Rls {
    fn new(n: usize, lambda: f64, delta: f64) -> Self {
        let mut p = vec![0.0; n * n];
        for i in 0..n {
            p[i * n + i] = 1.0 / delta;
        }
        Self {
            n,
            lambda,
            w: vec![0.0; n],
            p,
            hist: vec![0.0; n],
        }
    }
    fn predict(&self) -> f64 {
        (0..self.n).map(|k| self.w[k] * self.hist[k]).sum()
    }
    fn adapt(&mut self, x: f64, pred: f64) {
        let n = self.n;
        let mut px = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += self.p[i * n + j] * self.hist[j];
            }
            px[i] = s;
        }
        let mut denom = self.lambda;
        for j in 0..n {
            denom += self.hist[j] * px[j];
        }
        let inv = 1.0 / denom;
        let e = x - pred;
        for i in 0..n {
            self.w[i] += px[i] * inv * e;
        }
        let ilam = 1.0 / self.lambda;
        for i in 0..n {
            let ki = px[i] * inv;
            for j in 0..n {
                self.p[i * n + j] = (self.p[i * n + j] - ki * px[j]) * ilam;
            }
        }
    }
    fn push(&mut self, x: f64) {
        for k in (1..self.n).rev() {
            self.hist[k] = self.hist[k - 1];
        }
        self.hist[0] = x;
    }
    fn step(&mut self, x: i64) -> i64 {
        let pred = self.predict();
        let e = x - round_i64(pred);
        self.adapt(x as f64, pred);
        self.push(x as f64);
        e
    }
}

fn rls_residual(sig: &[i64], order: usize, lambda: f64, reset: usize) -> Vec<i64> {
    let mut r = Rls::new(order, lambda, 1.0);
    sig.iter()
        .enumerate()
        .map(|(i, &x)| {
            if i != 0 && i % reset == 0 {
                r = Rls::new(order, lambda, 1.0);
            }
            r.step(x)
        })
        .collect()
}

fn bps_of(resids: &[Vec<i64>], n_samples: f64) -> f64 {
    // Window like production (entropy::encode's single-k Golomb header caps ~65535).
    let bytes: usize = resids
        .iter()
        .flat_map(|r| r.chunks(16384))
        .map(|w| entropy::encode(w).unwrap().len())
        .sum();
    bytes as f64 * 8.0 / n_samples
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/ecg_100.bin".to_string());
    let sig = read_window(&path);
    let n_ch = sig.len();
    let t = sig[0].len();
    let nm = (n_ch * t) as f64;
    let name = path.rsplit('/').next().unwrap_or(&path);

    // Baseline: production RLS-8 (rls::residual) at reset=16384.
    let base: Vec<Vec<i64>> = sig.iter().map(|ch| rls::residual(ch)).collect();
    let base_bps = bps_of(&base, nm);

    println!(
        "# ECG predictor probe: {name} ({n_ch}ch × {t}). HHI=3.313 bps, prod RLS-8={base_bps:.4}"
    );
    println!("  {:>20} | {:>8} | {:>8}", "predictor", "bps", "vs prod");
    let report = |label: String, resids: &[Vec<i64>]| {
        let bps = bps_of(resids, nm);
        println!(
            "  {:>20} | {:>8.4} | {:>+7.1}%",
            label,
            bps,
            100.0 * (bps - base_bps) / base_bps
        );
    };

    for &order in &[8usize, 16, 32] {
        let resids: Vec<Vec<i64>> = sig
            .iter()
            .map(|ch| rls_residual(ch, order, 0.999, 16384))
            .collect();
        report(format!("RLS-{order}"), &resids);
    }

    // Long-term (inter-beat) predictor ON the RLS residual: the RLS residual is
    // itself quasi-periodic (the QRS spike RLS can't predict recurs every beat).
    // Per window, pick the LS-optimal lag in the beat range and subtract g·e[i-lag].
    // Causal (e[i-lag] already decoded) ⇒ reversible. Tests inter-beat redundancy.
    for &win in &[384usize, 512, 768, 2048] {
        let cascaded: Vec<Vec<i64>> = sig
            .iter()
            .map(|ch| ltp_on_residual(&rls::residual(ch), win, 180, 450))
            .collect();
        report(format!("RLS-8 + LTP/{win}"), &cascaded);
    }
    println!(
        "# negative vs-prod = beats production RLS-8. Gap to HHI (3.313) is -17.4% from here."
    );
}

/// Long-term predictor on a residual: per `win`-sample window, find the lag in
/// `[lo,hi]` (beat period) minimizing LS residual energy, subtract round(g·e[i-lag]).
fn ltp_on_residual(e: &[i64], win: usize, lo: usize, hi: usize) -> Vec<i64> {
    let n = e.len();
    let ef: Vec<f64> = e.iter().map(|&v| v as f64).collect();
    let mut out = e.to_vec();
    let mut start = 0;
    while start < n {
        let end = (start + win).min(n);
        // LS-optimal lag: maximize <e_i, e_{i-lag}>² / <e_{i-lag}, e_{i-lag}>.
        let (mut best_lag, mut best_score, mut best_g) = (0usize, 0.0f64, 0.0f64);
        let mut lag = lo;
        while lag <= hi {
            let (mut cross, mut energy) = (0.0f64, 0.0f64);
            for i in start.max(lag)..end {
                cross += ef[i] * ef[i - lag];
                energy += ef[i - lag] * ef[i - lag];
            }
            if energy > 0.0 {
                let score = cross * cross / energy;
                if score > best_score {
                    best_score = score;
                    best_lag = lag;
                    best_g = cross / energy;
                }
            }
            lag += 1;
        }
        if best_lag > 0 {
            let g = best_g.clamp(-1.0, 1.0);
            for i in start.max(best_lag)..end {
                out[i] = e[i] - round_i64(g * ef[i - best_lag]);
            }
        }
        start = end;
    }
    out
}
