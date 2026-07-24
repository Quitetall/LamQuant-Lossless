//! Multivariate (cross-channel) RLS probe (ADR 0054): the 2024 SOTA predicts each
//! channel from its own past AND the other channels jointly. Causal/decodable:
//! channels are coded in order, so channel c's regressor is [own K past] +
//! [already-coded channels 0..c at time n] — the same-instant spatial correlation
//! (volume conduction) plus temporal history, adapted by one RLS. Subsumes our
//! separate cross-channel + per-channel RLS into one predictor.
//!
//! Tunes K (own order), M (max cross-channel taps), reset period. vs the current
//! 5/3+LPC codec and a round-trip check.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example mv_rls_probe -- /tmp/ma_full.bin 49152
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lifting, lml, lpc};
use lamquant_lml_optimum::wavelet97::round_i64;

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

struct Rls {
    n: usize,
    lambda: f64,
    w: Vec<f64>,
    p: Vec<Vec<f64>>,
}
impl Rls {
    fn new(n: usize, lambda: f64) -> Self {
        let mut p = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            p[i][i] = 1.0;
        }
        Self {
            n,
            lambda,
            w: vec![0.0; n],
            p,
        }
    }
    fn predict(&self, reg: &[f64]) -> f64 {
        let mut s = 0.0;
        for k in 0..self.n {
            s += self.w[k] * reg[k];
        }
        s
    }
    fn adapt(&mut self, reg: &[f64], x: f64, pred: f64) {
        let n = self.n;
        let mut px = vec![0.0f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += self.p[i][j] * reg[j];
            }
            px[i] = s;
        }
        let mut denom = self.lambda;
        for j in 0..n {
            denom += reg[j] * px[j];
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
                self.p[i][j] = (self.p[i][j] - ki * px[j]) * ilam;
            }
        }
    }
}

/// Multivariate RLS residual for the whole signal: channel c uses its own K past
/// + the `m` most-recent prior channels' current sample. Returns total Golomb
/// bytes + round-trip flag.
fn mv_rls_bytes(sig: &[Vec<i64>], k: usize, m: usize, lambda: f64, reset: usize) -> (usize, bool) {
    let n_ch = sig.len();
    let t = sig[0].len();
    let mut total = 0usize;
    let mut rt_ok = true;
    for c in 0..n_ch {
        let xref = c.min(m); // # cross-channel taps for this channel
        let order = k + xref;
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect(); // recent prior channels
                                                                       // encode
        let mut rls = Rls::new(order, lambda);
        let mut own = vec![0.0f64; k];
        let mut res = Vec::with_capacity(t);
        for n in 0..t {
            if n != 0 && n % reset == 0 {
                rls = Rls::new(order, lambda);
            }
            let mut reg = vec![0.0f64; order];
            reg[..k].copy_from_slice(&own);
            for (i, &j) in refs.iter().enumerate() {
                reg[k + i] = sig[j][n] as f64;
            }
            let pred = rls.predict(&reg);
            let e = sig[c][n] - round_i64(pred);
            res.push(e);
            rls.adapt(&reg, sig[c][n] as f64, pred);
            for q in (1..k).rev() {
                own[q] = own[q - 1];
            }
            if k > 0 {
                own[0] = sig[c][n] as f64;
            }
        }
        total += golomb::encode_dense(&res).expect("g").len();
        // round-trip: decode channel c from res + prior channels (already exact)
        let mut drls = Rls::new(order, lambda);
        let mut down = vec![0.0f64; k];
        for n in 0..t {
            if n != 0 && n % reset == 0 {
                drls = Rls::new(order, lambda);
            }
            let mut reg = vec![0.0f64; order];
            reg[..k].copy_from_slice(&down);
            for (i, &j) in refs.iter().enumerate() {
                reg[k + i] = sig[j][n] as f64;
            }
            let pred = drls.predict(&reg);
            let xr = res[n] + round_i64(pred);
            if xr != sig[c][n] {
                rt_ok = false;
                break;
            }
            drls.adapt(&reg, xr as f64, pred);
            for q in (1..k).rev() {
                down[q] = down[q - 1];
            }
            if k > 0 {
                down[0] = xr as f64;
            }
        }
    }
    (total, rt_ok)
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
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/ma_full.bin".to_string());
    let w: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(49152);
    let full = read_window(&path);
    let t = full[0].len().min(w);
    let sig: Vec<Vec<i64>> = full.iter().map(|c| c[..t].to_vec()).collect();
    let nm = (sig.len() * t) as f64;
    let cur: usize = sig.iter().map(|c| current_bytes(c)).sum();
    println!(
        "# MV-RLS probe: {} ({}ch, {}). current 5/3+LPC = {:.3} bps",
        path,
        sig.len(),
        t,
        cur as f64 * 8.0 / nm
    );
    println!(
        "  {:<22} {:>8} {:>9} {:>5}",
        "K / M(xchan) / reset", "Δvs cur", "bps", "rt"
    );
    for &(k, m, reset) in &[
        (8usize, 0usize, 16384usize),
        (8, 8, 16384),
        (8, 16, 16384),
        (16, 16, 16384),
        (8, 32, 8192),
        (4, 16, 16384),
    ] {
        let (b, rt) = mv_rls_bytes(&sig, k, m, 0.999, reset);
        let d = -100.0 * (cur as f64 - b as f64) / cur as f64;
        println!(
            "  K={:<2} M={:<2} reset={:<5} {:>+7.1}% {:>9.3} {:>5}",
            k,
            m,
            reset,
            d,
            b as f64 * 8.0 / nm,
            if rt { "ok" } else { "FAIL" }
        );
    }
    println!("# M=0 = per-channel RLS (current production). M>0 adds cross-channel taps.");
}
