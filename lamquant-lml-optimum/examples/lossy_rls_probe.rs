//! Lossy-path RLS probe (ADR 0054): the lossy 9/7 path codes quantized wavelet
//! indices with static per-subband LPC + Golomb. Does adaptive RLS prediction of
//! those indices code them smaller at the SAME quantizer (same distortion ⇒ a
//! pure rate/RD win)? Compares index-coding bytes (LPC+Golomb vs RLS+entropy) on
//! the real 9/7 subbands across the lossy quantizer range.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example lossy_rls_probe -- /tmp/chb01_01_60s.bin
//! ```

#![allow(clippy::needless_range_loop)] // explicit matrix updates mirror the RLS equations

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lml, lpc};
use lamquant_lml_optimum::wavelet97::round_i64;
use lamquant_lml_optimum::{entropy, wavelet97};

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

const ORDER: usize = 8;
struct Rls {
    w: [f64; ORDER],
    p: [[f64; ORDER]; ORDER],
    hist: [f64; ORDER],
}
impl Rls {
    fn new() -> Self {
        let mut p = [[0.0f64; ORDER]; ORDER];
        for i in 0..ORDER {
            p[i][i] = 1.0;
        }
        Self {
            w: [0.0; ORDER],
            p,
            hist: [0.0; ORDER],
        }
    }
    fn step(&mut self, x: i64) -> i64 {
        let mut pred = 0.0;
        for k in 0..ORDER {
            pred += self.w[k] * self.hist[k];
        }
        let e = x - round_i64(pred);
        let mut px = [0.0f64; ORDER];
        for i in 0..ORDER {
            for j in 0..ORDER {
                px[i] += self.p[i][j] * self.hist[j];
            }
        }
        let mut denom = 0.999;
        for j in 0..ORDER {
            denom += self.hist[j] * px[j];
        }
        let inv = 1.0 / denom;
        let ef = x as f64 - pred;
        for i in 0..ORDER {
            self.w[i] += px[i] * inv * ef;
        }
        for i in 0..ORDER {
            let ki = px[i] * inv;
            for j in 0..ORDER {
                self.p[i][j] = (self.p[i][j] - ki * px[j]) / 0.999;
            }
        }
        for k in (1..ORDER).rev() {
            self.hist[k] = self.hist[k - 1];
        }
        self.hist[0] = x as f64;
        e
    }
}

fn rls_resid(idx: &[i64]) -> Vec<i64> {
    let mut r = Rls::new();
    idx.iter().map(|&v| r.step(v)).collect()
}

fn lpc_golomb(idx: &[i64], sb: usize) -> usize {
    let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(idx.len()));
    let (coeffs, residual, _o) = lpc::analyze_with_mode(idx, sb, scoped, lml::BIAS_CTX, None);
    1 + 4 * coeffs.len() + golomb::encode_dense(&residual).unwrap().len()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/chb01_01_60s.bin".to_string());
    let sig = read_window(&path);
    let t = sig[0].len();
    let n_levels = lml::compute_n_levels(t);
    let chan_subs: Vec<Vec<Vec<f64>>> = sig
        .iter()
        .map(|c| wavelet97::forward_97_levels(c, n_levels))
        .collect();

    println!(
        "# lossy-RLS probe: {} ({}ch). index-coding bytes at fixed q (same distortion).",
        path,
        sig.len()
    );
    println!(
        "  {:>5} | {:>11} {:>11} {:>11} {:>8}",
        "q", "LPC+gol", "RLS+gol", "RLS+ent", "best dLPC"
    );
    for &q in &[8i64, 16, 32, 64, 128] {
        let (mut lpc_b, mut rls_g, mut rls_e) = (0usize, 0usize, 0usize);
        for subs in &chan_subs {
            for (sb, sub) in subs.iter().enumerate() {
                let idx: Vec<i64> = sub.iter().map(|&c| round_i64(c / q as f64)).collect();
                lpc_b += lpc_golomb(&idx, sb);
                let r = rls_resid(&idx);
                rls_g += golomb::encode_dense(&r).unwrap().len();
                rls_e += entropy::encode(&r).unwrap().len();
            }
        }
        let best = rls_g.min(rls_e);
        println!(
            "  {:>5} | {:>11} {:>11} {:>11} {:>+7.1}%",
            q,
            lpc_b,
            rls_g,
            rls_e,
            -100.0 * (lpc_b as f64 - best as f64) / lpc_b as f64
        );
    }
    println!(
        "# negative best-dLPC = RLS beats LPC at index coding (same q ⇒ same distortion ⇒ RD win)."
    );
}
