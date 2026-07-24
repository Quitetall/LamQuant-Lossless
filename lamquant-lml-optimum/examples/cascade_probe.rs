//! Cascade / refinement probe (ADR 0054): on top of the shipped per-channel RLS,
//! test the MPEG-4-ALS-style RLS→LMS cascade (a sign-LMS stage on the RLS
//! residual) and block-adaptive Golomb on the RLS residual. Measure-first: does
//! anything beat RLS+single-k-Golomb?
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example cascade_probe -- /tmp/ma_full.bin 30000
//! ```

use std::fs;

use lamquant_lml_mcu::golomb;
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
        // adapt
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

fn rls_residual(x: &[i64]) -> Vec<i64> {
    let mut r = Rls::new();
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            if i != 0 && i % 16384 == 0 {
                r = Rls::new();
            }
            r.step(v)
        })
        .collect()
}

/// Sign-LMS refine stage with leakage (Q16 weights), integer/reversible.
fn signlms(res: &[i64], order: usize, mu: i64, leak: u32) -> Vec<i64> {
    let mut w = vec![0i64; order];
    let mut hist = vec![0i64; order];
    let mut out = Vec::with_capacity(res.len());
    for &x in res {
        let mut pred = 0i64;
        for k in 0..order {
            pred += w[k] * hist[k];
        }
        let p = pred >> 16;
        let e = x - p;
        out.push(e);
        let se = e.signum();
        for k in 0..order {
            w[k] += mu * se * hist[k].signum() - (w[k] >> leak);
        }
        for k in (1..order).rev() {
            hist[k] = hist[k - 1];
        }
        hist[0] = x;
    }
    out
}

#[inline]
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}
fn block_golomb_bytes(res: &[i64], block: usize) -> usize {
    let mut bits = 0u64;
    let mut off = 0;
    while off < res.len() {
        let end = (off + block).min(res.len());
        let b = &res[off..end];
        let best = (0..24)
            .map(|k| {
                b.iter()
                    .map(|&v| (zigzag(v) >> k) + 1 + k as u64)
                    .sum::<u64>()
            })
            .min()
            .unwrap_or(0);
        bits += 5 + best;
        off = end;
    }
    (bits as usize).div_ceil(8)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/ma_full.bin".to_string());
    let w: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30000);
    let sig = read_window(&path);
    let t = sig[0].len().min(w);

    let resids: Vec<Vec<i64>> = sig.iter().map(|c| rls_residual(&c[..t])).collect();
    let base: usize = resids
        .iter()
        .map(|r| golomb::encode_dense(r).unwrap().len())
        .sum();
    println!(
        "# cascade probe: {} ({}ch, {}). RLS+golomb baseline = {} B",
        path,
        sig.len(),
        t,
        base
    );

    let pct = |b: usize| -100.0 * (base as f64 - b as f64) / base as f64;
    // RLS → sign-LMS cascade
    for &(o, mu, leak) in &[
        (16usize, 2i64, 12u32),
        (32, 2, 12),
        (256, 1, 14),
        (32, 4, 10),
    ] {
        let tot: usize = resids
            .iter()
            .map(|r| {
                golomb::encode_dense(&signlms(r, o, mu, leak))
                    .unwrap()
                    .len()
            })
            .sum();
        println!(
            "  RLS→signLMS o={:<3} mu={} leak={:<2}   {:>9} {:>+7.2}%",
            o,
            mu,
            leak,
            tot,
            pct(tot)
        );
    }
    // RLS → block-adaptive Golomb
    for &blk in &[64usize, 128, 256] {
        let tot: usize = resids.iter().map(|r| block_golomb_bytes(r, blk)).sum();
        println!(
            "  RLS→block-golomb blk={:<4}        {:>9} {:>+7.2}%",
            blk,
            tot,
            pct(tot)
        );
    }
    println!("# negative = beats RLS+single-k-golomb (the current ship).");
}
