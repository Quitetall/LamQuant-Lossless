//! Diagnose the 5/3 result: does 5/3 "work" or not? Decompose the MCU pipeline into its decorrelators,
//! all coded with the SAME bare Golomb (no container overhead), windowed at 32768:
//!   RAW      = Golomb(raw samples)                    — no prediction, no transform (weak baseline).
//!   W53      = Σ Golomb(5/3 subband)                  — transform ALONE (no LPC).
//!   LPC      = Golomb(LPC residual on raw)            — temporal prediction ALONE (no transform).
//!   W53+LPC  = Σ Golomb(LPC residual per 5/3 subband) — both (what the MCU codec does).
//!
//! Reading: 5/3-alone = W53 vs RAW (does the transform decorrelate?); LPC-alone = LPC vs RAW; and the
//! decisive one — 5/3-on-LPC = W53+LPC vs LPC (does the transform add ANYTHING once you have a good
//! predictor?). If W53≪RAW and LPC≪RAW but W53+LPC≈LPC, then 5/3 and LPC are REDUNDANT decorrelators:
//! 5/3 "works" vs a weak baseline (the paper's regime) yet is ~0 on top of a strong LPC (this probe's
//! regime). Both true — different questions. The earlier bare-lml probe also carried container overhead
//! that this one removes.
//!
//! cargo run -p lamquant-lml-optimum --features encode --release --example mcu_transform_decomp_probe -- <bin>...

use std::fs;
use lamquant_lml_mcu::{golomb, lml, lpc};

const W: usize = 32768;
const LVL: u8 = 3;

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut s = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64);
            off += 4;
        }
        s.push(ch);
    }
    s
}

#[inline]
fn gr(v: &[i64]) -> usize {
    golomb::encode_dense(v).map(|g| g.len()).unwrap_or(1 << 30)
}

/// LPC residual on a slice at a given order (analyze then code residual). Returns golomb bytes + coeff.
fn lpc_gr(v: &[i64], order: usize) -> usize {
    let (coeffs, resid) = lpc::analyze(v, order, order);
    gr(&resid) + coeffs.len() * 2
}

fn main() {
    for path in &std::env::args().skip(1).collect::<Vec<_>>() {
        let sig = read_bin(path);
        let (c, t) = (sig.len(), sig[0].len());
        let nm = (c * t) as f64;
        let (mut raw, mut w53, mut lpc_b, mut w53lpc) = (0usize, 0usize, 0usize, 0usize);
        for ch in &sig {
            let mut s = 0;
            while s < t {
                let e = (s + W).min(t);
                let win = &ch[s..e];
                raw += gr(win);
                lpc_b += lpc_gr(win, 8);
                let subs = lml::forward_subbands(win, LVL);
                for (i, sb) in subs.iter().enumerate() {
                    w53 += gr(sb);
                    w53lpc += lpc_gr(sb, lpc::fixed_order_for_subband(i));
                }
                s = e;
            }
        }
        let bps = |x: usize| x as f64 * 8.0 / nm;
        let name = path.rsplit('/').next().unwrap_or(path);
        // 5/3-alone (vs raw), LPC-alone (vs raw), 5/3-ON-LPC (W53+LPC vs LPC — the decisive one)
        let d53 = 100.0 * (w53 as f64 - raw as f64) / raw as f64;
        let dlpc = 100.0 * (lpc_b as f64 - raw as f64) / raw as f64;
        let d53_on_lpc = 100.0 * (w53lpc as f64 - lpc_b as f64) / lpc_b as f64;
        println!("{name:>10}  RAW {:>6.3}  W53 {:>6.3} ({:+.1}%)  LPC {:>6.3} ({:+.1}%)  W53+LPC {:>6.3} | 5/3-on-LPC {:+.2}%",
                 bps(raw), bps(w53), d53, bps(lpc_b), dlpc, bps(w53lpc), d53_on_lpc);
    }
    println!("\n# W53≪RAW and LPC≪RAW but 5/3-on-LPC≈0 ⇒ 5/3 and LPC are REDUNDANT decorrelators: 5/3 'works'");
    println!("# vs a weak (no-LPC) baseline but adds ~nothing on top of a strong LPC. Explains the paper");
    println!("# (5/3 helps the pipeline) AND this probe (5/3 marginal given LPC) — both correct.");
}
