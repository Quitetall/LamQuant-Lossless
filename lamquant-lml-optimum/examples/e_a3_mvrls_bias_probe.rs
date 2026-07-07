//! Research E-A3 (from the E-A2 bias lead): does Sriraam context-adaptive bias cancellation
//! on the SHIPPED mv_rls residual recover the −1..−2% the E-A2 intercept probe estimated on
//! the referential lose-set? mv_rls's regressor has no intercept and its residual coder
//! (`entropy::encode`) does NOT apply the `bias_cancel` the LML-floor LPC path does
//! (`lml.rs:897`, `BIAS_CTX=32`), so a per-channel DC/baseline survives into the coded residual.
//!
//! This measures the REAL bias_cancel (a faithful copy of `lpc.rs::bias_cancel`, BIAS_CTX=32)
//! — a causal running-mean subtraction, exactly invertible (`bias_restore`), integer,
//! decode-replayable. Reports the ALWAYS-ON delta and the per-channel KEEP-BEST delta
//! (never-worse: code each channel with bias-cancel iff it is smaller).
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release --example e_a3_mvrls_bias_probe -- <bin>...
//! ```

use std::fs;

use lamquant_lml_optimum::{entropy, mv_rls};

const BIAS_CTX: usize = 32; // mirrors lml.rs::BIAS_CTX

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut sig = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64);
            off += 4;
        }
        sig.push(ch);
    }
    sig
}

/// Floor division (toward −∞) — mirrors `lpc.rs::floor_div`.
#[inline]
fn floor_div(a: i64, b: i64) -> i64 {
    let d = a / b;
    if (a ^ b) < 0 && d * b != a { d - 1 } else { d }
}

/// Faithful copy of `lpc.rs::bias_cancel` (causal running-mean subtraction, invertible).
fn bias_cancel(data: &mut [i64], ctx_len: usize) {
    let mask = if ctx_len.is_power_of_two() { ctx_len - 1 } else { 0 };
    let use_mask = mask != 0;
    let mut buf = vec![0i64; ctx_len];
    let mut running_sum = 0i64;
    let ctx = ctx_len as i64;
    for i in 0..data.len() {
        let bias = floor_div(running_sum, ctx);
        let val = data[i];
        data[i] -= bias;
        let slot = if use_mask { i & mask } else { i % ctx_len };
        let old = buf[slot];
        buf[slot] = val;
        running_sum += val - old;
    }
}

/// Windowed (W=32768, like production; entropy::encode has a u16 count cap) codelength, one channel.
fn codelen_ch(r: &[i64]) -> usize {
    const W: usize = 32768;
    let mut bits = 0usize;
    for chunk in r.chunks(W) {
        bits += entropy::encode(chunk).map(|v| 8 * v.len()).unwrap_or_else(|_| {
            chunk
                .iter()
                .map(|&v| (64 - (2 * v.unsigned_abs() + 1).leading_zeros()) as usize)
                .sum()
        });
    }
    bits
}

fn main() {
    let bins: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with("--")).collect();
    // sweep the bias context (power-of-two ⇒ fast mask path). A longer context tracks the slow
    // baseline the E-A2 intercept found; the ship form is a per-channel keep-best over these.
    let ctxs = [8usize, 16, BIAS_CTX, 64, 128];
    println!("E-A3 bias_cancel keep-best Δ% on the shipped mv_rls residual (cfg 1) — context sweep");
    print!("{:>14}", "recording");
    for &c in &ctxs {
        print!("  {:>9}", format!("ctx{c}"));
    }
    println!("  {:>9}", "best");
    for p in &bins {
        let sig = read_bin(p);
        let r = mv_rls::residuals(&sig, 1, 0); // shipped dominant config
        let base: usize = (0..sig.len()).map(|c| codelen_ch(&r[c])).sum();
        let name = p.rsplit('/').next().unwrap_or(p);
        print!("{name:>14}");
        let mut best = 0.0f64;
        for &ctx in &ctxs {
            let kb: usize = (0..sig.len())
                .map(|c| {
                    let l0 = codelen_ch(&r[c]);
                    let mut rc = r[c].clone();
                    bias_cancel(&mut rc, ctx);
                    l0.min(codelen_ch(&rc))
                })
                .sum();
            let d = 100.0 * (kb as f64 - base as f64) / base as f64;
            best = best.min(d);
            print!("  {d:>+9.3}");
        }
        println!("  {best:>+9.3}");
    }
    println!("   per-channel keep-best (never-worse). Ship form: a CODER_MV_RLS_BC flag/channel (+ a 2-bit ctx index).");
}
