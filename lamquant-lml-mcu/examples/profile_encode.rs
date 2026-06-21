//! Lossless encode stage profiler (ADR 0054 perf, 2026-06-21).
//!
//! No sampling profiler is available in this env, so this times the real default
//! per-channel lossless pipeline stage-by-stage via the public APIs the orchestrator
//! uses — `lifting::forward_3level` → per subband `lpc::analyze_with_mode`
//! (autocorr + Levinson + residual) → `golomb::encode_dense` — on a real CHB-MIT
//! window, to answer "what actually dominates encode?" (and put the autocorr SIMD
//! headroom in context).
//!
//! ```text
//! cargo run -p lamquant-lml-mcu --release --example profile_encode -- /tmp/chb01_01_60s.bin
//! ```

use std::time::Instant;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lifting, lml, lpc};

fn read_window(path: &str) -> Vec<Vec<i64>> {
    let bytes = std::fs::read(path).expect("read window dump");
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

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/chb01_01_60s.bin".to_string());
    let sig = read_window(&path);
    let n_ch = sig.len();
    let t = sig[0].len();
    let n_levels = lml::compute_n_levels(t);
    let mode = LpcMode::default();

    let iters = 200u32;
    let (mut t_lift, mut t_lpc, mut t_ent) = (0u128, 0u128, 0u128);
    let mut t_lpc_fixed = 0u128;
    let mut autocorr_calls = 0u64;
    let mut total_bytes = 0usize;

    let t_all = Instant::now();
    for it in 0..iters {
        for ch in &sig {
            // Stage 1: lifting (5/3 forward, 3-level).
            let t0 = Instant::now();
            let subbands: Vec<Vec<i64>> = match n_levels {
                3 => {
                    let (a3, d3, d2, d1) = lifting::forward_3level(ch);
                    vec![a3, d3, d2, d1]
                }
                _ => vec![ch.clone()],
            };
            t_lift += t0.elapsed().as_nanos();

            for (sb_idx, sub) in subbands.iter().enumerate() {
                // Stage 2: LPC analyze (autocorr + Levinson + residual filter).
                let scoped = lml::scope_lpc_mode(mode, lml::lpc_max_order(sub.len()));
                let t1 = Instant::now();
                let (_coeffs, residual, _order) =
                    lpc::analyze_with_mode(sub, sb_idx, scoped, lml::BIAS_CTX, None);
                t_lpc += t1.elapsed().as_nanos();
                if it == 0 {
                    autocorr_calls += 1;
                }

                // Same stage under Fixed mode (single [3,3,6,8] order, no adaptive
                // search) — the delta vs above is the cost of the order search.
                let t1f = Instant::now();
                let _ = lpc::analyze_with_mode(sub, sb_idx, LpcMode::Fixed, lml::BIAS_CTX, None);
                t_lpc_fixed += t1f.elapsed().as_nanos();

                // Stage 3: entropy (Golomb-Rice).
                let t2 = Instant::now();
                let g = golomb::encode_dense(&residual).expect("golomb");
                t_ent += t2.elapsed().as_nanos();
                if it == 0 {
                    total_bytes += g.len();
                }
            }
        }
    }
    let wall = t_all.elapsed().as_nanos();

    let per = |x: u128| x as f64 / iters as f64 / 1e6; // ms/window
    let lift = per(t_lift);
    let lpc_ms = per(t_lpc);
    let ent = per(t_ent);
    let sum = lift + lpc_ms + ent;
    let wall_ms = wall as f64 / iters as f64 / 1e6;

    println!("# window: {n_ch} ch x {t} samples ({path}), {n_levels}-level, {iters} iters");
    println!("# encoded ~{total_bytes} bytes/window ({:.3} bps)", total_bytes as f64 * 8.0 / (n_ch * t) as f64);
    println!("# {:<14} {:>10} {:>8}", "stage", "ms/window", "% sum");
    println!("  {:<14} {:>10.3} {:>7.1}%", "lifting", lift, 100.0 * lift / sum);
    let lpc_fixed = per(t_lpc_fixed);
    println!("  {:<14} {:>10.3} {:>7.1}%", "lpc_analyze", lpc_ms, 100.0 * lpc_ms / sum);
    println!(
        "  {:<14} {:>10.3}          (Fixed mode; adaptive order search ≈ {:.3} ms = {:.0}% of lpc)",
        "  └ lpc_fixed", lpc_fixed, lpc_ms - lpc_fixed, 100.0 * (lpc_ms - lpc_fixed) / lpc_ms
    );
    println!("  {:<14} {:>10.3} {:>7.1}%", "golomb", ent, 100.0 * ent / sum);
    println!("  {:<14} {:>10.3}", "stage sum", sum);
    println!("  {:<14} {:>10.3}", "wall total", wall_ms);

    // Autocorr share estimate: the microbench measured the current AVX2 autocorr at
    // ~1.106 µs/call (seg_len=256, order 16). Encode makes `autocorr_calls` of them.
    let autocorr_ms = autocorr_calls as f64 * 1.106 / 1000.0;
    println!(
        "\n# autocorr: {autocorr_calls} calls/window x ~1.106 µs (avx2_current) = ~{autocorr_ms:.3} ms/window\n\
         #   ≈ {:.1}% of lpc_analyze, {:.1}% of stage sum.  A 2.1x autocorr kernel saves ~{:.3} ms (~{:.1}% of encode).",
        100.0 * autocorr_ms / lpc_ms,
        100.0 * autocorr_ms / sum,
        autocorr_ms * (1.0 - 1.0 / 2.14),
        100.0 * (autocorr_ms * (1.0 - 1.0 / 2.14)) / sum
    );
}
