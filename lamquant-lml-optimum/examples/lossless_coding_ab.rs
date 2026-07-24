//! Lever-A probe (ADR 0054 Optimum-lossless, gate 0a): does arithmetic coding
//! beat Golomb-Rice on the **lossless** 5/3 LPC residuals?
//!
//! Replicates the real default lossless per-channel pipeline via public APIs —
//! `lifting::forward_3level` → per-subband `lpc::analyze_with_mode` (the exact
//! residual the floor codes) — then compares, per subband, the floor's coder
//! (Golomb-Rice) against the Lever-A keep-smallest over {Golomb, zRLE, arith_int
//! order-0, arith_int order-1}. This is the LOSSLESS analogue of the lossy
//! `coding_ab.rs`; the transform is held fixed so it isolates the entropy swap.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example lossless_coding_ab -- /tmp/chb01_01_60s.bin [more.bin ...]
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lifting, lml, lpc, zrle};
use lamquant_lml_optimum::arith_int;

fn read_window(path: &str) -> Vec<Vec<i64>> {
    let bytes = fs::read(path).expect("read window dump");
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

/// 5/3 forward subbands for the given level depth (mirrors `lml::forward_subbands`).
fn forward_subbands(ch: &[i64], n_levels: u8) -> Vec<Vec<i64>> {
    match n_levels {
        3 => {
            let (a3, d3, d2, d1) = lifting::forward_3level(ch);
            vec![a3, d3, d2, d1]
        }
        2 => {
            let (l1a, l1d) = lifting::forward(ch);
            let (l2a, l2d) = lifting::forward(&l1a);
            vec![l2a, l2d, l1d]
        }
        1 => {
            let (a, d) = lifting::forward(ch);
            vec![a, d]
        }
        _ => vec![ch.to_vec()],
    }
}

fn main() {
    let paths: Vec<String> = {
        let a: Vec<String> = std::env::args().skip(1).collect();
        if a.is_empty() {
            vec![
                "/tmp/chb01_01_60s.bin".into(),
                "/tmp/chb01_01_mid.bin".into(),
                "/tmp/chb01_01_end.bin".into(),
            ]
        } else {
            a
        }
    };

    println!("# Lever-A probe: lossless 5/3 LPC residuals — Golomb floor vs keep-smallest(+arith)");
    println!(
        "# {:<28} {:>12} {:>12} {:>9} {:>10}",
        "window", "floor B", "A B", "dCR%", "arith%sb"
    );

    let (mut tot_floor, mut tot_a) = (0usize, 0usize);
    for path in &paths {
        let sig = read_window(path);
        let n_ch = sig.len();
        let t = sig[0].len();
        let n_levels = lml::compute_n_levels(t);

        let (mut floor_b, mut a_b) = (0usize, 0usize);
        let (mut arith_sb, mut total_sb) = (0usize, 0usize);

        for ch in &sig {
            for (sb_idx, sub) in forward_subbands(ch, n_levels).iter().enumerate() {
                let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(sub.len()));
                let (_coeffs, residual, _o) =
                    lpc::analyze_with_mode(sub, sb_idx, scoped, lml::BIAS_CTX, None);

                // Floor coder: Golomb-Rice (the mcu default per-subband payload).
                let g = golomb::encode_dense(&residual).expect("golomb").len();
                floor_b += g;

                // Lever-A keep-smallest over the available lossless coders.
                let z = zrle::encode_dense(&residual)
                    .map(|v| v.len())
                    .unwrap_or(usize::MAX);
                let a0 = arith_int::encode_dense(&residual)
                    .map(|v| v.len())
                    .unwrap_or(usize::MAX);
                let a1 = arith_int::encode_dense_ctx(&residual)
                    .map(|v| v.len())
                    .unwrap_or(usize::MAX);
                let best = g.min(z).min(a0).min(a1);
                a_b += best;
                total_sb += 1;
                if best == a0 || best == a1 {
                    arith_sb += 1;
                }
            }
        }
        tot_floor += floor_b;
        tot_a += a_b;
        let name = path.rsplit('/').next().unwrap_or(path);
        println!(
            "  {:<28} {:>12} {:>12} {:>+8.2}% {:>9.0}%",
            format!("{name} ({n_ch}x{t})"),
            floor_b,
            a_b,
            -100.0 * (floor_b - a_b) as f64 / floor_b as f64,
            100.0 * arith_sb as f64 / total_sb.max(1) as f64
        );
    }
    println!(
        "  {:<28} {:>12} {:>12} {:>+8.2}%",
        "TOTAL",
        tot_floor,
        tot_a,
        -100.0 * (tot_floor as f64 - tot_a as f64) / tot_floor as f64
    );
    println!(
        "\n# dCR% = size reduction of Lever-A vs the Golomb floor (negative = smaller = better)."
    );
    println!("# arith%sb = fraction of subbands where an arithmetic coder won keep-smallest.");
}
