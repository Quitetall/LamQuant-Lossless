//! Stage-0 oracle: does choosing an MV-RLS adaptation config per channel beat
//! the shipped whole-window config choice?
//!
//! Current `mv_rls::encode_bc` tries six configs, but chooses one config for all
//! channels. Every channel is already coded as an independent entropy chunk and
//! only reads exact, previously decoded raw channels, so a per-channel config tag
//! is losslessly replayable. This probe charges the exact proposed wire cost:
//!
//! ```text
//! [n_ch u16][t u32][k u8]
//! per channel: [cfg u8][bias_ctx u8][payload_len u32][entropy payload]
//! ```
//!
//! It also charges the existing LMO lossless-body and outer-container overhead,
//! then compares against complete shipped `LmoCodec::encode(Lossless)` bytes.
//! Negative `PC/CONT` means a real candidate; positive means config granularity
//! cannot beat production even at its oracle ceiling.
//!
//! Run:
//! `cargo run -p lamquant-lml-optimum --features encode --release --example per_channel_config_oracle_probe -- <bin>...`

use std::fs;

use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_mcu::lpc;
use lamquant_lml_optimum::{entropy, lmo_lossless, mv_rls, LmoCodec};

const W: usize = 32768;
const N_CFG: usize = 6;
const BC_CTXS: [usize; 4] = [8, 16, 32, 64];

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read bin");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let expected = 8 + nch * t * 4;
    assert_eq!(b.len(), expected, "malformed bin length");
    let mut off = 8;
    let mut signal = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64);
            off += 4;
        }
        signal.push(ch);
    }
    signal
}

/// Exact `encode_one_bc` per-channel wire cost excluding the shared 9-byte
/// MV-RLS header: `[bc u8][glen u32][payload]`.
fn channel_bc_cost(residual: &[i64]) -> usize {
    let mut best = entropy::encode(residual).expect("entropy").len();
    for &ctx in &BC_CTXS {
        let mut corrected = residual.to_vec();
        lpc::bias_cancel(&mut corrected, ctx);
        best = best.min(entropy::encode(&corrected).expect("entropy bc").len());
    }
    1 + 4 + best
}

fn main() {
    println!("# Per-channel MV-RLS config oracle. PC includes cfg tag + exact framing.");
    println!("# CONT = complete shipped LmoCodec lossless stream. Negative PC/CONT = win.\n");
    println!(
        "{:>14} {:>4} {:>8} {:>8} {:>8} | {:>9} {:>9} {:>18}",
        "recording",
        "ch",
        "CONT bps",
        "MVBC bps",
        "PC bps",
        "PC/MVBC",
        "PC/CONT",
        "cfg channel wins"
    );

    for path in std::env::args().skip(1) {
        let signal = read_bin(&path);
        let (nch, t) = (signal.len(), signal[0].len());
        let mut cont_total = 0usize;
        let mut mvbc_total = 0usize;
        let mut pc_total = 0usize;
        let mut cfg_wins = [0usize; N_CFG];

        let mut start = 0usize;
        while start < t {
            let end = (start + W).min(t);
            let win: Vec<Vec<i64>> = signal.iter().map(|c| c[start..end].to_vec()).collect();

            let complete = LmoCodec
                .encode(&win, Mode::Lossless)
                .expect("container encode");
            let body = lmo_lossless::encode(&win).expect("lossless body");
            assert!(
                complete.len() >= body.len(),
                "outer container shorter than body"
            );
            let outer_overhead = complete.len() - body.len();
            cont_total += complete.len();

            let mvbc = mv_rls::encode_bc(&win).expect("mvrls bc");
            mvbc_total += outer_overhead + 5 + nch + mvbc.len();

            let residuals: Vec<Vec<Vec<i64>>> = (0..N_CFG)
                .map(|cfg| mv_rls::residuals(&win, cfg, 0))
                .collect();
            let mut per_channel = 0usize;
            for c in 0..nch {
                let mut best = usize::MAX;
                let mut best_cfg = 0usize;
                for cfg in 0..N_CFG {
                    let cost = channel_bc_cost(&residuals[cfg][c]);
                    if cost < best {
                        best = cost;
                        best_cfg = cfg;
                    }
                }
                cfg_wins[best_cfg] += 1;
                per_channel += 1 + best; // one cfg tag beyond existing [bc,len,payload]
            }
            // Existing LMO body header: version+feature+n_ch + raw n_refs bytes + coder mode.
            // Proposed per-channel stream header: n_ch+t+k = 7 bytes.
            pc_total += outer_overhead + (5 + nch) + 7 + per_channel;
            start = end;
        }

        let nm = (nch * t) as f64;
        let bps = |bytes: usize| bytes as f64 * 8.0 / nm;
        let pct = |a: usize, b: usize| 100.0 * (a as f64 - b as f64) / b as f64;
        let wins = cfg_wins
            .iter()
            .enumerate()
            .filter(|(_, n)| **n != 0)
            .map(|(cfg, n)| format!("{cfg}:{n}"))
            .collect::<Vec<_>>()
            .join(",");
        let name = path.rsplit('/').next().unwrap_or(&path);
        println!(
            "{name:>14} {nch:>4} {:>8.4} {:>8.4} {:>8.4} | {:>+8.3}% {:>+8.3}% {:>18}",
            bps(cont_total),
            bps(mvbc_total),
            bps(pc_total),
            pct(pc_total, mvbc_total),
            pct(pc_total, cont_total),
            wins
        );
    }
}
