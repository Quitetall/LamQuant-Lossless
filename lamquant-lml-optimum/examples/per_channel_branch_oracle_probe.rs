//! Stage-0 oracle: choose the shipped spatial or MV-RLS branch per channel.
//!
//! The production codec chooses one branch for an entire 32K-sample window.
//! This probe constructs a conservative, independently decodable research
//! packet: each spatial channel carries a complete single-channel LML/RLS
//! packet, while each MV-RLS channel carries its config, bias-cancel tag,
//! length, and entropy payload. All branch tags, reference metadata, lengths,
//! the research-body header, and the real outer-container overhead are charged.
//!
//! Negative `HYB/CONT` is the maximum deterministic gain available from this
//! seam without inventing a new predictor or entropy model.

use std::fs;

use lamquant_lml_mcu::{
    codec::{Codec, Mode},
    lml, lpc,
};
use lamquant_lml_optimum::{entropy, lmo_lossless, mv_rls, rls, LmoCodec};
use rayon::prelude::*;

const W: usize = 32768;
const MAX_REFS: usize = 3;
const N_CFG: usize = 6;
const BC_CTXS: [usize; 4] = [8, 16, 32, 64];

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read bin");
    assert!(b.len() >= 8, "bin header truncated");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    assert_eq!(b.len(), 8 + nch * t * 4, "malformed bin length");
    let mut off = 8;
    (0..nch)
        .map(|_| {
            (0..t)
                .map(|_| {
                    let x = i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64;
                    off += 4;
                    x
                })
                .collect()
        })
        .collect()
}

fn dot(a: &[i64], b: &[i64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

#[allow(clippy::needless_range_loop)]
fn joint_ls(target: &[i64], refs: &[usize], chans: &[Vec<i64>]) -> Vec<f64> {
    let k = refs.len();
    let mut a = vec![vec![0.0; k]; k];
    let mut b = vec![0.0; k];
    for r in 0..k {
        for c in 0..k {
            a[r][c] = dot(&chans[refs[r]], &chans[refs[c]]);
        }
        a[r][r] += 1e-6 * a[r][r].max(1.0);
        b[r] = dot(&chans[refs[r]], target);
    }
    for col in 0..k {
        let mut piv = col;
        for r in col + 1..k {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        if a[col][col].abs() < 1e-12 {
            continue;
        }
        for r in col + 1..k {
            let f = a[r][col] / a[col][col];
            for c in col..k {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut g = vec![0.0; k];
    for r in (0..k).rev() {
        let mut s = b[r];
        for c in r + 1..k {
            s -= a[r][c] * g[c];
        }
        g[r] = if a[r][r].abs() < 1e-12 {
            0.0
        } else {
            s / a[r][r]
        };
    }
    g
}

fn quantize(g: &[f64]) -> Option<Vec<i32>> {
    let q: Vec<i32> = g
        .iter()
        .map(|&v| {
            (v * 65536.0)
                .round()
                .clamp(i32::MIN as f64, i32::MAX as f64) as i32
        })
        .collect();
    (!q.iter().all(|&x| x == 0)).then_some(q)
}

fn residual(target: &[i64], refs: &[usize], gains: &[i32], chans: &[Vec<i64>]) -> Vec<i64> {
    (0..target.len())
        .map(|n| {
            let acc: i64 = refs
                .iter()
                .zip(gains)
                .map(|(&r, &g)| g as i64 * chans[r][n])
                .sum();
            target[n] - ((acc + (1 << 15)) >> 16)
        })
        .collect()
}

fn packet_cost(ch: &[i64]) -> usize {
    let one = vec![ch.to_vec()];
    let a = lml::compress(&one, 0)
        .map(|v| v.len())
        .unwrap_or(usize::MAX);
    let b = rls::encode(&one).map(|v| v.len()).unwrap_or(usize::MAX);
    a.min(b)
}

fn best_energy_ref(
    resid: &[i64],
    chans: &[Vec<i64>],
    n_prior: usize,
    chosen: &[usize],
) -> Option<usize> {
    (0..n_prior)
        .filter(|j| !chosen.contains(j))
        .filter_map(|j| {
            let den = dot(&chans[j], &chans[j]);
            (den > 0.0).then(|| (j, dot(resid, &chans[j]).powi(2) / den))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|x| x.0)
}

/// Exact production spatial search for one channel. Returns reference count and
/// residual. Its packet cost uses a complete single-channel packet, so the
/// research wire is executable without relying on private LML internals.
fn spatial_channel(i: usize, signal: &[Vec<i64>]) -> (usize, Vec<i64>) {
    let mut best_cost = packet_cost(&signal[i]);
    let mut best_refs = Vec::new();
    let mut best_resid = signal[i].clone();

    for j in 0..i {
        let Some(gq) = quantize(&joint_ls(&signal[i], &[j], signal)) else {
            continue;
        };
        let r = residual(&signal[i], &[j], &gq, signal);
        let cost = packet_cost(&r).saturating_add(7);
        if cost < best_cost {
            best_cost = cost;
            best_refs = vec![j];
            best_resid = r;
        }
    }
    while !best_refs.is_empty() && best_refs.len() < MAX_REFS.min(i) {
        let Some(j) = best_energy_ref(&best_resid, signal, i, &best_refs) else {
            break;
        };
        let mut refs = best_refs.clone();
        refs.push(j);
        let Some(gq) = quantize(&joint_ls(&signal[i], &refs, signal)) else {
            break;
        };
        let r = residual(&signal[i], &refs, &gq, signal);
        let cost = packet_cost(&r).saturating_add(1 + refs.len() * 6);
        if cost < best_cost {
            best_cost = cost;
            best_refs = refs;
            best_resid = r;
        } else {
            break;
        }
    }
    (best_refs.len(), best_resid)
}

fn mv_channel_cost(residuals: &[Vec<Vec<i64>>], c: usize) -> (usize, usize) {
    let mut best = (usize::MAX, 0);
    for cfg in 0..N_CFG {
        let res = &residuals[cfg][c];
        let mut payload = entropy::encode(res).expect("entropy").len();
        for &ctx in &BC_CTXS {
            let mut corrected = res.clone();
            lpc::bias_cancel(&mut corrected, ctx);
            payload = payload.min(entropy::encode(&corrected).expect("entropy bc").len());
        }
        // branch + cfg + bias tag + payload length + payload
        let cost = 1 + 1 + 1 + 4 + payload;
        if cost < best.0 {
            best = (cost, cfg);
        }
    }
    best
}

fn main() {
    println!("# Exact-cost per-channel spatial/MV-RLS branch oracle");
    println!("# Spatial channels carry full one-channel LML/RLS packets (conservative framing).\n");
    println!(
        "{:>14} {:>4} {:>9} {:>9} | {:>9} {:>9} {:>12}",
        "recording", "ch", "CONT bps", "HYB bps", "HYB/CONT", "MV chans", "MV cfg wins"
    );

    let mut grand_cont = 0usize;
    let mut grand_hybrid = 0usize;
    let mut recordings = 0usize;
    for path in std::env::args().skip(1) {
        let signal = read_bin(&path);
        let (nch, t) = (signal.len(), signal[0].len());
        let mut cont_total = 0usize;
        let mut hybrid_total = 0usize;
        let mut mv_wins = 0usize;
        let mut cfg_wins = [0usize; N_CFG];

        let mut start = 0;
        while start < t {
            let end = (start + W).min(t);
            let win: Vec<Vec<i64>> = signal.iter().map(|c| c[start..end].to_vec()).collect();
            let complete = LmoCodec.encode(&win, Mode::Lossless).expect("container");
            let body = lmo_lossless::encode(&win).expect("body");
            let outer = complete.len() - body.len();
            cont_total += complete.len();

            let mv_res: Vec<Vec<Vec<i64>>> = (0..N_CFG)
                .map(|cfg| mv_rls::residuals(&win, cfg, 0))
                .collect();
            let channel_costs: Vec<(usize, bool, usize)> = (0..nch)
                .into_par_iter()
                .map(|c| {
                    let (n_refs, spatial_res) = spatial_channel(c, &win);
                    // branch + n_refs + refs/gains + payload length + complete packet
                    let spatial = 1 + 1 + n_refs * 6 + 4 + packet_cost(&spatial_res);
                    let (mv, cfg) = mv_channel_cost(&mv_res, c);
                    if mv < spatial {
                        (mv, true, cfg)
                    } else {
                        (spatial, false, 0)
                    }
                })
                .collect();
            let mut channels = 0usize;
            for (cost, is_mv, cfg) in channel_costs {
                channels += cost;
                if is_mv {
                    mv_wins += 1;
                    cfg_wins[cfg] += 1;
                }
            }
            // Research body: version, feature, n_ch, t. Channel framing above.
            hybrid_total += outer + 8 + channels;
            start = end;
        }

        let nm = (nch * t) as f64;
        let bps = |bytes: usize| bytes as f64 * 8.0 / nm;
        let pct = 100.0 * (hybrid_total as f64 - cont_total as f64) / cont_total as f64;
        let cfgs = cfg_wins
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(cfg, n)| format!("{cfg}:{n}"))
            .collect::<Vec<_>>()
            .join(",");
        let name = path.rsplit('/').next().unwrap_or(&path);
        println!(
            "{name:>14} {nch:>4} {:>9.4} {:>9.4} | {pct:>+8.3}% {mv_wins:>9} {cfgs:>12}",
            bps(cont_total),
            bps(hybrid_total)
        );
        grand_cont += cont_total;
        grand_hybrid += hybrid_total;
        recordings += 1;
    }
    if recordings > 1 {
        let pct = 100.0 * (grand_hybrid as f64 - grand_cont as f64) / grand_cont as f64;
        println!("\nTOTAL {recordings} recordings: CONT={grand_cont} HYB={grand_hybrid} HYB/CONT={pct:+.3}%");
    }
}
