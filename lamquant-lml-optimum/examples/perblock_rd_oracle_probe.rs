//! Peer-codec Phase 1 (ADR 0085) — per-block RD ORACLE de-risk BEFORE the full IntDCT+coder build.
//!
//! The forensic mine proved single-lever grafts fail; the peer competitor must be H.BWC's INTEGRATED
//! per-block RD system. Before sinking weeks into a clean-room IntDCT + context coder, this probe
//! measures the CEILING of the core idea: per block, keep-best between
//!   SKIP = our time-domain mv_rls residual (the transform-skip side we already own), and
//!   DCT  = block DCT (float proxy) → CORRECTED intra prediction (spectral-neighbor LMS) + cross-channel,
//!          modeled by running mv_rls over the per-block COEFFICIENT VECTOR with FREQUENCY as the time
//!          axis: predicts coeff[c][k] from coeff[c][k-1..k-K] (adjacent-freq intra — the axis my earlier
//!          proxies got WRONG, using temporal) + co-located coeff of prior channels (cc).
//! Rice-optimal codelength per block per candidate (real, non-overfitting, per-block additive) + 1 mode
//! bit. Baselines: shipped container (CONT, real bytes) + SKIP-only Rice (calibrates Rice overhead).
//!
//! GATE: per-block-RD ORACLE ≤ CONT on referential (siena/eegmmidb/tuar) ⇒ the integrated architecture
//! can beat production ⇒ proceed to the full build (Phase 2 IntDCT). ORACLE ≈ SKIP-only on referential
//! (i.e. DCT never wins a block) ⇒ even the ceiling of per-block RD doesn't help us ⇒ stop and reconsider
//! before the heavy engineering. NB float-DCT proxy is OPTIMISTIC for the DCT side.
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example perblock_rd_oracle_probe -- [B] <bin>...

use std::fs;
use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{entropy, mv_rls, LmoCodec};

const WIN: usize = 32768;
const N_CFG: usize = 7;

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
fn enc_len(v: &[i64]) -> usize {
    entropy::encode(v).map(|g| g.len()).expect("entropy encode")
}
fn enc_windowed(v: &[i64]) -> usize {
    let (mut tot, mut s) = (0, 0);
    while s < v.len() {
        let e = (s + WIN).min(v.len());
        tot += enc_len(&v[s..e]);
        s = e;
    }
    tot
}

fn dct_basis(b: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0f64; b]; b];
    let (s0, s) = ((1.0 / b as f64).sqrt(), (2.0 / b as f64).sqrt());
    for k in 0..b {
        let sc = if k == 0 { s0 } else { s };
        for n in 0..b {
            m[k][n] = sc * (core::f64::consts::PI * (n as f64 + 0.5) * k as f64 / b as f64).cos();
        }
    }
    m
}

fn container_bytes(sig: &[Vec<i64>]) -> usize {
    let t = sig[0].len();
    let (mut tot, mut start) = (0, 0);
    while start < t {
        let end = (start + WIN).min(t);
        let win: Vec<Vec<i64>> = sig.iter().map(|c| c[start..end].to_vec()).collect();
        tot += LmoCodec.encode(&win, Mode::Lossless).map(|x| x.len()).unwrap_or(0);
        start = end;
    }
    tot
}

fn best_config_residual(sig: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let (mut best_res, mut best_bytes) = (Vec::new(), usize::MAX);
    for cfg in 0..N_CFG {
        let res = mv_rls::residuals(sig, cfg, 0);
        let bytes: usize = res.iter().map(|c| enc_windowed(c)).sum();
        if bytes < best_bytes {
            best_bytes = bytes;
            best_res = res;
        }
    }
    best_res
}

/// Rice-optimal codelength (bits) of an integer slice: zigzag → best Rice k over 0..=20.
fn rice_bits(vals: &[i64]) -> u64 {
    if vals.is_empty() {
        return 0;
    }
    let u: Vec<u64> = vals.iter().map(|&v| ((v << 1) ^ (v >> 63)) as u64).collect();
    let mut best = u64::MAX;
    for k in 0..=20u32 {
        let mut bits = 0u64;
        for &x in &u {
            bits += (x >> k) + 1 + k as u64;
        }
        if bits < best {
            best = bits;
        }
    }
    best
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cli_b = args.first().and_then(|a| a.parse::<usize>().ok());
    if cli_b.is_some() {
        args.remove(0);
    }
    let blocks: Vec<usize> = cli_b.map(|b| vec![b]).unwrap_or_else(|| vec![256, 512]);

    println!("# Peer-codec Phase-1 per-block RD ORACLE. Rice-codelength bps. CONT = shipped container (real).");
    println!("# SKIP = mv_rls time residual; DCT = block DCT + spectral-intra-LMS + cc (mv_rls over coeff");
    println!("# vector, freq=time). ORACLE = per-block keep-best(SKIP,DCT)+mode bit. Δ vs CONT + vs SKIP-only.\n");
    println!("{:>12} {:>4} {:>8} {:>8} {:>8} {:>8} | {:>9} {:>9} | {:>7}",
             "recording", "B", "CONT", "SKIP", "DCT", "ORACLE", "ORA/CONT", "ORA/SKIP", "DCTwin%");

    for path in &args {
        let sig = read_bin(path);
        let (c, t) = (sig.len(), sig[0].len());
        let nm = (c * t) as f64;
        let cont_bps = container_bytes(&sig) as f64 * 8.0 / nm;
        let res = best_config_residual(&sig);

        for &b in &blocks {
            let basis = dct_basis(b);
            let (mut skip_bits, mut dct_bits, mut ora_bits) = (0u64, 0u64, 0u64);
            let (mut blk_tot, mut dct_win) = (0usize, 0usize);
            let mut start = 0;
            while start + b <= t {
                // DCT each channel's block → coeff matrix (C × B), freq as the mv_rls "time" axis
                let coefm: Vec<Vec<i64>> = (0..c)
                    .map(|ci| {
                        (0..b)
                            .map(|k| {
                                let row = &basis[k];
                                let mut acc = 0.0f64;
                                for n in 0..b {
                                    acc += row[n] * sig[ci][start + n] as f64;
                                }
                                acc.round() as i64
                            })
                            .collect()
                    })
                    .collect();
                let rescoef = mv_rls::residuals(&coefm, 0, 0); // spectral-intra + cc prediction
                for ci in 0..c {
                    let s = rice_bits(&res[ci][start..start + b]);
                    let d = rice_bits(&rescoef[ci]);
                    skip_bits += s;
                    dct_bits += d;
                    ora_bits += s.min(d) + 1; // +1 mode bit
                    blk_tot += 1;
                    if d < s {
                        dct_win += 1;
                    }
                }
                start += b;
            }
            // tail (< b) coded as skip only, added to all three equally
            if start < t {
                for ci in 0..c {
                    let s = rice_bits(&res[ci][start..]);
                    skip_bits += s;
                    dct_bits += s;
                    ora_bits += s;
                }
            }
            let bps = |bits: u64| bits as f64 / nm;
            let skip_bps = bps(skip_bits);
            let ora_bps = bps(ora_bits);
            let dco = 100.0 * (ora_bps - cont_bps) / cont_bps;
            let dsk = 100.0 * (ora_bits as f64 - skip_bits as f64) / skip_bits as f64;
            let name = path.rsplit('/').next().unwrap_or(path);
            println!("{name:>12} {b:>4} {cont_bps:>8.4} {skip_bps:>8.4} {:>8.4} {ora_bps:>8.4} | {dco:>+8.3}% {dsk:>+8.3}% | {:>6.2}%",
                     bps(dct_bits), 100.0 * dct_win as f64 / blk_tot.max(1) as f64);
        }
    }
    println!("\n# ORA/SKIP = the per-block-RD lever (how much DCT-of-a-block ever wins over our residual).");
    println!("# ORA/CONT = does the per-block-RD ceiling beat production? ≤0 ⇒ build the full codec (Phase 2).");
    println!("# DCTwin% ≈ 0 on referential ⇒ even the CORRECTED spectral-intra DCT never wins a block ⇒ the");
    println!("# integrated transform path won't help us; the peer codec's gain (if any) must come elsewhere.");
}
