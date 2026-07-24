//! Component-1 de-risk: EXACT algebraic montage-graph dependency detection.
//!
//! The theoretical-architecture doc's Component 1 (and H.BWC's own montage-graph proposal, +3.185% on
//! CHB-MIT) exploits channels that are INTEGER-LINEAR-DEPENDENT on earlier channels: duplicates, and
//! bipolar-chain derivations where e.g. (A-C) = (A-B) + (B-C) EXACTLY, for every sample. Such a channel
//! carries ZERO new information; a statistical predictor (mv_rls) still pays coefficient + adaptation
//! cost to approximate it, but exact algebraic elimination codes it in ~0 bits. This lever is pure
//! integer algebra: deterministic, patent-clean, no generalization risk. But it ONLY pays where the
//! montage is redundant (bipolar), ~0 on clean referential recordings — so MEASURE before wiring.
//!
//! For each channel i (in channel-major order, so parents j<i are already decodable), we test:
//!   DUP    : ch_i == ch_j exactly for some j<i.
//!   PAIR   : ch_i == s_a*ch_a + s_b*ch_b exactly, s_* in {+1,-1}, a,b<i (classic montage derivation).
//!   INT-LS : integer-rounded least-squares of ch_i on its k best-correlated earlier channels; keep it
//!            only if the residual is EXACTLY zero for every sample (true algebraic dependency, not a fit).
//! A DC offset is allowed (montage derivations can differ by a constant): we test residual-is-constant,
//! not strictly-zero, and the constant costs one integer.
//!
//! Reported per corpus: #channels exactly eliminable, and the Rice-bits those channels currently cost
//! (their standalone best-residual under our config-0 mv_rls) as a % of the whole-recording Rice budget —
//! i.e. the ceiling on what Component 1 can save here. >~1% on any corpus ⇒ wire it as a container
//! keep-best pre-pass; ~0 everywhere ⇒ our corpora are referential and Component 1 is CHB-MIT-only.
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example montage_dependency_probe -- <bin>...

use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{mv_rls, LmoCodec};
use std::fs;

const K: usize = 6; // earlier-channel candidates for the INT-LS exact test
const WIN: usize = 32768;

/// Whole-recording shipped-container bytes (the REAL baseline — not config-0 mv_rls), windowed.
fn container_bytes(sig: &[Vec<i64>]) -> usize {
    if sig.is_empty() {
        return 0;
    }
    let t = sig[0].len();
    let (mut tot, mut start) = (0, 0);
    while start < t {
        let end = (start + WIN).min(t);
        let win: Vec<Vec<i64>> = sig.iter().map(|c| c[start..end].to_vec()).collect();
        tot += LmoCodec
            .encode(&win, Mode::Lossless)
            .map(|x| x.len())
            .unwrap_or(0);
        start = end;
    }
    tot
}

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

/// Rice-optimal codelength (bits) of an integer slice: zigzag → best Rice k over 0..=20.
fn rice_bits(vals: &[i64]) -> u64 {
    if vals.is_empty() {
        return 0;
    }
    let u: Vec<u64> = vals
        .iter()
        .map(|&v| ((v << 1) ^ (v >> 63)) as u64)
        .collect();
    let mut best = u64::MAX;
    for k in 0..=20u32 {
        let mut bits = 0u64;
        for &x in &u {
            bits += (x >> k) + 1 + k as u64;
        }
        best = best.min(bits);
    }
    best
}

/// Is `resid` constant (all equal)? Returns the constant if so. An exact algebraic dependency leaves a
/// constant DC offset (the derivation baseline); a genuine signal leaves a varying residual.
fn constant_residual(resid: &[i64]) -> Option<i64> {
    let first = *resid.first()?;
    if resid.iter().all(|&v| v == first) {
        Some(first)
    } else {
        None
    }
}

fn abscorr(a: &[i64], b: &[i64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().sum::<i64>() as f64 / n,
        b.iter().sum::<i64>() as f64 / n,
    );
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        0.0
    } else {
        (sab / (saa * sbb).sqrt()).abs()
    }
}

/// Try to find an EXACT integer-linear parent set for channel i among earlier channels.
/// Returns Some((description, parents)) if ch_i - Σ coeff·parent is constant for every sample.
fn exact_dependency(sig: &[Vec<i64>], i: usize) -> Option<String> {
    let ci = &sig[i];
    let t = ci.len();

    // DUP: exact duplicate of an earlier channel (constant offset allowed).
    for j in 0..i {
        let r: Vec<i64> = (0..t).map(|k| ci[k] - sig[j][k]).collect();
        if constant_residual(&r).is_some() {
            return Some(format!("DUP ch{j}"));
        }
    }

    // PAIR: ch_i == s_a*ch_a + s_b*ch_b exactly (montage derivation, ± only).
    for a in 0..i {
        for b in (a + 1)..i {
            for (sa, sb) in [(1i64, 1i64), (1, -1), (-1, 1), (-1, -1)] {
                let r: Vec<i64> = (0..t)
                    .map(|k| ci[k] - sa * sig[a][k] - sb * sig[b][k])
                    .collect();
                if constant_residual(&r).is_some() {
                    let sgn = |s: i64| if s < 0 { "-" } else { "+" };
                    return Some(format!("PAIR {}ch{a} {}ch{b}", sgn(sa), sgn(sb)));
                }
            }
        }
    }

    // INT-LS: integer-rounded LS on the k best-correlated earlier channels; keep only if EXACTLY reducible.
    let mut cand: Vec<(f64, usize)> = (0..i).map(|j| (abscorr(ci, &sig[j]), j)).collect();
    cand.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    let parents: Vec<usize> = cand.into_iter().take(K).map(|(_, j)| j).collect();
    if !parents.is_empty() {
        // Normal-equations LS (f64) of ci on {parents}, then round coeffs to integers.
        let p = parents.len();
        let mut ata = vec![vec![0.0f64; p]; p];
        let mut atb = vec![0.0f64; p];
        for k in 0..t {
            for (u, &pu) in parents.iter().enumerate() {
                atb[u] += sig[pu][k] as f64 * ci[k] as f64;
                for (v, &pv) in parents.iter().enumerate() {
                    ata[u][v] += sig[pu][k] as f64 * sig[pv][k] as f64;
                }
            }
        }
        if let Some(coef) = solve(&mut ata, &mut atb) {
            let icoef: Vec<i64> = coef.iter().map(|c| c.round() as i64).collect();
            if icoef.iter().any(|&c| c != 0) {
                let r: Vec<i64> = (0..t)
                    .map(|k| {
                        ci[k]
                            - parents
                                .iter()
                                .zip(&icoef)
                                .map(|(&pj, &c)| c * sig[pj][k])
                                .sum::<i64>()
                    })
                    .collect();
                if constant_residual(&r).is_some() {
                    let terms: String = parents
                        .iter()
                        .zip(&icoef)
                        .filter(|(_, &c)| c != 0)
                        .map(|(&pj, &c)| format!("{c:+}·ch{pj}"))
                        .collect::<Vec<_>>()
                        .join(" ");
                    return Some(format!("INT-LS {terms}"));
                }
            }
        }
    }
    None
}

/// Gaussian elimination solve A x = b (A p×p, destroyed in place). None if singular.
fn solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv =
            (col..n).max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap())?;
        if a[piv][col].abs() < 1e-6 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in (col + 1)..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0f64; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            s -= a[col][c] * x[c];
        }
        x[col] = s / a[col][col];
    }
    Some(x)
}

fn main() {
    println!(
        "# Component-1 exact algebraic montage dependency. #dep = channels EXACTLY eliminable; "
    );
    println!("# save% = their standalone (config-0 mv_rls) Rice-bits as a fraction of the whole recording");
    println!("# (the ceiling Component-1 can save here). >~1% ⇒ wire the keep-best pre-pass.\n");
    println!(
        "{:>12} {:>4} {:>6} {:>7}  detail",
        "recording", "C", "#dep", "save%"
    );

    for path in std::env::args().skip(1) {
        let sig = read_bin(&path);
        let c = sig.len();
        // Standalone residual bits per channel (config-0 mv_rls, the intra+joint default), for the ceiling.
        let res = mv_rls::residuals(&sig, 0, 0);
        let per_ch: Vec<u64> = res.iter().map(|r| rice_bits(r)).collect();
        let total: u64 = per_ch.iter().sum();

        let mut deps = Vec::new();
        let mut dep_idx: Vec<usize> = Vec::new();
        let mut saved = 0u64;
        for i in 0..c {
            if let Some(desc) = exact_dependency(&sig, i) {
                saved += per_ch[i];
                dep_idx.push(i);
                deps.push(format!("ch{i}={desc}"));
            }
        }
        let name = path.rsplit('/').next().unwrap_or(&path);
        let pct = if total > 0 {
            100.0 * saved as f64 / total as f64
        } else {
            0.0
        };
        let detail = if deps.is_empty() {
            "(none)".to_string()
        } else {
            deps.join("  ")
        };
        println!("{name:>12} {c:>4} {:>6} {pct:>6.2}%  {detail}", deps.len());

        // REAL baseline: does the SHIPPED container already capture these (multi-ref selection)?
        // Compare container(all channels) vs container(independent channels only) + derivation metadata.
        // ~16 bytes/derived channel covers [n_parents][parent idx×][coeff×][DC:i64] generously.
        if !dep_idx.is_empty() {
            let full = container_bytes(&sig);
            let reduced_sig: Vec<Vec<i64>> = (0..c)
                .filter(|i| !dep_idx.contains(i))
                .map(|i| sig[i].clone())
                .collect();
            let reduced = container_bytes(&reduced_sig) + dep_idx.len() * 16;
            let delta = full as f64 - reduced as f64;
            let dpct = if full > 0 {
                100.0 * delta / full as f64
            } else {
                0.0
            };
            println!(
                "{:>12}      container: full {full}B  elim {reduced}B  → REAL save {delta:+.0}B ({dpct:+.2}%)",
                ""
            );
        }
    }
    println!(
        "\n# Exact elimination sends a dependent channel to ~0 bits (one DC constant), which no"
    );
    println!("# statistical predictor fully reaches (it pays coeff+adaptation). Patent-clean integer algebra.");
}
