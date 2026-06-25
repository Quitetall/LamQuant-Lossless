//! STEP-0 probe (Phase A, eeg-codec-design-from-port §7): does **montage-geometry**
//! reference selection beat the production **energy-greedy** cross-channel selection
//! on lossless multichannel EEG? Decision-relevant inner gate — NO production change.
//!
//! Per channel i, three reference-selection policies feed the SAME residual machinery
//! (mirrors `lmo_lossless.rs` exactly: byte-greedy single + energy-greedy multi, Q16
//! gains, integer `predict_multi`, `lml::compress` per-channel cost):
//!   - ENERGY-GREEDY : the production selection (byte-greedy 1st ref over ALL priors,
//!                     then energy-greedy add up to MAX_REFS).
//!   - GEOMETRY      : refs = the K nearest PRIOR electrodes by 3D coord (10-20 table).
//!   - UNION         : per-channel keep-smaller{raw, energy-greedy, geometry} = the A1
//!                     production behavior (never worse than energy-greedy by construction).
//! Scored by `lml::compress` channel bytes (the production metric). GATE: does UNION
//! (i.e. adding geometry candidates) beat ENERGY-GREEDY? If not → geometry adds nothing,
//! STOP A1, pivot to A2.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example geometry_refs_ab -- <window>.bin [<window2>.bin ...]
//! ```
//! Reads `<window>.bin.coords` (one `x y z` electrode coord per row, meters, or
//! `nan nan nan` for unresolved) — produced by `tools/scripts/resolve_coords.py`.

use std::fs;

use lamquant_lml_mcu::lml;

const MAX_REFS: usize = 3; // matches lmo_lossless.rs
const GEO_K: usize = 3; // geometry nearest-neighbour reference count (≤ MAX_REFS)

/// Read the `<path>.coords` sidecar (N lines `x y z` meters, `nan` ⇒ unresolved).
fn read_coords(path: &str) -> Vec<Option<[f64; 3]>> {
    fs::read_to_string(format!("{path}.coords"))
        .unwrap_or_default()
        .lines()
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().filter_map(|t| t.parse().ok()).collect();
            if v.len() == 3 && v.iter().all(|x| x.is_finite()) {
                Some([v[0], v[1], v[2]])
            } else {
                None
            }
        })
        .collect()
}

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let bytes = fs::read(path).expect("read bin");
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

// ── residual machinery: copied to MATCH lmo_lossless.rs bit-for-bit ──────────────
fn dot(a: &[i64], b: &[i64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

#[allow(clippy::needless_range_loop)]
fn joint_ls(target: &[i64], refs: &[usize], chans: &[Vec<i64>]) -> Vec<f64> {
    let k = refs.len();
    let mut a = vec![vec![0.0f64; k]; k];
    let mut b = vec![0.0f64; k];
    for r in 0..k {
        for c in 0..k {
            a[r][c] = dot(&chans[refs[r]], &chans[refs[c]]);
        }
        a[r][r] += 1e-6 * a[r][r].max(1.0);
        b[r] = dot(&chans[refs[r]], target);
    }
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        if a[col][col].abs() < 1e-12 {
            continue;
        }
        for r in (col + 1)..k {
            let f = a[r][col] / a[col][col];
            for c in col..k {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut g = vec![0.0f64; k];
    for r in (0..k).rev() {
        let mut s = b[r];
        for c in (r + 1)..k {
            s -= a[r][c] * g[c];
        }
        g[r] = if a[r][r].abs() < 1e-12 { 0.0 } else { s / a[r][r] };
    }
    g
}

fn quantize_gains(g: &[f64]) -> Option<Vec<i32>> {
    let q: Vec<i32> = g
        .iter()
        .map(|&v| (v * 65536.0).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
        .collect();
    if q.iter().all(|&x| x == 0) {
        None
    } else {
        Some(q)
    }
}

fn predict_multi(refs: &[usize], gains_q: &[i32], chans: &[Vec<i64>], k: usize) -> i64 {
    let mut acc: i64 = 0;
    for (g, &r) in gains_q.iter().zip(refs) {
        acc += *g as i64 * chans[r][k];
    }
    (acc + (1 << 15)) >> 16
}

fn residual_multi(target: &[i64], refs: &[usize], gains_q: &[i32], chans: &[Vec<i64>]) -> Vec<i64> {
    (0..target.len()).map(|k| target[k] - predict_multi(refs, gains_q, chans, k)).collect()
}

fn channel_cost(ch: &[i64]) -> usize {
    lml::compress(core::slice::from_ref(&ch.to_vec()), 0).map(|v| v.len()).unwrap_or(usize::MAX)
}

const PER_REF_OVERHEAD: usize = 6;

#[allow(clippy::needless_range_loop)]
fn best_energy_ref(residual: &[i64], chans: &[Vec<i64>], n_prior: usize, chosen: &[usize]) -> Option<usize> {
    let mut best = (usize::MAX, 0.0f64);
    for j in 0..n_prior {
        if chosen.contains(&j) {
            continue;
        }
        let den = dot(&chans[j], &chans[j]);
        if den <= 0.0 {
            continue;
        }
        let num = dot(residual, &chans[j]);
        let red = num * num / den;
        if red > best.1 {
            best = (j, red);
        }
    }
    if best.0 == usize::MAX {
        None
    } else {
        Some(best.0)
    }
}

/// Fit Q16 gains for `refs`, return (cost incl. header overhead, residual). `raw`
/// fallback (cost = channel_cost(target), no refs) if refs empty or gains vanish.
fn cost_for_refs(target: &[i64], refs: &[usize], signal: &[Vec<i64>]) -> (usize, Vec<usize>, Vec<i32>) {
    if refs.is_empty() {
        return (channel_cost(target), Vec::new(), Vec::new());
    }
    let g = joint_ls(target, refs, signal);
    let Some(gq) = quantize_gains(&g) else {
        return (channel_cost(target), Vec::new(), Vec::new());
    };
    let r = residual_multi(target, refs, &gq, signal);
    let cost = channel_cost(&r).saturating_add(refs.len() * PER_REF_OVERHEAD + 1);
    (cost, refs.to_vec(), gq)
}

/// The production energy-greedy selection (mirrors lmo_lossless::encode per-channel).
fn energy_greedy(i: usize, signal: &[Vec<i64>]) -> (usize, Vec<usize>) {
    let target = &signal[i];
    let mut best_cost = channel_cost(target);
    let mut best_refs: Vec<usize> = Vec::new();
    let mut best_resid = target.clone();
    // (1) byte-greedy single ref over ALL priors
    for j in 0..i {
        let (cost, refs, _gq) = cost_for_refs(target, &[j], signal);
        if !refs.is_empty() && cost < best_cost {
            best_cost = cost;
            best_refs = refs;
            best_resid = residual_multi(target, &[j], &quantize_gains(&joint_ls(target, &[j], signal)).unwrap(), signal);
        }
    }
    // (2) energy-greedy add
    while best_refs.len() < MAX_REFS.min(i) && !best_refs.is_empty() {
        let Some(j) = best_energy_ref(&best_resid, signal, i, &best_refs) else { break };
        let mut refs = best_refs.clone();
        refs.push(j);
        let (cost, got, gq) = cost_for_refs(target, &refs, signal);
        if !got.is_empty() && cost < best_cost {
            best_cost = cost;
            best_refs = refs;
            best_resid = residual_multi(target, &best_refs, &gq, signal);
        } else {
            break;
        }
    }
    (best_cost, best_refs)
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: geometry_refs_ab <window>.bin [...]");
        std::process::exit(2);
    }
    println!(
        "  {:>34} | {:>4} {:>4} | {:>10} {:>10} {:>10} | {:>8} {:>7}",
        "window", "nch", "res", "energy-grdy", "geometry", "UNION", "geo<eg", "gain%"
    );
    let (mut tot_eg, mut tot_geo, mut tot_union) = (0usize, 0usize, 0usize);
    for path in &paths {
        let signal = read_bin(path);
        let n_ch = signal.len();
        let coords = read_coords(path);
        let n_res = coords.iter().filter(|c| c.is_some()).count();

        let (mut eg, mut geo, mut uni) = (0usize, 0usize, 0usize);
        let mut geo_wins = 0usize;
        for i in 0..n_ch {
            // ENERGY-GREEDY (production)
            let (eg_cost, _eg_refs) = energy_greedy(i, &signal);
            // GEOMETRY: K nearest PRIOR resolved electrodes by 3D distance
            let geo_cost = if let Some(ci) = coords[i] {
                let mut cand: Vec<(f64, usize)> = (0..i)
                    .filter_map(|j| coords[j].map(|cj| {
                        let d = (0..3).map(|k| (ci[k] - cj[k]).powi(2)).sum::<f64>();
                        (d, j)
                    }))
                    .collect();
                cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let refs: Vec<usize> = cand.iter().take(GEO_K).map(|&(_, j)| j).collect();
                cost_for_refs(&signal[i], &refs, &signal).0
            } else {
                channel_cost(&signal[i]) // unresolved ⇒ raw (no geometry)
            };
            let raw_cost = channel_cost(&signal[i]);
            let union_cost = eg_cost.min(geo_cost).min(raw_cost);
            if geo_cost < eg_cost {
                geo_wins += 1;
            }
            eg += eg_cost;
            geo += geo_cost;
            uni += union_cost;
        }
        let gain = 100.0 * (eg as f64 - uni as f64) / eg as f64;
        let short: String = std::path::Path::new(path)
            .file_name().unwrap().to_string_lossy().chars().rev().take(34).collect::<String>().chars().rev().collect();
        println!(
            "  {:>34} | {:>4} {:>4} | {:>10} {:>10} {:>10} | {:>8} {:>6.2}%",
            short, n_ch, n_res, eg, geo, uni, geo_wins, gain
        );
        tot_eg += eg;
        tot_geo += geo;
        tot_union += uni;
    }
    let gain = 100.0 * (tot_eg as f64 - tot_union as f64) / tot_eg as f64;
    println!(
        "  {:>34} | {:>4} {:>4} | {:>10} {:>10} {:>10} | {:>8} {:>6.2}%",
        "TOTAL", "", "", tot_eg, tot_geo, tot_union, "", gain
    );
    println!("\n# GATE: UNION gain% > 0 means geometry adds a never-worse byte win over the");
    println!("# production energy-greedy selection. ~0% ⇒ geometry adds nothing → STOP A1, pivot to A2.");
    println!("# (geometry total may exceed energy-greedy; what matters is UNION, the keep-smaller.)");
}
