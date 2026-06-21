//! Multi-reference cross-channel probe (ADR 0054 Lever C follow-up): does
//! predicting each channel from a small LS combo of K prior references beat the
//! shipped single-best-reference codec (gate 0c / Phase 2: −10.5% bipolar, −18%
//! 64ch referential)?
//!
//! Per channel: greedily select up to K prior references (OMP-style, by residual-
//! energy reduction), fit ONE joint LS over the selected set, code the exact
//! integer residual, measure end-to-end Golomb bytes (5/3 → LPC → Golomb) with a
//! 6-byte/ref overhead charged, keep-smaller vs raw. K=1 reproduces the shipped
//! codec; K=2,3 test the follow-up. Lossless ⇒ exact priors ⇒ no amplification.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example crosschan_multiref_ab -- /tmp/eeg64.bin /tmp/chb01_01_60s.bin
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lifting, lml, lpc};

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

fn pipeline_bytes(ch: &[i64], n_levels: u8) -> usize {
    let subs = match n_levels {
        3 => {
            let (a3, d3, d2, d1) = lifting::forward_3level(ch);
            vec![a3, d3, d2, d1]
        }
        _ => vec![ch.to_vec()],
    };
    let mut bytes = 0usize;
    for (sb_idx, sub) in subs.iter().enumerate() {
        let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(sub.len()));
        let (coeffs, residual, _o) = lpc::analyze_with_mode(sub, sb_idx, scoped, lml::BIAS_CTX, None);
        bytes += 1 + 4 * coeffs.len() + golomb::encode_dense(&residual).expect("golomb").len();
    }
    bytes
}

fn dot(a: &[i64], b: &[i64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// Greedy OMP: pick up to `k` prior refs from `priors` that best reduce the
/// running residual energy when predicting `target`.
fn greedy_refs(target: &[i64], priors: &[Vec<i64>], k: usize) -> Vec<usize> {
    let mut resid: Vec<f64> = target.iter().map(|&v| v as f64).collect();
    let mut chosen: Vec<usize> = Vec::new();
    for _ in 0..k {
        let (mut best_j, mut best_red) = (usize::MAX, 0.0f64);
        for (j, p) in priors.iter().enumerate() {
            if chosen.contains(&j) {
                continue;
            }
            let den: f64 = p.iter().map(|&x| x as f64 * x as f64).sum();
            if den <= 0.0 {
                continue;
            }
            let num: f64 = resid.iter().zip(p).map(|(&r, &x)| r * x as f64).sum();
            let red = num * num / den; // energy removed by the LS 1-tap on the residual
            if red > best_red {
                best_red = red;
                best_j = j;
            }
        }
        if best_j == usize::MAX || best_red <= 0.0 {
            break;
        }
        chosen.push(best_j);
        // Update residual = target − LS-projection onto the chosen set so far.
        let gains = joint_ls(target, &chosen, priors);
        for (t_idx, r) in resid.iter_mut().enumerate() {
            let mut p = 0.0;
            for (g, &j) in gains.iter().zip(&chosen) {
                p += g * priors[j][t_idx] as f64;
            }
            *r = target[t_idx] as f64 - p;
        }
    }
    chosen
}

/// Joint LS gains predicting `target` from the selected `refs` (Gaussian elim on
/// the small normal equations `X'X g = X'y`).
#[allow(clippy::needless_range_loop)] // index-based Gaussian elimination
fn joint_ls(target: &[i64], refs: &[usize], priors: &[Vec<i64>]) -> Vec<f64> {
    let k = refs.len();
    let mut a = vec![vec![0.0f64; k]; k];
    let mut b = vec![0.0f64; k];
    for r in 0..k {
        for c in 0..k {
            a[r][c] = dot(&priors[refs[r]], &priors[refs[c]]);
        }
        a[r][r] += 1e-6 * a[r][r].max(1.0); // ridge for numerical safety
        b[r] = dot(&priors[refs[r]], target);
    }
    // Gaussian elimination with partial pivoting.
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

fn multiref_residual(target: &[i64], refs: &[usize], gains: &[f64], priors: &[Vec<i64>]) -> Vec<i64> {
    (0..target.len())
        .map(|k| {
            let mut p = 0.0;
            for (g, &j) in gains.iter().zip(refs) {
                p += g * priors[j][k] as f64;
            }
            target[k] - p.round() as i64
        })
        .collect()
}

const PER_REF_OVERHEAD: usize = 6; // ref_idx(2) + gain(4)

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let paths = if paths.is_empty() {
        vec!["/tmp/eeg64.bin".into(), "/tmp/chb01_01_60s.bin".into()]
    } else {
        paths
    };
    println!("# Multi-reference cross-channel probe (end-to-end Golomb bytes, keep-smaller per channel)");
    println!("# {:<24} {:>10} {:>10} {:>10} {:>10}", "window", "floor B", "K=1", "K=2", "K=3");

    for path in &paths {
        let sig = read_window(path);
        let (n_ch, t) = (sig.len(), sig[0].len());
        let n_levels = lml::compute_n_levels(t);
        let base: Vec<usize> = sig.iter().map(|c| pipeline_bytes(c, n_levels)).collect();
        let base_sum: usize = base.iter().sum();

        let mut tot = [base[0]; 3]; // ch0 raw, for K=1,2,3
        for i in 1..n_ch {
            let priors: Vec<Vec<i64>> = sig[..i].to_vec();
            for (ki, &k) in [1usize, 2, 3].iter().enumerate() {
                let refs = greedy_refs(&sig[i], &priors, k.min(i));
                let chosen = if refs.is_empty() {
                    base[i]
                } else {
                    let gains = joint_ls(&sig[i], &refs, &priors);
                    let r = multiref_residual(&sig[i], &refs, &gains, &priors);
                    (pipeline_bytes(&r, n_levels) + refs.len() * PER_REF_OVERHEAD + 1).min(base[i])
                };
                tot[ki] += chosen;
            }
        }
        let name = path.rsplit('/').next().unwrap_or(path);
        let pct = |b: usize| -100.0 * (base_sum as f64 - b as f64) / base_sum as f64;
        println!(
            "  {:<24} {:>10} {:>9.2}% {:>9.2}% {:>9.2}%",
            format!("{name} ({n_ch}x{t})"),
            base_sum,
            pct(tot[0]),
            pct(tot[1]),
            pct(tot[2])
        );
    }
    println!("\n# K=1 = shipped single-best codec; K=2/3 = multi-reference follow-up. % = vs floor.");
}
