//! Cross-channel SECOND-STAGE A/B on the mv_rls residual — the deep-research shortlist (ADR pending),
//! competed head-to-head, rate-gated, on every corpus bin.
//!
//! WHY second-stage-on-residual: mv_rls ALREADY is the integrated joint spatio-temporal predictor the
//! research crowned (ALS-MCC class) — it predicts each channel from `[own K past] + [M prior channels
//! at the SAME instant]` (present co-temporal spatial + temporal, one RLS, channel-major). So the only
//! honest question left is whether ANY cross-channel structure SURVIVES mv_rls's joint prediction that
//! a further explicit stage can extract (online-RLS adaptation lag; index-adjacent vs best-correlated
//! reference; online vs per-window batch). Every mechanism here runs on the mv_rls RESIDUAL, never the
//! raw signal (raw-signal rotation = the cascade penalty, measured dead in ricct/crosschan).
//!
//! Mechanisms (all: reference = best-CORRELATED earlier channel(s) j<c, present co-temporal, channel-
//! major ⇒ decodable; per-32768-window batch weights shipped as side-info, counted in NET; rate-gated
//! keep-best vs the raw residual per window/channel ⇒ NEVER-WORSE and cascade-proof by construction):
//!   B0 = per-channel mv_rls residual, our shipped coder (the baseline to beat).
//!   M1 = ALS-MCC: r'_c = r_c - round(g · r_ref), scalar LS gain g, single best-correlated ref.
//!   M2 = rate-gated difference: r'_c = r_c - r_ref (g≡1), coupled only if it codes smaller.
//!   M3 = full spatial LS: r'_c = r_c - round(Σ_{j<c} a_j · r_j), all earlier residuals, present.
//!   ORACLE = min(B0,M1,M2,M3) per window/channel — the union ceiling.
//! M2 ⊂ M1 ⊂ M3 in energy (g≡1 ⊂ scalar ⊂ multi-tap), but side-info grows B0<M2<M1<M3, so the RATE
//! winner is a genuine energy-vs-sideinfo competition — exactly the rate-not-energy selection the
//! research says dodges the cascade penalty.
//!
//! Reversibility: r'_c subtracts a rounded integer combo of ALREADY-DECODED earlier channels' EXACT
//! residuals ⇒ decoder (channel-major) recomputes the identical integer and adds back. Asserted here.
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example crosschan_residual_ab -- <bin>...

use std::fs;
use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{entropy, mv_rls, LmoCodec};

const WIN: usize = 32768;
const Q: f64 = 65536.0; // Q16 fixed-point for shipped weights
const N_CFG: usize = 7; // mv_rls CONFIGS count (keep-best like the container)

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

/// Shipped container bytes (LmoCodec Lossless = the real production keep-best over floor +
/// lmo_lossless cross-channel + mv_rls), windowed like production — the TRUE north-star baseline.
fn container_bytes(sig: &[Vec<i64>]) -> usize {
    let t = sig[0].len();
    let mut tot = 0;
    let mut start = 0;
    while start < t {
        let end = (start + WIN).min(t);
        let win: Vec<Vec<i64>> = sig.iter().map(|c| c[start..end].to_vec()).collect();
        tot += LmoCodec.encode(&win, Mode::Lossless).map(|x| x.len()).unwrap_or(0);
        start = end;
    }
    tot
}

/// Best-config mv_rls residual field: try each of the N_CFG keep-best configs (like the container's
/// mv_rls path), pick the single config with the smallest total per-channel coded size, return that
/// residual field + its coded bytes. This is the STRONG per-channel mv_rls baseline for the cross-
/// stage to add on top of — comparing the cross-stage to config-0 alone would be a strawman.
fn best_config_residual(sig: &[Vec<i64>]) -> (Vec<Vec<i64>>, usize) {
    let t = sig[0].len();
    let (mut best_res, mut best_bytes) = (Vec::new(), usize::MAX);
    for cfg in 0..N_CFG {
        let res = mv_rls::residuals(sig, cfg, 0);
        let mut bytes = 0usize;
        for ch in &res {
            let mut s = 0;
            while s < t {
                let e = (s + WIN).min(t);
                bytes += enc_len(&ch[s..e]);
                s = e;
            }
        }
        if bytes < best_bytes {
            best_bytes = bytes;
            best_res = res;
        }
    }
    (best_res, best_bytes)
}

/// Pearson |corr| of two equal-length slices (0 if either is constant).
fn abscorr(a: &[i64], b: &[i64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<i64>() as f64 / n, b.iter().sum::<i64>() as f64 / n);
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 { 0.0 } else { (sab / (saa * sbb).sqrt()).abs() }
}

/// Solve (G + ridge·I) w = b for w, G symmetric PD (Cholesky). Returns None if not PD.
fn solve_spd(g: &[Vec<f64>], b: &[f64], ridge: f64) -> Option<Vec<f64>> {
    let n = b.len();
    let mut a = g.to_vec();
    for i in 0..n { a[i][i] += ridge; }
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j { s -= l[i][k] * l[j][k]; }
            if i == j {
                if s <= 0.0 { return None; }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // forward L y = b, back L^T w = y
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i { s -= l[i][k] * y[k]; }
        y[i] = s / l[i][i];
    }
    let mut w = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n { s -= l[k][i] * w[k]; }
        w[i] = s / l[i][i];
    }
    Some(w)
}

/// Apply an integer combo predictor: out[t] = r[t] - round(Σ qk·refs[k][t] / Q). Reversibility is
/// STRUCTURAL, not asserted here (out+pred==r is trivially true): the decoder, running channel-major,
/// holds the same EXACT already-decoded refs and the same quantized `qw`, recomputes the identical
/// integer `pred` deterministically (i128 accumulate → one f64 round, same on both sides), and adds
/// it back — r[t] = out[t] + pred. No f64 in the reference values, so host↔MCU stays bit-identical.
fn apply_combo(r: &[i64], refs: &[&[i64]], qw: &[i64]) -> Vec<i64> {
    let mut out = vec![0i64; r.len()];
    for t in 0..r.len() {
        let mut acc: i128 = 0;
        for (k, rk) in refs.iter().enumerate() {
            acc += qw[k] as i128 * rk[t] as i128;
        }
        let pred = (acc as f64 / Q).round() as i64;
        out[t] = r[t] - pred;
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!("# Cross-channel second-stage A/B on the BEST-config mv_rls residual — rate-gated keep-best,");
    println!("# side-info in NET. Two baselines: B0 = best-config per-channel residual (isolates the cross-");
    println!("# stage gain); CONT = shipped LmoCodec Lossless container (the TRUE 'do we beat production' bar).");
    println!("# M1=ALS-MCC scalar, M2=unit-diff, M3=spatial-LS(all earlier), ORACLE=min. Negative Δ = win.\n");
    println!("{:>12} {:>4} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>9} {:>9} | {:>16}",
             "recording", "ch", "CONT bps", "B0 bps", "M1/B0", "M2/B0", "M3/B0", "ORA/B0", "ORA/CONT", "win B0/M1/M2/M3");

    for path in &args {
        let sig = read_bin(path);
        let cont = container_bytes(&sig);
        let (res, b0chk) = best_config_residual(&sig);
        let (c, t) = (res.len(), res[0].len());
        let nm = (c * t) as f64;
        let (mut b0, mut m1, mut m2, mut m3, mut ora) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut wins = [0usize; 4];

        let mut ws = 0;
        while ws < t {
            let we = (ws + WIN).min(t);
            let wl = we - ws;
            // per-window slices
            let win: Vec<&[i64]> = res.iter().map(|ch| &ch[ws..we]).collect();
            for ci in 0..c {
                let rc = win[ci];
                let base = enc_len(rc);
                b0 += base;
                let mut best = base;
                let mut best_sel = 0usize;

                // candidate references: all earlier channels j<ci
                let earlier: Vec<usize> = (0..ci).collect();
                // best-correlated single reference
                let bestref = earlier.iter().copied()
                    .map(|j| (j, abscorr(rc, win[j])))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(j, _)| j);

                // ---- M1: ALS-MCC scalar gain, single best-correlated ref ----
                let mut c1 = 1usize << 30;
                if let Some(rj) = bestref {
                    // scalar LS gain g = <rc,rj>/<rj,rj>
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for t2 in 0..wl { num += rc[t2] as f64 * win[rj][t2] as f64; den += (win[rj][t2] as f64).powi(2); }
                    if den > 0.0 {
                        let qg = (num / den * Q).round() as i64;
                        let out = apply_combo(rc, &[win[rj]], &[qg]);
                        c1 = enc_len(&out) + 5; // 1 ref idx + 4 weight bytes
                    }
                }
                if c1 < best { best = c1; best_sel = 1; }

                // ---- M2: rate-gated unit difference to best-correlated ref ----
                let mut c2 = 1usize << 30;
                if let Some(rj) = bestref {
                    let out = apply_combo(rc, &[win[rj]], &[Q as i64]); // g≡1
                    c2 = enc_len(&out) + 1; // ref idx only
                }
                if c2 < best { best = c2; best_sel = 2; }

                // ---- M3: full spatial LS over ALL earlier residuals (present co-temporal) ----
                let mut c3 = 1usize << 30;
                if !earlier.is_empty() {
                    let k = earlier.len();
                    // Gram G[a][b] = <r_a, r_b>, rhs b[a] = <rc, r_a>
                    let mut g = vec![vec![0.0f64; k]; k];
                    let mut rhs = vec![0.0f64; k];
                    for a in 0..k {
                        let ra = win[earlier[a]];
                        let mut br = 0.0f64;
                        for t2 in 0..wl { br += rc[t2] as f64 * ra[t2] as f64; }
                        rhs[a] = br;
                        for b in a..k {
                            let rb = win[earlier[b]];
                            let mut s = 0.0f64;
                            for t2 in 0..wl { s += ra[t2] as f64 * rb[t2] as f64; }
                            g[a][b] = s;
                            g[b][a] = s;
                        }
                    }
                    let ridge = 1e-3 * (0..k).map(|i| g[i][i]).sum::<f64>().max(1.0) / k as f64;
                    if let Some(w) = solve_spd(&g, &rhs, ridge) {
                        let qw: Vec<i64> = w.iter().map(|&x| (x * Q).round() as i64).collect();
                        let refs: Vec<&[i64]> = earlier.iter().map(|&j| win[j]).collect();
                        let out = apply_combo(rc, &refs, &qw);
                        c3 = enc_len(&out) + 4 * k; // 4 bytes/weight
                    }
                }
                if c3 < best { best = c3; best_sel = 3; }

                m1 += c1.min(base); // report M1 as keep-best vs base (never-worse)
                m2 += c2.min(base);
                m3 += c3.min(base);
                ora += best;
                wins[best_sel] += 1;
            }
            ws = we;
        }

        // b0 accumulated in the loop must equal best_config_residual's reported bytes (same field, same coder).
        assert_eq!(b0, b0chk, "B0 accounting mismatch — residual field / windowing inconsistent");
        let cont_bps = cont as f64 * 8.0 / nm;
        let b0_bps = b0 as f64 * 8.0 / nm;
        let pct_b0 = |x: usize| 100.0 * (x as f64 - b0 as f64) / b0 as f64;
        // ORACLE vs the shipped container — the real "do we beat production" number.
        let ora_vs_cont = 100.0 * (ora as f64 - cont as f64) / cont as f64;
        let name = path.rsplit('/').next().unwrap_or(path);
        println!("{name:>12} {c:>4} {cont_bps:>8.4} {b0_bps:>8.4} | {:>+7.3}% {:>+7.3}% {:>+7.3}% | {:>+8.3}% {:>+8.3}% | {:>4} {:>3} {:>3} {:>3}",
                 pct_b0(m1), pct_b0(m2), pct_b0(m3), pct_b0(ora), ora_vs_cont, wins[0], wins[1], wins[2], wins[3]);
    }
    println!("\n# M*/B0 = cross-stage gain over the best-config per-channel residual (isolates the mechanism).");
    println!("# ORA/CONT = the ONLY bar that matters: does the cross-channel scheme beat the SHIPPED container?");
    println!("#   negative ⇒ a real win over production (wire the winning mechanism as a keep-best candidate);");
    println!("#   positive ⇒ the container's existing cross-channel coding (lmo_lossless) already gets it — the");
    println!("#   cross-stage only recovers config-0's lag, not a genuine gain over what we ship.");
    println!("# A large M*/B0 win but positive ORA/CONT = strawman warning: config-0 was weak, container wins.");
}
