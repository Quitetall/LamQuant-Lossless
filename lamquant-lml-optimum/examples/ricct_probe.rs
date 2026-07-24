//! RICCT — Reversible Integer Cross-Channel Transform (SHOT 2, ADR 0064).
//!
//! Every spatial-decorrelation attempt so far has been PREDICTIVE (candidate A
//! lower-triangular LS, mv_rls) and hit the documented cascade penalty
//! (`crosschan.rs:23-36`): decorrelation cuts residual energy to ~40% but flattens
//! the intra-channel spectrum the 9/7 wavelet exploits, so per-bit it loses.
//!
//! RICCT is the TRANSFORM dual: a full ORTHOGONAL KLT (rotation into channel
//! principal components), made bit-exact lossless by factoring the rotation into
//! Givens rotations, each applied as 3 rounded shears (lifting) → exactly
//! invertible integer map. A rotation REDISTRIBUTES energy (orthonormal) instead
//! of subtracting a prediction — the question is whether the container codes the
//! rotated channels in fewer total bytes than the originals.
//!
//! Decorrelator = the KLT V^T where C = V D V^T. Jacobi eigen-decomposition
//! produces V as a product of Givens rotations J_1..J_K (C_{m+1}=J_m^T C_m J_m,
//! V=J_1..J_K), so V^T X = J_K^T..J_1^T X: apply each J_m transposed, in order.
//! We REPLAY the same (p,q,theta) list for forward (encode) and inverse (decode),
//! so reversibility holds regardless of the math — the round-trip gate proves it.
//!
//! Granular diagnostics, to EXPLAIN the result (not just report it):
//!  - per-recording: container bytes orig vs rotated (+ side-info), % vs baseline
//!  - energy concentration: variance distribution across channels orig vs rotated
//!  - cascade-penalty proxy: sum of entropy::encode(first-difference) per channel,
//!    orig vs rotated — if rotated >> orig, the rotation flattened the temporal
//!    spectrum (the predicted failure mechanism); if rotated < orig the spatial
//!    win survives the temporal coder.
//!
//! Run UNDER the memory cap (probes can OOM): `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example ricct_probe -- <bin>...

use lamquant_lml_mcu::codec::{Codec, LmlCodec, Mode};
use lamquant_lml_optimum::{entropy, LmoCodec};
use std::fs;

const W: usize = 32768; // production window

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
fn rnd(x: f64) -> i64 {
    x.round() as i64
}

/// A Givens rotation in plane (p,q) by angle theta, stored for replay.
#[derive(Clone, Copy)]
struct Rot {
    p: usize,
    q: usize,
    theta: f64,
}

/// Mean-removed population channel covariance (n x n).
fn covariance(sig: &[Vec<i64>]) -> Vec<Vec<f64>> {
    let n = sig.len();
    let t = sig[0].len();
    let mus: Vec<f64> = sig
        .iter()
        .map(|c| c.iter().map(|&v| v as f64).sum::<f64>() / t as f64)
        .collect();
    let mut cov = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for k in 0..t {
                s += (sig[i][k] as f64 - mus[i]) * (sig[j][k] as f64 - mus[j]);
            }
            let c = s / t as f64;
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }
    cov
}

/// Cyclic-Jacobi eigen-decomposition of symmetric `cov`. Returns the ordered list
/// of Givens rotations J_m (plane p<q, angle theta) that diagonalize it. We only
/// emit rotations with |theta| above `eps` (tiny ones don't decorrelate and cost
/// side-info). Caps total sweeps so side-info stays bounded.
fn jacobi_rotations(cov: &[Vec<f64>], max_sweeps: usize, eps: f64) -> Vec<Rot> {
    let n = cov.len();
    let mut a: Vec<Vec<f64>> = cov.to_vec();
    let mut rots = Vec::new();
    for _sweep in 0..max_sweeps {
        // largest off-diagonal magnitude this sweep (convergence check)
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off.sqrt() < eps {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                // Jacobi angle: theta zeroing a[p][q]
                let phi = 0.5 * (2.0 * apq).atan2(app - aqq);
                if phi.abs() < eps {
                    continue;
                }
                let (c, s) = (phi.cos(), phi.sin());
                // Apply J^T A J  (rotate rows/cols p,q by phi)
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp + s * akq;
                    a[k][q] = -s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk + s * aqk;
                    a[q][k] = -s * apk + c * aqk;
                }
                rots.push(Rot { p, q, theta: phi });
            }
        }
    }
    rots
}

/// Reversible integer rotation of two channels by `phi`, in place, via 3 rounded
/// shears: a += round(t*b); b += round(s*a); a += round(t*b), with t=(cos-1)/sin,
/// s=sin. Each shear is exactly invertible (inverse subtracts in reverse order).
#[inline]
fn lift_rotate(a: &mut [i64], b: &mut [i64], phi: f64) {
    let (c, s) = (phi.cos(), phi.sin());
    if s.abs() < 1e-15 {
        return;
    } // ~identity (phi≈0 or pi); skip (no-op kept identical on decode)
    let t = (c - 1.0) / s;
    for k in 0..a.len() {
        a[k] += rnd(t * b[k] as f64);
    }
    for k in 0..a.len() {
        b[k] += rnd(s * a[k] as f64);
    }
    for k in 0..a.len() {
        a[k] += rnd(t * b[k] as f64);
    }
}

#[inline]
fn lift_rotate_inv(a: &mut [i64], b: &mut [i64], phi: f64) {
    let (c, s) = (phi.cos(), phi.sin());
    if s.abs() < 1e-15 {
        return;
    }
    let t = (c - 1.0) / s;
    for k in 0..a.len() {
        a[k] -= rnd(t * b[k] as f64);
    }
    for k in 0..a.len() {
        b[k] -= rnd(s * a[k] as f64);
    }
    for k in 0..a.len() {
        a[k] -= rnd(t * b[k] as f64);
    }
}

/// Forward RICCT: apply J_m transposed (rotate by -theta_m), in order m=0..K.
fn ricct_forward(sig: &mut [Vec<i64>], rots: &[Rot]) {
    for r in rots {
        let (lo, hi) = (r.p.min(r.q), r.p.max(r.q));
        let (left, right) = sig.split_at_mut(hi);
        let a = &mut left[lo];
        let b = &mut right[0];
        lift_rotate(a, b, -r.theta);
    }
}

/// Inverse RICCT: undo each rotation in reverse order.
fn ricct_inverse(sig: &mut [Vec<i64>], rots: &[Rot]) {
    for r in rots.iter().rev() {
        let (lo, hi) = (r.p.min(r.q), r.p.max(r.q));
        let (left, right) = sig.split_at_mut(hi);
        let a = &mut left[lo];
        let b = &mut right[0];
        lift_rotate_inv(a, b, -r.theta);
    }
}

/// Per-WINDOW RICCT: recompute the KLT per 32768-window (tracks non-stationarity,
/// the most likely reason a static transform underperforms an adaptive predictor),
/// rotate that window, code with the no-spatial floor. Returns (bytes, side_bytes).
fn ricct_perwindow_floor(sig: &[Vec<i64>]) -> (usize, usize) {
    let t = sig[0].len();
    let (mut bytes, mut side) = (0usize, 0usize);
    let mut s = 0;
    while s < t {
        let e = (s + W).min(t);
        let mut win: Vec<Vec<i64>> = sig.iter().map(|ch| ch[s..e].to_vec()).collect();
        let cov = covariance(&win);
        let rots = jacobi_rotations(&cov, 12, 1e-7);
        ricct_forward(&mut win, &rots);
        bytes += LmlCodec
            .encode(&win, Mode::Lossless)
            .map(|x| x.len())
            .unwrap_or(0);
        side += rots.len() * 10;
        s = e;
    }
    (bytes, side)
}

/// Windowed bytes for any `Codec`. `LmoCodec` = full keep-best (incl. mv_rls
/// spatial candidate); `LmlCodec` = the per-channel floor (5/3+LPC+Golomb, NO
/// cross-channel) — the reference that isolates whether RICCT decorrelates at all.
fn coded_bytes<C: Codec>(codec: &C, sig: &[Vec<i64>]) -> usize {
    let t = sig[0].len();
    let mut tot = 0;
    let mut s = 0;
    while s < t {
        let e = (s + W).min(t);
        let win: Vec<Vec<i64>> = sig.iter().map(|ch| ch[s..e].to_vec()).collect();
        tot += codec
            .encode(&win, Mode::Lossless)
            .map(|x| x.len())
            .unwrap_or(0);
        s = e;
    }
    tot
}

/// Cascade-penalty proxy + energy: returns (sum entropy(first-diff) bytes, total variance).
fn temporal_codability(sig: &[Vec<i64>]) -> (usize, f64) {
    let mut bytes = 0usize;
    let mut var = 0.0f64;
    for ch in sig {
        let mut d = Vec::with_capacity(ch.len());
        let mut prev = 0i64;
        for &x in ch {
            d.push(x - prev);
            prev = x;
        }
        bytes += entropy::encode(&d).map(|g| g.len()).unwrap_or(1 << 30);
        let t = ch.len() as f64;
        let mu = ch.iter().map(|&v| v as f64).sum::<f64>() / t;
        var += ch
            .iter()
            .map(|&v| {
                let z = v as f64 - mu;
                z * z
            })
            .sum::<f64>()
            / t;
    }
    (bytes, var)
}

fn main() {
    println!("# RICCT — reversible integer cross-channel KLT (Jacobi/Givens lifting)\n");
    for path in std::env::args().skip(1) {
        let orig = read_bin(&path);
        let (nch, t) = (orig.len(), orig[0].len());
        let name = path.rsplit('/').next().unwrap_or(&path);
        if nch < 2 {
            println!("# {} ({}ch) — skip (need >=2ch)", name, nch);
            continue;
        }

        let cov = covariance(&orig);
        let eps = 1e-7;
        let rots = jacobi_rotations(&cov, 12, eps);

        // forward
        let mut rot = orig.clone();
        ricct_forward(&mut rot, &rots);
        // round-trip gate
        let mut back = rot.clone();
        ricct_inverse(&mut back, &rots);
        let rt_ok = back == orig;

        let base = coded_bytes(&LmoCodec, &orig); // best on original (incl. mv_rls spatial)
        let rric = coded_bytes(&LmoCodec, &rot); // full container on rotated
        let floor_o = coded_bytes(&LmlCodec, &orig); // no-spatial floor on original
        let floor_r = coded_bytes(&LmlCodec, &rot); // no-spatial floor on rotated = RICCT spatial stage
        let side = rots.len() * 10; // (p:u8,q:u8,theta:f64) upper bound, uncompressed
        let total = rric + side;
        let nm = (nch * t) as f64;
        let dpct = 100.0 * (total as f64 - base as f64) / base as f64;

        let (cod_o, var_o) = temporal_codability(&orig);
        let (cod_r, var_r) = temporal_codability(&rot);
        // energy concentration: fraction of total variance in the top channel
        let per_var_o = orig
            .iter()
            .map(|c| {
                let t = c.len() as f64;
                let mu = c.iter().map(|&v| v as f64).sum::<f64>() / t;
                c.iter()
                    .map(|&v| {
                        let z = v as f64 - mu;
                        z * z
                    })
                    .sum::<f64>()
                    / t
            })
            .collect::<Vec<_>>();
        let per_var_r = rot
            .iter()
            .map(|c| {
                let t = c.len() as f64;
                let mu = c.iter().map(|&v| v as f64).sum::<f64>() / t;
                c.iter()
                    .map(|&v| {
                        let z = v as f64 - mu;
                        z * z
                    })
                    .sum::<f64>()
                    / t
            })
            .collect::<Vec<_>>();
        let top_o = per_var_o.iter().cloned().fold(0.0, f64::max) / var_o.max(1.0);
        let top_r = per_var_r.iter().cloned().fold(0.0, f64::max) / var_r.max(1.0);

        let floor_dec = 100.0 * (floor_r as f64 - floor_o as f64) / floor_o as f64; // <0 ⇒ rotation decorrelates usefully at floor
        let vs_base = 100.0 * ((floor_r + side) as f64 - base as f64) / base as f64; // RICCT-spatial+side vs adaptive mv_rls
        println!("## {} ({}ch x {})", name, nch, t);
        println!(
            "  rotations={} (side~{}B={:.3}% of base)  rt={}",
            rots.len(),
            side,
            100.0 * side as f64 / base as f64,
            if rt_ok { "OK" } else { "FAIL!!" }
        );
        println!(
            "  LmoCodec base (mv_rls) = {:>10} ({:.4} bps)",
            base,
            base as f64 * 8.0 / nm
        );
        println!(
            "  LmoCodec on RICCT      = {:>10} ({:.4} bps)  + side = {:>10}  => {:+.2}% vs base",
            rric,
            rric as f64 * 8.0 / nm,
            total,
            dpct
        );
        println!("  --- isolate the spatial stage (no-spatial floor) ---");
        println!("  LmlCodec floor orig    = {:>10}", floor_o);
        println!(
            "  LmlCodec floor RICCT   = {:>10}  ({:+.2}% vs floor_orig ⇒ {} )",
            floor_r,
            floor_dec,
            if floor_dec < 0.0 {
                "rotation DECORRELATES"
            } else {
                "rotation HURTS floor"
            }
        );
        println!(
            "  RICCT-spatial + side vs adaptive mv_rls base: {:+.2}%",
            vs_base
        );
        // per-window RICCT (tracks non-stationarity) — the last variant that could rescue it
        let (pw_bytes, pw_side) = ricct_perwindow_floor(&orig);
        let pw_total = pw_bytes + pw_side;
        println!("  per-window RICCT floor = {:>10} + side {:>8} = {:>10}  ({:+.2}% vs floor_orig, {:+.2}% vs base)",
                 pw_bytes, pw_side, pw_total, 100.0*(pw_bytes as f64-floor_o as f64)/floor_o as f64, 100.0*(pw_total as f64-base as f64)/base as f64);
        println!(
            "  cascade proxy (Σ entropy(Δ) bytes):  orig={:>10}  rotated={:>10}  ({:+.1}%)",
            cod_o,
            cod_r,
            100.0 * (cod_r as f64 - cod_o as f64) / cod_o as f64
        );
        println!(
            "  energy in top channel:                orig={:.3}     rotated={:.3}",
            top_o, top_r
        );
        println!();
    }
    println!("# GATE: RICCT wins iff (container RICCT + side) < container base WITHOUT the cascade proxy blowing up.");
    println!("# If rotated codes fewer bytes but the cascade proxy rises, energy compaction is fighting temporal flattening.");
}
