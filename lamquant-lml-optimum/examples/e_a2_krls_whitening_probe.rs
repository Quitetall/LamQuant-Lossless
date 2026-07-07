//! Research E-A2 (spec Part B): does an ONLINE nonlinear predictor extract generalizing
//! codelength from the SHIPPED mv_rls residual — i.e. structure OUTSIDE the linear class
//! Theorem A1 covers? E-A1 killed FROZEN nonlinear models (2-6% in-sample, non-stationary);
//! an ONLINE model refits per-recording, sidestepping that generalization wall — the one
//! nonlinear rung the kill-record does not cover.
//!
//! Predictor: **RFF-NLMS** — normalized LMS in a random-Fourier-feature space
//! φ(u)=sqrt(2/D)·cos(Ω u + b), Ω~N(0,1), b~U(0,2π) (Rahimi–Recht). A tractable O(D)/sample
//! kernel-LMS — the cheap-decisive instance of the spec's RFF-KRLS (full RFF-RLS is O(D²),
//! infeasible over millions of samples; it is the heavier CONFIRM iff this shows signal).
//! It predicts the mv_rls residual r[c][n] from a NONLINEAR map of the causal signal context
//! (own K past + NREF cross-channel same-instant) that mv_rls's LINEAR filter could not use.
//!
//! GATE (docs/proposals/lossless-frontier-krls-regret-2026-07.md Part B): net codelen
//! reduction on the referential lose-set (siena/eegmmidb) ≥ 0.5% AND beating the
//! cross-channel-magnitude lever (−0.23% eegmmidb / −0.19% siena). Else Front-A-nonlinear
//! is DEAD and Theorem A1's linear class is the deterministic frontier.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release --example e_a2_krls_whitening_probe -- <bin>...
//! ```

use std::f64::consts::PI;
use std::fs;

use lamquant_lml_optimum::{entropy, mv_rls};

const K: usize = 8; // own past taps in the feature vector
const NREF: usize = 3; // cross-channel same-instant taps (c-1, c-2, c-3)
const D: usize = 512; // random-Fourier-feature dimension
const MU: f64 = 0.20; // NLMS step size

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut sig = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64);
            off += 4;
        }
        sig.push(ch);
    }
    sig
}

/// Codelength in bits, windowed at W=32768 like production (entropy::encode carries a u16
/// count cap, so a whole 65536-sample channel overflows it — must chunk).
fn codelen_bits(res: &[Vec<i64>]) -> f64 {
    const W: usize = 32768;
    let mut bits = 0usize;
    for r in res {
        for chunk in r.chunks(W) {
            bits += entropy::encode(chunk).map(|v| 8 * v.len()).unwrap_or_else(|_| {
                chunk
                    .iter()
                    .map(|&v| (64 - (2 * v.unsigned_abs() + 1).leading_zeros()) as usize)
                    .sum()
            });
        }
    }
    bits as f64
}

/// Deterministic LCG → uniform [0,1); Box–Muller for the Gaussian Ω. Fixed seed ⇒ the RFF
/// map (Ω,b) is reproducible (decode-replayable if ever shipped).
struct Lcg(u64);
impl Lcg {
    fn u01(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.u01().max(1e-12);
        let u2 = self.u01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Online RFF-NLMS second stage on the mv_rls residual `r`, using nonlinear features of the
/// causal signal context. Returns the whitened residual r' = r − round(krls_pred).
fn krls_whiten(signal: &[Vec<i64>], r: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let n_ch = signal.len();
    let t = signal[0].len();
    let p = K + NREF;
    let mut lcg = Lcg(0x1234_5678_9abc_def1);
    let omega: Vec<f64> = (0..D * p).map(|_| lcg.gauss()).collect(); // D×p, γ=1 (u is normalized)
    let bias: Vec<f64> = (0..D).map(|_| 2.0 * PI * lcg.u01()).collect();
    let scale = (2.0 / D as f64).sqrt();

    let mut out = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        // normalize the feature by the channel RMS so Ω·u is O(1) (cos not saturated)
        let sigma = {
            let m2 = signal[c].iter().map(|&v| { let f = v as f64; f * f }).sum::<f64>() / t as f64;
            m2.sqrt().max(1.0)
        };
        let refs: Vec<usize> = (0..NREF.min(c)).map(|i| c - 1 - i).collect();
        let mut w = vec![0.0f64; D];
        let mut u = vec![0.0f64; p];
        let mut phi = vec![0.0f64; D];
        let mut rp = Vec::with_capacity(t);
        for n in 0..t {
            for j in 0..K {
                let idx = n as isize - 1 - j as isize;
                u[j] = if idx >= 0 { signal[c][idx as usize] as f64 / sigma } else { 0.0 };
            }
            for i in 0..NREF {
                u[K + i] = refs.get(i).map(|&rf| signal[rf][n] as f64 / sigma).unwrap_or(0.0);
            }
            let mut pn2 = 0.0f64;
            let mut pred = 0.0f64;
            for d in 0..D {
                let mut s = bias[d];
                let base = d * p;
                for j in 0..p {
                    s += omega[base + j] * u[j];
                }
                let v = scale * s.cos();
                phi[d] = v;
                pn2 += v * v;
                pred += w[d] * v;
            }
            rp.push(r[c][n] - (pred.round() as i64));
            let err = r[c][n] as f64 - pred;
            let g = MU * err / (pn2 + 1e-6);
            for d in 0..D {
                w[d] += g * phi[d];
            }
        }
        out.push(rp);
    }
    out
}

/// LINEARITY CONTROL: online NLMS predicting r[c][n] from the RAW feature u_n (own K +
/// NREF cross-channel same-instant), NO random-Fourier map. If this matches `krls_whiten`,
/// the gain is LINEAR (mv_rls's λ=0.997 forgetting discarded recoverable linear structure a
/// complementary slow predictor reclaims) — NOT nonlinear. If RFF ≫ linear ⇒ genuinely nonlinear.
fn linear_whiten(signal: &[Vec<i64>], r: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let n_ch = signal.len();
    let t = signal[0].len();
    let p = K + NREF + 1; // +1 = intercept (constant feature) — rules out "RFF just fits a bias"
    let mut out = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        let sigma = {
            let m2 = signal[c].iter().map(|&v| { let f = v as f64; f * f }).sum::<f64>() / t as f64;
            m2.sqrt().max(1.0)
        };
        let refs: Vec<usize> = (0..NREF.min(c)).map(|i| c - 1 - i).collect();
        let mut w = vec![0.0f64; p];
        let mut u = vec![0.0f64; p];
        u[p - 1] = 1.0; // intercept
        let mut rp = Vec::with_capacity(t);
        for n in 0..t {
            for j in 0..K {
                let idx = n as isize - 1 - j as isize;
                u[j] = if idx >= 0 { signal[c][idx as usize] as f64 / sigma } else { 0.0 };
            }
            for i in 0..NREF {
                u[K + i] = refs.get(i).map(|&rf| signal[rf][n] as f64 / sigma).unwrap_or(0.0);
            }
            let un2: f64 = u.iter().map(|&x| x * x).sum();
            let pred: f64 = (0..p).map(|j| w[j] * u[j]).sum();
            rp.push(r[c][n] - (pred.round() as i64));
            let g = MU * (r[c][n] as f64 - pred) / (un2 + 1e-6);
            for j in 0..p {
                w[j] += g * u[j];
            }
        }
        out.push(rp);
    }
    out
}

fn main() {
    let bins: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with("--")).collect();
    println!("E-A2 online 2nd-stage whitening of the shipped mv_rls residual (cfg 1)  [D={D}, μ={MU}]");
    println!(
        "{:>14}  {:>9}  {:>9}  {}",
        "recording", "linearΔ%", "RFFΔ%", "read"
    );
    for p in &bins {
        let sig = read_bin(p);
        let r = mv_rls::residuals(&sig, 1, 0); // shipped dominant config
        let lr = codelen_bits(&r);
        let lin = 100.0 * (codelen_bits(&linear_whiten(&sig, &r)) - lr) / lr;
        let rff = 100.0 * (codelen_bits(&krls_whiten(&sig, &r)) - lr) / lr;
        let name = p.rsplit('/').next().unwrap_or(p);
        // interpretation: nonlinear iff RFF beats the gate AND clearly beats the linear control
        let read = if rff <= -0.5 && rff < lin - 0.2 {
            "NONLINEAR (RFF beats linear)"
        } else if rff <= -0.5 || lin <= -0.5 {
            "LINEAR (2nd-stage helps, not nonlinear)"
        } else {
            "no gain"
        };
        println!("{name:>14}  {lin:>+9.3}  {rff:>+9.3}  {read}");
    }
    println!("   GATE (Part B): RFF ≤ −0.5% AND < linear control ⇒ genuinely-nonlinear lever LIVES (Front A);");
    println!("   RFF≈linear ⇒ the gain is LINEAR (mv_rls forgetting too fast — a complementary slow linear stage).");
}
