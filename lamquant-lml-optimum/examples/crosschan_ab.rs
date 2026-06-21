//! Lever-4 end-to-end A/B (ADR 0054 Phase 3): does closed-loop cross-channel
//! prediction's ~1.5 bit/sample headroom survive quantization?
//!
//! Baseline = the current best LMO lossy path (joint 9/7 + PCRD + integer
//! arithmetic). Cross-channel = fit a causal cross-channel predictor, then encode
//! each channel's residual against the *reconstructed* prior channels (closed
//! loop, no drift) through the SAME 9/7+arith path. The predictor-coefficient
//! header bytes are charged to the cross-channel rate, so the BPS is honest.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example crosschan_ab -- /tmp/chb01_01_60s.bin
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_optimum::{crosschan, lmo_pcrd97};

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

fn prd(orig: &[Vec<i64>], recon: &[Vec<i64>]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (o, r) in orig.iter().zip(recon.iter()) {
        let m = o.iter().sum::<i64>() as f64 / o.len().max(1) as f64;
        for (a, b) in o.iter().zip(r.iter()) {
            let e = (*a - *b) as f64;
            num += e * e;
            den += (*a as f64 - m) * (*a as f64 - m);
        }
    }
    if den == 0.0 { 0.0 } else { 100.0 * (num / den).sqrt() }
}

fn energy(x: &[Vec<i64>]) -> f64 {
    let mut s = 0.0;
    for ch in x {
        let m = ch.iter().sum::<i64>() as f64 / ch.len().max(1) as f64;
        for &v in ch {
            s += (v as f64 - m) * (v as f64 - m);
        }
    }
    s
}

/// Closed-loop cross-channel encode→decode at `target` BPS with ridge
/// `ridge_frac`. Returns (total bytes incl. coeff header, reconstruction,
/// residual-energy / original-energy ratio).
fn crosschan_roundtrip(sig: &[Vec<i64>], target: f64, ridge_frac: f64) -> (usize, Vec<Vec<i64>>, f64) {
    let n_ch = sig.len();
    let t = sig[0].len();
    let pred = crosschan::fit_predictor_ridged(sig, ridge_frac);
    let mut total = pred.to_bytes().len();
    let mut x_hat: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    let mut residuals: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for i in 0..n_ch {
        let p = if i == 0 { vec![0i64; t] } else { pred.predict_channel(i, &x_hat, t) };
        let resid: Vec<i64> = (0..t).map(|k| sig[i][k] - p[k]).collect();
        residuals.push(resid.clone());
        let body = lmo_pcrd97::encode_target_bps_97(&[resid], target, LpcMode::default())
            .expect("cc residual encode");
        total += body.len();
        let rhat = lmo_pcrd97::decode_97(&body).expect("cc residual decode");
        let xh: Vec<i64> = (0..t).map(|k| p[k] + rhat[0][k]).collect();
        x_hat.push(xh);
    }
    let ratio = energy(&residuals) / energy(sig).max(1.0);
    (total, x_hat, ratio)
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/chb01_01_60s.bin".to_string());
    let sig = read_window(&path);
    let n_ch = sig.len();
    let t = sig[0].len();
    let nm = (n_ch * t) as f64;
    println!("# window: {n_ch} ch x {t} samples ({path})");

    // The closed-loop prediction operates on lossy priors, so the clean-signal LS
    // coefficients amplify quant noise on the near-singular montage. Sweep the
    // ridge (Wiener regularization) at each operating point to find the sweet spot
    // between decorrelation and noise amplification. resE = residual/orig energy.
    for &target in &[3.0f64, 2.0, 1.5] {
        let b97 = lmo_pcrd97::encode_target_bps_97(&sig, target, LpcMode::default()).expect("9/7");
        let r97 = lmo_pcrd97::decode_97(&b97).expect("9/7 decode");
        let prd97 = prd(&sig, &r97);
        println!(
            "\n# target {target:.1} BPS  baseline 9/7+arith: {:.3} bps, PRD {prd97:.3}",
            b97.len() as f64 * 8.0 / nm
        );
        println!("# {:>8} | {:>8} {:>9} {:>8} | {:>8}", "ridge", "cc bps", "cc PRD", "resE", "dPRD%");
        for &ridge in &[1e-6f64, 1e-2, 1e-1, 0.3, 1.0, 3.0] {
            let (cc_bytes, r_cc, ratio) = crosschan_roundtrip(&sig, target, ridge);
            let prd_cc = prd(&sig, &r_cc);
            println!(
                "  {:>8.0e} | {:>8.3} {:>9.3} {:>8.3} | {:>+8.1}",
                ridge,
                cc_bytes as f64 * 8.0 / nm,
                prd_cc,
                ratio,
                (prd_cc - prd97) / prd97 * 100.0
            );
        }
    }
}
