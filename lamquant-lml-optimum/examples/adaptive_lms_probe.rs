//! Adaptive predictor probe (ADR 0054): HHI's edge on ECG/EEG is its per-sample
//! adaptive LMS-16 predictor; ours is block-static LPC (fit on 256 samples,
//! applied statically) — which can't track non-stationary ECG. Test an integer
//! sign-sign LMS predictor (deterministic ⇒ losslessly reversible: the decoder
//! re-runs the same adaptation on the reconstructed history) vs the current path.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example adaptive_lms_probe -- /tmp/ecg_100.bin 30000
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lifting, lml, lpc};

fn read_window(path: &str) -> Vec<Vec<i64>> {
    let bytes = fs::read(path).expect("read");
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

fn wavelet_lpc_golomb(x: &[i64]) -> usize {
    let (a3, d3, d2, d1) = lifting::forward_3level(x);
    let mut bytes = 0;
    for (sb, sub) in [a3, d3, d2, d1].iter().enumerate() {
        let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(sub.len()));
        let (coeffs, residual, _o) = lpc::analyze_with_mode(sub, sb, scoped, lml::BIAS_CTX, None);
        bytes += 1 + 4 * coeffs.len() + golomb::encode_dense(&residual).expect("g").len();
    }
    bytes
}

/// Sign-sign LMS adaptive prediction. Q16 weights; pred = (Σ w_k·x[n-1-k])>>16;
/// e[n]=x[n]-pred; w_k += mu·sign(e)·sign(x[n-1-k]). Fully integer/deterministic
/// ⇒ the decoder reproduces it from reconstructed history (lossless).
fn signlms_residual(x: &[i64], order: usize, mu: i64) -> Vec<i64> {
    let mut w = vec![0i64; order];
    let mut e = vec![0i64; x.len()];
    for n in 0..x.len() {
        let mut pred = 0i64;
        for k in 0..order {
            if n > k {
                pred += w[k] * x[n - 1 - k];
            }
        }
        pred >>= 16;
        let err = x[n] - pred;
        e[n] = err;
        let se = err.signum();
        if se != 0 {
            for k in 0..order {
                if n > k {
                    w[k] += mu * se * x[n - 1 - k].signum();
                }
            }
        }
    }
    e
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/ecg_100.bin".to_string());
    let w: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30000);
    let sig = read_window(&path);
    let t = sig[0].len().min(w);
    let nm = (sig.len() * t) as f64;
    println!(
        "# adaptive-LMS probe: {} ({}ch, {}). bytes; A=current 5/3+LPC.",
        path,
        sig.len(),
        t
    );

    let mut ta = 0usize;
    let mut best: Vec<(String, usize)> = Vec::new();
    for full in &sig {
        ta += wavelet_lpc_golomb(&full[..t]);
    }
    for &(order, mu) in &[
        (16usize, 8i64),
        (16, 32),
        (32, 8),
        (32, 32),
        (32, 128),
        (16, 128),
    ] {
        let mut tot = 0usize;
        for full in &sig {
            let e = signlms_residual(&full[..t], order, mu);
            tot += golomb::encode_dense(&e).expect("g").len();
        }
        best.push((format!("LMS o={order} mu={mu}"), tot));
    }
    println!(
        "  {:<18} {:>9} {:>8} {:>9}",
        "config", "bytes", "Δvs A", "bps"
    );
    println!(
        "  {:<18} {:>9} {:>8} {:>9.3}",
        "A (5/3+LPC)",
        ta,
        "—",
        ta as f64 * 8.0 / nm
    );
    for (name, b) in &best {
        let dpct = -100.0 * (ta as f64 - *b as f64) / ta as f64;
        println!(
            "  {:<18} {:>9} {:>+7.1}% {:>9.3}",
            name,
            b,
            dpct,
            *b as f64 * 8.0 / nm
        );
    }
    println!("# HHI ECG ≈ 3.31 bps. (LMS golomb only — no wavelet, no LPC header.)");
}
