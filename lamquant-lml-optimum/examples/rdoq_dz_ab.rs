//! RDOQ-lite screen (ADR 0054 lossy): does a sub-0.5 deadzone quantizer offset
//! beat round-nearest at matched BPS on the 9/7 lossy path? The deadzone offset
//! is the dominant practical RDOQ gain for Laplacian wavelet coefficients
//! (JPEG2000/VVC use ≈0.375); reconstruction is unchanged (idx·q), so decode is
//! identical — δ is a pure encode-side RD choice.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example rdoq_dz_ab -- /tmp/chb01_01_60s.bin
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_optimum::lmo_pcrd97;

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
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (o, r) in orig.iter().zip(recon) {
        let m = o.iter().sum::<i64>() as f64 / o.len().max(1) as f64;
        for (a, b) in o.iter().zip(r) {
            let e = (*a - *b) as f64;
            num += e * e;
            den += (*a as f64 - m) * (*a as f64 - m);
        }
    }
    if den == 0.0 {
        0.0
    } else {
        100.0 * (num / den).sqrt()
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/chb01_01_60s.bin".to_string());
    let sig = read_window(&path);
    let nm = (sig.len() * sig[0].len()) as f64;
    let offsets = [0.5f64, 0.45, 0.40, 0.375, 0.35, 0.30, 0.25];

    println!(
        "# RDOQ-lite deadzone sweep ({path}); PRD at each target BPS per offset δ (0.5=round)"
    );
    print!("# {:>4} |", "BPS");
    for d in offsets {
        print!(" {:>13}", format!("δ={d:.3}"));
    }
    println!();

    for &target in &[3.0f64, 2.5, 2.0, 1.5, 1.0] {
        print!("  {target:>4.2} |");
        let mut base_prd = 0.0;
        for (i, &d) in offsets.iter().enumerate() {
            let body =
                lmo_pcrd97::encode_target_bps_97_dz(&sig, target, LpcMode::default(), d).unwrap();
            let recon = lmo_pcrd97::decode_97(&body).unwrap();
            let bps = body.len() as f64 * 8.0 / nm;
            let p = prd(&sig, &recon);
            if i == 0 {
                base_prd = p;
            }
            // show PRD and (for δ<0.5) the %Δ vs round-nearest at ~matched BPS
            if i == 0 {
                print!(" {bps:>5.2}/{p:>6.2} ");
            } else {
                print!(
                    " {bps:>5.2}/{p:>5.2}{:+>+.0} ",
                    -100.0 * (base_prd - p) / base_prd
                );
            }
        }
        println!();
    }
    println!("\n# cell = achieved BPS / PRD ; for δ<0.5 the trailing number = %ΔPRD vs δ=0.5 (negative = better).");
}
