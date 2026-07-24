//! Our lossy RD points for the H.BWC head-to-head: `LmoCodec.encode(TargetBps)`
//! (the current 9/7 + arithmetic + deadzone + TCQ keep-best path, auto-picked vs
//! the 5/3 floor) at each target BPS on one window; prints achieved BPS + CfP
//! mean-removed PRD. Pair with the HHI EncoderApp QP sweep on the matched EDF.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example lossy_points -- /tmp/eeg64.bin
//! ```

use std::fs;

use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{decode_any, LmoCodec};

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
        .unwrap_or_else(|| "/tmp/eeg64.bin".to_string());
    let sig = read_window(&path);
    let nm = (sig.len() * sig[0].len()) as f64;
    println!(
        "# {} ({}ch x {})  LamQuant lossy (9/7+arith+deadzone+TCQ, auto-pick)",
        path,
        sig.len(),
        sig[0].len()
    );
    println!("# {:>8} {:>10} {:>9}", "target", "bps", "PRD%");
    for &tb in &[4.0f64, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.75] {
        let body = LmoCodec.encode(&sig, Mode::TargetBps(tb)).expect("encode");
        let recon = decode_any(&body).expect("decode");
        println!(
            "  {:>8.2} {:>10.4} {:>9.4}",
            tb,
            body.len() as f64 * 8.0 / nm,
            prd(&sig, &recon)
        );
    }
}
