//! 9/7 vs 5/3 transform A/B on a real recording (ADR 0054 Phase 3, lever 2).
//!
//! Reads a raw window dump `[n_ch:u32 LE][t:u32 LE][i32 LE samples, channel-major]`,
//! encodes it at each target BPS with BOTH the integer 5/3 PCRD path (the `mcu`
//! floor, `compress_target_bps_pcrd`) and the float 9/7 PCRD path
//! (`lmo_pcrd97::encode_target_bps_97`), decodes each, and prints the achieved
//! bits-per-sample and CfP mean-removed PRD for each. Isolates the transform —
//! identical quant / LPC / entropy chain on both sides.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example transform_ab -- /tmp/chb01_01_60s.bin
//! ```

use std::fs;
use std::path::PathBuf;

use lamquant_lml_mcu::lml;
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
            let v = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            ch.push(v as i64);
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
    let _ = PathBuf::from(&path);
    let sig = read_window(&path);
    let n_ch = sig.len();
    let t = sig[0].len();
    let nm = (n_ch * t) as f64;
    println!("# window: {n_ch} ch x {t} samples  ({path})");
    println!(
        "# {:>4} | {:>10} {:>9} | {:>10} {:>9} | {:>8}",
        "BPS", "5/3 bps", "5/3 PRD", "9/7 bps", "9/7 PRD", "dPRD%"
    );

    for &target in &[3.5f64, 3.0, 2.5, 2.0, 1.5, 1.0] {
        let b53 = lml::compress_target_bps_pcrd(&sig, target, LpcMode::default()).expect("5/3");
        let r53 = lml::decompress(&b53).expect("5/3 decode");
        let bps53 = b53.len() as f64 * 8.0 / nm;
        let prd53 = prd(&sig, &r53);

        let b97 = lmo_pcrd97::encode_target_bps_97(&sig, target, LpcMode::default()).expect("9/7");
        let r97 = lmo_pcrd97::decode_97(&b97).expect("9/7 decode");
        let bps97 = b97.len() as f64 * 8.0 / nm;
        let prd97 = prd(&sig, &r97);

        let dprd = (prd97 - prd53) / prd53 * 100.0;
        println!(
            "  {:>4.2} | {:>10.3} {:>9.3} | {:>10.3} {:>9.3} | {:>+8.1}",
            target, bps53, prd53, bps97, prd97, dprd
        );
    }
}
