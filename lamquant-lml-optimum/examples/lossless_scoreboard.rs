//! Optimum-lossless scoreboard (ADR 0054 Lever C): the full LMO container
//! `Mode::Lossless` auto-pick (id=2 cross-channel vs id=0 floor) vs the raw 5/3
//! floor, end-to-end with bit-exact round-trip verification, on real recordings.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example lossless_scoreboard -- /tmp/chb01_01_60s.bin [more.bin ...]
//! ```

use std::fs;

use lamquant_lml_mcu::codec::{Codec, LmlCodec, Mode};
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

fn main() {
    let paths: Vec<String> = {
        let a: Vec<String> = std::env::args().skip(1).collect();
        if a.is_empty() {
            vec!["/tmp/chb01_01_60s.bin".into(), "/tmp/chb01_01_mid.bin".into(), "/tmp/chb01_01_end.bin".into()]
        } else { a }
    };
    println!("# Optimum-lossless scoreboard: LMO Lossless (auto-pick id=2 vs id=0) vs 5/3 floor");
    println!("# {:<26} {:>11} {:>11} {:>8} {:>8} {:>6}", "window", "floor B", "LMO B", "dCR%", "floor bps", "rt");

    let (mut tf, mut tl) = (0usize, 0usize);
    for path in &paths {
        let sig = read_window(path);
        let (n_ch, t) = (sig.len(), sig[0].len());
        let nm = (n_ch * t) as f64;

        let floor = LmlCodec.encode(&sig, Mode::Lossless).expect("floor");
        let lmo = LmoCodec.encode(&sig, Mode::Lossless).expect("lmo");
        // Bit-exact round-trip through the universal dispatch.
        let back = decode_any(&lmo).expect("decode");
        let rt = back == sig;

        tf += floor.len();
        tl += lmo.len();
        let name = path.rsplit('/').next().unwrap_or(path);
        println!(
            "  {:<26} {:>11} {:>11} {:>+7.2}% {:>8.3} {:>6}",
            format!("{name} ({n_ch}x{t})"),
            floor.len(), lmo.len(),
            -100.0 * (floor.len() as f64 - lmo.len() as f64) / floor.len() as f64,
            floor.len() as f64 * 8.0 / nm,
            if rt { "ok" } else { "FAIL" }
        );
        assert!(rt, "round-trip must be bit-exact");
    }
    println!(
        "  {:<26} {:>11} {:>11} {:>+7.2}%",
        "TOTAL", tf, tl, -100.0 * (tf as f64 - tl as f64) / tf as f64
    );
    println!("\n# dCR% = LMO lossless vs the 5/3 floor (negative = smaller). rt = bit-exact round-trip.");
}
