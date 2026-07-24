//! End-to-end verification of adaptive transform-skip through the REAL MCU codec (`lml::compress`),
//! not the per-subband proxy — so it exercises adaptive-AIC LPC + bias-cancel + the actual keep-best.
//! Reads the LAMQUANT_TRY_TRANSFORM_SKIP env flag from the environment (do NOT set it in-process); run
//! twice to compare:
//!   LAMQUANT_TRY_TRANSFORM_SKIP=0 cargo run ... --example mcu_transform_skip_verify -- <W> <bin>...
//!   LAMQUANT_TRY_TRANSFORM_SKIP=1 cargo run ... --example mcu_transform_skip_verify -- <W> <bin>...
//! The delta in total bytes is the shippable per-packet win (never-worse: flag=1 total ≤ flag=0 total).

use lamquant_lml_mcu::lml;
use std::fs;

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

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let w: usize = args.remove(0).parse().expect("first arg = window size");
    let flag = std::env::var("LAMQUANT_TRY_TRANSFORM_SKIP").unwrap_or_default();
    println!("# real lml::compress  W={w}  LAMQUANT_TRY_TRANSFORM_SKIP={flag:?}");
    for path in &args {
        let sig = read_bin(path);
        let (c, t) = (sig.len(), sig[0].len());
        let mut bytes = 0usize;
        let mut start = 0;
        while start < t {
            let end = (start + w).min(t);
            let win: Vec<Vec<i64>> = sig.iter().map(|ch| ch[start..end].to_vec()).collect();
            bytes += lml::compress(&win, 0).map(|v| v.len()).unwrap_or(1 << 30);
            start = end;
        }
        let bps = bytes as f64 * 8.0 / (c * t) as f64;
        let name = path.rsplit('/').next().unwrap_or(path);
        println!("{name:>12}  {bytes:>10} B   {bps:>8.4} bps");
    }
}
