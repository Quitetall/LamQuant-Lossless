//! Full-recording lossless BPS for the H.BWC head-to-head: windows a long
//! recording into ≤W-sample chunks, runs the Optimum codec (LmoCodec Lossless,
//! cross-channel auto-pick) per chunk, verifies bit-exact round-trip, and reports
//! total BPS = Σbytes·8/(N·M) — directly comparable to HHI's bits/sample on the
//! same EDF.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example lossless_full -- /tmp/ecg_full.bin [window_size]
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

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/ecg_full.bin".to_string());
    let w: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32768);
    let sig = read_window(&path);
    let (n_ch, t) = (sig.len(), sig[0].len());
    let nm = (n_ch * t) as f64;

    let mut total_bytes = 0usize;
    let mut ok = true;
    let mut off = 0usize;
    while off < t {
        let end = (off + w).min(t);
        let chunk: Vec<Vec<i64>> = sig.iter().map(|ch| ch[off..end].to_vec()).collect();
        let body = LmoCodec.encode(&chunk, Mode::Lossless).expect("encode");
        total_bytes += body.len();
        if decode_any(&body).expect("decode") != chunk {
            ok = false;
        }
        off = end;
    }
    let name = path.rsplit('/').next().unwrap_or(&path);
    println!(
        "{:<24} {}ch x {} (W={w}) : {} bytes, {:.4} bps  rt={}",
        name,
        n_ch,
        t,
        total_bytes,
        total_bytes as f64 * 8.0 / nm,
        if ok { "ok" } else { "FAIL" }
    );
}
