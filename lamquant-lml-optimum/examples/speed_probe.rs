//! Throughput + CR: Optimum (LmoCodec) vs the MCU floor (LmlCodec, 5/3+LPC+Golomb)
//! on the same windows. Reports bps (compression) and encode/decode Msample/s +
//! MiB/s (16-bit input). Pair with `time EncoderApp …` on the matching EDF for the
//! Optimum-vs-HHI speed head-to-head.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example speed_probe -- /tmp/eeg64.bin [window_size]
//! ```

use std::fs;
use std::time::Instant;

use lamquant_lml_mcu::codec::{Codec, LmlCodec, Mode};
use lamquant_lml_optimum::{decode_any, LmoCodec};

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

/// Encode + decode all windows of `sig`, returning (total_bytes, encode_secs,
/// decode_secs). `enc`/`dec` are the codec's encode/decode closures.
fn run(
    sig: &[Vec<i64>],
    w: usize,
    enc: &dyn Fn(&[Vec<i64>]) -> Vec<u8>,
    dec: &dyn Fn(&[u8]) -> Vec<Vec<i64>>,
) -> (usize, f64, f64, bool) {
    let t = sig[0].len();
    let chunks: Vec<Vec<Vec<i64>>> = (0..t)
        .step_by(w)
        .map(|off| {
            let end = (off + w).min(t);
            sig.iter().map(|ch| ch[off..end].to_vec()).collect()
        })
        .collect();

    let t0 = Instant::now();
    let bodies: Vec<Vec<u8>> = chunks.iter().map(|c| enc(c)).collect();
    let enc_s = t0.elapsed().as_secs_f64();
    let total: usize = bodies.iter().map(|b| b.len()).sum();

    let t1 = Instant::now();
    let mut ok = true;
    for (b, c) in bodies.iter().zip(&chunks) {
        if &dec(b) != c {
            ok = false;
        }
    }
    let dec_s = t1.elapsed().as_secs_f64();
    (total, enc_s, dec_s, ok)
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/eeg64.bin".to_string());
    let w: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(32768);
    let sig = read_window(&path);
    let (n_ch, t) = (sig.len(), sig[0].len());
    let nm = (n_ch * t) as f64;

    let opt = run(
        &sig, w,
        &|c| LmoCodec.encode(c, Mode::Lossless).expect("opt enc"),
        &|b| decode_any(b).expect("opt dec"),
    );
    let mcu = run(
        &sig, w,
        &|c| LmlCodec.encode(c, Mode::Lossless).expect("mcu enc"),
        &|b| LmlCodec.decode(b).expect("mcu dec"),
    );

    // Throughput in Msample/s and MiB/s (16-bit input = 2 bytes/sample).
    let msa = |secs: f64| nm / secs / 1e6;
    let mibs = |secs: f64| nm * 2.0 / secs / (1024.0 * 1024.0);
    let name = path.rsplit('/').next().unwrap_or(&path);

    println!("# speed+CR: {name}  {n_ch}ch × {t}  (W={w}, 16-bit input)");
    println!("  {:>10} | {:>8} | {:>9} {:>8} | {:>9} {:>8} | {}",
        "codec", "bps", "enc Msa/s", "enc MiB/s", "dec Msa/s", "dec MiB/s", "rt");
    let row = |label: &str, r: &(usize, f64, f64, bool)| {
        println!("  {:>10} | {:>8.4} | {:>9.2} {:>8.1} | {:>9.2} {:>8.1} | {}",
            label, r.0 as f64 * 8.0 / nm, msa(r.1), mibs(r.1), msa(r.2), mibs(r.2),
            if r.3 { "ok" } else { "FAIL" });
    };
    row("Optimum", &opt);
    row("MCU floor", &mcu);
    println!("  Optimum vs MCU: CR {:+.1}%   enc {:.2}× {}   dec {:.2}× {}",
        100.0 * (opt.0 as f64 - mcu.0 as f64) / mcu.0 as f64,
        opt.1 / mcu.1, if opt.1 > mcu.1 { "slower" } else { "faster" },
        opt.2 / mcu.2, if opt.2 > mcu.2 { "slower" } else { "faster" });
}
