//! Confirm the SHIPPED mv_rls BoundedMae win end-to-end: window the recording at W (≤ the
//! entropy coder's u16 cap), run the real `lmo::encode(Mode::BoundedMae(δ))` per chunk, and
//! report bps + which transform won (3 = mv_rls-bounded, 0 = floor) + the decoded max-error
//! (the hard guarantee). This is the shipped path, not a probe.
//!
//! cargo run -p lamquant-lml-optimum --features encode --release --example nl_ship_measure -- <bin>...

use std::fs;

use lamquant_lml_mcu::codec::Mode;
use lamquant_lml_optimum::{decode_any, lmo};

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

fn main() {
    const W: usize = 32768;
    let bins: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .collect();
    println!(
        "Shipped lmo::encode(Mode::BoundedMae) — windowed at {W}  (tid 3 = mv_rls-bounded won)"
    );
    println!(
        "{:>14}  {:>4}  {:>9}  {:>12}  {:>8}",
        "recording", "δ", "bps", "mv_rls won", "maxErr"
    );
    for p in &bins {
        let sig = read_bin(p);
        let (nch, t) = (sig.len(), sig[0].len());
        for &d in &[1u64, 8, 16, 32] {
            let (mut bytes, mut maxerr, mut mv_wins, mut nwin) = (0usize, 0i64, 0usize, 0usize);
            let mut start = 0;
            while start < t {
                let end = (start + W).min(t);
                let chunk: Vec<Vec<i64>> = sig.iter().map(|c| c[start..end].to_vec()).collect();
                let body = lmo::encode(&chunk, Mode::BoundedMae(d)).expect("encode");
                bytes += body.len();
                if body[6] == 3 {
                    mv_wins += 1;
                }
                nwin += 1;
                let dec = decode_any(&body).expect("decode");
                for c in 0..nch {
                    for n in 0..(end - start) {
                        maxerr = maxerr.max((chunk[c][n] - dec[c][n]).abs());
                    }
                }
                start = end;
            }
            let bps = bytes as f64 * 8.0 / (nch * t) as f64;
            let name = p.rsplit('/').next().unwrap_or(p);
            println!("{name:>14}  {d:>4}  {bps:>9.4}  {mv_wins:>5}/{nwin:<5}  {maxerr:>8} (<={d})");
        }
    }
}
