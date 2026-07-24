//! Introspect the LSB-split decision on a real window.
use lamquant_lml_optimum::lmo_lossless;
use std::fs;
fn read_bin(p: &str) -> Vec<Vec<i64>> {
    let b = fs::read(p).unwrap();
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut o = 8;
    let mut s = vec![];
    for _ in 0..nch {
        let mut c = vec![];
        for _ in 0..t {
            c.push(i32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as i64);
            o += 4;
        }
        s.push(c);
    }
    s
}
fn h_bit(sig: &[Vec<i64>], s: usize) -> f64 {
    let (mut n0, mut n1) = (0u64, 0u64);
    for ch in sig {
        for &x in ch {
            if (x >> s) & 1 == 0 {
                n0 += 1
            } else {
                n1 += 1
            }
        }
    }
    let n = (n0 + n1) as f64;
    let p = n1 as f64 / n;
    if p <= 0.0 || p >= 1.0 {
        0.0
    } else {
        -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
    }
}
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let sig = read_bin(&path);
    let w = 32768.min(sig[0].len());
    let win: Vec<Vec<i64>> = sig.iter().map(|c| c[..w].to_vec()).collect();
    println!(
        "window {}ch x {}  bit0_H={:.3} bit1_H={:.3}",
        win.len(),
        w,
        h_bit(&win, 0),
        h_bit(&win, 1)
    );
    let base = lmo_lossless::encode_with_geometry(&win, None).unwrap();
    let best = lmo_lossless::encode(&win).unwrap();
    println!(
        "base(no-split)={} bytes  encode(keep-best)={} bytes  split_used={}  delta={:+}",
        base.len(),
        best.len(),
        best[0] == 0xFE,
        best.len() as i64 - base.len() as i64
    );
}
