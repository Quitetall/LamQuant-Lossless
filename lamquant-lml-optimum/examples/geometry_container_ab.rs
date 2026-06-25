//! Phase A1 CONTAINER measurement: does `lmo_lossless::encode_with_geometry` shrink the
//! actual id=2 lossless body vs production `encode`, on held-out windows? (The id=2 body
//! is the container winner on correlated multichannel EEG.) Roundtrip-verified.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example geometry_container_ab -- <window>.bin [...]
//! ```
//! Reads `<window>.bin.coords` (from tools/scripts/resolve_coords.py).

use std::fs;

use lamquant_lml_optimum::lmo_lossless;
use lamquant_lml_optimum::montage::MontageGeometry;

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let bytes = fs::read(path).expect("read bin");
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

fn read_coords(path: &str) -> Vec<Option<[f64; 3]>> {
    fs::read_to_string(format!("{path}.coords"))
        .unwrap_or_default()
        .lines()
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().filter_map(|t| t.parse().ok()).collect();
            if v.len() == 3 && v.iter().all(|x| x.is_finite()) {
                Some([v[0], v[1], v[2]])
            } else {
                None
            }
        })
        .collect()
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    println!("  {:>36} | {:>4} | {:>11} {:>11} | {:>7}", "window", "nch", "encode", "+geometry", "gain%");
    let (mut tb, mut tg) = (0usize, 0usize);
    for path in &paths {
        let sig = read_bin(path);
        let coords = read_coords(path);
        let base = lmo_lossless::encode(&sig).unwrap();
        let geom = MontageGeometry::new(coords);
        let g = lmo_lossless::encode_with_geometry(&sig, Some(&geom)).unwrap();
        assert_eq!(lmo_lossless::decode(&g).unwrap(), sig, "geometry body must roundtrip");
        let gain = 100.0 * (base.len() as f64 - g.len() as f64) / base.len() as f64;
        let short: String = std::path::Path::new(path).file_name().unwrap().to_string_lossy().into_owned();
        let short = short.chars().rev().take(36).collect::<String>().chars().rev().collect::<String>();
        println!("  {:>36} | {:>4} | {:>11} {:>11} | {:>6.2}%", short, sig.len(), base.len(), g.len(), gain);
        tb += base.len();
        tg += g.len();
    }
    let gain = 100.0 * (tb as f64 - tg as f64) / tb as f64;
    println!("  {:>36} | {:>4} | {:>11} {:>11} | {:>6.2}%", "TOTAL", "", tb, tg, gain);
    println!("\n# id=2 body bytes. The full container keep-bests this vs the 5/3 floor; on");
    println!("# correlated multichannel EEG the id=2 body is the winner, so this ~ container.");
}
