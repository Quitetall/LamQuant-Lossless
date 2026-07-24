//! Per-candidate cost (time) vs benefit (bytes) on one window — what's wasted in the keep-best.
use lamquant_lml_mcu::lml;
use lamquant_lml_optimum::{lmo_lossless, mv_rls, rls};
use std::fs;
use std::time::Instant;
const W: usize = 32768;
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
fn timed<F: Fn() -> usize>(name: &str, f: F) {
    let t0 = Instant::now();
    let mut by = 0;
    for _ in 0..1 {
        by = f();
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("    {:<22} {:>9} B   {:>7.2} s", name, by, dt);
}
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let sig = read_bin(&path);
    let w = W.min(sig[0].len());
    let win: Vec<Vec<i64>> = sig.iter().map(|c| c[..w].to_vec()).collect();
    println!(
        "  {} ({}ch x {}) — per-candidate on one window:",
        path.rsplit('/').next().unwrap(),
        win.len(),
        w
    );
    timed("LML floor", || lml::compress(&win, 0).unwrap().len());
    timed("B raw-RLS", || {
        rls::encode(&win).map(|x| x.len()).unwrap_or(0)
    });
    timed("D RLS-seg", || {
        rls::encode_seg(&win).map(|x| x.len()).unwrap_or(0)
    });
    timed("C MV-RLS(14cfg)", || {
        mv_rls::encode(&win).map(|x| x.len()).unwrap_or(0)
    });
    timed("A cross-ch+keepbest(full)", || {
        lmo_lossless::encode_with_geometry(&win, None)
            .unwrap()
            .len()
    });
}
