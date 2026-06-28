//! TUH per-candidate diagnostic: which keep-best coder actually BINDS on TUH?
//! Per W-window, sum each directly-callable candidate vs the container min:
//!   B = rls::encode (raw per-channel RLS)
//!   C = mv_rls::encode (multivariate cross-channel RLS — the config grid)
//!   D = rls::encode_seg (RLS + change-point segmentation)
//!   container = LmoCodec.encode (= min over A/B/C/D; A = cross-channel+lml/rls is
//!               internal, so `container < min(B,C,D)` ⇒ the cross-channel cand A binds)
//! Localizes the TUH lever: tuning a coder that doesn't bind can't help.
//!
//! cargo run -p lamquant-lml-optimum --features encode --release --example tuh_candidates -- <bin>...

use std::fs;
use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{mv_rls, rls, LmoCodec};

const W: usize = 32768;

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut s = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t { ch.push(i32::from_le_bytes(b[off..off+4].try_into().unwrap()) as i64); off += 4; }
        s.push(ch);
    }
    s
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!("  {:<22} {:>10} {:>10} {:>10} {:>10} | binds", "recording", "container", "rls(B)", "mvrls(C)", "rlsSeg(D)");
    let (mut sc, mut sb, mut scc, mut sd) = (0u64, 0u64, 0u64, 0u64);
    for path in &args {
        let sig = read_bin(path);
        let t = sig[0].len();
        let (mut cont, mut b, mut c, mut d) = (0usize, 0usize, 0usize, 0usize);
        let mut start = 0;
        while start < t {
            let end = (start + W).min(t);
            let win: Vec<Vec<i64>> = sig.iter().map(|ch| ch[start..end].to_vec()).collect();
            cont += LmoCodec.encode(&win, Mode::Lossless).map(|x| x.len()).unwrap_or(0);
            b += rls::encode(&win).map(|x| x.len()).unwrap_or(usize::MAX);
            c += mv_rls::encode(&win).map(|x| x.len()).unwrap_or(usize::MAX);
            d += rls::encode_seg(&win).map(|x| x.len()).unwrap_or(usize::MAX);
            start = end;
        }
        // which directly-callable candidate is closest to the container; if container
        // is below all of them, the internal cross-channel candidate A is binding.
        let trio = [("B/rls", b), ("C/mvrls", c), ("D/rlsSeg", d)];
        let min_named = trio.iter().min_by_key(|x| x.1).unwrap();
        let binds = if cont < min_named.1 { "A/cross-channel" } else { min_named.0 };
        let name = path.rsplit('/').next().unwrap_or(path);
        println!("  {:<22} {:>10} {:>10} {:>10} {:>10} | {}", name, cont, b, c, d, binds);
        sc += cont as u64; sb += b as u64; scc += c as u64; sd += d as u64;
    }
    println!("\n  TOTALS  container {} | rls {} | mvrls {} | rlsSeg {}", sc, sb, scc, sd);
    println!("  # container below ALL of B/C/D ⇒ the cross-channel candidate A is the TUH lever (not MV-RLS).");
}
