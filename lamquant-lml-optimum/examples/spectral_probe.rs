//! Spectral candidate de-risk: does a per-channel block DCT-II + our REAL entropy
//! coder (`entropy::encode`, the scale_cond range coder the container ships) beat the
//! current container on TUH? Float DCT rounded to i64 (a faithful entropy proxy for a
//! reversible integer DCT — the real lifting transform costs ~1-3% more). Coefficients
//! are emitted POSITION-MAJOR (all DC, then all k=1, …) so the adaptive coder sees each
//! frequency band as a run and locks onto its scale. Compares to LmoCodec (container).
//!
//! cargo run -p lamquant-lml-optimum --features encode --release --example spectral_probe -- <bin>...

use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{entropy, LmoCodec};

static DBG: AtomicBool = AtomicBool::new(true);

const WIN: usize = 32768; // container window (for the container baseline)

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut s = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t { ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64); off += 4; }
        s.push(ch);
    }
    s
}

/// DCT-II basis (ortho-normalized) for block size b: basis[k][n].
fn dct_basis(b: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0f64; b]; b];
    let s0 = (1.0 / b as f64).sqrt();
    let s = (2.0 / b as f64).sqrt();
    for k in 0..b {
        let sc = if k == 0 { s0 } else { s };
        for n in 0..b {
            m[k][n] = sc * (core::f64::consts::PI * (n as f64 + 0.5) * k as f64 / b as f64).cos();
        }
    }
    m
}

/// Container (shipped) byte count, mirroring lossless_full's W-windowing.
fn container_bytes(sig: &[Vec<i64>]) -> usize {
    let t = sig[0].len();
    let mut tot = 0;
    let mut start = 0;
    while start < t {
        let end = (start + WIN).min(t);
        let win: Vec<Vec<i64>> = sig.iter().map(|ch| ch[start..end].to_vec()).collect();
        tot += LmoCodec.encode(&win, Mode::Lossless).map(|x| x.len()).unwrap_or(0);
        start = end;
    }
    tot
}

/// Per-channel block-DCT coded position-major with entropy::encode — operating
/// PER WINDOW (WIN=32768) like the container, so each coded stream is ≤ window size
/// (golomb's header limit). Within a window: blocks of `b`, coeffs emitted DC-first.
fn dct_bytes(sig: &[Vec<i64>], b: usize) -> usize {
    let basis = dct_basis(b);
    let t = sig[0].len();
    let mut tot = 0usize;
    for ch in sig {
        let mut start = 0;
        while start < t {
            let end = (start + WIN).min(t);
            let win = &ch[start..end];
            let nb = win.len() / b;
            let mut coeffs = vec![vec![0i64; nb]; b];
            for blk in 0..nb {
                let base = blk * b;
                for k in 0..b {
                    let mut acc = 0.0f64;
                    let row = &basis[k];
                    for n in 0..b { acc += row[n] * win[base + n] as f64; }
                    coeffs[k][blk] = acc.round() as i64;
                }
            }
            // position-major within the window (≤ WIN values), DC band first.
            let mut stream = Vec::with_capacity(nb * b);
            for k in 0..b { stream.extend_from_slice(&coeffs[k]); }
            // tail samples in this window (win.len() - nb*b) appended raw.
            stream.extend_from_slice(&win[nb * b..]);
            let enc = entropy::encode(&stream).expect("entropy encode");
            let dec = entropy::decode(&enc).expect("entropy decode");
            assert_eq!(dec, stream, "entropy NOT lossless on DCT coeffs"); // correctness gate
            if DBG.swap(false, Ordering::Relaxed) {
                let mx = stream.iter().map(|&x| x.unsigned_abs()).max().unwrap_or(0);
                eprintln!("    [dbg b={b}] win_stream_len={} max|coef|={} enc_bytes={}",
                          stream.len(), mx, enc.len());
            }
            tot += enc.len() + 8;
            start = end;
        }
    }
    tot
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    for path in &args {
        let sig = read_bin(path);
        let (c, t) = (sig.len(), sig[0].len());
        let nm = (c * t) as f64;
        let cont = container_bytes(&sig);
        // per-channel time-domain baseline (order-1 diff + real entropy coder, per window)
        // — isolates "DCT vs time-domain prediction" with NO cross-channel, same coder.
        let mut td = 0usize;
        for ch in &sig {
            let mut start = 0;
            while start < t {
                let end = (start + WIN).min(t);
                let mut d = Vec::with_capacity(end - start);
                let mut prev = 0i64;
                for &x in &ch[start..end] { d.push(x - prev); prev = x; }
                td += entropy::encode(&d).map(|g| g.len()).unwrap_or(usize::MAX).min(1 << 30) + 8;
                start = end;
            }
        }
        let name = path.rsplit('/').next().unwrap_or(path);
        print!("  {:<20} {}ch x{:6} | container {:.2} | td-diff {:.2} bps",
               name, c, t, cont as f64 * 8.0 / nm, td as f64 * 8.0 / nm);
        for &b in &[256usize, 512, 1024, 2048] {
            let d = dct_bytes(&sig, b);
            print!("  | DCT-{} {:.2} ({:+.1}%)", b, d as f64 * 8.0 / nm, 100.0 * (d as f64 - cont as f64) / cont as f64);
        }
        println!();
    }
    println!("\n  # DCT bps < container ⇒ a reversible integer-DCT keep-best candidate is worth building.");
    println!("  # (float-DCT-rounded is a slight UNDER-estimate; the reversible lifting transform costs ~1-3% more.)");
}
