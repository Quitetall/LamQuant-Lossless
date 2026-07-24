//! TUH tuning (diagnostic + config search). For each TUH `.bin`:
//!  - container  = `lmo_lossless::encode` (the shipped codec — what loses to H.BWC)
//!  - mvrls_grid = `mv_rls::encode` (current CONFIGS keep-best)
//!  - search     = best of a BROAD (λ, reset, m, seg) grid via `encode_len_params`
//! Tells us (a) whether MV-RLS even binds on TUH (vs the container min), and
//! (b) whether configs NOT in the grid would cut TUH bytes (→ add them, never-worse).
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example tuh_tune -- <bin> [<bin> ...]
//! ```

#![allow(clippy::doc_lazy_continuation)]

use std::collections::BTreeMap;
use std::fs;

use lamquant_lml_mcu::codec::{Codec, Mode};
use lamquant_lml_optimum::{mv_rls, LmoCodec};

const W: usize = 32768; // the shipped window size (mirror lossless_full)

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

/// Split a recording into W-sample windows (last partial), mirroring the codec.
fn windows(sig: &[Vec<i64>]) -> Vec<Vec<Vec<i64>>> {
    let t = sig[0].len();
    let mut out = Vec::new();
    let mut start = 0;
    while start < t {
        let end = (start + W).min(t);
        out.push(sig.iter().map(|ch| ch[start..end].to_vec()).collect());
        start = end;
    }
    out
}

const LAMBDAS: &[f64] = &[0.9997, 0.9990, 0.9980, 0.9970, 0.9950, 0.9900, 0.9850];
const RESETS: &[usize] = &[16384, 8192, 4096, 2048, 1024, 512];
const MS: &[usize] = &[16, 32];
const SEGS: &[usize] = &[0, 1];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // winners: (lambda_milli, reset, m, seg) -> count of recordings it wins
    let mut winners: BTreeMap<(u64, usize, usize, usize), usize> = BTreeMap::new();
    let mut sum_grid = 0u64;
    let mut sum_search = 0u64;
    let mut sum_container = 0u64;
    println!(
        "  {:<26} {:>9} {:>9} {:>9} | search vs grid | best (λ,reset,m,seg)",
        "recording", "container", "mvrls", "search"
    );
    for path in &args {
        let sig = read_bin(path);
        let (mut container, mut grid, mut search) = (0usize, 0usize, 0usize);
        // per-window winners for THIS recording (a config can win on some windows)
        for win in windows(&sig) {
            container += LmoCodec
                .encode(&win, Mode::Lossless)
                .map(|b| b.len())
                .unwrap_or(0);
            grid += mv_rls::encode(&win).map(|b| b.len()).unwrap_or(usize::MAX);
            let mut best = usize::MAX;
            let mut bestp = (0.0f64, 0usize, 0usize, 0usize);
            for &lam in LAMBDAS {
                for &rst in RESETS {
                    for &m in MS {
                        for &seg in SEGS {
                            let len = mv_rls::encode_len_params(&win, lam, rst, m, seg);
                            if len < best {
                                best = len;
                                bestp = (lam, rst, m, seg);
                            }
                        }
                    }
                }
            }
            search += best;
            *winners
                .entry(((bestp.0 * 10000.0) as u64, bestp.1, bestp.2, bestp.3))
                .or_insert(0) += 1;
        }
        let name = path.rsplit('/').next().unwrap_or(path);
        let vs_grid = 100.0 * (search as f64 - grid as f64) / grid as f64;
        let binds = if grid <= container { "MV*" } else { "   " };
        println!(
            "  {:<26} {:>9} {:>9}{} {:>9} | {:>+6.2}%        | (per-window best below)",
            name, container, grid, binds, search, vs_grid
        );
        sum_grid += grid as u64;
        sum_search += search as u64;
        sum_container += container as u64;
    }
    println!(
        "\n  TOTALS: container {} | mvrls_grid {} | search {} ({:+.2}% vs grid)",
        sum_container,
        sum_grid,
        sum_search,
        100.0 * (sum_search as f64 - sum_grid as f64) / sum_grid as f64
    );
    println!("\n  winning params (count) — candidates to ADD to CONFIGS (keep-best, never-worse):");
    let mut wv: Vec<_> = winners.into_iter().collect();
    wv.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    for ((lm, rst, m, seg), cnt) in wv {
        println!(
            "    ({:.4}, {}, {}) seg={}   x{}",
            lm as f64 / 10000.0,
            rst,
            m,
            seg,
            cnt
        );
    }
    println!(
        "\n  # 'search vs grid' < 0 ⇒ configs outside the current grid cut TUH bytes ⇒ add them."
    );
    println!(
        "  # 'MV*' marks recordings where MV-RLS already binds (≤ container); only there does"
    );
    println!(
        "  # tuning MV-RLS configs change the shipped bytes. Validate on held-out + no-regression."
    );
}
