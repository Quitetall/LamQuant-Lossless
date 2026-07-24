//! Best-basis PER-BLOCK competition — the capstone that formally retires (or resurrects) the
//! fixed-transform family for lossless EEG. Prior probes measured each fixed basis GLOBALLY:
//!   * spectral_probe (DCT-II vs container)         — cosine, whole-window.
//!   * transform_ab / lmo_pcrd97 (9/7 vs 5/3)       — wavelet, lossy PCRD.
//!   * ricct_probe / crosschan (KLT rotation, LS)   — spatial, DOMINATED by mv_rls (cascade penalty).
//! All lost or tied vs mv_rls. But a GLOBAL loss doesn't refute the *mixture* hypothesis: EEG is
//! 1/f background (wavelet-optimal, KLT of self-similar) + narrowband rhythm (DCT-optimal, KLT of
//! AR(1)) + transients (wavelet-optimal). If those live in DIFFERENT blocks, a per-block CHOICE of
//! basis could beat any single fixed one — even one the predictor globally dominates.
//!
//! This probe measures exactly that ceiling. For each B-sample block (per channel, per 32768 window):
//!   P = mv_rls residual for that block (adaptive predictor — runs continuously, warm state).
//!   D = block DCT-II, float-rounded to i64, DC first-differenced across blocks (JPEG DC-DPCM).
//!   W = block 5/3 integer lifting (LeGall), multi-level, coefficients subband-major.
//! len_X(block) = entropy::encode(coeffs).len()  — the REAL ship coder (scale_cond/golomb keep-best).
//!
//! Oracle best-basis = Σ_blocks min(len_P, len_D, len_W); baseline = Σ_blocks len_P — measured at the
//! SAME per-block granularity, so the per-block framing tax cancels and the Δ isolates pure basis
//! SELECTION gain. Reported before selector, then net after a conservative 2-bit/block selector AND
//! an entropy-coded selector (H of the empirical selector distribution).
//!
//! ROBUST-NEGATIVE bias by construction: D uses the float-DCT-rounded proxy, which UNDER-states a
//! reversible integer DCT's real lossless cost by ~1-3% (spectral_probe note) — i.e. it makes cosine
//! look BETTER than shippable. If best-basis still can't clear the selector cost with DCT inflated
//! and wavelet free, the fixed-transform family is definitively spent.
//!
//! Gate to escalate to a real switchable codec: net <= -1.0% on the referential lose-set
//! (siena/eegmmidb). Also reports the per-basis WIN RATE — how often D or W actually beats P — which
//! is the direct empirical answer to "is EEG rhythmic/periodic enough for a frequency basis to win?".
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example best_basis_probe -- [B] <bin>...

use lamquant_lml_optimum::{entropy, mv_rls};
use std::fs;

const WIN: usize = 32768; // entropy window (u16 cap on entropy::encode)

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

#[inline]
fn enc_len(v: &[i64]) -> usize {
    entropy::encode(v).map(|g| g.len()).unwrap_or(1 << 30)
}

/// Ortho-normalized DCT-II basis[k][n] for block size b.
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

/// One forward LeGall 5/3 lifting level, in place over `x[0..n]` (n even), with symmetric
/// (mirror) boundary extension. Result: even indices hold the approx, odd hold the detail.
fn dwt53_level(x: &mut [i64], n: usize) {
    let at = |x: &[i64], i: isize| -> i64 {
        // mirror-extend index into [0,n)
        let mut i = i;
        let m = n as isize;
        if i < 0 {
            i = -i;
        }
        if i >= m {
            i = 2 * (m - 1) - i;
        }
        x[i.max(0).min(m - 1) as usize]
    };
    // predict: odd -= floor((left+right)/2)
    let mut d = vec![0i64; n];
    for k in (1..n).step_by(2) {
        let p = (at(x, k as isize - 1) + at(x, k as isize + 1)) as f64 / 2.0;
        d[k] = x[k] - p.floor() as i64;
    }
    for k in (1..n).step_by(2) {
        x[k] = d[k];
    }
    // update: even += floor((dleft+dright+2)/4)
    let mut a = vec![0i64; n];
    for k in (0..n).step_by(2) {
        let dl = if k == 0 { x[1] } else { x[k - 1] };
        let dr = if k + 1 < n { x[k + 1] } else { x[k - 1] };
        a[k] = x[k] + ((dl + dr + 2) as f64 / 4.0).floor() as i64;
    }
    for k in (0..n).step_by(2) {
        x[k] = a[k];
    }
}

/// Multi-level 5/3 of a block, returned subband-major: [approx(coarsest), detail_L-1 .. detail_0].
/// De-interleaves each level (evens→approx front, odds→detail) so bands are contiguous runs the
/// entropy coder can lock a scale onto. `levels` capped so the coarsest band stays >= 4 samples.
fn dwt53_block(block: &[i64], levels: usize) -> Vec<i64> {
    let b = block.len();
    let mut cur = block.to_vec();
    let mut details: Vec<Vec<i64>> = Vec::new(); // fine→coarse
    let mut n = b;
    for _ in 0..levels {
        if n < 8 {
            break;
        }
        dwt53_level(&mut cur[..n], n);
        let half = n / 2;
        let mut approx = Vec::with_capacity(half);
        let mut detail = Vec::with_capacity(half);
        for k in 0..n {
            if k % 2 == 0 {
                approx.push(cur[k]);
            } else {
                detail.push(cur[k]);
            }
        }
        details.push(detail);
        cur[..half].copy_from_slice(&approx);
        n = half;
    }
    // emit: coarsest approx (cur[..n]) then details coarse→fine
    let mut out = Vec::with_capacity(b);
    out.extend_from_slice(&cur[..n]);
    for det in details.iter().rev() {
        out.extend_from_slice(det);
    }
    out
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cli_b = args.first().and_then(|a| a.parse::<usize>().ok());
    if cli_b.is_some() {
        args.remove(0);
    }
    let block_sizes: Vec<usize> = cli_b
        .map(|b| vec![b])
        .unwrap_or_else(|| vec![256, 512, 1024]);

    println!("# Best-basis per-block competition: P=mv_rls  D=DCT-II(DC-DPCM)  W=5/3 lifting");
    println!("# len via the REAL ship coder. Δ = best-basis vs predictor-only, SAME per-block granularity.");
    println!("# DCT uses the float-rounded proxy (UNDER-states real integer-DCT cost) ⇒ negative is robust.\n");
    println!(
        "{:>12} {:>5} {:>10} {:>10} {:>8} {:>8} {:>8} | {:>13} {:>13}",
        "recording", "B", "pred B", "best B", "grossΔ%", "net2b%", "netHb%", "win D%", "win W%"
    );

    for path in &args {
        let sig = read_bin(path);
        let res = mv_rls::residuals(&sig, 0, 0); // P: adaptive-predictor residual, per channel
        let t = sig[0].len();

        for &b in &block_sizes {
            let basis = dct_basis(b);
            let levels = (b.trailing_zeros() as usize).saturating_sub(2); // coarsest band >= 4
            let (mut pred_b, mut best_b) = (0usize, 0usize);
            let (mut nblk, mut win_d, mut win_w) = (0usize, 0usize, 0usize);
            let mut sel_counts = [0usize; 3]; // P, D, W

            for (ci, ch) in sig.iter().enumerate() {
                let rch = &res[ci];
                let mut wstart = 0;
                while wstart < t {
                    let wend = (wstart + WIN).min(t);
                    let mut prev_dc = 0i64; // JPEG DC-DPCM, reset per window
                    let mut pos = wstart;
                    while pos + b <= wend {
                        let raw = &ch[pos..pos + b];
                        // P: predictor residual for this block
                        let lp = enc_len(&rch[pos..pos + b]);
                        // D: DCT-II, DC first-differenced across blocks
                        let mut coeffs = vec![0i64; b];
                        for k in 0..b {
                            let row = &basis[k];
                            let mut acc = 0.0f64;
                            for n in 0..b {
                                acc += row[n] * raw[n] as f64;
                            }
                            coeffs[k] = acc.round() as i64;
                        }
                        let dc = coeffs[0];
                        coeffs[0] = dc - prev_dc;
                        prev_dc = dc;
                        let ld = enc_len(&coeffs);
                        // W: 5/3 lifting, subband-major
                        let wco = dwt53_block(raw, levels);
                        let lw = enc_len(&wco);

                        pred_b += lp;
                        let (mut m, mut sel) = (lp, 0usize);
                        if ld < m {
                            m = ld;
                            sel = 1;
                        }
                        if lw < m {
                            m = lw;
                            sel = 2;
                        }
                        best_b += m;
                        sel_counts[sel] += 1;
                        if ld < lp {
                            win_d += 1;
                        }
                        if lw < lp {
                            win_w += 1;
                        }
                        nblk += 1;
                        pos += b;
                    }
                    wstart = wend;
                }
            }

            // selector cost: conservative 2 bits/block, and entropy-coded H(sel).
            let sel_2b = (nblk * 2 + 7) / 8; // bytes
            let hsel: f64 = sel_counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / nblk.max(1) as f64;
                    -p * p.log2()
                })
                .sum();
            let sel_h = ((nblk as f64 * hsel / 8.0).ceil()) as usize;

            let gross = 100.0 * (best_b as f64 - pred_b as f64) / pred_b as f64;
            let net2 = 100.0 * ((best_b + sel_2b) as f64 - pred_b as f64) / pred_b as f64;
            let neth = 100.0 * ((best_b + sel_h) as f64 - pred_b as f64) / pred_b as f64;
            let name = path.rsplit('/').next().unwrap_or(path);
            println!("{name:>12} {b:>5} {pred_b:>10} {best_b:>10} {gross:>+7.3}% {net2:>+7.3}% {neth:>+7.3}% | {:>12.2}% {:>12.2}%",
                     100.0 * win_d as f64 / nblk.max(1) as f64,
                     100.0 * win_w as f64 / nblk.max(1) as f64);
        }
    }
    println!("\n# grossΔ% = pure basis-selection ceiling (before selector). net2b/netHb = after 2-bit / entropy selector.");
    println!(
        "# Gate: net <= -1.0% on the lose-set (siena/eegmmidb) ⇒ build a real switchable codec."
    );
    println!("# Else the fixed-transform family (cosine, wavelet, KLT) is spent vs the adaptive predictor —");
    println!("# and the only lever left with lossless headroom is the nonlinear/learned class (separate LMQ thread).");
}
