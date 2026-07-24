//! Backward-adaptive long-term (block-matching / LTP) predictor on the mv_rls residual —
//! the ZERO-side-info temporal variant that the two prior block probes did NOT cover.
//!
//! Prior art (measured, against our ceiling):
//!   * av2_ltp_probe (671416d) — FORWARD-adaptive temporal LTP: estimates lag+gain on the
//!     CURRENT block, codes them as side-info. siena −0.72% / tusz −0.12% / ma +0.07% net;
//!     "side-info eats it". MARGINAL.
//!   * block_backadapt_ls_probe (4ff41cb) — backward-adaptive per-block CROSS-CHANNEL (spatial)
//!     LS, zero side-info. DOMINATED by mv_rls (+13..84%): mv_rls already IS the joint
//!     spatio-temporal predictor, so a decoupled spatial-then-floor cascade loses.
//!
//! This probe removes BOTH failure modes at once:
//!   1. ZERO side-info — estimate (lag τ, gain g) on the PREVIOUS block's residual (all past,
//!      reconstructed == original in lossless ⇒ the decoder re-derives τ,g identically), apply
//!      to the CURRENT block. Bets that the periodicity explaining the recent past also explains
//!      the near future (true for stationary rhythm; weaker under drift — that is what we measure).
//!   2. It cascades on the mv_rls RESIDUAL (temporal-lag), not a decoupled spatial floor —
//!      so it only has to catch periodicity mv_rls's K=8 short taps could not reach (ECG R-R,
//!      3 Hz spike-wave ~83 @ 250, spindles), not re-do the spatial decorrelation.
//!
//! Open-loop bits go/no-go (entropy::encode on the LTP residual vs the base residual). NET == GROSS
//! here (no side-info term), so a negative % is a *real, held-out, zero-cost* codelength win.
//! The apply/skip gate keys ONLY on the trailing (past) window ⇒ decoder-reproducible.
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example ltp_backadapt_probe -- [BLK] <bin>...

use lamquant_lml_optimum::{entropy, mv_rls};
use std::fs;

const W: usize = 32768; // entropy window (u16 cap on entropy::encode)
const TMIN: usize = 16;
const TMAX: usize = 512; // covers ECG R-R + slow rhythmic EEG at typical fs

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

fn entropy_bytes(q: &[i64]) -> usize {
    let mut tot = 0;
    let mut s = 0;
    while s < q.len() {
        let e = (s + W).min(q.len());
        tot += entropy::encode(&q[s..e])
            .map(|g| g.len())
            .unwrap_or(1 << 30)
            + 4;
        s = e;
    }
    tot
}

/// Estimate (best lag τ, gain g, energy-reduction) over the window [lo,hi) of `r`,
/// using only samples i with i-τ >= floor (causal / in-range). Closed-form LS gain.
fn best_lag(r: &[i64], lo: usize, hi: usize, floor: usize) -> (usize, f64, f64) {
    let e0: f64 = (lo..hi).map(|i| (r[i] * r[i]) as f64).sum();
    if e0 <= 0.0 {
        return (0, 0.0, 0.0);
    }
    let (mut best_t, mut best_red, mut best_g) = (0usize, 0.0f64, 0.0f64);
    for t in TMIN..=TMAX {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in lo..hi {
            if i < t || i - t < floor {
                continue;
            }
            let rt = r[i - t] as f64;
            num += r[i] as f64 * rt;
            den += rt * rt;
        }
        if den <= 0.0 {
            continue;
        }
        let red = num * num / den; // LS residual-energy drop
        if red > best_red {
            best_red = red;
            best_t = t;
            best_g = num / den;
        }
    }
    (best_t, best_g, best_red)
}

/// Backward-adaptive LTP one channel. (τ,g) estimated on the PREVIOUS block, applied to the
/// current block ⇒ zero side-info. Returns (ltp_residual, fraction_of_blocks_applied).
fn ltp_backadapt(r: &[i64], blk: usize) -> (Vec<i64>, f64) {
    let n = r.len();
    let mut out = r.to_vec();
    let (mut blocks, mut applied) = (0usize, 0usize);
    let mut bs = blk; // first block has no past ⇒ passthrough
    while bs < n {
        let be = (bs + blk).min(n);
        blocks += 1;
        // Estimate on the trailing (already-past) block [bs-blk, bs), referencing only samples
        // >= bs-blk-TMAX which are all < bs (past). floor = 0 (all of r before bs is reconstructed).
        let (t, g, red) = best_lag(r, bs - blk, bs, 0);
        // Apply only if the past predicts a real drop (> a small margin vs trailing energy).
        // Decision keys on past-only data ⇒ decoder makes the identical call.
        let e_prev: f64 = (bs - blk..bs).map(|i| (r[i] * r[i]) as f64).sum();
        if t != 0 && red > 0.01 * e_prev.max(1.0) {
            for i in bs..be {
                if i >= t {
                    out[i] = r[i] - (g * r[i - t] as f64).round() as i64;
                }
            }
            applied += 1;
        }
        bs = be;
    }
    (out, applied as f64 / blocks.max(1) as f64)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // optional first arg = block size
    let blk = args
        .first()
        .and_then(|a| a.parse::<usize>().ok())
        .unwrap_or(4096);
    if args
        .first()
        .map(|a| a.parse::<usize>().is_ok())
        .unwrap_or(false)
    {
        args.remove(0);
    }
    println!("# Backward-adaptive LTP (zero side-info) on the mv_rls residual — BLK={blk}\n");
    println!(
        "{:>12}  {:>10}  {:>10}  {:>9}  {:>8}",
        "recording", "base B", "+LTP B", "Δ%", "applied"
    );
    for path in &args {
        let sig = read_bin(path);
        let (nch, t) = (sig.len(), sig[0].len());
        let nm = (nch * t) as f64;
        let res = mv_rls::residuals(&sig, 0, 0);
        let base: usize = res.iter().map(|c| entropy_bytes(c)).sum();
        let (mut ltp, mut act) = (0usize, 0.0f64);
        for c in &res {
            let (r2, a) = ltp_backadapt(c, blk);
            ltp += entropy_bytes(&r2);
            act += a;
        }
        let d = 100.0 * (ltp as f64 - base as f64) / base as f64;
        let name = path.rsplit('/').next().unwrap_or(path);
        println!(
            "{name:>12}  {base:>10}  {ltp:>10}  {d:>+8.3}%  {:>6.1}%   ({:.3} bps base)",
            100.0 * act / nch as f64,
            base as f64 * 8.0 / nm
        );
    }
    println!("\n# NET == GROSS (zero side-info). Negative Δ% = real held-out zero-cost win.");
    println!("# Gate to escalate to a closed-loop ship: net <= -1.0% on the referential lose-set");
    println!("# (siena/eegmmidb) — else backward-adaptive block-matching is spent too.");
}
