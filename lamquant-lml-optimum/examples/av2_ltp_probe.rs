//! AV2 LML pull #1 — long-term (CELP-LTP-style) predictor cascade on the MV-RLS residual.
//!
//! The ONE mechanism mv_rls structurally lacks: it is short-context AR (K=8 own
//! taps + cross-channel), so it cannot reach a *periodic* lag — ECG R-R (~250
//! samples), 3 Hz spike-wave (~83 @ 250 Hz), spindles. A long-term predictor
//! (from speech coding, public domain — NOT the AV2 IBC spec) re-predicts the
//! residual from itself one period back. If mv_rls already whitened the period out,
//! the gain is ~0; if periodicity survives in the residual, this shaves it.
//!
//! Per channel, per block: pick the lag T in [TMIN,TMAX] minimising post-LTP energy
//! (closed-form LS gain g = Σr·r_T / Σr_T²); apply r2[n] = r[n] − round(g·r[n−T]);
//! entropy::encode(r2) vs entropy::encode(r). Open-loop bits estimate (tight; the
//! shipping form is closed-loop). Round-trip not needed for a go/no-go on bits.
//!
//! Strongest on ECG/rhythmic signals; expected ~0 on desynchronised scalp EEG —
//! we measure it to settle which.
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example av2_ltp_probe -- <bin>...

use std::fs;
use lamquant_lml_optimum::{entropy, mv_rls};

const W: usize = 32768;      // entropy window
const BLK: usize = 8192;     // LTP lag re-estimation block
const TMIN: usize = 16;
const TMAX: usize = 512;     // covers ECG-ish + slow rhythmic EEG at typical fs

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

fn entropy_bytes(q: &[i64]) -> usize {
    let mut tot = 0; let mut s = 0;
    while s < q.len() {
        let e = (s + W).min(q.len());
        tot += entropy::encode(&q[s..e]).map(|g| g.len()).unwrap_or(1 << 30) + 4;
        s = e;
    }
    tot
}

/// LTP one channel residual. Returns (ltp_residual, fraction_of_blocks_with_gain).
fn ltp_channel(r: &[i64]) -> (Vec<i64>, f64) {
    let n = r.len();
    let mut out = r.to_vec();
    let mut blocks = 0usize;
    let mut active = 0usize;
    let mut bs = 0;
    while bs < n {
        let be = (bs + BLK).min(n);
        blocks += 1;
        // best lag over [TMIN,TMAX], constrained so n-T >= 0 in-block
        let mut best_t = 0usize;
        let mut best_red = 0.0f64; // energy reduction
        let e0: f64 = (bs..be).map(|i| (r[i] * r[i]) as f64).sum();
        if e0 <= 0.0 { bs = be; continue; }
        for t in TMIN..=TMAX {
            let mut num = 0.0f64; // Σ r·r_T
            let mut den = 0.0f64; // Σ r_T²
            for i in bs..be {
                if i < t { continue; }
                let rt = r[i - t] as f64;
                num += r[i] as f64 * rt;
                den += rt * rt;
            }
            if den <= 0.0 { continue; }
            let g = num / den;
            // energy after LTP ≈ e0_block - num²/den (LS residual energy drop)
            let red = num * num / den;
            if red > best_red { best_red = red; best_t = t; }
            let _ = g;
        }
        // apply if the reduction is worth the ~per-block side info (lag u16 + gain ~ 3 bytes ≈ 24 bits)
        if best_t != 0 && best_red > 24.0 * 8.0 {
            // recompute gain at best_t
            let mut num = 0.0; let mut den = 0.0;
            for i in bs..be { if i >= best_t { let rt = r[i - best_t] as f64; num += r[i] as f64 * rt; den += rt * rt; } }
            let g = num / den;
            for i in bs..be { if i >= best_t { out[i] = r[i] - (g * r[i - best_t] as f64).round() as i64; } }
            active += 1;
        }
        bs = be;
    }
    (out, active as f64 / blocks.max(1) as f64)
}

fn main() {
    println!("# AV2 LML #1 — long-term predictor cascade on the MV-RLS residual\n");
    for path in std::env::args().skip(1) {
        let sig = read_bin(&path);
        let (nch, t) = (sig.len(), sig[0].len());
        let nm = (nch * t) as f64;
        let res = mv_rls::residuals(&sig, 0, 0);
        let name = path.rsplit('/').next().unwrap_or(&path);
        let base: usize = res.iter().map(|c| entropy_bytes(c)).sum();
        let mut ltp = 0usize;
        let mut act = 0.0f64;
        for c in &res {
            let (r2, a) = ltp_channel(c);
            ltp += entropy_bytes(&r2);
            act += a;
        }
        // side info: per active block, lag(u16)+gain(i16) = 4 bytes; estimate from activity
        let nblocks = res.iter().map(|c| (c.len() + BLK - 1) / BLK).sum::<usize>();
        let side = (act / nch as f64 * nblocks as f64 * 4.0) as usize; // rough
        let total = ltp + side;
        let d = 100.0 * (total as f64 - base as f64) / base as f64;
        println!("## {} ({}ch x {})", name, nch, t);
        println!("   baseline (mv_rls resid)  = {:>10} ({:.4} bps)", base, base as f64 * 8.0 / nm);
        println!("   + LTP cascade            = {:>10} + side {:>7} = {:>10}  ({:+.2}% vs baseline)", ltp, side, total, d);
        println!("   blocks with LTP gain     = {:.1}%  (high ⇒ residual periodicity mv_rls missed)", 100.0 * act / nch as f64);
        println!();
    }
    println!("# negative ⇒ LTP shaves residual periodicity mv_rls couldn't reach. Expected strong on");
    println!("# ECG/rhythmic, ~0 on desynchronised scalp EEG. (ECG untested — no ECG data on hand.)");
}
