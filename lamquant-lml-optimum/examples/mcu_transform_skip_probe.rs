//! Adaptive 5/3 (transform-skip) de-risk: does PER-WINDOW keep-best between the 5/3-transform path and
//! transform-skip (n_levels=0, LPC-on-raw) beat the shipped always-transform MCU codec?
//!
//! The decomposition probe (mcu_transform_decomp) showed 5/3-on-LPC is signal-dependent: it HELPS
//! non-stationary windows (ma −7.9%) and HURTS stationary ones (+0.7 to +7.5%). A recording mixes both,
//! so a per-window choice of {transform, skip} — keep-best by real coded length — is never-worse and
//! should recover the loss on stationary windows. This measures that win at two granularities:
//!   ALWAYS  = shipped behavior: 5/3 (n_levels=3) + per-subband LPC + Golomb on every window/channel.
//!   PKT     = per WINDOW, all channels share one n_levels ∈ {0,3}, keep smaller total (+1 mode bit/win).
//!             WIRE-COMPATIBLE NOW: the LML1 header already carries a per-packet n_levels; decode of
//!             n_levels=0 is already supported. This is the shippable never-worse win.
//!   CH      = per WINDOW per CHANNEL choose {0,3} (+1 mode bit/ch/win). Needs a wire change (per-channel
//!             n_levels); this is the UPPER BOUND that says whether that wire change is worth it.
//!
//! Faithful to the MCU codec: forward_subbands(win,3) → per-subband lpc::analyze at fixed_order_for_subband
//! → golomb, exactly as encode_one_channel; skip is the n_levels=0 path (single full-length subband).
//!
//! cargo run -p lamquant-lml-optimum --features encode --release --example mcu_transform_skip_probe -- [W] <bin>...

use std::fs;
use lamquant_lml_mcu::{golomb, lml, lpc};

const LVL: u8 = 3; // shipped default depth for clinical-scale windows (matches decomp probe)

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
fn gr(v: &[i64]) -> usize {
    golomb::encode_dense(v).map(|g| g.len()).unwrap_or(1 << 30)
}

/// LPC residual golomb bytes + coeff side-info (2 B/coeff), matching encode_one_channel per subband.
fn lpc_gr(v: &[i64], order: usize) -> usize {
    let (coeffs, resid) = lpc::analyze(v, order, order);
    gr(&resid) + coeffs.len() * 2
}

/// Bytes to code one channel-window with the 5/3 transform (n_levels=LVL), per-subband LPC.
fn full_bytes(ch_win: &[i64]) -> usize {
    lml::forward_subbands(ch_win, LVL)
        .iter()
        .enumerate()
        .map(|(i, sb)| lpc_gr(sb, lpc::fixed_order_for_subband(i)))
        .sum()
}

/// Bytes to code one channel-window with transform SKIP (n_levels=0): LPC on the raw window.
fn skip_bytes(ch_win: &[i64]) -> usize {
    lpc_gr(ch_win, lpc::fixed_order_for_subband(0))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cli_w = args.first().and_then(|a| a.parse::<usize>().ok());
    if cli_w.is_some() {
        args.remove(0);
    }
    // Realistic MCU packet scales (~5s/10s/20s @ 256 Hz). Per-window stationarity varies at these sizes.
    let wins: Vec<usize> = cli_w.map(|w| vec![w]).unwrap_or_else(|| vec![1280, 2560, 5120]);

    println!("# Adaptive 5/3 transform-skip. ALWAYS = shipped (5/3 every window). PKT = per-window keep-best");
    println!("# n_levels∈{{0,3}} shared by all channels (wire-compatible NOW). CH = per-window-per-channel");
    println!("# (needs a per-channel n_levels wire change). Δ vs ALWAYS; skip%% = windows PKT picked skip.\n");
    println!("{:>12} {:>5} {:>9} {:>9} {:>9} | {:>8} {:>8} | {:>7}",
             "recording", "W", "ALWAYS", "PKT", "CH", "PKT/ALW", "CH/ALW", "skip%");

    for path in &args {
        let sig = read_bin(path);
        let (c, t) = (sig.len(), sig[0].len());
        let nm = (c * t) as f64;
        for &w in &wins {
            let (mut always, mut pkt, mut ch_lvl) = (0usize, 0usize, 0usize);
            let (mut n_win, mut n_skip) = (0usize, 0usize);
            let mut start = 0;
            while start < t {
                let end = (start + w).min(t);
                let (mut win_full, mut win_skip) = (0usize, 0usize);
                for chan in &sig {
                    let seg = &chan[start..end];
                    let (f, s) = (full_bytes(seg), skip_bytes(seg));
                    win_full += f;
                    win_skip += s;
                    always += f;
                    ch_lvl += f.min(s); // per-channel keep-best (+mode bit accounted below)
                }
                // per-packet: one n_levels for the whole window
                let picked_skip = win_skip < win_full;
                pkt += win_full.min(win_skip);
                if picked_skip {
                    n_skip += 1;
                }
                n_win += 1;
                start = end;
            }
            // mode-bit overhead: PKT = 1 bit/window; CH = 1 bit/channel/window.
            let pkt_tot = pkt + (n_win + 7) / 8;
            let ch_tot = ch_lvl + (n_win * c + 7) / 8;
            let bps = |x: usize| x as f64 * 8.0 / nm;
            let name = path.rsplit('/').next().unwrap_or(path);
            let d = |x: usize| 100.0 * (x as f64 - always as f64) / always as f64;
            println!("{name:>12} {w:>5} {:>9.4} {:>9.4} {:>9.4} | {:>+7.3}% {:>+7.3}% | {:>6.1}%",
                     bps(always), bps(pkt_tot), bps(ch_tot), d(pkt_tot), d(ch_tot),
                     100.0 * n_skip as f64 / n_win.max(1) as f64);
        }
    }
    println!("\n# PKT/ALW < 0 ⇒ wire-compatible per-window transform-skip is a real never-worse win (ship it).");
    println!("# CH/ALW materially below PKT/ALW ⇒ per-channel n_levels (a wire change) buys enough more to");
    println!("# justify the schema bump; CH ≈ PKT ⇒ per-packet skip captures it, no wire change needed.");
}
