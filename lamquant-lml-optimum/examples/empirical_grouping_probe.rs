//! Probe 0a — empirical subband decomposition, Approach 1 (research-direction
//! §6.8.1): deep standard 5/3 + finer band grouping. Tests the core thesis of the
//! strategic plan (§8.4: adaptivity beats uniformity) on the cheapest possible
//! lever: does decomposing the signal into MORE / FINER subbands (so each band
//! holds a more coherent single rhythm) reduce the total LOSSLESS coded size?
//!
//! For each window we run L-level integer 5/3 lifting for L = 3 (the current
//! dyadic baseline) .. 6, code every subband with the production per-subband
//! LPC + Golomb cost model, and sum bytes. A "raw" no-transform LPC+Golomb row
//! is the floor. If deeper monotonically helps, the band-structure thesis holds
//! and Approach 2 (alpha-anchored adaptive boundaries) is worth building.
//!
//! Measure-first: this is a GATE, not a product. A flat/negative result is a
//! valid logged outcome (cf. the deeper-5/3 LOSSY rejection in the technique
//! mine §C — this probe asks the *lossless entropy* question that mine left open).
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example empirical_grouping_probe -- /tmp/chb01_01_60s.bin
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lifting, lml, lpc};

/// Window length in samples. ~10 s at 250 Hz — matches the codec's working
/// window so the band-structure question is asked at a realistic granularity.
/// Depth 6 needs ≥ 4·2⁶ = 256 samples on the smallest subband; 2500 ≫ that.
const WINDOW: usize = 2500;

fn read_window(path: &str) -> Vec<Vec<i64>> {
    let bytes = fs::read(path).expect("read");
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

/// Production per-subband cost model: 1 byte order + 4 bytes/coeff + Golomb
/// residual (same accounting the lossy 9/7 path uses, `lossy_rls_probe.rs`).
fn lpc_golomb(idx: &[i64], sb: usize) -> usize {
    if idx.is_empty() {
        return 0;
    }
    let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(idx.len()));
    let (coeffs, residual, _o) = lpc::analyze_with_mode(idx, sb, scoped, lml::BIAS_CTX, None);
    1 + 4 * coeffs.len() + golomb::encode_dense(&residual).unwrap().len()
}

/// L-level integer 5/3 decomposition → subbands `[d1, d2, …, dL, aL]`.
/// Each level splits the running approximation; reuses the reversible primitive.
fn forward_levels(signal: &[i64], levels: u8) -> Vec<Vec<i64>> {
    let mut subs = Vec::with_capacity(levels as usize + 1);
    let mut approx = signal.to_vec();
    for _ in 0..levels {
        let (a, d) = lifting::forward(&approx);
        subs.push(d);
        approx = a;
    }
    subs.push(approx);
    subs
}

/// Full integer 5/3 wavelet PACKET to `depth` → 2^depth equal-width leaf bands
/// (decomposes BOTH approx and detail at every level — the finest integer-
/// reversible analogue of an EEG-band partition). Each split reuses the
/// reversible primitive, so the whole packet is bit-exact.
fn forward_packet(signal: &[i64], depth: u8) -> Vec<Vec<i64>> {
    if depth == 0 || signal.len() < 4 {
        return alloc_vec(signal);
    }
    let (a, d) = lifting::forward(signal);
    let mut out = forward_packet(&a, depth - 1);
    out.extend(forward_packet(&d, depth - 1));
    out
}

fn alloc_vec(s: &[i64]) -> Vec<Vec<i64>> {
    vec![s.to_vec()]
}

/// Total coded bytes for a whole channel, windowed at `WINDOW` samples.
/// `Mode::Dyadic(L)` = standard L-level (approx-chain only); `Mode::Packet(D)`
/// = full wavelet packet of depth D; `Mode::Raw` = no transform.
enum Mode {
    Raw,
    Dyadic(u8),
    Packet(u8),
}

fn channel_bytes(ch: &[i64], mode: &Mode) -> usize {
    let mut total = 0usize;
    for win in ch.chunks(WINDOW) {
        let leaves = match mode {
            Mode::Raw => vec![win.to_vec()],
            Mode::Dyadic(l) => forward_levels(win, *l),
            Mode::Packet(d) => forward_packet(win, *d),
        };
        for (sb, sub) in leaves.iter().enumerate() {
            total += lpc_golomb(sub, sb);
        }
    }
    total
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/chb01_01_60s.bin".to_string());
    let sig = read_window(&path);
    let n_ch = sig.len();
    let t = sig[0].len();
    let n_samples = (n_ch * t) as f64;

    println!(
        "# empirical-grouping probe (Approach 1): {} ({}ch × {}, window={})",
        path, n_ch, t, WINDOW
    );
    println!("# total LOSSLESS coded bytes (per-subband LPC+Golomb). Baseline = dyadic L=3.");
    println!(
        "  {:>10} | {:>12} {:>9} {:>10}",
        "mode", "bytes", "bps", "vs L=3"
    );

    // raw floor + dyadic L=3..6 (approx-chain) + full wavelet packet depth 3/4.
    let modes: [(&str, Mode); 7] = [
        ("raw", Mode::Raw),
        ("dyadic L=3", Mode::Dyadic(3)),
        ("dyadic L=4", Mode::Dyadic(4)),
        ("dyadic L=6", Mode::Dyadic(6)),
        ("packet D=3", Mode::Packet(3)),
        ("packet D=4", Mode::Packet(4)),
        ("packet D=5", Mode::Packet(5)),
    ];
    // baseline = dyadic L=3 (computed first explicitly so all rows compare to it).
    let baseline = sig
        .iter()
        .map(|ch| channel_bytes(ch, &Mode::Dyadic(3)))
        .sum::<usize>() as f64;
    for (label, mode) in &modes {
        let bytes: usize = sig.iter().map(|ch| channel_bytes(ch, mode)).sum();
        let bps = bytes as f64 * 8.0 / n_samples;
        let vs = format!("{:+.1}%", 100.0 * (bytes as f64 - baseline) / baseline);
        println!("  {:>10} | {:>12} {:>9.4} {:>10}", label, bytes, bps, vs);
    }
    println!("# negative vs-L=3 = finer/packet band separation codes smaller ⇒ helps lossless.");
}
