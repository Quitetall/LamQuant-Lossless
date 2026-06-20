//! Track 2 keystone: the bounded-MAE near-lossless mode MUST guarantee
//! `max|orig − recon| ≤ δ` at the packet level (encode → decode), for every
//! δ and a range of signal shapes, and reduce to exact lossless at δ = 0.
//!
//! This is the load-bearing correctness contract for the H.BWC near-lossless
//! tier (ADR 0051). The bound is structural (closed-loop DPCM, decoder
//! replays the identical prediction loop), so it must hold regardless of the
//! signal or the predictor coefficients chosen.

use lamquant_lossless_core::lml::{compress_bounded_mae, decompress};
use lamquant_lossless_core::lpc::LpcMode;

/// Deterministic pseudo-signal generator (no rng dependency): a mix of a
/// ramp, a periodic component, and a sparse-spike term so subbands and
/// residuals are non-trivial.
fn make_signal(n_ch: usize, t: usize, seed: i64) -> Vec<Vec<i64>> {
    (0..n_ch)
        .map(|c| {
            let phase = seed.wrapping_add(c as i64 * 1009);
            (0..t)
                .map(|i| {
                    let i = i as i64;
                    let ramp = (i.wrapping_mul(37).wrapping_add(phase)) % 8000 - 4000;
                    let spike = if (i + phase) % 53 == 0 { 1500 } else { 0 };
                    ramp + spike
                })
                .collect()
        })
        .collect()
}

#[test]
fn bounded_mae_packet_respects_delta() {
    for &delta in &[0u64, 1, 2, 5, 13, 100, 1000] {
        for &(n_ch, t) in &[(1usize, 1usize), (1, 2), (3, 250), (8, 2560), (21, 625)] {
            let signal = make_signal(n_ch, t, (delta as i64) * 17 + t as i64);
            let bytes = compress_bounded_mae(&signal, delta, LpcMode::default())
                .unwrap_or_else(|e| panic!("encode failed δ={} n_ch={} t={}: {:?}", delta, n_ch, t, e));
            let recon = decompress(&bytes)
                .unwrap_or_else(|e| panic!("decode failed δ={} n_ch={} t={}: {:?}", delta, n_ch, t, e));

            assert_eq!(recon.len(), n_ch, "channel count");
            for c in 0..n_ch {
                assert_eq!(recon[c].len(), t, "sample count ch={}", c);
                let mae = signal[c]
                    .iter()
                    .zip(&recon[c])
                    .map(|(a, b)| (a - b).abs())
                    .max()
                    .unwrap_or(0);
                assert!(
                    mae <= delta as i64,
                    "MAE bound VIOLATED δ={} n_ch={} t={} ch={} mae={}",
                    delta, n_ch, t, c, mae
                );
                if delta == 0 {
                    assert_eq!(signal[c], recon[c], "δ=0 must be byte-exact ch={}", c);
                }
            }
        }
    }
}

#[test]
fn bounded_mae_encode_is_deterministic() {
    // Same input → byte-identical packet (guards against any nondeterminism
    // in the encode path). Cross-backend (Firmware vs Desktop) equality is
    // inherited: the only SIMD in this path is the autocorrelation kernel,
    // which is bit-identical to its scalar form by construction and already
    // locked by the `byte_equal_backends` golden gate.
    let signal = make_signal(8, 2560, 7);
    let a = compress_bounded_mae(&signal, 8, LpcMode::default()).unwrap();
    let b = compress_bounded_mae(&signal, 8, LpcMode::default()).unwrap();
    assert_eq!(a, b, "bounded-MAE encode must be deterministic");
}

#[test]
fn bounded_mae_higher_delta_compresses_smaller() {
    // Sanity: a looser error budget should not produce a LARGER packet than
    // a tighter one on the same signal (monotone rate-distortion direction).
    let signal = make_signal(8, 2560, 4242);
    let lossless = compress_bounded_mae(&signal, 0, LpcMode::default()).unwrap().len();
    let loose = compress_bounded_mae(&signal, 64, LpcMode::default()).unwrap().len();
    assert!(
        loose <= lossless,
        "δ=64 packet ({} B) should be ≤ δ=0 packet ({} B)",
        loose, lossless
    );
}
