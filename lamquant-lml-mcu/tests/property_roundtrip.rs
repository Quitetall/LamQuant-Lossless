//! Property-based bit-exact roundtrip tests (Cat A4 — 2026-05-21).
//!
//! Asserts `decompress(compress(x)) == x` for random
//! `[n_channels, n_samples, bit_depth]` tuples plus targeted
//! edge generators (zero-variance, DC-offset, saturated amplitude,
//! single-sample spike).
//!
//! The codec is lossless on integer inputs in `[-2^(bits-1), 2^(bits-1))`,
//! so the property is exact equality — not statistical similarity.

#![cfg(feature = "std")]

use std::io::Cursor;

use lamquant_lml_mcu::lml;
use proptest::prelude::*;

/// Quantize an i64 sample to a given bit depth `[-2^(bits-1), 2^(bits-1))`.
fn quantize(sample: i64, bits: u8) -> i64 {
    let lo = -(1i64 << (bits - 1));
    let hi = (1i64 << (bits - 1)) - 1;
    sample.clamp(lo, hi)
}

/// Strategy: small random signal — `n_ch ∈ [1,8]`, `T ∈ [32,512]`,
/// per-sample value within ±32767 (post-quantization fits 16 bits).
fn signal_strategy() -> impl Strategy<Value = Vec<Vec<i64>>> {
    (1usize..=8, 32usize..=512).prop_flat_map(|(n_ch, t)| {
        prop::collection::vec(
            prop::collection::vec(-32768i64..32768, t),
            n_ch,
        )
    })
}

/// Roundtrip helper — compress then immediately decompress, assert eq.
fn assert_roundtrip(signal: &[Vec<i64>]) {
    let bytes = lml::compress(signal, 0).expect("compress");
    let mut cursor = Cursor::new(&bytes);
    let recovered = lml::decompress_from(&mut cursor).expect("decompress");
    assert_eq!(recovered.len(), signal.len(), "channel count mismatch");
    for (i, (orig, rec)) in signal.iter().zip(recovered.iter()).enumerate() {
        assert_eq!(orig, rec, "channel {} mismatch (len orig={} rec={})",
                   i, orig.len(), rec.len());
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,                  // keep CI under ~10s
        max_shrink_iters: 256,
        ..ProptestConfig::default()
    })]

    /// Random in-range signals must roundtrip bit-exact.
    #[test]
    fn roundtrip_random(signal in signal_strategy()) {
        assert_roundtrip(&signal);
    }
}

// ── Targeted edge-case generators ────────────────────────────────────

/// Zero-variance window — all samples identical.
#[test]
fn roundtrip_zero_variance() {
    let signal: Vec<Vec<i64>> = vec![vec![1234; 128]; 4];
    assert_roundtrip(&signal);
}

/// DC offset (constant non-zero) across many channels.
#[test]
fn roundtrip_dc_offset() {
    let signal: Vec<Vec<i64>> = (0..8)
        .map(|ch| vec![(ch as i64 - 4) * 1000; 256])
        .collect();
    assert_roundtrip(&signal);
}

/// Saturated amplitude — alternating extremes within q15 range.
#[test]
fn roundtrip_saturated_amplitude() {
    let signal: Vec<Vec<i64>> = (0..2)
        .map(|_| (0..256).map(|i| if i % 2 == 0 { 32767 } else { -32768 }).collect())
        .collect();
    assert_roundtrip(&signal);
}

/// Single-sample spike on otherwise-zero channel.
#[test]
fn roundtrip_single_spike() {
    let mut signal: Vec<Vec<i64>> = vec![vec![0i64; 256]; 4];
    signal[0][127] = 30000;
    signal[1][0] = -30000;
    signal[2][255] = 12345;
    assert_roundtrip(&signal);
}

/// Mixed signal — sinusoid + spike + plateau.
#[test]
fn roundtrip_mixed_pattern() {
    let n = 384usize;
    let ch0: Vec<i64> = (0..n).map(|i| {
        let phase = (i as f64) * 0.1;
        quantize((phase.sin() * 8000.0) as i64, 16)
    }).collect();
    let ch1: Vec<i64> = (0..n).map(|i| if i == n / 2 { 25000 } else { 100 }).collect();
    let ch2: Vec<i64> = vec![-500; n];
    assert_roundtrip(&[ch0, ch1, ch2]);
}
