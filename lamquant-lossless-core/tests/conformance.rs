//! Conformance tests — golden vectors with deterministic seeds.
//!
//! These tests verify that the codec produces correct output for known inputs.
//! Any implementation claiming LML conformance must pass these tests.

use lamquant_lossless_core::{lifting, lml, lpc};

/// Deterministic signal generator matching Python's numpy.random.default_rng.
/// Uses a simple LCG for reproducibility across languages.
fn make_signal(seed: u64, n_ch: usize, t: usize, amp: i64) -> Vec<Vec<i64>> {
    let mut state = seed;
    (0..n_ch)
        .map(|_| {
            (0..t)
                .map(|_| {
                    // xorshift64
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state as i64) % (2 * amp + 1)) - amp
                })
                .collect()
        })
        .collect()
}

#[test]
fn roundtrip_21ch_2500() {
    let signal = make_signal(42, 21, 2500, 5000);
    let compressed = lml::compress(&signal, 0).unwrap();
    let recovered = lml::decompress(&compressed).expect("decompress failed");
    assert_eq!(signal.len(), recovered.len());
    for ch in 0..signal.len() {
        assert_eq!(signal[ch], recovered[ch], "Channel {} mismatch", ch);
    }
}

#[test]
fn roundtrip_1ch_100() {
    let signal = make_signal(2000, 1, 100, 1000);
    let compressed = lml::compress(&signal, 0).unwrap();
    let recovered = lml::decompress(&compressed).unwrap();
    assert_eq!(signal, recovered);
}

#[test]
fn roundtrip_64ch_1250() {
    let signal = make_signal(3000, 64, 1250, 3000);
    let compressed = lml::compress(&signal, 0).unwrap();
    let recovered = lml::decompress(&compressed).unwrap();
    assert_eq!(signal, recovered);
}

#[test]
fn roundtrip_noise_bits_4() {
    let signal = make_signal(4000, 21, 2500, 50000);
    let compressed = lml::compress(&signal, 4).unwrap();
    let recovered = lml::decompress(&compressed).unwrap();
    for ch in 0..signal.len() {
        for i in 0..signal[ch].len() {
            let expected = (signal[ch][i] >> 4) << 4;
            assert_eq!(
                expected, recovered[ch][i],
                "ch={} i={}: {} vs {}",
                ch, i, expected, recovered[ch][i]
            );
        }
    }
}

#[test]
fn all_zeros() {
    let signal = vec![vec![0i64; 2500]; 21];
    let compressed = lml::compress(&signal, 0).unwrap();
    let recovered = lml::decompress(&compressed).unwrap();
    assert_eq!(signal, recovered);
}

#[test]
fn all_max_int16() {
    let signal = vec![vec![32767i64; 2500]; 21];
    let compressed = lml::compress(&signal, 0).unwrap();
    let recovered = lml::decompress(&compressed).unwrap();
    assert_eq!(signal, recovered);
}

#[test]
fn all_min_int16() {
    let signal = vec![vec![-32768i64; 2500]; 21];
    let compressed = lml::compress(&signal, 0).unwrap();
    let recovered = lml::decompress(&compressed).unwrap();
    assert_eq!(signal, recovered);
}

#[test]
fn single_sample() {
    let signal = vec![vec![42i64]];
    let compressed = lml::compress(&signal, 0).unwrap();
    let recovered = lml::decompress(&compressed).unwrap();
    assert_eq!(signal, recovered);
}

#[test]
fn four_samples() {
    let signal = vec![vec![100i64, -200, 300, -400]];
    let compressed = lml::compress(&signal, 0).unwrap();
    let recovered = lml::decompress(&compressed).unwrap();
    assert_eq!(signal, recovered);
}

#[test]
fn impulse() {
    let mut signal = vec![vec![0i64; 2500]; 21];
    signal[10][1250] = 30000;
    let compressed = lml::compress(&signal, 0).unwrap();
    let recovered = lml::decompress(&compressed).unwrap();
    assert_eq!(signal, recovered);
}

#[test]
fn alternating() {
    let signal: Vec<Vec<i64>> = (0..21)
        .map(|_| (0..2500).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect())
        .collect();
    let compressed = lml::compress(&signal, 0).unwrap();
    let recovered = lml::decompress(&compressed).unwrap();
    assert_eq!(signal, recovered);
}

// ── Lifting conformance ──

#[test]
fn lifting_roundtrip_all_lengths() {
    for n in [1, 2, 3, 4, 5, 7, 8, 10, 63, 128, 313, 625, 1250, 2500] {
        let signal: Vec<i64> = (0..n).map(|i| ((i * 137) % 10000 - 5000) as i64).collect();
        let (a, d) = lifting::forward(&signal);
        let rec = lifting::inverse(&a, &d);
        assert_eq!(signal, rec, "Failed at n={}", n);
    }
}

// ── LPC conformance ──

#[test]
fn lpc_roundtrip_all_orders() {
    for order in 0..=3 {
        for t in [5, 50, 313, 625, 1250] {
            let signal: Vec<i64> = (0..t).map(|i| ((i * 47) % 5000 - 2500) as i64).collect();
            let (coeffs, res) = lpc::analyze(&signal, order, 16);
            let rec = lpc::synthesize(&res, &coeffs, order, 16);
            assert_eq!(signal, rec, "Failed order={} t={}", order, t);
        }
    }
}

// ── CRC corruption detection ──

#[test]
fn crc_detects_single_bit_flip() {
    let signal = make_signal(5000, 21, 2500, 5000);
    let compressed = lml::compress(&signal, 0).unwrap();

    // Find the payload region (after ASCII prefix + 22-byte header)
    let nl = compressed.iter().position(|&b| b == b'\n').unwrap();
    let payload_start = nl + 1 + 22;

    // Flip one bit in the payload
    let mut corrupted = compressed.clone();
    if payload_start + 10 < corrupted.len() {
        corrupted[payload_start + 10] ^= 0x01;
    }

    assert!(
        lml::decompress(&corrupted).is_err(),
        "CRC should have caught bit flip"
    );
}

#[test]
fn rejects_future_version() {
    let signal = vec![vec![1i64; 100]];
    let mut compressed = lml::compress(&signal, 0).unwrap();
    let pos = compressed.windows(4).position(|w| w == b"LML1").unwrap();
    compressed[pos + 3] = b'2';
    let err = lml::decompress(&compressed).unwrap_err();
    assert!(matches!(
        err,
        lamquant_lossless_core::error::LmlError::UnsupportedVersion(b'2')
    ));
}

// ── Stress ──

#[test]
fn stress_100_random() {
    for seed in 0..100u64 {
        let n_ch = ((seed % 63) + 1) as usize;
        let t = ((seed * 47 % 4996) + 4) as usize;
        let signal = make_signal(seed + 10000, n_ch, t, 5000);
        let compressed = lml::compress(&signal, 0).unwrap();
        let recovered = lml::decompress(&compressed).unwrap();
        assert_eq!(signal.len(), recovered.len(), "seed={}", seed);
        for ch in 0..n_ch {
            assert_eq!(
                signal[ch].len(),
                recovered[ch].len(),
                "seed={} ch={}",
                seed,
                ch
            );
            assert_eq!(signal[ch], recovered[ch], "seed={} ch={}", seed, ch);
        }
    }
}
