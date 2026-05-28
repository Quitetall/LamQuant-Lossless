//! Fuzz the roundtrip: compress → decompress must be bit-exact.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Interpret fuzzer data as a small signal
    if data.len() < 4 { return; }
    let n_ch = (data[0] as usize % 8) + 1;  // 1-8 channels
    let remaining = &data[1..];
    let n_samp = remaining.len() / (n_ch * 2);
    if n_samp < 1 || n_samp > 10000 { return; }

    // Build signal from fuzzer bytes
    let signal: Vec<Vec<i64>> = (0..n_ch)
        .map(|ch| {
            (0..n_samp)
                .map(|s| {
                    let off = (ch * n_samp + s) * 2;
                    if off + 1 < remaining.len() {
                        i16::from_le_bytes([remaining[off], remaining[off + 1]]) as i64
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect();

    // `compress` may reject malformed input (zero channels, t mismatch,
    // adversarial noise_bits). Those Err cases are not fuzz-target bugs —
    // skip them so we only assert the roundtrip on payloads the encoder
    // actually accepted.
    let compressed = match lamquant_core::lml::compress(&signal, 0) {
        Ok(c) => c,
        Err(_) => return,
    };
    let recovered = lamquant_core::lml::decompress(&compressed)
        .expect("decompress of our own compressed data must not fail");

    assert_eq!(signal.len(), recovered.len(), "channel count mismatch");
    for ch in 0..signal.len() {
        assert_eq!(signal[ch], recovered[ch], "channel {} data mismatch", ch);
    }
});
