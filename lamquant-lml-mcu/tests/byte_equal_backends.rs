//! Byte-equal cross-backend conformance.
//!
//! Pins the invariant `Scalar.encode(x) == Vectorized.encode(x)` for
//! a small set of deterministic input fixtures. This is the gate the
//! perf-hardening workstream sits behind: when AVX2 / NEON paths
//! land inside the `Vectorized` arm, ANY drift from the scalar
//! reference output makes this test fail.
//!
//! TDD ordering:
//!   1. (this commit) — write the gate. Both backends currently
//!      dispatch to the same scalar code, so the test passes
//!      trivially.
//!   2. (later commit) — add real SIMD inside `ComputeBackend::
//!      Vectorized` impl. If outputs diverge by even one byte, this
//!      file fails.
//!
//! If a legitimate wire-format change lands (e.g. a flag bit bumps
//! from 0 to 1 in the header), this test will fail because the
//! SHA changes. Re-generate the golden table by running with
//! `LAMQUANT_REGEN_GOLDENS=1 cargo test --test byte_equal_backends`
//! and commit the new SHAs alongside the wire-format change.
//!
//! Endianness: the LML wire format is little-endian by spec (see
//! `docs/lml-format-v1.md` and `lamquant-core/src/lml.rs`'s
//! `to_le_bytes` / `from_le_bytes` calls). The compressor normalises
//! on emit, so these golden SHAs hold on both little- and big-endian
//! hosts. Cross-architecture CI (the macOS + Windows lanes from
//! Phase 2.1 / 2.2) re-validate this every commit.
//!
//! Publishing the goldens externally: they live alongside the
//! source for now. If a future third-party reader wants to run the
//! same gate without compiling Rust, mirror the table into
//! `specs/conformance/byte_equal_v1.json` at that time.

use lamquant_lml_mcu::backend::{
    compress_with_backend, decompress_with_backend, ComputeBackend,
};
use lamquant_lml_mcu::lpc::LpcMode;
use sha2::{Digest, Sha256};

/// xorshift64 — deterministic across machines + architectures.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Match the bench harness's signal generator so the conformance
/// table can extend to cover bench shapes too.
fn synth_signal(n_ch: usize, t: usize, seed: u64) -> Vec<Vec<i64>> {
    (0..n_ch)
        .map(|c| {
            let mut state =
                seed.wrapping_add((c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut ch = Vec::with_capacity(t);
            let mut prev: i64 = 0;
            for _ in 0..t {
                let step = (xorshift64(&mut state) as i32 >> 24) as i64;
                prev = (prev + step).clamp(-2000, 2000);
                ch.push(prev);
            }
            ch
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// One conformance vector — fixed input shape + locked SHA-256 of
/// the encoded `.lml` payload under `Firmware` backend (reference).
#[derive(Debug)]
struct GoldenVector {
    name: &'static str,
    n_ch: usize,
    t: usize,
    seed: u64,
    noise_bits: u8,
    lpc_mode_name: &'static str, // for human-readable failure messages
    lpc_mode: LpcMode,
    /// SHA-256 of `Firmware.compress(synth_signal(seed, ...))`
    /// recorded by running the test once with
    /// `LAMQUANT_REGEN_GOLDENS=1` then pasting the printed values
    /// back here. The wire format is unchanged across backends, so
    /// the `Desktop` backend MUST produce the same SHAs.
    sha256_firmware: &'static str,
}

/// Golden table. Cover three shapes intentionally:
///   * 1ch_100      — minimum-viable signal
///   * 4ch_2500     — single-window 10s @ 250Hz, common clinical
///   * 32ch_2500    — multi-channel scalp EEG width
///
/// Larger shapes are out of scope for this gate — they're covered
/// by the criterion bench + the existing per-format integration
/// tests. The gate measures BACKEND DRIFT, not corpus coverage.
const GOLDEN_VECTORS: &[GoldenVector] = &[
    GoldenVector {
        name: "1ch_100",
        n_ch: 1,
        t: 100,
        seed: 0x1111_2222_3333_4444,
        noise_bits: 0,
        lpc_mode_name: "default(anytime)",
        lpc_mode: LpcMode::Anytime {
            max_order: 16,
            deadline: None,
        },
        // ADR 0023 Track B1+: feature-gated as of the post-perf-
        // regression fix. Default builds (no `experimental_bit_pack`
        // / `experimental_arithmetic` features) skip the per-subband
        // selection entirely → golden SHA matches the pre-B1 codec.
        // When the features ARE compiled in + the env vars are set,
        // 1ch_100 produces a different (smaller) output, but that
        // path is exercised by the dedicated experimental test
        // setup not by this byte-equal-backends golden.
        sha256_firmware: "aeacda4c3b675cc58118810063009f5b84f421da951375bcbfa2f6da206df5b6",
    },
    GoldenVector {
        name: "4ch_2500",
        n_ch: 4,
        t: 2500,
        seed: 0xDEAD_BEEF_CAFE_BABE,
        noise_bits: 0,
        lpc_mode_name: "default(anytime)",
        lpc_mode: LpcMode::Anytime {
            max_order: 16,
            deadline: None,
        },
        sha256_firmware: "fde1e7f711483e30d4332f07bcf928da8a351a0c67e9f3844f9ebd7c153114f1",
    },
    GoldenVector {
        name: "4ch_2500_fixed",
        n_ch: 4,
        t: 2500,
        seed: 0xDEAD_BEEF_CAFE_BABE,
        noise_bits: 0,
        lpc_mode_name: "fixed",
        lpc_mode: LpcMode::Fixed,
        sha256_firmware: "074d62518de8fdc3d4dc547c95e5631974a0e9e471d95112574193e1279c110a",
    },
    GoldenVector {
        name: "32ch_2500",
        n_ch: 32,
        t: 2500,
        seed: 0xCAFE_BABE_F00D_BEEF,
        noise_bits: 0,
        lpc_mode_name: "default(anytime)",
        lpc_mode: LpcMode::Anytime {
            max_order: 16,
            deadline: None,
        },
        sha256_firmware: "1e92724c7c9d522f8b342c52b54227010d0f758bff602023a0600cfe9d25cc1c",
    },
];

/// Set this to bypass golden assertions and just emit SHAs for paste.
fn regen_mode() -> bool {
    std::env::var("LAMQUANT_REGEN_GOLDENS")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[test]
fn firmware_backend_matches_golden_shas() {
    let mut failures = Vec::new();
    let regen = regen_mode();
    if regen {
        println!("\n## Regenerated golden SHAs (paste into GOLDEN_VECTORS):");
    }
    for v in GOLDEN_VECTORS {
        let signal = synth_signal(v.n_ch, v.t, v.seed);
        let encoded = compress_with_backend(
            &signal,
            v.noise_bits,
            v.lpc_mode,
            ComputeBackend::Firmware,
        )
        .expect("firmware compress");
        let actual = sha256_hex(&encoded);
        if regen {
            println!(
                "    {}  ({}, lpc={}) -> sha256_firmware: \"{}\",",
                v.name, v.n_ch, v.lpc_mode_name, actual
            );
            continue;
        }
        if actual != v.sha256_firmware {
            failures.push(format!(
                "vector `{}` drifted:\n  expected sha256 = {}\n  actual   sha256 = {}",
                v.name, v.sha256_firmware, actual
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "firmware backend output drifted from goldens:\n{}",
        failures.join("\n")
    );
}

/// The load-bearing test for the perf workstream: every backend
/// available on this build MUST produce the same encoded bytes for
/// every golden vector. When real SIMD lands in the `Desktop`
/// backend, this test will fail the moment outputs diverge.
#[test]
#[cfg(feature = "archive")]
fn desktop_backend_matches_firmware_bytes() {
    let mut failures = Vec::new();
    for v in GOLDEN_VECTORS {
        let signal = synth_signal(v.n_ch, v.t, v.seed);
        let firmware = compress_with_backend(
            &signal,
            v.noise_bits,
            v.lpc_mode,
            ComputeBackend::Firmware,
        )
        .expect("firmware compress");
        let desktop = compress_with_backend(
            &signal,
            v.noise_bits,
            v.lpc_mode,
            ComputeBackend::Desktop,
        )
        .expect("desktop compress");
        if firmware != desktop {
            failures.push(format!(
                "vector `{}` diverged ({} bytes firmware vs {} bytes desktop; sha {} vs {})",
                v.name,
                firmware.len(),
                desktop.len(),
                sha256_hex(&firmware),
                sha256_hex(&desktop),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Desktop backend diverged from Firmware:\n{}",
        failures.join("\n")
    );
}

/// Cross-backend decode equality: feed bytes produced by Firmware
/// into both backends' decompressor, assert identical recovered
/// signal. This sister test catches drift in the decoder hot path.
#[test]
#[cfg(feature = "archive")]
fn desktop_backend_decode_matches_firmware() {
    for v in GOLDEN_VECTORS {
        let signal = synth_signal(v.n_ch, v.t, v.seed);
        let bytes = compress_with_backend(
            &signal,
            v.noise_bits,
            v.lpc_mode,
            ComputeBackend::Firmware,
        )
        .expect("compress");
        let dec_firmware = decompress_with_backend(&bytes, ComputeBackend::Firmware)
            .expect("firmware decompress");
        let dec_desktop = decompress_with_backend(&bytes, ComputeBackend::Desktop)
            .expect("desktop decompress");
        assert_eq!(
            dec_firmware, dec_desktop,
            "vector `{}` decode diverged across backends",
            v.name
        );
        // Sanity: decode must roundtrip the input.
        assert_eq!(
            dec_firmware, signal,
            "vector `{}` roundtrip failed",
            v.name
        );
    }
}
