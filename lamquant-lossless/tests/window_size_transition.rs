//! Track B3 transition regression — encoded size scans across signal
//! lengths via the **real** ingest pipeline (pack_archive on
//! synthetic ASCII files) so `auto_window_size`'s window-size pick
//! is exercised end-to-end.
//!
//! ADR 0023 Track B3: `auto_window_size` has a regime-2 / regime-3
//! boundary at `MAX_AUTO_WINDOW = 16384` ref samples. Below the
//! threshold, the encoder picks a single window covering the whole
//! signal; above, it falls back to the legacy 2500-sample chunking.
//! This test:
//!
//!   1. Synthesises ASCII int-per-line files of lengths spanning
//!      regimes 1, 2, and 3 (including ±1 sample around the
//!      boundary).
//!   2. Packs each into its own LMA archive (real ingest →
//!      synth EDF → encode_edf_to_lml → window pick → compress).
//!   3. Extracts the archive + verifies byte-exact roundtrip.
//!   4. Asserts bits-per-sample doesn't dip pathologically at any
//!      transition — the regime-2→3 cliff in particular must stay
//!      within ~15 % so the cliff is harmless to CR.
//!
//! Why pack_archive (not compress_with_mode) — `auto_window_size`
//! is only invoked on the ingest path, not on the bare `compress`
//! call. To exercise the B3 logic we have to go through the full
//! `pack_archive` → `try_ingest_to_lml` → `encode_edf_to_lml` chain.

use lamquant_core::lma::{list_archive, pack_archive, unpack_archive};
use sha2::{Digest, Sha256};
use std::fs;

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// Render `n` AR(1) integer samples as an ASCII int-per-line file
/// with CRLF line endings (matches the Bonn dataset format). The
/// PRNG is deterministic; the same `n` always produces the same
/// bytes. Returns the encoded file content.
fn synth_ascii_signal(n: usize) -> Vec<u8> {
    let mut state: i64 = 0;
    let mut seed: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let mut out = Vec::with_capacity(n * 6);
    for _ in 0..n {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let noise = ((seed as i32) % 11) as i64;
        state = ((state * 31) / 32) + noise;
        // Clamp into i16 so the ingest detector's i16 fit-check passes.
        let v = state.clamp(-30_000, 30_000);
        out.extend_from_slice(format!("{}\r\n", v).as_bytes());
    }
    out
}

#[test]
fn cr_does_not_dip_across_window_size_transitions() {
    let src = tempfile::tempdir().expect("src tempdir");

    // Lengths spanning regimes 1, 2, and 3. Includes ±1 sample
    // around the 16384-sample regime-2→3 boundary so a 1-sample-
    // wide cliff in `auto_window_size` would show up.
    //
    // Filenames match the Bonn `[A-Z]\d{3}` pattern so
    // `guess_synth_sample_rate` picks 173.61 Hz (Bonn standard).
    // At 173.61 Hz: ref_signal = n × 250 / 173.61. The regime
    // boundary at ref = 16384 corresponds to actual n ≈ 11377
    // samples. Probe near there for direct cliff coverage.
    let lengths: &[usize] = &[
        // Regime 1 / 2 (short signals, single-window).
        500, 1_000, 2_500, 4_097, // ← Bonn dataset record length
        6_000, 8_000, 10_000,
        // Boundary at actual ≈ 11377.
        11_376, 11_378, 11_400,
        // Regime 3 (multi-window default chunking).
        12_000, 16_000, 24_000, 32_000,
    ];

    // Write one .txt per length into the source directory.
    let mut originals: Vec<(String, Vec<u8>)> = Vec::with_capacity(lengths.len());
    for (i, &n) in lengths.iter().enumerate() {
        // Bonn-style filename so guess_synth_sample_rate picks 173.61
        // Hz for the EDF synthesis. Single uppercase + 3 digits.
        let name = format!("L{:03}.txt", i + 1);
        let bytes = synth_ascii_signal(n);
        fs::write(src.path().join(&name), &bytes).expect("write");
        originals.push((name, bytes));
    }

    // Pack into one archive — every file goes through the ingest
    // pipeline (sniff → synth EDF → encode_edf_to_lml → window
    // pick via auto_window_size).
    let archive = tempfile::Builder::new()
        .prefix("window_transition_")
        .suffix(".lma")
        .tempfile()
        .expect("archive tempfile");
    let summary =
        pack_archive(src.path(), archive.path(), 9, false, None).expect("pack");
    // Most entries should hit the ingest path; very small files
    // (sub-2 KiB) can fall back to zstd when zstd's smaller output
    // beats the LML's encode overhead. Either path is correct;
    // we just need a majority on LML so the transition test actually
    // exercises auto_window_size.
    assert!(
        summary.counts_lml >= lengths.len() * 2 / 3,
        "expected ≥2/3 of entries on LML path; got lml={} zstd={} store={} of {}",
        summary.counts_lml,
        summary.counts_zstd,
        summary.counts_store,
        lengths.len()
    );

    // List entries + map to (n, compressed_size).
    let entries = list_archive(archive.path()).expect("list");
    let mut scan: Vec<(usize, u64)> = Vec::with_capacity(lengths.len());
    for (i, &n) in lengths.iter().enumerate() {
        let name = format!("L{:03}.txt", i + 1);
        let entry = entries
            .iter()
            .find(|e| e.path == name)
            .unwrap_or_else(|| panic!("missing entry {}", name));
        scan.push((n, entry.compressed_size));
    }

    // 1. Byte-exact roundtrip.
    let dst = tempfile::tempdir().expect("dst tempdir");
    let unpack = unpack_archive(archive.path(), dst.path(), true, false, None).expect("unpack");
    assert!(unpack.errors.is_empty(), "unpack errors: {:?}", unpack.errors);
    for (name, expected) in &originals {
        let got = fs::read(dst.path().join(name)).expect("read extracted");
        assert_eq!(
            sha256_hex(&got),
            sha256_hex(expected),
            "byte mismatch on {}",
            name
        );
    }

    // 2. CR smoothness — bits/sample doesn't EXPLODE across any
    //    transition. 1.25x cap is the loosest reasonable threshold:
    //    anything tighter triggers on legitimate per-window header
    //    overhead at regime boundaries; anything looser misses
    //    real pathological window picks.
    let bps_series: Vec<(usize, f64, u64)> = scan
        .iter()
        .map(|(n, sz)| (*n, (*sz as f64) * 8.0 / (*n as f64), *sz))
        .collect();
    for w in bps_series.windows(2) {
        let (n_a, bps_a, _) = w[0];
        let (n_b, bps_b, _) = w[1];
        let ratio = bps_b / bps_a;
        assert!(
            ratio <= 1.25 && ratio >= 0.80,
            "bits/sample changed too sharply across n={} → n={}: \
             {:.3} → {:.3} bps (ratio {:.3})",
            n_a,
            n_b,
            bps_a,
            bps_b,
            ratio
        );
    }

    // 3. Focused regime-boundary check (n=11376 vs n=11378 — one
    //    sample on each side of actual = 11377 ≈ ref-rate-16384).
    let at_boundary = bps_series.iter().find(|(n, _, _)| *n == 11_376).expect("11376");
    let past_boundary = bps_series.iter().find(|(n, _, _)| *n == 11_378).expect("11378");
    let cliff_ratio = past_boundary.1 / at_boundary.1;
    assert!(
        (0.85..=1.15).contains(&cliff_ratio),
        "regime 2→3 boundary CR cliff is too large: \
         n=11376 {:.3} bps vs n=11378 {:.3} bps (ratio {:.3})",
        at_boundary.1,
        past_boundary.1,
        cliff_ratio
    );

    // Debug dump on `--nocapture`.
    eprintln!("window-size transition scan (via ingest + auto_window_size):");
    eprintln!("  {:>7}  {:>10}  {:>8}", "n", "bytes", "bps");
    for (n, bps, sz) in &bps_series {
        eprintln!("  {:>7}  {:>10}  {:>8.3}", n, sz, bps);
    }
}
