//! ADR 0069/0071 L9 — BCS1 read-side completion, end-to-end CLI proof.
//!
//! `write_abir` (`abir_container.rs`) now emits `BCS1` by default; this
//! file is the durable regression suite proving the REST of the read side
//! (the `lml` CLI, not just the whole-file `container::read_bytes`/
//! `read_file` facade) actually understands it:
//!
//!   - `split_on_bcs1_file_reports_correct_sample_rate` — the explicit
//!     regression guard for the silent bug in
//!     `bin/lml.rs::read_sample_rate_from_header`: pre-fix, its
//!     legacy-shaped version probe (`hdr[4..6] == 1`) happened to also
//!     pass on a BCS1 file (byte coincidence — BCS1's
//!     `version_major`/`version_minor` are `01 00` too), so it silently
//!     read BCS1's `total_samples` field (offset 16..20) reinterpreted as
//!     a millihertz sample rate instead of erroring or reading the real
//!     field at offset 22..26. No exception was raised; `lml split`
//!     would have produced chunks tagged with a bogus, unrelated
//!     sample rate.
//!   - `encode_decode_round_trip_on_bcs1_is_lossless` — the PRIMARY
//!     workflow: `lml encode` (which now writes BCS1) followed by
//!     `lml decode` must reconstruct the original signal exactly.
//!   - `cmd_info_and_stats_succeed_on_bcs1_file` — `lml info`/`lml stats`
//!     must not reject a BCS1 file.
//!   - `legacy_lml1_conformance_vector_still_works_via_every_cli_path` —
//!     back-compat: a REAL pre-BCS1 `.lml` conformance vector
//!     (`specs/conformance/vectors/basic_4ch_1024s_250hz.lml`, committed,
//!     untouched) must still decode/info/stats/split unchanged.

#![cfg(feature = "archive")]

use abir::{Bcs1Header, BCS1_HEADER_LEN, BCS1_MAGIC, CODEC_LMQ_FSQ};
use lamquant_core::container;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Locate the `lml` binary cargo just built for THIS invocation's feature
/// set. Mirrors `tests/cli_metadata_snapshot.rs::lml_bin` /
/// `tests/op_e2e.rs::lml_path` — same lookup convention, duplicated per
/// existing precedent in this test suite (each integration test binary is
/// self-contained).
fn lml_bin() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = manifest_dir
        .parent()
        .expect("workspace root")
        .join("target");
    for c in &[
        target.join("debug").join("lml"),
        target.join("debug").join("lml.exe"),
        target.join("release").join("lml"),
        target.join("release").join("lml.exe"),
    ] {
        if c.exists() {
            return c.clone();
        }
    }
    panic!("lml binary not built; run `cargo build --bin lml --features host` first");
}

fn synth_signal(n_ch: usize, t: usize, seed: u64) -> Vec<Vec<i64>> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut sig = vec![Vec::with_capacity(t); n_ch];
    for ch in 0..n_ch {
        for _ in 0..t {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            sig[ch].push(((state >> 33) as i32) as i64 % 8000);
        }
    }
    sig
}

const LEGACY_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../specs/conformance/vectors/basic_4ch_1024s_250hz.lml"
);

/// The explicit regression guard requested by ADR 0069/0071 L9: `lml
/// split` on a BCS1 file must report/embed the CORRECT sample rate, never
/// the silently-wrong one the pre-fix `read_sample_rate_from_header` bug
/// produced.
#[test]
fn split_on_bcs1_file_reports_correct_sample_rate() {
    let tmp = tempfile::tempdir().unwrap();
    let input_path = tmp.path().join("input.lml");

    // A deliberately distinctive, non-default sample rate: if the silent
    // bug regresses (reading `total_samples` instead of
    // `sample_rate_mhz`), the split chunks would carry the total_samples
    // count (512) reinterpreted as a millihertz rate (0.512 Hz) instead
    // of 733.0 Hz — an unmistakable mismatch.
    let distinctive_sr = 733.0_f64;
    let sig = synth_signal(2, 512, 7);
    container::write_file(&input_path, &sig, distinctive_sr, 128, 0, "{}").unwrap();

    // Confirm the fixture really is BCS1 — guards this test itself
    // against a future write-path regression silently reverting to
    // LML1, which would make this regression guard meaningless.
    let bytes = std::fs::read(&input_path).unwrap();
    assert_eq!(
        &bytes[0..4],
        BCS1_MAGIC,
        "fixture must be BCS1 for this regression guard to mean anything"
    );

    let out_dir = tmp.path().join("chunks");
    let out = Command::new(lml_bin())
        .arg("split")
        .arg(&input_path)
        .arg("--chunks")
        .arg("2")
        .arg("-o")
        .arg(&out_dir)
        .output()
        .expect("spawn lml split");
    assert!(
        out.status.success(),
        "lml split failed: status={:?}\nstdout={}\nstderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let mut chunks: Vec<PathBuf> = std::fs::read_dir(&out_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().map(|e| e == "lml").unwrap_or(false))
        .collect();
    chunks.sort();
    assert_eq!(chunks.len(), 2, "expected 2 split chunks, got {:?}", chunks);

    for chunk_path in &chunks {
        let chunk_bytes = std::fs::read(chunk_path).unwrap();
        assert_eq!(
            &chunk_bytes[0..4],
            BCS1_MAGIC,
            "split chunk must also be BCS1 (write_into always emits BCS1 today)"
        );
        let hdr = Bcs1Header::parse(&chunk_bytes[..BCS1_HEADER_LEN]).unwrap();
        let sr = hdr.sample_rate_mhz as f64 / 1000.0;
        assert!(
            (sr - distinctive_sr).abs() < 1e-6,
            "chunk {} sample_rate {sr} != source {distinctive_sr} — the ADR 0069/0071 L9 \
             silent-bug regression: read_sample_rate_from_header must read BCS1's \
             sample_rate_mhz field (offset 22..26), not total_samples (offset 16..20)",
            chunk_path.display()
        );
    }
}

/// The PRIMARY workflow: `lml encode` (BCS1 by default) -> `lml decode`
/// reconstructs the original signal byte-exact.
#[test]
fn encode_decode_round_trip_on_bcs1_is_lossless() {
    let tmp = tempfile::tempdir().unwrap();

    // Raw int16 multiplexed + JSON sidecar — mirrors
    // `cli_metadata_snapshot.rs::raw_fixture`'s shape.
    let n_ch = 2usize;
    let n_samples = 200usize;
    let mut raw_bytes = Vec::with_capacity(n_ch * n_samples * 2);
    let mut expected: Vec<Vec<i32>> = vec![Vec::with_capacity(n_samples); n_ch];
    for s in 0..n_samples {
        for ch in 0..n_ch {
            let v = ((s as i64 * (ch as i64 + 1) * 37) % 4001 - 2000) as i16;
            raw_bytes.extend_from_slice(&v.to_le_bytes());
            expected[ch].push(v as i32);
        }
    }
    let raw_path = tmp.path().join("data.raw");
    std::fs::write(&raw_path, &raw_bytes).unwrap();
    let sidecar = format!(
        "{{\"n_channels\":{n_ch},\"sample_rate\":250.0,\"dtype\":\"int16\",\
         \"orientation\":\"multiplexed\",\"channels\":[\"ch0\",\"ch1\"],\
         \"phys_min\":[-200.0,-200.0],\"phys_max\":[200.0,200.0],\"phys_dim\":\"uV\"}}"
    );
    std::fs::write(tmp.path().join("data.json"), sidecar).unwrap();

    let lml_path = tmp.path().join("data.lml");
    let encode_out = Command::new(lml_bin())
        .arg("encode")
        .arg(&raw_path)
        .arg("-o")
        .arg(&lml_path)
        .arg("--no-bundle")
        .arg("--i-understand-data-loss")
        .output()
        .expect("spawn lml encode");
    assert!(
        encode_out.status.success(),
        "lml encode failed: stdout={}\nstderr={}",
        String::from_utf8_lossy(&encode_out.stdout),
        String::from_utf8_lossy(&encode_out.stderr),
    );

    let encoded_bytes = std::fs::read(&lml_path).unwrap();
    assert_eq!(
        &encoded_bytes[0..4],
        BCS1_MAGIC,
        "lml encode must emit BCS1 by default (write_abir, ADR 0069/0071 L9)"
    );

    let decoded_path = tmp.path().join("data.decoded.raw");
    let decode_out = Command::new(lml_bin())
        .arg("decode")
        .arg(&lml_path)
        .arg("-o")
        .arg(&decoded_path)
        .output()
        .expect("spawn lml decode");
    assert!(
        decode_out.status.success(),
        "lml decode failed on a BCS1 file: stdout={}\nstderr={}",
        String::from_utf8_lossy(&decode_out.stdout),
        String::from_utf8_lossy(&decode_out.stderr),
    );

    // `lml decode`'s default raw output is channel-major int32 LE.
    let decoded_bytes = std::fs::read(&decoded_path).unwrap();
    assert_eq!(decoded_bytes.len(), n_ch * n_samples * 4);
    for ch in 0..n_ch {
        for s in 0..n_samples {
            let off = (ch * n_samples + s) * 4;
            let v = i32::from_le_bytes([
                decoded_bytes[off],
                decoded_bytes[off + 1],
                decoded_bytes[off + 2],
                decoded_bytes[off + 3],
            ]);
            assert_eq!(
                v, expected[ch][s],
                "decoded sample mismatch at ch={ch} s={s}: encode->decode round trip \
                 through BCS1 must be byte-exact"
            );
        }
    }
}

/// `lml info` / `lml stats` on a BCS1 file must succeed (not "Not LML or
/// LMA").
#[test]
fn cmd_info_and_stats_succeed_on_bcs1_file() {
    let tmp = tempfile::tempdir().unwrap();
    let input_path = tmp.path().join("input.lml");
    let sig = synth_signal(3, 640, 19);
    container::write_file(&input_path, &sig, 256.0, 128, 0, "{}").unwrap();
    assert_eq!(&std::fs::read(&input_path).unwrap()[0..4], BCS1_MAGIC);

    let info_out = Command::new(lml_bin())
        .arg("info")
        .arg(&input_path)
        .output()
        .expect("spawn lml info");
    assert!(
        info_out.status.success(),
        "lml info failed on BCS1: stdout={}\nstderr={}",
        String::from_utf8_lossy(&info_out.stdout),
        String::from_utf8_lossy(&info_out.stderr),
    );
    let info_stdout = String::from_utf8_lossy(&info_out.stdout);
    assert!(
        info_stdout.contains("BCS1"),
        "lml info stdout should identify the BCS1 format: {info_stdout}"
    );
    assert!(info_stdout.contains("Channels:   3"), "{info_stdout}");

    let stats_out = Command::new(lml_bin())
        .arg("stats")
        .arg(&input_path)
        .output()
        .expect("spawn lml stats");
    assert!(
        stats_out.status.success(),
        "lml stats failed on BCS1: stdout={}\nstderr={}",
        String::from_utf8_lossy(&stats_out.stdout),
        String::from_utf8_lossy(&stats_out.stderr),
    );
}

/// Back-compat (ADR 0069/0071 L9): a REAL pre-BCS1 `.lml` conformance
/// vector, committed to the repo and untouched by this change, must still
/// decode/info/stats/split via every CLI path.
#[test]
fn legacy_lml1_conformance_vector_still_works_via_every_cli_path() {
    let fixture = Path::new(LEGACY_FIXTURE);
    assert!(
        fixture.exists(),
        "conformance fixture missing: {}",
        fixture.display()
    );
    let fixture_bytes = std::fs::read(fixture).unwrap();
    assert_eq!(
        &fixture_bytes[0..4],
        b"LML1",
        "sanity: fixture must be the legacy LML1 format for this back-compat test to mean anything"
    );

    let tmp = tempfile::tempdir().unwrap();

    // decode
    let decoded_path = tmp.path().join("legacy.decoded.raw");
    let decode_out = Command::new(lml_bin())
        .arg("decode")
        .arg(fixture)
        .arg("-o")
        .arg(&decoded_path)
        .output()
        .expect("spawn lml decode");
    assert!(
        decode_out.status.success(),
        "lml decode regressed on a legacy LML1 file: stdout={}\nstderr={}",
        String::from_utf8_lossy(&decode_out.stdout),
        String::from_utf8_lossy(&decode_out.stderr),
    );
    assert!(decoded_path.exists());

    // info
    let info_out = Command::new(lml_bin())
        .arg("info")
        .arg(fixture)
        .output()
        .expect("spawn lml info");
    assert!(
        info_out.status.success(),
        "lml info regressed on a legacy LML1 file: stdout={}\nstderr={}",
        String::from_utf8_lossy(&info_out.stdout),
        String::from_utf8_lossy(&info_out.stderr),
    );
    let info_stdout = String::from_utf8_lossy(&info_out.stdout);
    assert!(
        info_stdout.contains("LML1"),
        "lml info stdout should identify the legacy LML1 format: {info_stdout}"
    );

    // stats
    let stats_out = Command::new(lml_bin())
        .arg("stats")
        .arg(fixture)
        .output()
        .expect("spawn lml stats");
    assert!(
        stats_out.status.success(),
        "lml stats regressed on a legacy LML1 file: stdout={}\nstderr={}",
        String::from_utf8_lossy(&stats_out.stdout),
        String::from_utf8_lossy(&stats_out.stderr),
    );

    // split (into a fresh output dir; never writes back over the fixture)
    let split_out_dir = tmp.path().join("legacy_chunks");
    let split_out = Command::new(lml_bin())
        .arg("split")
        .arg(fixture)
        .arg("--chunks")
        .arg("2")
        .arg("-o")
        .arg(&split_out_dir)
        .output()
        .expect("spawn lml split");
    assert!(
        split_out.status.success(),
        "lml split regressed on a legacy LML1 file: stdout={}\nstderr={}",
        String::from_utf8_lossy(&split_out.stdout),
        String::from_utf8_lossy(&split_out.stderr),
    );
    let n_chunks = std::fs::read_dir(&split_out_dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .map(|x| x == "lml")
                .unwrap_or(false)
        })
        .count();
    assert_eq!(n_chunks, 2, "lml split must still produce 2 chunks for a legacy input");

    // The fixture itself must be untouched (read-only proof — this test
    // must never mutate the committed golden).
    assert_eq!(
        std::fs::read(fixture).unwrap(),
        fixture_bytes,
        "legacy conformance fixture must not be mutated by this test"
    );
}

/// ADR 0074 Track N — the lossy-signing gate: a BCS1 file carrying the LMQ neural
/// descriptor (`CODEC_LMQ_FSQ = 0x10`) is refused FAIL-CLOSED by the lossless
/// reader — it can never be mis-decoded as integer samples. This pins the actual
/// reader behavior (not just `bcs1_gate_decodable` in isolation): a `.lmq`
/// produced by the neural shell is permanently un-decodable by every lossless path.
#[test]
fn lossless_reader_refuses_the_bcs1_lmq_neural_descriptor() {
    let header = Bcs1Header {
        version_major: 1,
        version_minor: 0,
        modality_tag: 0, // Eeg
        modality_source: 2, // Manual
        codec_descriptor: CODEC_LMQ_FSQ, // 0x10 — the neural body
        mode: 0,
        tier: 0,
        decode_capability: 1, // also non-zero → a second fail-closed reason
        n_channels: 2,
        n_windows: 1,
        total_samples: 10,
        window_size: 5,
        sample_rate_mhz: 256_000,
        bit_depth: 16,
        flags: 0,
        metadata_length: 0,
    };
    let mut bytes = header.to_bytes().to_vec();
    bytes.extend_from_slice(&[0u8; 16]); // a dummy neural body — never reached

    // The lossless whole-file reader (magic dispatch → BCS1 → gate) must REFUSE it.
    let via_dispatch = container::read_bytes(&bytes);
    assert!(via_dispatch.is_err(), "lossless read_bytes must refuse the LMQ descriptor");
    let via_bcs1 = container::bcs1_read_bytes(&bytes);
    assert!(via_bcs1.is_err(), "bcs1_read_bytes must refuse the LMQ descriptor");
    // Sanity: the descriptor byte really is at offset 8 and really is LmqFsq.
    assert_eq!(bytes[8], CODEC_LMQ_FSQ);
}
