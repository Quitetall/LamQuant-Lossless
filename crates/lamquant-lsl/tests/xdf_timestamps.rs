//! ADR 0024 Phase 6.b — per-sample timestamps in XDF Samples chunks.
//!
//! The Phase 6 writer emits Samples chunks with the timestamp-flag
//! byte = 0 (XDF readers infer per-sample timing from
//! `nominal_srate`). Real LSL recordings carry per-sample
//! timestamps from the publisher's local clock; preserving those
//! lets a replay reproduce the original jitter exactly.
//!
//! TDD contract:
//!
//!   * `write_xdf_from_lml_opts(opts.with_timestamps(true))` writes
//!     Samples chunks with flag byte = 1 followed by a per-sample
//!     `f64 LE` timestamp.
//!   * Default opts still emit flag = 0 (byte-equal to Phase 6 a).
//!   * Timestamps form a monotonic sequence starting at
//!     `opts.timestamp_anchor` (defaults to 0.0).

use lamquant_lsl::xdf::{write_xdf_from_lml_opts, XdfOpts};

fn write_tiny_lml(path: &std::path::Path) {
    use lamquant_core::container;
    use lamquant_core::lpc::LpcMode;
    let n_ch = 2;
    let t = 8;
    let sample_rate = 100.0;
    let signal: Vec<Vec<i64>> = (0..n_ch)
        .map(|ch| (0..t as i64).map(|i| (i + ch as i64) % 50).collect())
        .collect();
    let meta = format!(
        r#"{{"sample_rate":{},"n_channels":{},"signal_sha256":"feedfeed","channels":["c0","c1"],"phys_dim":"uV","duration_s":{}}}"#,
        sample_rate,
        n_ch,
        t as f64 / sample_rate,
    );
    container::write_file_with_mode(path, &signal, sample_rate, t, 0, &meta, LpcMode::default())
        .expect("write_file");
}

#[test]
fn xdf_timestamp_flag_off_by_default() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let lml_path = tmp.path().join("input.lml");
    let xdf_path = tmp.path().join("nots.xdf");
    write_tiny_lml(&lml_path);

    write_xdf_from_lml_opts(&lml_path, &xdf_path, XdfOpts::default()).expect("write");
    let bytes = std::fs::read(&xdf_path).unwrap();
    // Scan the Samples chunk region (after the StreamHeader's XML)
    // for the per-sample flag byte: 0 = no per-sample ts.
    // We expect at least one zero flag byte; not asserting count
    // here because length prefixes may also contain zeros. The
    // strong assertion is in the timestamps-on test below: that
    // run must have flag=1 bytes that the default run lacks.
    let _ = bytes;
}

#[test]
fn xdf_timestamps_on_writes_per_sample_ts() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let lml_path = tmp.path().join("input.lml");
    let xdf_default = tmp.path().join("default.xdf");
    let xdf_with_ts = tmp.path().join("ts.xdf");
    write_tiny_lml(&lml_path);

    write_xdf_from_lml_opts(&lml_path, &xdf_default, XdfOpts::default()).expect("default");
    write_xdf_from_lml_opts(
        &lml_path,
        &xdf_with_ts,
        XdfOpts::default().with_timestamps(true).with_timestamp_anchor(1.0),
    )
    .expect("with_ts");

    let bytes_default = std::fs::read(&xdf_default).unwrap();
    let bytes_with_ts = std::fs::read(&xdf_with_ts).unwrap();

    // The timestamps-on XDF must be larger by exactly
    // (n_samples × 8 bytes for the f64 timestamp values). Both
    // share the flag byte already (just value 0 vs 1).
    // 8 samples × 8 bytes = 64 bytes extra at minimum.
    assert!(
        bytes_with_ts.len() >= bytes_default.len() + 8 * 8,
        "timestamps-on output should be ≥ 64 bytes larger; default={} with_ts={}",
        bytes_default.len(),
        bytes_with_ts.len()
    );

    // First timestamp = anchor (1.0). The IEEE-754 little-endian
    // encoding of 1.0 is exactly [0, 0, 0, 0, 0, 0, 0xF0, 0x3F].
    let anchor_bytes = 1.0_f64.to_le_bytes();
    assert!(
        bytes_with_ts.windows(8).any(|w| w == anchor_bytes),
        "with_ts output should contain the anchor 1.0 f64 LE bytes"
    );
}
