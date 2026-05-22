//! ADR 0024 Phase 6.d — Multi-stream XDF: pack multiple `.lml`
//! sources into one `.xdf` file, each as its own LSL-shaped stream.
//!
//! LabRecorder + pyxdf surface each as a separate stream when
//! they read the file. Useful for archived sessions where several
//! amplifiers ran simultaneously.
//!
//! TDD contract:
//!
//!   * `write_xdf_multistream(&[lml_a, lml_b], xdf)` emits a
//!     single XDF file containing two distinct StreamHeader
//!     chunks (different stream_ids).
//!   * Each stream's Samples + StreamFooter chunk references the
//!     same stream_id as its StreamHeader.

use lamquant_lsl::xdf::{write_xdf_multistream, XdfOpts};

fn write_tiny_lml(path: &std::path::Path, sha_seed: &str, n_ch: usize) {
    use lamquant_core::container;
    use lamquant_core::lpc::LpcMode;
    let t = 8;
    let signal: Vec<Vec<i64>> = (0..n_ch)
        .map(|ch| (0..t as i64).map(|i| (i + ch as i64) % 30).collect())
        .collect();
    let channels_json: Vec<String> =
        (0..n_ch).map(|i| format!("\"ch{}\"", i)).collect();
    let meta = format!(
        r#"{{"sample_rate":100.0,"n_channels":{},"signal_sha256":"{}","channels":[{}],"phys_dim":"uV","duration_s":0.08}}"#,
        n_ch,
        sha_seed,
        channels_json.join(",")
    );
    container::write_file_with_mode(path, &signal, 100.0, t, 0, &meta, LpcMode::default())
        .expect("write");
}

#[test]
fn multistream_writes_distinct_streamheaders() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.lml");
    let b = tmp.path().join("b.lml");
    let xdf = tmp.path().join("combined.xdf");
    write_tiny_lml(&a, "aaa111", 2);
    write_tiny_lml(&b, "bbb222", 3);

    write_xdf_multistream(&[a.clone(), b.clone()], &xdf, XdfOpts::default())
        .expect("multistream");

    let bytes = std::fs::read(&xdf).unwrap();
    let body = String::from_utf8_lossy(&bytes);

    // Two unique source_ids should appear in the StreamHeader XML.
    let sources_count = body.matches("<source_id>lamquant:").count();
    assert_eq!(
        sources_count, 2,
        "expected 2 distinct StreamHeaders in multi-stream XDF; found {} `<source_id>` tags",
        sources_count
    );

    // Two distinct channel_count values must show up too:
    assert!(body.contains("<channel_count>2</channel_count>"));
    assert!(body.contains("<channel_count>3</channel_count>"));
}

#[test]
fn multistream_empty_input_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let xdf = tmp.path().join("out.xdf");
    let err = write_xdf_multistream(&[], &xdf, XdfOpts::default())
        .expect_err("should error on empty input");
    assert!(
        err.to_string().contains("at least one"),
        "error should mention requiring at least one input; got: {}",
        err
    );
}
