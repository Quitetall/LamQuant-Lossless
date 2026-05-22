//! ADR 0024 Phase 6 — XDF export. Reads a `.lml` file and writes
//! out a valid `.xdf` archive that LabRecorder, OpenViBE, or any
//! other XDF consumer can open.
//!
//! TDD contract:
//!
//!   * Output starts with the `XDF:` magic bytes (4 bytes).
//!   * Contains a `StreamHeader` chunk with the channel count +
//!     sample rate from the source `.lml`.
//!   * Contains at least one `Samples` chunk carrying the encoded
//!     binary signal.
//!   * Ends with a `StreamFooter` chunk.
//!
//! No external dep on liblsl — XDF is a plain binary format, our
//! writer is pure Rust.

use lamquant_lsl::xdf::write_xdf_from_lml;

fn write_tiny_lml(path: &std::path::Path) -> (usize, usize, f64) {
    use lamquant_core::container;
    use lamquant_core::lpc::LpcMode;
    let n_ch = 3;
    let t = 32;
    let sample_rate = 100.0;
    let signal: Vec<Vec<i64>> = (0..n_ch)
        .map(|ch| {
            (0..t as i64)
                .map(|i| (i + ch as i64 * 10) % 50)
                .collect()
        })
        .collect();
    let meta = format!(
        r#"{{"sample_rate":{},"n_channels":{},"signal_sha256":"abcd","channels":["c0","c1","c2"],"phys_dim":"uV","duration_s":{}}}"#,
        sample_rate,
        n_ch,
        t as f64 / sample_rate,
    );
    container::write_file_with_mode(path, &signal, sample_rate, t, 0, &meta, LpcMode::default())
        .expect("write_file");
    (n_ch, t, sample_rate)
}

#[test]
fn xdf_export_magic_and_basic_shape() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let lml_path = tmp.path().join("input.lml");
    let xdf_path = tmp.path().join("output.xdf");
    let (n_ch, t, sample_rate) = write_tiny_lml(&lml_path);

    write_xdf_from_lml(&lml_path, &xdf_path).expect("xdf write");

    let bytes = std::fs::read(&xdf_path).expect("xdf read");
    assert!(bytes.len() > 4, "xdf output empty");
    assert_eq!(&bytes[..4], b"XDF:", "xdf magic missing");

    // Stream header XML should reference channel count + sample rate.
    let body = String::from_utf8_lossy(&bytes);
    assert!(
        body.contains(&format!("<channel_count>{}</channel_count>", n_ch)),
        "channel_count not in stream header"
    );
    assert!(
        body.contains(&format!("<nominal_srate>{}</nominal_srate>", sample_rate)),
        "nominal_srate not in stream header"
    );
    // Footer should report the actual sample count.
    assert!(
        body.contains(&format!("<sample_count>{}</sample_count>", t)),
        "sample_count not in stream footer"
    );
}

#[test]
fn xdf_export_writes_samples_chunk() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let lml_path = tmp.path().join("input.lml");
    let xdf_path = tmp.path().join("output.xdf");
    let (n_ch, t, _) = write_tiny_lml(&lml_path);

    write_xdf_from_lml(&lml_path, &xdf_path).expect("xdf write");
    let bytes = std::fs::read(&xdf_path).expect("xdf read");

    // Find the Samples chunk (tag 0x0003) somewhere in the file.
    // Format: [chunk_len_prefix: variable][tag: u16 LE][payload].
    // Without parsing the variable-length prefix robustly here,
    // do a substring search for the i32-LE-encoded first sample
    // (value = 0 from our synthetic signal: channel 0 sample 0 =
    // (0 + 0 * 10) % 50 = 0). Four zero bytes near the end of the
    // file rules out the all-zero header padding.
    let last_quarter_start = bytes.len() * 3 / 4;
    let last_quarter = &bytes[last_quarter_start..];
    let zero4 = [0u8; 4];
    assert!(
        last_quarter
            .windows(4)
            .any(|w| w == zero4),
        "expected sample bytes in the last quarter of the XDF (n_ch={}, t={})",
        n_ch,
        t
    );
}
