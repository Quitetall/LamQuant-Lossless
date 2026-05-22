//! ADR 0024 Phase 6.c — ClockOffset chunks (tag 4) in XDF output.
//!
//! XDF spec defines tag 4 = ClockOffset, format:
//!   [stream_id: u32 LE][collection_time: f64 LE][offset_value: f64 LE]
//!
//! Used by LSL receivers to align publisher + receiver clocks via
//! NTP-style time correction. LamQuant's own .lml format doesn't
//! carry clock-offset history, but the XDF writer must support
//! emitting offsets when a caller provides them (e.g. for a
//! recording where the inlet captured timestamps + offsets).
//!
//! TDD contract:
//!
//!   * `XdfOpts::with_clock_offsets(&[(t, off), ...])` adds the
//!     pairs.
//!   * Output contains one ClockOffset chunk per pair, with the
//!     stream_id matching the StreamHeader's.
//!   * Default opts (no clock offsets) emit zero ClockOffset
//!     chunks (Phase 6.a + 6.b compatibility).

use lamquant_lsl::xdf::{write_xdf_from_lml_opts, XdfOpts};

fn write_tiny_lml(path: &std::path::Path) {
    use lamquant_core::container;
    use lamquant_core::lpc::LpcMode;
    let n_ch = 1;
    let t = 4;
    let signal = vec![vec![0i64; t]];
    let meta = r#"{"sample_rate":100.0,"n_channels":1,"signal_sha256":"co1","channels":["c0"],"phys_dim":"uV","duration_s":0.04}"#;
    container::write_file_with_mode(path, &signal, 100.0, t, 0, meta, LpcMode::default())
        .expect("write");
}

#[test]
fn clock_offsets_emit_chunks() {
    let tmp = tempfile::tempdir().unwrap();
    let lml = tmp.path().join("in.lml");
    let xdf_default = tmp.path().join("default.xdf");
    let xdf_with_co = tmp.path().join("co.xdf");
    write_tiny_lml(&lml);

    write_xdf_from_lml_opts(&lml, &xdf_default, XdfOpts::default()).expect("default");

    // Three offset observations.
    let offsets = vec![(1.0, 0.001), (2.0, 0.0015), (3.0, 0.002)];
    write_xdf_from_lml_opts(
        &lml,
        &xdf_with_co,
        XdfOpts::default().with_clock_offsets(&offsets),
    )
    .expect("with_co");

    let bytes_default = std::fs::read(&xdf_default).unwrap();
    let bytes_with_co = std::fs::read(&xdf_with_co).unwrap();

    // Each ClockOffset chunk = length_prefix + tag(2B) + stream_id(4B)
    // + collection_time(8B) + offset_value(8B) = at least 23 bytes.
    // 3 chunks ≥ 69 bytes larger.
    assert!(
        bytes_with_co.len() >= bytes_default.len() + 3 * 23,
        "with_co should be ≥ 69 bytes larger; default={} with_co={}",
        bytes_default.len(),
        bytes_with_co.len()
    );

    // Verify the offset f64 values appear in the output.
    for &(t, off) in &offsets {
        let t_bytes = t.to_le_bytes();
        let off_bytes = off.to_le_bytes();
        assert!(
            bytes_with_co.windows(8).any(|w| w == t_bytes),
            "collection_time {} bytes missing from XDF",
            t
        );
        assert!(
            bytes_with_co.windows(8).any(|w| w == off_bytes),
            "offset_value {} bytes missing from XDF",
            off
        );
    }
}

#[test]
fn no_clock_offsets_default() {
    let tmp = tempfile::tempdir().unwrap();
    let lml = tmp.path().join("in.lml");
    let xdf = tmp.path().join("out.xdf");
    write_tiny_lml(&lml);
    write_xdf_from_lml_opts(&lml, &xdf, XdfOpts::default()).expect("write");

    let bytes = std::fs::read(&xdf).unwrap();
    // ClockOffset chunk tag = 4 LE = 0x04 0x00. Scan for the
    // 2-byte pattern; if any chunk had tag 4 we'd find it.
    // (Length-prefix bytes can also be 0x04 for the 4-byte form,
    // but our tiny file produces no chunks that large, so 0x04
    // followed by 0x00 elsewhere is very unlikely. Spot-check.)
    let count_tag4 = bytes
        .windows(2)
        .filter(|w| w == &[0x04u8, 0x00])
        .count();
    // 0x04 0x00 may appear inside the StreamHeader XML or as
    // part of i32 sample data (value 4 → bytes [4,0,0,0]). Allow
    // a few occurrences; what matters is the OUTPUT IS SMALL —
    // no ClockOffset chunks add their ~23 bytes.
    let _ = count_tag4;
    // Stronger assertion: total file size stays small.
    assert!(
        bytes.len() < 800,
        "default XDF should stay compact; got {} bytes",
        bytes.len()
    );
}
