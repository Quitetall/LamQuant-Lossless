//! Integration test: spawn the `lml` binary with `--emit-json-events` and
//! parse its stdout against the canonical OpEvent schema.
//!
//! This is the contract gate for Phase 1: any front-end that consumes
//! `lml --emit-json-events` (Tauri GUI, Python TUI) gets the exact same
//! wire format the schema_parity test pins. If the binary's output ever
//! drifts (added a field, renamed a variant), this test fails.

use lamquant_ops::OpEvent;
use std::process::Command;

/// Locate the `lml` debug binary built by the workspace. Tests rely on
/// `cargo test` having built it via a workspace target dependency.
fn lml_path() -> std::path::PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // CARGO_MANIFEST_DIR is .../lamquant-core; the workspace target is
    // one level up.
    let target = manifest_dir
        .parent()
        .expect("workspace root")
        .join("target");
    let candidates = [
        target.join("debug").join("lml"),
        target.join("debug").join("lml.exe"),
        target.join("release").join("lml"),
        target.join("release").join("lml.exe"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    panic!(
        "could not find lml binary; run `cargo build --bin lml` first (looked at {:?})",
        candidates
    );
}

#[test]
fn emit_json_events_on_failed_info_yields_started_then_error() {
    let bin = lml_path();
    let out = Command::new(&bin)
        .arg("--emit-json-events")
        .arg("info")
        .arg("/definitely/does/not/exist.lml")
        .output()
        .expect("spawn lml");

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
    assert!(
        lines.len() >= 2,
        "expected ≥2 OpEvent lines, got {} (stdout={:?})",
        lines.len(),
        stdout
    );

    // First line MUST be Started.
    let first = OpEvent::from_json_line(lines[0])
        .unwrap_or_else(|e| panic!("parse first line: {} ({})", e, lines[0]));
    match first {
        OpEvent::Started { op_id, .. } => assert_eq!(op_id, "info"),
        other => panic!("expected Started, got {:?}", other),
    }

    // Last line MUST be terminal Error (since the file does not exist).
    let last_line = lines.last().unwrap();
    let last = OpEvent::from_json_line(last_line)
        .unwrap_or_else(|e| panic!("parse last line: {} ({})", e, last_line));
    match last {
        OpEvent::Error { message, .. } => {
            assert!(
                message.contains("info"),
                "error message should reference op id, got {:?}",
                message
            );
        }
        other => panic!("expected terminal Error, got {:?}", other),
    }

    // Process must exit non-zero on the error.
    assert!(
        !out.status.success(),
        "lml should exit non-zero on missing input"
    );
}

#[test]
fn emit_json_events_stdout_contains_only_json_lines() {
    let bin = lml_path();
    let out = Command::new(&bin)
        .arg("--emit-json-events")
        .arg("info")
        .arg("/definitely/does/not/exist.lml")
        .output()
        .expect("spawn lml");

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    for (i, line) in stdout.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        OpEvent::from_json_line(line).unwrap_or_else(|e| {
            panic!(
                "stdout line {} is not a valid OpEvent JSON line: {} ({:?})",
                i + 1,
                e,
                line
            )
        });
    }
    // Pretty error message lives on stderr.
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stderr.contains("Error"),
        "expected pretty error on stderr, got {:?}",
        stderr
    );
}
