//! CLI binary smoke tests — make sure each binary's argv parser
//! produces useful help output + sane usage errors. Doesn't
//! exercise live LSL I/O (that's covered by
//! `outlet_inlet_roundtrip.rs`); just verifies the CLI surface
//! is wired correctly.
//!
//! Test driven development: this file defined the CLI contract
//! BEFORE the binaries were implemented. Every assertion below
//! reflects user-facing behaviour the binaries are required to
//! provide.

#![cfg(feature = "liblsl")]

use std::process::Command;

fn binary_path(name: &str) -> std::path::PathBuf {
    // CARGO env var points at the cargo binary; sibling
    // `target/debug` (or `target/release`) holds our bins.
    let target_dir = env!("CARGO_MANIFEST_DIR")
        .parse::<std::path::PathBuf>()
        .unwrap()
        .ancestors()
        .nth(2)
        .unwrap()
        .join("target")
        .join("debug");
    target_dir.join(name)
}

fn run_with_args(bin: &str, args: &[&str]) -> std::process::Output {
    let path = binary_path(bin);
    assert!(path.exists(), "binary {} not built — run `cargo build -p lamquant-lsl --features liblsl --bins` first", path.display());
    Command::new(path).args(args).output().expect("spawn")
}

#[test]
fn lml_stream_no_args_exits_with_usage() {
    let out = run_with_args("lml-stream", &[]);
    assert!(!out.status.success(), "missing path should fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("usage"), "stderr should show usage; got: {}", stderr);
}

#[test]
fn lml_stream_help_flag() {
    let out = run_with_args("lml-stream", &["--help"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let combined = format!("{}{}", stderr, stdout);
    assert!(
        combined.contains("usage") && combined.contains("--rate"),
        "help should mention usage + --rate flag; got: {}",
        combined
    );
}

#[test]
fn lml_record_no_args_exits_with_usage() {
    let out = run_with_args("lml-record", &[]);
    assert!(!out.status.success(), "missing args should fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("usage"),
        "stderr should show usage; got: {}",
        stderr
    );
}

#[test]
fn lml_discover_runs_with_short_timeout() {
    // Should run and exit cleanly even when no LSL streams are
    // discoverable on the network (it should just emit a header
    // line + an empty list).
    let out = run_with_args("lml-discover", &["--timeout", "0.5"]);
    assert!(out.status.success(), "lml-discover should exit 0 even with empty network; stderr: {}", String::from_utf8_lossy(&out.stderr));
}
