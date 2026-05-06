//! Cross-process parity: write a history.json from Python via the
//! `lamquant_codec.cli.menu` module, then read it from Rust via the
//! shared `lamquant-history` crate. Round-trip must preserve every field
//! covered by `specs/history-schema.json`.
//!
//! Skipped automatically if `python3` is not on PATH so this doesn't fail
//! Windows CI runners that intentionally exclude Python.

use lamquant_history::History;
use std::process::Command;

fn python_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn python_writer_round_trips_through_rust_reader() {
    if !python_available() {
        eprintln!("python3 not available — skipping cross-process test");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("history.json");

    // Manifest dir is .../crates/lamquant-history. Walk up two levels to
    // reach the workspace root where the Python package lives.
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");

    let script = format!(
        r#"
import os, sys
os.environ['LAMQUANT_HISTORY'] = r'{path}'
sys.path.insert(0, r'{root}')
from lamquant_codec.cli.menu import update_history, add_recent_path
update_history('encode', 'a.edf', 'ok')
update_history('decode', 'a.lml', 'partial')
add_recent_path('inputs', '/tmp/a.edf')
add_recent_path('outputs', '/tmp/a.lml')
"#,
        path = path.display(),
        root = workspace_root.display(),
    );

    let out = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .output()
        .expect("spawn python3");
    assert!(
        out.status.success(),
        "python writer failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let h = History::load_from(&path);
    assert_eq!(
        h.recent_operations.len(),
        2,
        "expected 2 ops, got {}",
        h.recent_operations.len()
    );
    // Newest first: decode then encode.
    assert_eq!(h.recent_operations[0].action, "decode");
    assert_eq!(h.recent_operations[0].result, "partial");
    assert_eq!(h.recent_operations[1].action, "encode");
    assert_eq!(h.recent_operations[1].result, "ok");

    assert!(h.recent_paths.inputs.iter().any(|p| p == "/tmp/a.edf"));
    assert!(h.recent_paths.outputs.iter().any(|p| p == "/tmp/a.lml"));
}
