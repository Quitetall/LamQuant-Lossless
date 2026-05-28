//! Phase E.1 / C.3c — end-to-end OpEvent integration test.
//!
//! Spawns the `lml` binary with `--emit-json-events`, parses its
//! stdout into `OpEvent`s, and feeds each through
//! `AppState::apply_op_event`. Asserts that the C.3c reducer drives
//! `op_log`, `op_progress`, and `op_terminal_ok` to the right
//! terminal values.
//!
//! Uses `info /nonexistent` so the test runs fast and doesn't need
//! a fixture EDF — the resulting Started → Error pair exercises the
//! reducer's clear/normalize logic without disk dependencies.

use lamquant_core::tui::state::AppState;
use lamquant_ops::OpEvent;
use std::process::Command;

fn lml_path() -> std::path::PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
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
    panic!("lml binary not built; run `cargo build --bin lml` first");
}

#[test]
fn op_event_stream_drives_appstate_reducer_to_failed_terminal() {
    let bin = lml_path();
    let out = Command::new(&bin)
        .arg("--emit-json-events")
        .arg("info")
        .arg("/definitely/does/not/exist.lml")
        .output()
        .expect("spawn lml");

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let events: Vec<OpEvent> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| OpEvent::from_json_line(l).ok())
        .collect();

    assert!(
        events.len() >= 2,
        "expected ≥2 events, got {}: {:?}",
        events.len(),
        stdout,
    );

    // Drive the reducer with the live event stream.
    let mut state = AppState::new();
    for ev in &events {
        state.apply_op_event(ev);
    }

    // After Error event the reducer must have:
    assert_eq!(
        state.op_terminal_ok,
        Some(false),
        "Error event should mark op_terminal_ok = Some(false)",
    );
    assert_eq!(
        state.op_progress, None,
        "terminal Error should clear op_progress",
    );
    assert!(
        !state.op_log.is_empty(),
        "op_log should contain at least Started + Error lines",
    );
    let joined = state.op_log.join("\n");
    assert!(
        joined.contains("started: info"),
        "op_log missing started line: {:?}",
        state.op_log,
    );
    assert!(
        joined.contains("error:"),
        "op_log missing error line: {:?}",
        state.op_log,
    );
}

#[test]
fn fresh_appstate_has_no_op_state() {
    // Sanity check the default — without dispatching any OpEvent the
    // reducer fields are empty / None. Catches a future regression
    // where AppState::new() accidentally seeds op state.
    let s = AppState::new();
    assert!(s.op_log.is_empty());
    assert_eq!(s.op_progress, None);
    assert_eq!(s.op_terminal_ok, None);
}
