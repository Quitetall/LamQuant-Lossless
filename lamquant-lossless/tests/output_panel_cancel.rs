//! Output panel terminal-state projection contract.

#![cfg(target_os = "linux")]

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use lamquant_ops::{bounded_channel, spawn_advanced_command};
use lamquant_tui::panel::Panel;
use lamquant_tui::panels::output::OutputPanel;

fn with_current_history<T>(run: impl FnOnce() -> T) -> T {
    static HISTORY_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = HISTORY_ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let directory = tempfile::tempdir().expect("temporary history directory");
    let path = directory.path().join("history.json");
    std::fs::write(
        &path,
        r#"{
  "schema_version": "2.0",
  "parity_version": 2,
  "recent_operations": [],
  "recent_paths": {"inputs": [], "outputs": []},
  "interrupted": false,
  "last_op": null,
  "last_input": null,
  "last_output": null
}"#,
    )
    .expect("write current history");
    let previous = std::env::var_os("LAMQUANT_HISTORY");
    unsafe {
        std::env::set_var("LAMQUANT_HISTORY", &path);
    }
    let result = run();
    match previous {
        Some(value) => unsafe { std::env::set_var("LAMQUANT_HISTORY", value) },
        None => unsafe { std::env::remove_var("LAMQUANT_HISTORY") },
    }
    result
}

fn run_to_terminal(program: &str, arguments: Vec<String>, cancel: bool) -> OutputPanel {
    let (sink, receiver) = bounded_channel();
    let handle = with_current_history(|| {
        spawn_advanced_command("info".into(), program.into(), arguments, sink)
            .expect("compile supervising plan")
    });
    let mut panel = OutputPanel::new();
    panel.start("info".into(), receiver);
    if cancel {
        handle.kill();
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        while let Some(projection) = panel.try_recv_projection() {
            panel.consume(projection);
        }
        panel.tick();
        if panel.is_done() {
            return panel;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("expected executor-issued terminal projection");
}

#[test]
fn cancelled_failure_marks_panel_cancelled_not_failed() {
    let started = Instant::now();
    let panel = run_to_terminal("sleep", vec!["30".into()], true);

    assert!(panel.is_done());
    assert!(panel.is_cancelled());
    assert!(!panel.is_failed());
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "immediate cancellation waited for the containment fallback timeout"
    );
}

#[test]
fn real_failure_marks_panel_failed_not_cancelled() {
    let panel = run_to_terminal("false", vec![], false);

    assert!(panel.is_done());
    assert!(panel.is_failed());
    assert!(!panel.is_cancelled());
}

#[test]
fn receipt_marks_neither_failed_nor_cancelled() {
    let panel = run_to_terminal("true", vec![], false);

    assert!(panel.is_done());
    assert!(!panel.is_failed());
    assert!(!panel.is_cancelled());
}
