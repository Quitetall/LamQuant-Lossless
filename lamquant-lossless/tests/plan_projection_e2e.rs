//! End-to-end compiled-plan projection flow from codec process to shared state.

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use lamquant_ops::{bounded_channel, spawn_advanced_command, PlanUpdate};
use lamquant_tui::state::AppState;

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

#[test]
fn supervising_failure_receipt_drives_shared_state_terminal() {
    let (sink, receiver) = bounded_channel();
    let _handle = with_current_history(|| {
        spawn_advanced_command("info".into(), "false".into(), vec![], sink)
            .expect("compile supervising plan")
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut state = AppState::new();
    while std::time::Instant::now() < deadline {
        if let Ok(projection) = receiver.recv_timeout(Duration::from_millis(100)) {
            let terminal = matches!(projection.update, PlanUpdate::Failure { .. });
            state.apply_plan_projection(&projection);
            if terminal {
                assert_eq!(state.op_terminal_ok, Some(false));
                assert_eq!(state.op_progress, None);
                let log = state.op_log.join("\n");
                assert!(log.contains("planned:"), "log={log:?}");
                assert!(log.contains("error:"), "log={log:?}");
                return;
            }
        }
    }
    panic!("expected terminal failure projection");
}

#[test]
fn fresh_appstate_has_no_plan_run_state() {
    let state = AppState::new();
    assert!(state.op_log.is_empty());
    assert_eq!(state.op_progress, None);
    assert_eq!(state.op_terminal_ok, None);
}
