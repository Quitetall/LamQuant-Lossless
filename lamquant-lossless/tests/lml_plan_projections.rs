//! Codec CLI plan-projection contract.

use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use lamquant_ops::{
    bounded_channel, spawn_advanced_command, DiagnosticLevel, PlanProjection, PlanUpdate,
};

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

fn lml_path() -> std::path::PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
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
    candidates
        .iter()
        .find(|candidate| candidate.exists())
        .cloned()
        .unwrap_or_else(|| panic!("lml binary not built; looked at {candidates:?}"))
}

fn direct_command() -> Command {
    let mut command = Command::new(lml_path());
    command
        .env_remove("LAMQUANT_HISTORY")
        .env("LAMQUANT_GRAPH_ID", "11".repeat(32))
        .env("LAMQUANT_PLAN_ID", "22".repeat(32))
        .env("LAMQUANT_INVOCATION_ID", "33".repeat(32))
        .arg("--emit-plan-projections")
        .arg("info")
        .arg("/definitely/does/not/exist.lml");
    command
}

#[test]
fn direct_codec_stream_is_identity_bound_observations() {
    let output = direct_command().output().expect("spawn lml");
    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");
    let projections: Vec<PlanProjection> = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| PlanProjection::from_json_line(line).expect("valid projection"))
        .collect();
    assert!(projections.len() >= 2, "stdout={stdout:?}");
    assert!(matches!(
        &projections[0].update,
        PlanUpdate::Planned { operation, .. } if operation == "info"
    ));
    let identity = &projections[0].plan;
    assert!(projections
        .iter()
        .all(|projection| &projection.plan == identity));
    assert!(matches!(
        &projections.last().expect("terminal diagnostic").update,
        PlanUpdate::Diagnostic {
            level: DiagnosticLevel::Error,
            ..
        }
    ));
}

#[test]
fn supervising_plan_owns_terminal_failure_receipt() {
    let (sink, receiver) = bounded_channel();
    let _handle = with_current_history(|| {
        spawn_advanced_command("info".into(), "false".into(), vec![], sink)
            .expect("compile supervising plan")
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if let Ok(projection) = receiver.recv_timeout(Duration::from_millis(100)) {
            if let PlanUpdate::Failure {
                receipt, cancelled, ..
            } = &projection.update
            {
                assert!(!cancelled);
                assert_eq!(receipt.graph_id, projection.plan.graph_id);
                assert_eq!(receipt.plan_id, projection.plan.plan_id);
                assert_eq!(receipt.invocation_id, projection.plan.invocation_id);
                projection.validate().expect("valid terminal receipt");
                return;
            }
        }
    }
    panic!("expected supervising terminal failure receipt");
}

#[test]
fn direct_projection_stdout_contains_only_json() {
    let output = direct_command().output().expect("spawn lml");
    for (index, line) in String::from_utf8_lossy(&output.stdout).lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        PlanProjection::from_json_line(line)
            .unwrap_or_else(|error| panic!("line {}: {error}: {line:?}", index + 1));
    }
    assert!(String::from_utf8_lossy(&output.stderr).contains("Error"));
}
