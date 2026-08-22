//! Codec CLI plan-projection contract.
//!
//! These assert what the `lml` BINARY emits: that `--emit-plan-projections`
//! writes nothing but conforming JSON to stdout, that every line shares one
//! plan identity, and that the canonical operation ids agree across the Rust
//! registry, the JSON Schema and the UI parity spec.
//!
//! The vocabulary comes from `lamquant-plan`, which is in THIS repository.
//! It used to come from `lamquant_ops`, in the private meta-repository, which
//! made a public crate unbuildable for anyone outside the owning account
//! (ADR 0185, issue #120). `lamquant_ops` only ever re-exported these types
//! from `lamquant_plan` -- `crates/lamquant-ops/src/lib.rs:33` is a
//! `pub use lamquant_plan::{...}` -- so this is the same type, reached
//! directly instead of through a private hop.
//!
//! A fourth test lived here and has MOVED rather than been dropped:
//! `supervising_plan_owns_terminal_failure_receipt` exercised
//! `spawn_advanced_command` + `bounded_channel`, which are the launcher's
//! process runner and channel sink. Those deliberately stayed in
//! `lamquant-ops` when the vocabulary moved out, because they spawn processes
//! and open sockets. A test of the launcher supervising a plan is not a test
//! of this codec, and it now lives beside the code it exercises at
//! `crates/lamquant-ops/tests/supervising_plan_receipt.rs`.

use std::process::Command;

use lamquant_plan::{DiagnosticLevel, PlanProjection, PlanUpdate};

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
fn canonical_operation_ids_match_rust_schema_and_ui_spec() {
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../../specs/plan-projections.schema.json"))
            .expect("plan projection schema");
    let schema_ids: Vec<&str> = schema["properties"]["operation"]["enum"]
        .as_array()
        .expect("operation enum")
        .iter()
        .map(|value| value.as_str().expect("operation id string"))
        .collect();

    let ui = include_str!("../../specs/ui-parity.md");
    let marker = ui
        .split_once("<!-- canonical-operation-ids:start -->")
        .expect("operation marker start")
        .1
        .split_once("<!-- canonical-operation-ids:end -->")
        .expect("operation marker end")
        .0;
    let ui_ids: Vec<&str> = marker
        .lines()
        .filter_map(|line| line.trim().strip_prefix("- `")?.strip_suffix('`'))
        .collect();

    assert_eq!(lamquant_plan::canonical_operation_ids(), schema_ids);
    assert_eq!(schema_ids, ui_ids);
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
