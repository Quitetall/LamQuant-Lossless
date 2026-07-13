#![cfg(feature = "host")]

use lamquant_core::lpc::LpcMode;
use lamquant_core::tui::state::AppState;
use lamquant_core::workflows::{self, VerificationOutcome, VerificationTarget};
use lamquant_core::{container, lma};
use lamquant_ops::OpEvent;
use std::process::Command;

struct Fixture {
    _root: tempfile::TempDir,
    lml: std::path::PathBuf,
    archive: std::path::PathBuf,
}

fn fixture() -> Fixture {
    let root = tempfile::tempdir().unwrap();
    let lml = root.path().join("signal.lml");
    let signal = vec![
        (0..256).map(|value| value as i64 - 128).collect(),
        (0..256).map(|value| 128 - value as i64).collect(),
    ];
    let mut encoded = Vec::new();
    container::write_into(
        &mut encoded,
        &signal,
        250.0,
        128,
        0,
        "{}",
        LpcMode::default(),
    )
    .unwrap();
    std::fs::write(&lml, encoded).unwrap();

    let archive_input = root.path().join("archive-input");
    std::fs::create_dir(&archive_input).unwrap();
    std::fs::write(archive_input.join("notes.txt"), b"archive workflow").unwrap();
    let archive = root.path().join("bundle.lma");
    lma::pack_archive(&archive_input, &archive, 3, false, None).unwrap();
    Fixture {
        _root: root,
        lml,
        archive,
    }
}

fn lml_binary() -> std::path::PathBuf {
    let target = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target");
    [
        target.join("debug/lml"),
        target.join("debug/lml.exe"),
        target.join("release/lml"),
        target.join("release/lml.exe"),
    ]
    .into_iter()
    .find(|candidate| candidate.exists())
    .expect("lml binary must be built for CLI contract tests")
}

#[test]
fn mixed_codec_and_archive_workflow_returns_one_structured_report() {
    let fixture = fixture();
    let report = workflows::verify_path(fixture._root.path(), false).unwrap();

    assert_eq!(report.items.len(), 2);
    assert_eq!(report.passed(), 2);
    assert_eq!(report.failed(), 0);
    assert!(report.is_success());
    assert!(report.has_archives());
    assert!(report.items.iter().any(|item| {
        matches!(
            item.outcome,
            VerificationOutcome::Lml {
                channels: 2,
                samples: 256
            }
        )
    }));
    assert!(report.items.iter().any(|item| {
        item.target() == VerificationTarget::Lma
            && matches!(&item.outcome, VerificationOutcome::Lma(result) if result.passed())
    }));
}

#[test]
fn corruption_is_data_in_the_report_not_a_process_exit() {
    let fixture = fixture();
    let mut bytes = std::fs::read(&fixture.archive).unwrap();
    bytes[16] ^= 0x40;
    std::fs::write(&fixture.archive, bytes).unwrap();

    let report = workflows::verify_path(fixture._root.path(), false).unwrap();
    assert_eq!(report.items.len(), 2);
    assert_eq!(report.failed(), 1);
    assert!(!report.is_success());
    let archive = report.archive_item().unwrap();
    assert!(!archive.passed());
    assert!(matches!(archive.outcome, VerificationOutcome::Lma(_)));
}

#[test]
fn op_events_and_tui_reducer_consume_the_same_report() {
    let fixture = fixture();
    let report = workflows::verify_path(fixture._root.path(), false).unwrap();
    let events = report.op_events();

    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, OpEvent::FileDone { .. }))
            .count(),
        report.items.len()
    );
    let mut state = AppState::new();
    for event in &events {
        state.apply_op_event(event);
    }
    let log = state.op_log.join("\n");
    assert!(log.contains("signal.lml"));
    assert!(log.contains("bundle.lma"));
}

#[test]
fn cli_adapters_do_not_reimplement_the_workflow() {
    // This is an architecture contract, not an output test: it deliberately
    // fails when parsing or process policy migrates back into a CLI adapter.
    let cli = include_str!("../src/bin/lml.rs");
    let workflows = include_str!("../src/workflows.rs");
    assert!(!workflows.contains("b\"LMA1\""));
    assert!(!workflows.contains("b\"LMA2\""));
    assert!(!workflows.contains("SeekFrom"));

    let info_start = cli.find("fn cmd_info(input:").unwrap();
    let info_end = cli[info_start..]
        .find("fn render_container_inspection")
        .unwrap()
        + info_start;
    let info_adapter = &cli[info_start..info_end];
    assert!(info_adapter.contains("workflows::inspect_path(input)"));
    assert!(!info_adapter.contains("read_exact"));
    assert!(!info_adapter.contains("Bcs1Header"));

    let start = cli.find("fn cmd_verify(input:").unwrap();
    let end = cli[start..]
        .find("// ── Paranoid roundtrip verification")
        .unwrap()
        + start;
    let adapter = &cli[start..end];
    assert!(adapter.contains("workflows::verify_path(input, recursive)"));
    assert!(!adapter.contains("container::read_file"));
    assert!(!adapter.contains("WalkDir"));
    assert!(!adapter.contains("process::exit"));

    let archive_start = cli.find("fn cmd_verify_archive_explain").unwrap();
    let archive_end = cli[archive_start..].find("fn archive_method_name").unwrap() + archive_start;
    let archive_adapter = &cli[archive_start..archive_end];
    assert!(archive_adapter.contains("workflows::verify_archive(input)"));
    assert!(!archive_adapter.contains("lma::verify_archive"));
}

#[test]
fn compact_explain_json_and_tui_keep_cli_compatibility() {
    let fixture = fixture();
    let binary = lml_binary();

    let info = Command::new(&binary)
        .args(["info", fixture.lml.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(info.status.success());
    let info_stdout = String::from_utf8_lossy(&info.stdout);
    assert!(info_stdout.contains("Format:     BCS1 v1.0"));
    assert!(info_stdout.contains("Channels:   2"));
    assert!(info_stdout.contains("Seek table: 2 entries (LMLFOOT1 magic OK)"));

    let archive_info = Command::new(&binary)
        .args(["info", fixture.archive.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(archive_info.status.success());
    assert!(String::from_utf8_lossy(&archive_info.stderr).contains("archive inspector"));

    let unknown = fixture._root.path().join("unknown.bin");
    std::fs::write(&unknown, b"NOPE-not-a-container").unwrap();
    let unknown_info = Command::new(&binary)
        .args(["info", unknown.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!unknown_info.status.success());
    assert!(String::from_utf8_lossy(&unknown_info.stderr).contains(
        "Not LML or LMA (magic: [78, 79, 80, 69]). Expected leading bytes `LML1` or `BCS1` or `LMA1`."
    ));

    let codec = Command::new(&binary)
        .args(["verify", fixture.lml.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(codec.status.success());
    let codec_stdout = String::from_utf8_lossy(&codec.stdout);
    assert!(codec_stdout.contains("OK  2ch × 256"));
    assert!(codec_stdout.contains("1/1 verified, 0 failed"));

    // LMA2 historically uses the batch renderer through `lml verify`; retain
    // its extra per-file and aggregate lines while both formats share logic.
    let archive_via_verify = Command::new(&binary)
        .args(["verify", fixture.archive.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(archive_via_verify.status.success());
    let archive_verify_stdout = String::from_utf8_lossy(&archive_via_verify.stdout);
    assert!(archive_verify_stdout.contains("OK  lma"));
    assert!(archive_verify_stdout.contains("1/1 verified, 0 failed"));

    let compact = Command::new(&binary)
        .args(["verify-archive", fixture.archive.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(compact.status.success());
    let compact_stdout = String::from_utf8_lossy(&compact.stdout);
    assert!(compact_stdout.contains("Archive SHA-256... OK"));
    assert!(compact_stdout.contains("1 files verified, 0 failed"));
    assert!(compact_stdout.contains("INTEGRITY OK — archive is valid."));

    let explained = Command::new(&binary)
        .args([
            "verify-archive",
            fixture.archive.to_str().unwrap(),
            "--explain",
        ])
        .output()
        .unwrap();
    assert!(explained.status.success());
    let explained_stdout = String::from_utf8_lossy(&explained.stdout);
    assert!(explained_stdout.contains("[1/5] Archive size:"));
    assert!(explained_stdout.contains("Verified:           1/1"));
    assert!(explained_stdout.contains("Result: PASS"));

    let json = Command::new(&binary)
        .args([
            "--emit-json-events",
            "verify-archive",
            fixture.archive.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(json.status.success());
    let events = String::from_utf8_lossy(&json.stdout)
        .lines()
        .map(|line| OpEvent::from_json_line(line).unwrap())
        .collect::<Vec<_>>();
    assert!(events
        .iter()
        .any(|event| matches!(event, OpEvent::FileDone { success: true, .. })));
    assert!(matches!(events.last(), Some(OpEvent::Done { .. })));

    let mut state = AppState::new();
    for event in &events {
        state.apply_op_event(event);
    }
    assert_eq!(state.op_terminal_ok, Some(true));

    let mixed_json = Command::new(&binary)
        .args([
            "--emit-json-events",
            "verify",
            fixture._root.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(mixed_json.status.success());
    let mixed_events = String::from_utf8_lossy(&mixed_json.stdout)
        .lines()
        .map(|line| OpEvent::from_json_line(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        mixed_events
            .iter()
            .filter(|event| matches!(event, OpEvent::FileDone { .. }))
            .count(),
        2
    );
    assert!(matches!(mixed_events.last(), Some(OpEvent::Done { .. })));
}
