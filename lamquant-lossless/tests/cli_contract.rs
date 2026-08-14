#[cfg(feature = "archive")]
use lamquant_core::workflows::{VerificationOutcome, VerificationTarget};
#[cfg(feature = "archive")]
use lamquant_core::{lma, lma_forensic, workflows};
#[cfg(feature = "archive")]
use std::process::Command;
#[cfg(feature = "archive")]
mod common;

#[cfg(feature = "archive")]
use lamquant_lml_mcu::lpc::LpcMode;

#[cfg(feature = "archive")]
use tempfile::TempDir;

#[cfg(feature = "archive")]
fn write_container_fixture(dir: &TempDir) -> std::path::PathBuf {
    let signal = vec![vec![0_i64, 1, 2, 3, 4, 5, 6, 7, 8, 9]];
    let bytes = common::encode_uniform(&signal, 250.0, 256, "{}", LpcMode::default());
    let path = dir.path().join("valid.lml");
    std::fs::write(&path, &bytes).expect("write BCS2 container fixture");
    path
}

#[cfg(feature = "archive")]
fn write_archive_fixture(dir: &TempDir) -> std::path::PathBuf {
    let data_root = dir.path().join("pack");
    std::fs::create_dir(&data_root).expect("make archive source dir");
    std::fs::write(data_root.join("a.bin"), b"alpha").expect("write source file");
    std::fs::write(data_root.join("b.bin"), b"beta").expect("write source file");
    let archive_path = dir.path().join("valid.lma");
    lma::pack_archive(&data_root, &archive_path, 3, false, None).expect("pack LMA2");
    archive_path
}

#[cfg(feature = "archive")]
fn write_corrupted_archive(path: &std::path::Path) -> std::path::PathBuf {
    let mut bytes = std::fs::read(path).expect("read source archive fixture");
    assert!(bytes.len() > 64);
    let index = bytes.len() - 33;
    bytes[index] ^= 0x24;
    let out = path.with_file_name("corrupt.lma");
    std::fs::write(&out, &bytes).expect("write corrupted archive fixture");
    out
}

#[cfg(feature = "archive")]
#[test]
fn cli_convert_archive_writes_verified_capsule_without_touching_source() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let archive = write_archive_fixture(&tmp);
    let source_before = std::fs::read(&archive).expect("read source archive");
    let capsule = tmp.path().join("converted.bcs2");

    let output = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("convert-archive")
        .arg(&archive)
        .arg("--output")
        .arg(&capsule)
        .output()
        .expect("spawn archive converter");

    assert!(
        output.status.success(),
        "converter failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(lma_forensic::is_capsule(&capsule));
    assert_eq!(
        std::fs::read(&archive).expect("reread source archive"),
        source_before,
        "conversion must be non-destructive"
    );

    let verification = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("verify-capsule")
        .arg(&capsule)
        .output()
        .expect("spawn capsule verifier");
    assert!(
        verification.status.success(),
        "capsule verification failed: {}",
        String::from_utf8_lossy(&verification.stderr)
    );
    assert!(String::from_utf8_lossy(&verification.stdout).contains("2 files verified"));

    let restored = tmp.path().join("restored");
    std::fs::create_dir_all(&restored).expect("make empty restore destination");
    lma_forensic::unpack_capsule(&capsule, &restored, false).expect("restore capsule");
    assert_eq!(std::fs::read(restored.join("a.bin")).unwrap(), b"alpha");
    assert_eq!(std::fs::read(restored.join("b.bin")).unwrap(), b"beta");
}

#[cfg(feature = "archive")]
#[test]
fn cli_convert_archive_rejects_out_of_contract_zstd_level() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let archive = write_archive_fixture(&tmp);
    let capsule = tmp.path().join("converted.bcs2");

    let output = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("convert-archive")
        .arg(&archive)
        .arg("--output")
        .arg(&capsule)
        .arg("--zstd-level")
        .arg("0")
        .output()
        .expect("spawn archive converter");

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value '0'"));
    assert!(!capsule.exists());
}

#[cfg(feature = "archive")]
#[test]
fn cli_convert_archive_refuses_source_as_output() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let archive = write_archive_fixture(&tmp);
    let source_before = std::fs::read(&archive).expect("read source archive");

    let output = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("convert-archive")
        .arg(&archive)
        .arg("--output")
        .arg(&archive)
        .output()
        .expect("spawn archive converter");

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("convert-archive output must differ from input"));
    assert_eq!(std::fs::read(&archive).unwrap(), source_before);
}

#[cfg(feature = "archive")]
#[test]
fn cli_convert_archive_does_not_publish_output_after_source_corruption() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let archive = write_archive_fixture(&tmp);
    let corrupt = write_corrupted_archive(&archive);
    let capsule = tmp.path().join("converted.bcs2");

    let output = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("convert-archive")
        .arg(&corrupt)
        .arg("--output")
        .arg(&capsule)
        .output()
        .expect("spawn archive converter");

    assert!(!output.status.success());
    assert!(!capsule.exists(), "failed conversion published output");
}

#[cfg(feature = "archive")]
#[test]
fn cli_convert_archive_refuses_invalid_archive_custody_hash() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let archive = write_archive_fixture(&tmp);
    let mut bytes = std::fs::read(&archive).expect("read archive fixture");
    *bytes.last_mut().expect("archive has custody hash") ^= 0x01;
    let corrupt = tmp.path().join("bad-custody-hash.lma");
    std::fs::write(&corrupt, bytes).expect("write bad custody hash fixture");
    let capsule = tmp.path().join("converted.bcs2");

    let output = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("convert-archive")
        .arg(&corrupt)
        .arg("--output")
        .arg(&capsule)
        .output()
        .expect("spawn archive converter");

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Archive SHA-256 mismatch"));
    assert!(!capsule.exists(), "failed custody gate published output");
}

#[cfg(feature = "archive")]
fn normalize_verification_stdout(stdout: &str, archive: &std::path::Path) -> String {
    let replaced = stdout.replace(archive.to_string_lossy().as_ref(), "<ARCHIVE>");
    let mut normalized = String::new();
    for line in replaced.lines() {
        if line.starts_with("[2/5] Archive SHA-256:     OK  sha256:") {
            normalized.push_str("[2/5] Archive SHA-256:     OK  sha256:<HASH>");
        } else if line.starts_with("[1/5] Archive size:        ") {
            normalized.push_str("[1/5] Archive size:        <SIZE>");
        } else if line.starts_with("  2 files verified, 0 failed, ") {
            normalized.push_str("  2 files verified, 0 failed, <ELAPSED>");
        } else if line.starts_with("       Elapsed:            ") {
            normalized.push_str("       Elapsed:            <ELAPSED>");
        } else {
            normalized.push_str(line);
        }
        normalized.push('\n');
    }
    normalized
}

#[cfg(feature = "archive")]
#[test]
fn mixed_bcs2_and_lma2_result_without_process_capture() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let _lma_archive = write_archive_fixture(&tmp);
    let _lml_path = write_container_fixture(&tmp);

    let report = workflows::verify_path(tmp.path(), false).expect("verify mixed path");
    assert_eq!(report.items.len(), 2);

    let mut saw_bcs2 = false;
    let mut saw_lma2 = false;
    for item in report.items {
        match item.target() {
            VerificationTarget::Archive => {
                saw_lma2 = true;
                assert!(item.passed());
            }
            VerificationTarget::Container => {
                saw_bcs2 = true;
                assert!(item.passed());
            }
        }
    }

    assert!(saw_bcs2, "batch missed BCS2 container path");
    assert!(saw_lma2, "batch missed LMA archive path");
}

#[cfg(feature = "archive")]
#[test]
fn corruption_is_explicit_in_workflow_report() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let _good_archive = write_archive_fixture(&tmp);
    let _lml_path = write_container_fixture(&tmp);
    let _corrupted = write_corrupted_archive(&tmp.path().join("valid.lma"));

    let report = workflows::verify_path(tmp.path(), false).expect("mixed verify");
    assert_eq!(report.items.len(), 3);

    let failed = report.items.iter().find(|item| !item.passed());
    assert!(failed.is_some(), "expected failed item in report");

    let failed = failed.unwrap();
    let updates = failed.to_plan_updates();
    assert!(updates.iter().any(|update| matches!(
        update,
        lamquant_ops::PlanUpdate::Diagnostic {
            level: lamquant_ops::DiagnosticLevel::Error,
            message,
            ..
        } if message.contains("verification failed")
    )));
}

#[cfg(all(feature = "archive", feature = "tui"))]
#[test]
fn tui_plan_projection_from_workflows_can_update_appstate() {
    use lamquant_ops::PlanIdentity;
    use lamquant_tui::state::AppState;

    let tmp = tempfile::tempdir().expect("temp dir");
    let lma_archive = write_archive_fixture(&tmp);
    let _lml_path = write_container_fixture(&tmp);
    std::fs::rename(&lma_archive, tmp.path().join("input.lma")).expect("rename archive fixture");

    let report = workflows::verify_path(tmp.path(), false).expect("mixed verify");
    let expected_last = report
        .items
        .last()
        .expect("report item")
        .to_artifact_projection();
    let identity = PlanIdentity {
        graph_id: "11".repeat(32),
        plan_id: "22".repeat(32),
        invocation_id: "33".repeat(32),
    };

    let projections = report.to_plan_projections(&identity);
    let mut state = AppState::new();
    for projection in projections {
        match projection.update {
            lamquant_ops::PlanUpdate::Artifact { .. } => {}
            _ => panic!("expected workflow to emit artifact updates"),
        }
        projection.validate().expect("valid projection");
        state.apply_plan_projection(&projection);
    }
    assert_eq!(state.op_last_artifact.as_ref(), Some(&expected_last));
}

#[cfg(feature = "archive")]
#[test]
fn architecture_assert_cli_should_dispatch_to_workflows() {
    let cli = include_str!("../src/bin/lml.rs");
    assert!(
        cli.contains("workflows::verify_path") && cli.contains("workflows::inspect_path"),
        "expected CLI to dispatch verify/info through workflows; currently out of scope",
    );
    let verify_adapter = cli
        .split_once("fn cmd_verify(input:")
        .expect("verify adapter")
        .1
        .split_once("fn emit_verification_projections")
        .expect("end verify adapter")
        .0;
    assert!(!verify_adapter.contains("std::process::exit"));
    assert!(!verify_adapter.contains("WalkDir"));
    assert!(!verify_adapter.contains("File::open"));
    assert!(!verify_adapter.contains("container::"));
}

#[cfg(all(feature = "archive", feature = "tui"))]
#[test]
fn compact_explain_json_and_tui_share_verification_semantics() {
    use lamquant_ops::{PlanIdentity, PlanProjection, PlanUpdate};
    use lamquant_tui::state::AppState;

    let tmp = tempfile::tempdir().expect("temp dir");
    let archive = write_archive_fixture(&tmp);
    let report = workflows::verify_archive(&archive).expect("structured verification");
    let item = report.items.first().expect("verification item");
    let expected = item.to_artifact_projection();
    match &item.outcome {
        VerificationOutcome::Archive(_) => {}
        other => panic!("expected archive verification, got {other:?}"),
    }

    let compact = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("verify-archive")
        .arg(&archive)
        .output()
        .expect("spawn compact verifier");
    assert!(compact.status.success());
    let compact_stdout = String::from_utf8(compact.stdout).expect("compact stdout utf8");
    assert_eq!(
        normalize_verification_stdout(&compact_stdout, &archive),
        concat!(
            "Verifying <ARCHIVE>\n",
            "  Archive SHA-256... OK\n",
            "  Manifest: 2 files\n",
            "\n",
            "  2 files verified, 0 failed, <ELAPSED>\n",
            "  INTEGRITY OK — archive is valid.\n",
        )
    );

    let explained = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("verify-archive")
        .arg(&archive)
        .arg("--explain")
        .output()
        .expect("spawn explained verifier");
    assert!(explained.status.success());
    let explained_stdout = String::from_utf8(explained.stdout).expect("explain stdout utf8");
    assert_eq!(
        normalize_verification_stdout(&explained_stdout, &archive),
        concat!(
            "Verifying <ARCHIVE> (auditable readout)\n",
            "─────────────────────────────────────────────────────────\n",
            "[1/5] Archive size:        <SIZE>\n",
            "[2/5] Archive SHA-256:     OK  sha256:<HASH>\n",
            "[3/5] Manifest:            OK (2 entries enumerated)\n",
            "[4/5] Per-entry verify:\n",
            "       [1/2] ✓ a.bin                                   14 B  zstd    sha256:8ed3f6ad685b  CR 0.36x  (zstd OK)\n",
            "       [2/2] ✓ b.bin                                   13 B  zstd    sha256:f44e64e75f39  CR 0.31x  (zstd OK)\n",
            "[5/5] Summary:\n",
            "       Compressed total:   27 B (27 bytes)\n",
            "       Decompressed total: 9 B (9 bytes)\n",
            "       Archive CR:         0.33x\n",
            "       Verified:           2/2\n",
            "       Failed:             0/2\n",
            "       Elapsed:            <ELAPSED>\n",
            "─────────────────────────────────────────────────────────\n",
            "Result: PASS (archive-wide hash OK + all entries verified)\n",
        )
    );

    let json = Command::new(env!("CARGO_BIN_EXE_lml"))
        .env("LAMQUANT_GRAPH_ID", "11".repeat(32))
        .env("LAMQUANT_PLAN_ID", "22".repeat(32))
        .env("LAMQUANT_INVOCATION_ID", "33".repeat(32))
        .arg("--emit-plan-projections")
        .arg("verify-archive")
        .arg(&archive)
        .output()
        .expect("spawn JSON verifier");
    assert!(json.status.success());
    let json_stdout = String::from_utf8(json.stdout).expect("json stdout utf8");
    let observed = json_stdout
        .lines()
        .map(|line| PlanProjection::from_json_line(line).expect("valid JSON projection"))
        .find_map(|projection| match projection.update {
            PlanUpdate::Artifact { artifact, .. } => Some(artifact),
            _ => None,
        })
        .expect("verification artifact projection");
    assert_eq!(observed.path, expected.path);
    assert_eq!(observed.success, expected.success);
    assert_eq!(observed.compression_ratio, expected.compression_ratio);
    assert_eq!(observed.bytes_in, expected.bytes_in);
    assert_eq!(observed.bytes_out, expected.bytes_out);

    let identity = PlanIdentity {
        graph_id: "11".repeat(32),
        plan_id: "22".repeat(32),
        invocation_id: "33".repeat(32),
    };
    let mut state = AppState::new();
    state.apply_plan_projection(&item.to_plan_projection(&identity));
    assert_eq!(state.op_last_artifact.as_ref(), Some(&expected));
}

#[cfg(feature = "archive")]
#[test]
fn cli_info_dispatches_lma2_through_archive_inspection() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let archive = write_archive_fixture(&tmp);
    let output = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("info")
        .arg(&archive)
        .output()
        .expect("spawn lml info");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    let stderr = String::from_utf8(output.stderr).expect("stderr utf8");
    assert!(stdout.contains("valid.lma"));
    assert!(stdout.contains("a.bin"));
    assert!(stderr.contains("archive inspector"));
}

#[cfg(feature = "archive")]
#[test]
fn cli_info_outputs_bcs2_fields() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let lml_path = write_container_fixture(&tmp);
    let status = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("info")
        .arg(&lml_path)
        .output()
        .expect("spawn lml");

    assert!(status.status.success());
    let stdout = String::from_utf8(status.stdout).expect("stdout utf8");
    let stderr = String::from_utf8(status.stderr).expect("stderr utf8");

    let normalized = stdout.replace(lml_path.to_string_lossy().as_ref(), "<INPUT>");
    assert_eq!(
        normalized,
        "File:       <INPUT>\n\
Format:     BCS2 / bcs.lml.lossless.v1\n\
Channels:   1\n\
Windows:    1\n\
Samples:    10 (0.0s @ 250 Hz)\n\
Duration:   0s\n\
Window:     10 samples\n\
Size:       3.5 KB\n\
CR:         0.02:1  (80 B raw → 3.5 KB)\n"
    );
    assert_eq!(stderr, "");
}

#[cfg(feature = "archive")]
#[test]
fn cli_verify_smoke_reports_expected_count_and_bcs2_in_batch() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let _archive = write_archive_fixture(&tmp);
    let lml_path = write_container_fixture(&tmp);
    std::fs::rename(lml_path, tmp.path().join("batch.lml")).expect("rename bcs2");

    let output = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("verify")
        .arg(tmp.path())
        .output()
        .expect("spawn lml");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert!(stdout.contains("verified, 0 failed"));
    assert!(stdout.contains("OK"));
}

#[cfg(feature = "archive")]
#[test]
fn cli_verify_single_lma2_preserves_batch_summary() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let archive = write_archive_fixture(&tmp);
    let output = Command::new(env!("CARGO_BIN_EXE_lml"))
        .arg("verify")
        .arg(&archive)
        .output()
        .expect("spawn lml verify");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    let stderr = String::from_utf8(output.stderr).expect("stderr utf8");
    assert!(stdout.contains("OK  lma"));
    assert!(stdout.contains("1/1 verified, 0 failed"));
    assert_eq!(stderr, "");
}
