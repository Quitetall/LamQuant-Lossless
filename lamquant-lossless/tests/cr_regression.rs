//! CR-regression gate for codec adaptivity work (ADR 0023 Track B).
//!
//! Reads `tests/cr_baselines.json`. For each entry in `regression_set`,
//! shells out to the installed `lml` binary, encodes the source EDF to a
//! bare LML, and asserts the output size is ≤ baseline_compressed_size_bytes.
//!
//! Behaviour when the binary or source EDF isn't available:
//!   - `lml` not on PATH                 → eprintln + return (test passes).
//!   - source EDF path doesn't exist     → eprintln + skip that entry.
//!   - ALL entries skipped               → eprintln + return (test passes).
//!   - At least one ran and any regressed → panic with a clear diff.
//!
//! Rationale: CI nodes that don't ship the reference dataset shouldn't fail
//! the gate; local-dev workstations that DO have the EDFs catch regressions
//! before they ship. Bonn stretch-target is intentionally not gated here —
//! see ADR 0023 + cr_baselines.json `stretch_targets`.

use serde::Deserialize;
use std::path::Path;
use std::process::Command;

#[derive(Deserialize)]
struct Baselines {
    regression_set: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    name: String,
    src_path: String,
    baseline_compressed_size_bytes: u64,
}

#[test]
fn no_regression_on_reference_edfs() {
    let json_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cr_baselines.json");
    let raw = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("read {}: {}", json_path.display(), e));
    let baselines: Baselines =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse cr_baselines.json: {}", e));

    // Locate the lml binary. Prefer the installed CLI for parity with how
    // baselines were measured. Fall back to skipping the test if absent.
    let lml = if let Ok(p) = std::env::var("LML_BIN") {
        p
    } else if Path::new("/home/brianklam/.cargo/bin/lml").exists() {
        "/home/brianklam/.cargo/bin/lml".to_string()
    } else if Command::new("which")
        .arg("lml")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .is_some()
    {
        "lml".to_string()
    } else {
        eprintln!("[cr_regression] no `lml` binary on PATH; skipping");
        return;
    };

    let tmp = tempfile::Builder::new()
        .prefix("cr_regression_")
        .tempdir()
        .expect("tempdir");

    let mut ran = 0;
    let mut regressions = Vec::new();

    for entry in &baselines.regression_set {
        let src = Path::new(&entry.src_path);
        if !src.exists() {
            eprintln!(
                "[cr_regression] {}: source missing at {}; skipping",
                entry.name, entry.src_path
            );
            continue;
        }
        let out = tmp.path().join(format!("{}.lml", entry.name));
        let status = Command::new(&lml)
            .args(["encode", "--bare-lml", "--i-understand-data-loss", "-o"])
            .arg(&out)
            .arg(src)
            .output()
            .unwrap_or_else(|e| panic!("spawn lml encode: {}", e));
        if !status.status.success() {
            panic!(
                "[cr_regression] {}: lml encode failed (exit {:?})\nstdout: {}\nstderr: {}",
                entry.name,
                status.status.code(),
                String::from_utf8_lossy(&status.stdout),
                String::from_utf8_lossy(&status.stderr),
            );
        }
        let got = std::fs::metadata(&out)
            .unwrap_or_else(|e| panic!("stat {}: {}", out.display(), e))
            .len();
        ran += 1;
        let baseline = entry.baseline_compressed_size_bytes;
        if got > baseline {
            regressions.push(format!(
                "  {:35} baseline={:>12}  got={:>12}  Δ=+{} ({:+.3}%)",
                entry.name,
                baseline,
                got,
                got - baseline,
                100.0 * (got as f64 - baseline as f64) / baseline as f64
            ));
        } else {
            eprintln!(
                "[cr_regression] {:35} baseline={:>12} got={:>12} Δ={:>+12} ({:+.3}%)",
                entry.name,
                baseline,
                got,
                got as i64 - baseline as i64,
                100.0 * (got as f64 - baseline as f64) / baseline as f64
            );
        }
    }

    if ran == 0 {
        eprintln!("[cr_regression] no reference EDFs available; skipping (not a failure)");
        return;
    }

    if !regressions.is_empty() {
        panic!(
            "CR regression on {} of {} reference EDFs:\n{}\n\nIf this is intentional, update \
             tests/cr_baselines.json in the same commit and justify in the message.",
            regressions.len(),
            ran,
            regressions.join("\n")
        );
    }
}
