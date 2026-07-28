//! A hung helper must not hang the caller.
//!
//! `PyBackend` drives inference over a subprocess, and until now a helper that
//! stalled stalled its caller with it, forever. The comment in `py_backend.rs`
//! said as much and called a watchdog a follow-up "before any unattended use".
//!
//! There are two distinct ways the exchange can hang, and a test that only
//! covers one of them would leave the other in place looking fixed:
//!
//!   * the helper never exits — the classic case, and the one a naive watchdog
//!     around `wait` would catch;
//!   * the helper never READS its stdin. A pipe holds 64 KiB on Linux and a
//!     real request is far larger, so `write_all` blocks once the buffer fills
//!     — before any wait is reached. A timeout guarding only the wait never
//!     fires here.
//!
//! Both are exercised below against real Python processes, because both are
//! properties of pipes and process lifetimes that a mock would define away.

#![cfg(feature = "python")]

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use lamquant_lmq::backend::NeuralBackend;
use lamquant_lmq::py_backend::PyBackend;
use semantic_abir::ContentId;
use semantic_abir_bcs::{ModelProvenance, PccpStatus};

const TIMEOUT: Duration = Duration::from_millis(700);
/// Generous headroom over TIMEOUT: this asserts the call returns promptly rather
/// than never, so it must not fail merely because a loaded machine was slow.
const PATIENCE: Duration = Duration::from_secs(20);

fn python3_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn model() -> ModelProvenance {
    ModelProvenance {
        checkpoint_content_id: ContentId::from_bytes([7; 32]),
        checkpoint_sha256: [8; 32],
        pccp_change_id: "LMQ-TIMEOUT-TEST".to_owned(),
        pccp_evidence_id: ContentId::from_bytes([9; 32]),
        pccp_status: PccpStatus::Candidate,
    }
}

/// Write a throwaway helper script and return its path.
fn helper_script(name: &str, body: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("lmq_timeout_{name}_{}.py", std::process::id()));
    let mut file = fs::File::create(&path).expect("create helper script");
    file.write_all(body.as_bytes()).expect("write helper script");
    path
}

fn signal() -> Vec<Vec<i64>> {
    // Comfortably past a 64 KiB pipe buffer once serialised as JSON, which is
    // the whole point of the never-reads case.
    (0..21)
        .map(|c| (0..2500).map(|i| ((i + c) % 200) as i64 - 100).collect())
        .collect()
}

#[test]
fn a_helper_that_never_exits_is_killed_at_the_deadline() {
    if !python3_available() {
        eprintln!("SKIP: python3 not available");
        return;
    }
    // Reads the whole request (so the write completes), then never exits.
    let script = helper_script(
        "never_exits",
        "import sys, time\nsys.stdin.buffer.read()\ntime.sleep(3600)\n",
    );
    let backend = PyBackend::model("python3", script.clone(), model()).with_timeout(TIMEOUT);

    let started = Instant::now();
    let outcome = backend.encode(&signal(), 250.0);
    let elapsed = started.elapsed();
    let _ = fs::remove_file(&script);

    let error = outcome.expect_err("a helper that never exits must not succeed");
    assert!(
        error.0.contains("exceeded"),
        "the failure must name the deadline, not some downstream symptom: {}",
        error.0
    );
    assert!(
        elapsed < PATIENCE,
        "returned only after {elapsed:?}; the deadline is meant to bound this"
    );
    // Lower bound too: returning instantly would mean some unrelated failure
    // produced the error and the deadline was never the thing that fired.
    assert!(
        elapsed >= TIMEOUT,
        "returned after {elapsed:?}, sooner than the {TIMEOUT:?} deadline, so \
         something other than the timeout ended this call"
    );
}

#[test]
fn a_helper_that_never_reads_stdin_is_killed_at_the_deadline() {
    if !python3_available() {
        eprintln!("SKIP: python3 not available");
        return;
    }
    // Never touches stdin, so the pipe fills and the WRITE is what blocks.
    // Before the rework this deadlocked with the caller stuck in `write_all`,
    // never reaching any wait, so no watchdog on the wait could have helped.
    let script = helper_script("never_reads", "import time\ntime.sleep(3600)\n");
    let backend = PyBackend::model("python3", script.clone(), model()).with_timeout(TIMEOUT);

    let started = Instant::now();
    let outcome = backend.encode(&signal(), 250.0);
    let elapsed = started.elapsed();
    let _ = fs::remove_file(&script);

    let error = outcome.expect_err("a helper that never reads must not succeed");
    assert!(
        error.0.contains("exceeded"),
        "the failure must name the deadline: {}",
        error.0
    );
    assert!(
        elapsed < PATIENCE,
        "returned only after {elapsed:?}; the write path is inside the deadline too"
    );
    assert!(
        elapsed >= TIMEOUT,
        "returned after {elapsed:?}, sooner than the {TIMEOUT:?} deadline, so \
         something other than the timeout ended this call"
    );
}

#[test]
fn a_helper_that_fails_on_import_still_reports_its_own_traceback() {
    if !python3_available() {
        eprintln!("SKIP: python3 not available");
        return;
    }
    // Regression guard for the diagnosis, not the deadline. This helper dies
    // instantly without reading stdin, so our write fails with a broken pipe.
    // Reporting that would bury the traceback that says what actually happened
    // — and it is exactly how a missing `lamquant_neural` presents.
    let script = helper_script(
        "import_error",
        "raise ModuleNotFoundError(\"No module named 'lamquant_neural'\")\n",
    );
    let backend = PyBackend::model("python3", script.clone(), model()).with_timeout(TIMEOUT);

    let outcome = backend.encode(&signal(), 250.0);
    let _ = fs::remove_file(&script);

    let error = outcome.expect_err("a helper that raises must not succeed");
    assert!(
        error.0.contains("lamquant_neural"),
        "the helper's own diagnosis must survive, not be replaced by a pipe error: {}",
        error.0
    );
    assert!(
        !error.0.contains("write request"),
        "a broken pipe is a consequence of the helper dying, not the reason: {}",
        error.0
    );
}
