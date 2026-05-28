//! Phase E.3 — `lamquant` launcher argv integration tests.
//!
//! Spawns the cargo-built `lamquant` binary as a subprocess and
//! checks the user-visible contracts:
//!
//!   - `--help` / `--version` exit 0 with expected output
//!   - `--gui` exits 127 + install hint when sibling binary missing
//!   - `--tui` would launch TUI (skipped — no TTY in test env;
//!     existence and exit-on-error path covered indirectly)
//!
//! Uses `env!("CARGO_BIN_EXE_lamquant")` which Cargo sets for
//! integration tests.

use std::process::{Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_lamquant");

#[test]
fn help_exits_zero_with_usage() {
    let out = Command::new(BIN)
        .arg("--help")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn lamquant --help");
    assert!(out.status.success(), "exit code {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("lamquant") && stdout.contains("USAGE"),
        "stdout missing usage banner: {}",
        stdout,
    );
}

#[test]
fn version_prints_pkg_version() {
    let out = Command::new(BIN)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn lamquant --version");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(env!("CARGO_PKG_VERSION")),
        "stdout missing version `{}`: {}",
        env!("CARGO_PKG_VERSION"),
        stdout,
    );
}

#[test]
fn h_short_flag_works() {
    // -h should equal --help (both route to Mode::Help).
    let out = Command::new(BIN)
        .arg("-h")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn lamquant -h");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("USAGE"), "stdout: {}", stdout);
}

#[test]
fn gui_without_sibling_emits_install_hint() {
    // CARGO_BIN_EXE_lamquant points at target/debug/lamquant. If
    // lamquant-gui was built earlier in the same workspace,
    // target/debug/lamquant-gui exists too — and exec_lamquant_gui's
    // current_exe()-relative resolve would happily find it,
    // bypassing the NotFound branch this test is meant to exercise.
    //
    // Copy the launcher to a clean tempdir so the sibling lookup is
    // guaranteed to miss.
    let tmp = tempfile::tempdir().expect("tempdir");
    let dest = tmp.path().join(if cfg!(windows) {
        "lamquant.exe"
    } else {
        "lamquant"
    });
    std::fs::copy(BIN, &dest).expect("copy binary");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&dest).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&dest, perms).unwrap();
    }

    let out = Command::new(&dest)
        .arg("--gui")
        // PATH cleared so the bare-name fallback can't find a
        // pre-installed lamquant-gui anywhere on the system.
        .env("PATH", "")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn isolated lamquant --gui");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(127),
        "expected exit 127 (NotFound branch); stderr: {}",
        stderr,
    );
    assert!(
        stderr.contains("lamquant-gui") && stderr.contains("install"),
        "stderr missing install hint: {}",
        stderr,
    );
}
