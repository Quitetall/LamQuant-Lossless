//! lamquant — top-level launcher binary.
//!
//! Entry point users invoke after `cargo install --path lamquant-core --bin
//! lamquant` (or downloading the release artifact). This binary is the
//! single front-door for both the TUI and the GUI:
//!
//!   lamquant                    smart-detect (today: TUI; A.4 adds GUI)
//!   lamquant --tui              force TUI
//!   lamquant --gui              force GUI (A.3 wires real launch path)
//!   lamquant <subcommand> ...   pass through to `lml <subcommand>`
//!   lamquant --help             this launcher's help
//!   lamquant --version          version
//!
//! Subcommand passthrough execs the `lml` binary so users keep using one
//! verb (`lamquant`) for the codec stack instead of remembering two binary
//! names. Requires `lml` on PATH; `cargo install --path lamquant-core`
//! installs both binaries side-by-side, so the common case Just Works.
//!
//! Phase A.1 lands argv parsing + passthrough. Phase A.3 wires the GUI
//! Cargo feature so `--gui` actually launches the Tauri runtime; until
//! then `--gui` prints a helpful build-instruction stub.

#[cfg(not(feature = "host"))]
fn main() {
    eprintln!("lamquant: built without host feature — recompile with `--features host`");
    std::process::exit(1);
}

#[cfg(feature = "host")]
fn main() {
    use std::process::exit;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mode = if argv.is_empty() {
        smart_detect_mode()
    } else {
        parse_mode(&argv)
    };
    let code = match mode {
        Mode::Tui => lamquant_core::tui::run_interactive(),
        Mode::Gui(extra) => exec_lamquant_gui(&extra),
        Mode::Help => {
            print_usage();
            0
        }
        Mode::Version => {
            println!("lamquant {}", env!("CARGO_PKG_VERSION"));
            0
        }
        Mode::Passthrough(args) => exec_lml(&args),
    };
    exit(code);
}

#[cfg(feature = "host")]
enum Mode {
    /// Launch the interactive TUI.
    Tui,
    /// Launch the desktop GUI by exec-ing the sibling `lamquant-gui`
    /// binary. We can't link against it as a library (gui/src-tauri
    /// already depends on lamquant-core, and reversing creates a Cargo
    /// cycle), but the cargo-installed sibling is just as good once
    /// it's installed. Carries any extra argv (after `--gui`) through
    /// to the GUI binary so e.g. `lamquant --gui --debug` works once
    /// the GUI grows arg support.
    Gui(Vec<String>),
    Help,
    Version,
    /// Forward all argv to `lml`. The launcher's argv is unchanged for
    /// `lml` so flag handling lands in clap there, not here.
    Passthrough(Vec<String>),
}

/// Smart-detect for no-arg invocation. Env var > persisted cfg > default
/// TUI. The "default TUI" choice is intentional even on systems with a
/// display: GUI is opt-in until A.4 ships the first-run prompt that
/// upgrades users who want it. Avoids breaking flows where users already
/// type `lamquant` expecting the TUI.
#[cfg(feature = "host")]
fn smart_detect_mode() -> Mode {
    // 1. Env override — highest priority. Useful for CI / scripts /
    //    "I want the GUI today, but don't change my saved preference".
    //    Trim + lowercase before matching so " GUI " behaves like "gui".
    if let Ok(v) = std::env::var("LAMQUANT_UI") {
        match v.trim().to_ascii_lowercase().as_str() {
            "tui" => return Mode::Tui,
            "gui" => return Mode::Gui(Vec::new()),
            "ask" | "auto" => {} // fall through to cfg
            other => eprintln!(
                "lamquant: ignoring LAMQUANT_UI=`{}` (expected tui|gui|ask|auto)",
                other,
            ),
        }
    }
    // 2. Persisted preference from lamquant.toml.
    //    Same trim + lowercase normalization as env so a TOML value of
    //    "GUI" or " tui " is interpreted the same as "gui" / "tui".
    let cfg = lamquant_core::tui::config::LamQuantConfig::load();
    // Apply config-time runtime choices (compute backend selector,
    // etc.) once at startup so the very first encode -- not just
    // post-save edits -- uses the user's persisted preference.
    cfg.apply_to_runtime();
    match cfg.ui_preference.trim().to_ascii_lowercase().as_str() {
        "tui" => Mode::Tui,
        "gui" => Mode::Gui(Vec::new()),
        // "ask" or unrecognized → TUI default. A.4 wires the wizard
        // first-run prompt that upgrades this to a real choice.
        _ => Mode::Tui,
    }
}

#[cfg(feature = "host")]
fn parse_mode(argv: &[String]) -> Mode {
    let Some(first) = argv.first().map(String::as_str) else {
        return Mode::Tui;
    };
    // Bare `help` (no leading `--`) is intentionally NOT captured here —
    // it falls through to passthrough so users can run `lamquant help
    // encode` and reach `lml help encode` (clap's built-in help subcommand
    // form). Only `--help`/`-h` print the launcher's own help.
    match first {
        "--help" | "-h" => Mode::Help,
        "--version" | "-V" => Mode::Version,
        "--tui" => Mode::Tui,
        "--gui" => Mode::Gui(argv[1..].to_vec()),
        _ => Mode::Passthrough(argv.to_vec()),
    }
}

#[cfg(feature = "host")]
fn print_usage() {
    print!(
        "lamquant {ver} — LamQuant launcher

USAGE:
    lamquant                    interactive (TUI today; GUI auto-detect in v1.1)
    lamquant --tui              force TUI
    lamquant --gui              force GUI (requires lamquant-gui sibling binary)
    lamquant <subcommand> ...   pass through to `lml <subcommand>`
    lamquant --help             this help
    lamquant --version          version

For codec subcommands (encode, decode, verify, info, stats, bench, ...),
run `lml --help` or `lamquant <subcommand> --help`.
",
        ver = env!("CARGO_PKG_VERSION"),
    );
}

/// Exec the sibling `lamquant-gui` binary — the Tauri desktop app.
/// Resolved relative to current_exe() so the cargo-installed sibling is
/// always preferred over a PATH binary (defends against shadowing).
/// Forwards any args after `--gui` so future GUI flags don't need
/// launcher changes.
#[cfg(feature = "host")]
fn exec_lamquant_gui(args: &[String]) -> i32 {
    use std::io::ErrorKind;
    use std::process::Command;
    let gui = sibling_binary("lamquant-gui");
    match Command::new(&gui).args(args).status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            // NotFound is the most common failure (GUI not installed yet)
            // and warrants the install hint. PermissionDenied / other
            // errors usually mean a different problem (corrupt install,
            // SELinux policy, etc.) where the install hint is misleading.
            if e.kind() == ErrorKind::NotFound {
                eprintln!(
                    "lamquant: `{}` not found.

To install the GUI alongside the TUI:
  cargo install --path gui/src-tauri --bin lamquant-gui

Or run the GUI directly from a source checkout:
  cd gui && npm run tauri dev",
                    gui.display(),
                );
            } else {
                eprintln!("lamquant: failed to exec `{}`: {}", gui.display(), e,);
            }
            127
        }
    }
}

#[cfg(feature = "host")]
fn exec_lml(args: &[String]) -> i32 {
    use std::process::Command;
    // Resolve `lml` next to our own binary first (the cargo-installed
    // sibling). Falls back to PATH only if current_exe() fails (rare —
    // mostly happens in unprivileged containers without /proc). Pinning
    // to the sibling defends against a malicious `lml` shadowing on PATH.
    let lml = sibling_binary("lml");
    match Command::new(&lml).args(args).status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!(
                "lamquant: failed to exec `{}`: {} \
                 (cargo install --path lamquant-core installs both binaries)",
                lml.display(),
                e,
            );
            127
        }
    }
}

/// Path to a sibling binary — resolved next to our own executable so the
/// cargo-installed sibling is always preferred over PATH. Falls back to
/// bare-name (PATH lookup) when current_exe() fails (rare —
/// unprivileged containers without /proc); emits a stderr warning so
/// users notice they're outside the hardened sibling-resolution path.
///
/// No `.is_file()` check on the candidate: that would introduce a
/// TOCTOU race between stat and exec. We let `Command::new` fail
/// naturally on a bad path; the calling exec_* handles the error.
#[cfg(feature = "host")]
fn sibling_binary(name: &str) -> std::path::PathBuf {
    use std::path::PathBuf;
    if let Ok(self_path) = std::env::current_exe() {
        if let Some(dir) = self_path.parent() {
            // Append .exe on Windows to match cargo install's output.
            #[cfg(windows)]
            let exec_name = format!("{}.exe", name);
            #[cfg(not(windows))]
            let exec_name = name.to_string();
            return dir.join(exec_name);
        }
    }
    eprintln!(
        "lamquant: warning — could not resolve installed-binary directory; \
         falling back to PATH lookup for `{}` (reduced shadowing protection)",
        name,
    );
    PathBuf::from(name)
}
