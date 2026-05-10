//! Launcher table — external commands that aren't `lml` subcommands.
//!
//! Used for training, eagle validator, pytest, install scripts, GUI launch
//! — anything where the front-end shells out to a different binary. Keep
//! this table in sync with `specs/ui-parity.md::Op IDs`.

/// Look up an external command spec by op id. Returns `(program, args, label)`.
pub fn launcher(id: &str) -> Option<(&'static str, Vec<&'static str>, &'static str)> {
    Some(match id {
        // Training (Python)
        "train_encoder" => ("python", vec!["lamquant.py", "train", "--mode=encoder"], "encoder training"),
        "train_snn"     => ("python", vec!["lamquant.py", "train", "--mode=snn"],     "SNN training"),
        "train_tnn"     => ("python", vec!["lamquant.py", "train", "--mode=tnn"],     "TNN training"),
        "train_resume"  => ("python", vec!["lamquant.py", "train", "--resume"],       "resume training"),

        // Eagle validator (Python entry point)
        "eagle_quick" => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=quick"], "eagle: quick"),
        "eagle_full"  => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=full"],  "eagle: full"),
        "eagle_bench" => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=bench"], "eagle: bench"),
        "eagle_lqs_l" => ("python", vec!["-m", "lamquant_codec.eagle", "--lqs=L"],       "eagle: LQS-L"),
        "eagle_lqs_c" => ("python", vec!["-m", "lamquant_codec.eagle", "--lqs=C"],       "eagle: LQS-C"),
        "eagle_lqs_m" => ("python", vec!["-m", "lamquant_codec.eagle", "--lqs=M"],       "eagle: LQS-M"),
        "eagle_lqs_a" => ("python", vec!["-m", "lamquant_codec.eagle", "--lqs=A"],       "eagle: LQS-A"),
        "eagle_perf"  => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=perf"],  "eagle: perf"),
        "eagle_rd"    => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=rd"],    "eagle: rate-distortion"),
        "eagle_h2h"   => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=h2h"],   "eagle: head-to-head"),

        // pytest menu
        "test_conformance" => ("pytest", vec!["tests/conformance/", "-q"],   "tests: conformance"),
        "test_full"        => ("pytest", vec!["tests/", "-q"],               "tests: full"),
        "test_paranoid"    => ("pytest", vec!["tests/", "-q", "--paranoid"], "tests: paranoid"),
        "test_codec"       => ("pytest", vec!["tests/codec/", "-q"],         "tests: codec"),

        // Setup / install
        "setup_pip"    => ("pip",   vec!["install", "-e", ".[dev]"],                       "pip install -e .[dev]"),
        "setup_extras" => ("pip",   vec!["install", "hypothesis", "prompt_toolkit", "zstandard"], "pip extras"),
        "setup_cargo"  => ("cargo", vec!["build", "--release", "--manifest-path", "lamquant-core/Cargo.toml", "--bin", "lml"], "cargo build lml"),
        "setup_musl"   => ("cargo", vec!["build", "--release", "--manifest-path", "lamquant-core/Cargo.toml", "--bin", "lml", "--target", "x86_64-unknown-linux-musl"], "static linux build"),
        "setup_windows"=> ("cargo", vec!["build", "--release", "--manifest-path", "lamquant-core/Cargo.toml", "--bin", "lml", "--target", "x86_64-pc-windows-gnu"], "windows build"),

        // GUI / visualization launchers
        "gui"                => ("lamquant-gui",  vec![], "Vision GUI"),
        "viz_lamquant-gui"   => ("lamquant-gui",  vec![], "Vision GUI"),
        "viz_eeglab"         => ("eeglab",        vec![], "EEGLab"),
        "viz_mne"            => ("python3",       vec!["-c", "import mne; mne.gui.browse_raw()"], "MNE-Python viewer"),
        "viz_OpenBCIGUI"     => ("OpenBCIGUI",    vec![], "OpenBCI GUI"),
        "viz_BVAnalyzer"     => ("BVAnalyzer",    vec![], "BrainVision Analyzer"),
        "viz_besa"           => ("besa",          vec![], "BESA"),

        // Auto-install launchers fired by the visualization panel's
        // [Enter]/[i] key on missing tools. Each runs the standard
        // package-manager install for that tool. Network access
        // required; failures show in the output panel and the
        // visualization panel's [r] re-probe surfaces success.
        "viz_install_mne"        => ("pip",   vec!["install", "mne"],         "install: pip install mne"),
        "viz_install_scope_tui"  => ("cargo", vec!["install", "scope-tui"],   "install: cargo install scope-tui"),
        "viz_install_bottom"     => ("cargo", vec!["install", "bottom"],      "install: cargo install bottom"),
        "viz_install_television" => ("cargo", vec!["install", "television"],  "install: cargo install television"),
        "viz_install_csvlens"    => ("cargo", vec!["install", "csvlens"],     "install: cargo install csvlens"),
        "viz_install_gitui"      => ("cargo", vec!["install", "gitui"],       "install: cargo install gitui"),

        // Cockpit utilities (Phase B.2 wiring of [r/c/m] keys).
        // Linux/macOS only — sh -c shell pipelines. Windows users see
        // the same status sidebar entry; the launcher itself fails fast.
        "cockpit_reset" => (
            "sh",
            vec![
                "-c",
                // Two destructive operations, each with honest exit
                // reporting. tmux missing-session is normal (no error);
                // tmux command-not-installed is reported. rm failures
                // are reported. Final exit code reflects whether either
                // step encountered a real error.
                "rc=0; \
                 if [ -e ~/.cache/lamquant ]; then \
                     if rm -rf ~/.cache/lamquant; then \
                         echo '✓ ~/.cache/lamquant cleared'; \
                     else \
                         echo '✗ rm -rf ~/.cache/lamquant failed (permissions?)'; \
                         rc=1; \
                     fi; \
                 else \
                     echo '— ~/.cache/lamquant did not exist; nothing to clear'; \
                 fi; \
                 if command -v tmux >/dev/null 2>&1; then \
                     out=$(tmux kill-session -t lamquant-train 2>&1); \
                     case \"$out\" in \
                         *\"can't find session\"*|'') \
                             echo '✓ tmux: no lamquant-train session running' ;; \
                         *) \
                             echo \"✗ tmux kill-session failed: $out\"; \
                             rc=1 ;; \
                     esac; \
                 else \
                     echo '— tmux not installed; skipped session kill'; \
                 fi; \
                 if [ \"$rc\" = 0 ]; then \
                     echo 'reset complete'; \
                 else \
                     echo 'reset finished with errors'; \
                 fi; \
                 exit $rc",
            ],
            "reset: cache + tmux",
        ),
        "cockpit_checkpoints" => (
            "sh",
            vec![
                "-c",
                // Tighter glob: -path 'runs/*/checkpoints/*' so we match
                // only files under a `checkpoints/` directory inside a
                // run, not arbitrary files with "checkpoint" in the name.
                "if [ -d runs ]; then \
                     hits=$(find runs -maxdepth 6 -path 'runs/*/checkpoints/*' -type f 2>/dev/null | sort -r | head -100); \
                     if [ -z \"$hits\" ]; then \
                         echo 'no checkpoints found under runs/*/checkpoints/'; \
                     else \
                         echo \"$hits\"; \
                     fi; \
                 else \
                     echo 'no runs/ directory in cwd ('\"$PWD\"')'; \
                 fi",
            ],
            "list checkpoints",
        ),
        "cockpit_metrics" => (
            "sh",
            vec![
                "-c",
                // Distinguishes "no runs/ at all" from "runs/ exists but
                // empty"; both are non-error exits since training just
                // hasn't started yet.
                "if [ ! -d runs ]; then \
                     echo 'no runs/ directory in cwd ('\"$PWD\"')'; \
                     exit 0; \
                 fi; \
                 latest=$(ls -td runs/*/ 2>/dev/null | head -1); \
                 if [ -z \"$latest\" ]; then \
                     echo 'no training runs found under runs/'; \
                     exit 0; \
                 fi; \
                 log=\"${latest}log.txt\"; \
                 if [ -f \"$log\" ]; then \
                     echo \"# tailing $log\"; \
                     tail -200 \"$log\"; \
                 else \
                     echo \"no log.txt in $latest (looked for $log)\"; \
                 fi",
            ],
            "tail latest log",
        ),

        // Firmware exports
        "fw_export" => ("python", vec!["scripts/export_weights.py"], "export weights"),

        // Syscheck (handled in-process for the most part, but expose Python self-test)
        "syscheck_py" => ("python", vec!["-m", "lamquant_codec.cli.syscheck"], "syscheck (python)"),

        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_op_returns_some() {
        assert!(launcher("train_encoder").is_some());
        assert!(launcher("eagle_lqs_l").is_some());
    }

    #[test]
    fn unknown_op_returns_none() {
        assert!(launcher("not_a_real_op").is_none());
    }
}
