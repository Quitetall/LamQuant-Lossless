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
