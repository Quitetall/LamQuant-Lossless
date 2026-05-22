//! `lml-stream` — replay an `.lml` archive as an LSL outlet.
//!
//! ADR 0024 Phase 4. Reads a `.lml` file, constructs an LSL outlet
//! with channel metadata propagated from the EDF source, paces the
//! sample stream at the chosen rate, drains every sample. Any
//! LSL-aware consumer (LabRecorder, pylsl, NeuroPype) sees the
//! replayed stream identically to a live source.
//!
//! Usage:
//!     lml-stream <path/to.lml> [--rate <multiplier>] [--burst]
//!         [--name <stream_name>]
//!
//! Defaults: stream name = derived from file stem; rate = real-time
//! (match source sample rate).

use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage: lml-stream <path/to.lml> [--rate <x>] [--burst] [--name <stream>]\n\n\
         options:\n\
         \x20 --rate <x>      playback rate multiplier (e.g. 2.0 = 2x speed)\n\
         \x20 --burst         push as-fast-as-possible (no pacing)\n\
         \x20 --name <name>   override the LSL stream name\n\
         \x20 --help          show this help text"
    );
    std::process::exit(2)
}

fn main() -> ExitCode {
    use lamquant_lsl::{Outlet, Rate};

    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        usage();
    }

    let mut path: Option<PathBuf> = None;
    let mut rate = Rate::RealTime;
    let mut name: Option<String> = None;

    let mut iter = argv.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => usage(),
            "--burst" => rate = Rate::Burst,
            "--rate" => {
                let Some(v) = iter.next() else {
                    eprintln!("--rate requires a value");
                    usage();
                };
                let m: f64 = match v.parse() {
                    Ok(x) => x,
                    Err(e) => {
                        eprintln!("invalid --rate {}: {}", v, e);
                        usage();
                    }
                };
                rate = Rate::Multiplier(m);
            }
            "--name" => {
                let Some(v) = iter.next() else {
                    eprintln!("--name requires a value");
                    usage();
                };
                name = Some(v);
            }
            other if other.starts_with("--") => {
                eprintln!("unknown flag: {}", other);
                usage();
            }
            other => {
                if path.is_some() {
                    eprintln!("multiple positional args: {} and {}", path.unwrap().display(), other);
                    usage();
                }
                path = Some(PathBuf::from(other));
            }
        }
    }

    let Some(path) = path else {
        eprintln!("missing path");
        usage();
    };

    eprintln!(
        "lml-stream: opening {} (rate = {:?}, name = {:?})",
        path.display(),
        rate,
        name.as_deref().unwrap_or("<auto>")
    );
    let outlet = match Outlet::from_lml_with_rate(&path, name.as_deref(), rate) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("outlet open failed: {}", e);
            return ExitCode::from(1);
        }
    };
    eprintln!(
        "lml-stream: ready ({} samples @ {} Hz nominal)",
        outlet.sample_count(),
        outlet.nominal_srate()
    );
    match outlet.push_all() {
        Ok(n) => {
            eprintln!("lml-stream: pushed {} samples", n);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("lml-stream: push failed: {}", e);
            ExitCode::from(1)
        }
    }
}
