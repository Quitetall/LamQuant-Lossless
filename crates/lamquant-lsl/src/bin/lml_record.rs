//! `lml-record` — subscribe to an LSL stream + record to `.lml`.
//!
//! ADR 0024 Phase 4. Resolves a discoverable LSL stream by name,
//! pulls samples in a tight loop, accumulates them through the
//! codec's window machinery, and writes a `.lml` container to disk
//! when capture completes.
//!
//! Usage:
//!     lml-record --name <stream> --out <path.lml>
//!         [--max-samples <N>] [--window <size>]
//!
//! Defaults: window = 2500 samples (matches `lml encode`'s default
//! at 250 Hz); max-samples = unlimited (Ctrl-C to stop, or hit
//! --max-samples).

use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage: lml-record --name <stream> --out <path.lml> \\\n\
         \x20      [--max-samples <N>] [--window <size>]\n\n\
         options:\n\
         \x20 --name <stream>    LSL stream name to subscribe to\n\
         \x20 --out <path>       output .lml file (will be overwritten)\n\
         \x20 --max-samples <N>  stop after N samples (default: unlimited)\n\
         \x20 --window <size>    codec window size (default: 2500)\n\
         \x20 --timeout <sec>    discovery timeout (default: 5.0)\n\
         \x20 --help             show this help text"
    );
    std::process::exit(2)
}

fn main() -> ExitCode {
    use lamquant_lsl::inlet::{InletEncodeOpts, RecordSession};
    use lamquant_lsl::Inlet;

    let mut name: Option<String> = None;
    let mut out: Option<PathBuf> = None;
    let mut max_samples: Option<usize> = None;
    let mut window_size: usize = 2500;
    let mut timeout: f64 = 5.0;

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => usage(),
            "--name" => name = iter.next(),
            "--out" => out = iter.next().map(PathBuf::from),
            "--max-samples" => {
                max_samples = match iter.next().and_then(|v| v.parse().ok()) {
                    Some(n) => Some(n),
                    None => {
                        eprintln!("--max-samples needs a positive integer");
                        usage();
                    }
                };
            }
            "--window" => {
                window_size = match iter.next().and_then(|v| v.parse().ok()) {
                    Some(n) if n > 0 => n,
                    _ => {
                        eprintln!("--window needs a positive integer");
                        usage();
                    }
                };
            }
            "--timeout" => {
                timeout = match iter.next().and_then(|v| v.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--timeout needs a number");
                        usage();
                    }
                };
            }
            other => {
                eprintln!("unknown arg: {}", other);
                usage();
            }
        }
    }
    let (name, out) = match (name, out) {
        (Some(n), Some(o)) => (n, o),
        _ => usage(),
    };

    eprintln!("lml-record: resolving stream name='{}' (timeout {} s)", name, timeout);
    let inlet = match Inlet::resolve_by_name(&name, timeout) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("resolve failed: {}", e);
            return ExitCode::from(1);
        }
    };
    eprintln!(
        "lml-record: subscribed ({} channels @ {} Hz)",
        inlet.channel_count(),
        inlet.nominal_srate()
    );

    let mut session = match RecordSession::new(inlet, window_size, InletEncodeOpts::default()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("session new failed: {}", e);
            return ExitCode::from(1);
        }
    };

    let cap_target = max_samples.unwrap_or(usize::MAX);
    eprintln!("lml-record: capturing up to {} samples...", if cap_target == usize::MAX { "∞".to_string() } else { cap_target.to_string() });
    match session.capture(cap_target) {
        Ok(n) => eprintln!("lml-record: encoded {} windows", n),
        Err(e) => {
            eprintln!("capture failed: {}", e);
            return ExitCode::from(1);
        }
    }

    let windows = match session.finish() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("session finish failed: {}", e);
            return ExitCode::from(1);
        }
    };
    let total: usize = windows.iter().map(|w| w.len()).sum();
    if let Err(e) = std::fs::write(&out, windows.concat()) {
        eprintln!("write {} failed: {}", out.display(), e);
        return ExitCode::from(1);
    }
    eprintln!("lml-record: wrote {} bytes to {}", total, out.display());
    ExitCode::SUCCESS
}
