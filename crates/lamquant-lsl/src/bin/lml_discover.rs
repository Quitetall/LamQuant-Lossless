//! `lml-discover` — list LSL streams on the local network.
//!
//! ADR 0024 Phase 4. Resolves all LSL streams visible within the
//! configured timeout, prints one line per stream with
//! source_id / name / type / channel_count / sample_rate.
//! Useful for debugging discovery + verifying that a publisher is
//! actually reachable before running `lml-record`.
//!
//! Usage:
//!     lml-discover [--timeout <sec>]

use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage: lml-discover [--timeout <sec>]\n\n\
         options:\n\
         \x20 --timeout <sec>  discovery wait time (default: 1.0)\n\
         \x20 --help           show this help text"
    );
    std::process::exit(2)
}

fn main() -> ExitCode {
    let mut timeout: f64 = 1.0;
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => usage(),
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

    // `lsl::resolve_streams` discovers everything visible within
    // the timeout. Empty result is normal on quiet networks; we
    // still print the header so scripts can grep deterministically.
    let streams = match lsl::resolve_streams(timeout) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("resolve failed: {:?}", e);
            return ExitCode::from(1);
        }
    };
    println!("source_id\tname\ttype\tchannels\tsrate");
    for info in &streams {
        println!(
            "{}\t{}\t{}\t{}\t{}",
            info.source_id(),
            info.stream_name(),
            info.stream_type(),
            info.channel_count(),
            info.nominal_srate()
        );
    }
    eprintln!("lml-discover: {} stream(s) within {:.1} s", streams.len(), timeout);
    ExitCode::SUCCESS
}
