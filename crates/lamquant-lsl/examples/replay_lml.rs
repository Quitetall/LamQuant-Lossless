//! Replay an `.lml` archive as an LSL outlet. Requires the
//! `--features liblsl` build (system liblsl must be installed —
//! see crates/lamquant-lsl/README.md).
//!
//! Usage:
//!
//!     cargo run -p lamquant-lsl --features liblsl --example replay_lml -- \
//!         /path/to/recording.lml [--burst | --rate 2.0]
//!
//! A pylsl subscriber in another terminal:
//!
//!     python -c "from pylsl import StreamInlet, resolve_streams; \
//!         s = resolve_streams()[0]; inlet = StreamInlet(s); \
//!         print(inlet.pull_sample())"
//!
//! When built without `--features liblsl`, the example prints an
//! install hint and exits 1.

#[cfg(feature = "liblsl")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use lamquant_lsl::{Outlet, Rate};
    use std::env;
    use std::path::PathBuf;
    use std::process::ExitCode;

    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!(
                "usage: replay_lml <path/to.lml> [--burst | --rate <multiplier>]"
            );
            std::process::exit(2);
        }
    };
    let mut rate = Rate::RealTime;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--burst" => rate = Rate::Burst,
            "--rate" => {
                let Some(v) = args.next() else {
                    eprintln!("--rate requires a value (e.g. 2.0)");
                    std::process::exit(2);
                };
                let m: f64 = v.parse().expect("invalid --rate multiplier");
                rate = Rate::Multiplier(m);
            }
            other => {
                eprintln!("unknown flag {}", other);
                std::process::exit(2);
            }
        }
    }

    eprintln!("opening {} (rate = {:?})", path.display(), rate);
    let outlet = Outlet::from_lml_with_rate(&path, None, rate)?;
    eprintln!(
        "outlet ready: {} samples at {} Hz",
        outlet.sample_count(),
        outlet.nominal_srate()
    );
    eprintln!("pushing to LSL network...");
    let pushed = outlet.push_all()?;
    eprintln!("done: pushed {} samples", pushed);
    Ok(())
}

#[cfg(not(feature = "liblsl"))]
fn main() {
    eprintln!(
        "lamquant-lsl example: built without the `liblsl` feature.\n\n\
         Rebuild with `--features liblsl` after installing the system\n\
         liblsl library. See crates/lamquant-lsl/README.md for\n\
         platform-specific install instructions."
    );
    std::process::exit(1);
}
