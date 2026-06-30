//! Front-end bit-exact baseline — freeze current behavior so the upcoming IR
//! refactor (frontends → SignalBundle IR) can be proven byte-identical.
//!
//! Two locks bracket the refactor:
//!   * **codec-wire** (always runs, no fixtures): deterministic signals →
//!     `lml::compress` → sha256. Locks the LML backend bytes — the invariant
//!     that must NEVER change regardless of how the front-end is rewritten.
//!   * **front-end (NWB)** (`--features nwb`): an h5py-authored fixture →
//!     `nwb::read_bundle` → sha256 of the parsed signal. Locks that the reader
//!     produces the same IR signal. Skips when python3+h5py absent.
//!
//! Regenerate goldens after an INTENTIONAL change:
//!   LAMQUANT_REGEN_FRONTEND=1 cargo test --features nwb --test front_end_bit_exact -- --nocapture
//! then paste the printed shas into the GOLDEN_* tables below.

use sha2::{Digest, Sha256};

/// sha256 over channel-major signal: count, then per-channel (len + i64-LE samples).
fn sha_signal(signal: &[Vec<i64>]) -> String {
    let mut h = Sha256::new();
    h.update((signal.len() as u64).to_le_bytes());
    for ch in signal {
        h.update((ch.len() as u64).to_le_bytes());
        for &s in ch {
            h.update(s.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

fn sha_bytes(b: &[u8]) -> String {
    format!("{:x}", Sha256::new().chain_update(b).finalize())
}

fn regen() -> bool {
    std::env::var("LAMQUANT_REGEN_FRONTEND").is_ok()
}

/// Deterministic test signals — varied shapes the codec must encode identically.
fn fixtures() -> Vec<(&'static str, Vec<Vec<i64>>)> {
    let ramp: Vec<i64> = (0..2000).map(|t| (t % 257) as i64 - 128).collect();
    let sine: Vec<i64> = (0..2000)
        .map(|t| ((t as f64 * 0.13).sin() * 1000.0) as i64)
        .collect();
    let flat: Vec<i64> = vec![42; 2000];
    vec![
        ("single_ramp", vec![ramp.clone()]),
        ("multi_4ch", vec![ramp.clone(), sine.clone(), flat.clone(), ramp.clone()]),
        ("flat_const", vec![flat]),
    ]
}

// Codec-wire goldens (lossless, noise_bits=0). Captured on main pre-IR-refactor.
const GOLDEN_WIRE: &[(&str, &str)] = &[
    ("single_ramp", "f3a890974f399ea71cde88ca7073c4bfa31765bd46d0c5d2e75170d75111d3d4"),
    ("multi_4ch", "fb45d32714be7746c24be2d2c8cba83da7722f4cc69ea0848c7bf59a83cca64b"),
    ("flat_const", "32faeba1ce70f364add756a863cad497da3ed48715fa12653832916b33226a3b"),
];

#[test]
fn codec_wire_bytes_locked() {
    for (name, signal) in fixtures() {
        let bytes = lamquant_core::lml::compress(&signal, 0).expect("compress");
        let got = sha_bytes(&bytes);
        if regen() {
            println!("WIRE {name} = {got}");
            continue;
        }
        let want = GOLDEN_WIRE.iter().find(|(n, _)| *n == name).unwrap().1;
        assert_eq!(got, want, "codec wire bytes changed for {name}");
    }
}

#[cfg(feature = "nwb")]
mod nwb_frontend {
    use super::*;
    use std::process::Command;

    const GOLDEN_NWB_SIGNAL: &str =
        "7dbf6db220c3e098e3a07b22f3e8a27d8d8fe88ab4e27665677f7c76cb7ca102";

    fn make_fixture(path: &std::path::Path) -> bool {
        let script = format!(
            r#"
import sys
try:
    import h5py, numpy as np
except Exception:
    sys.exit(42)
T, C = 800, 6
data = (np.arange(T).reshape(T,1)*7 + np.arange(C).reshape(1,C)).astype('<i2')
flags = (np.arange(300) % 11).astype('u1')
with h5py.File(r"{}", "w") as f:
    f.create_group("acquisition").create_group("ElectricalSeries").create_dataset("data", data=data, chunks=(200,C))
    f.create_dataset("flags", data=flags)
sys.exit(0)
"#,
            path.display()
        );
        matches!(
            Command::new("python3").arg("-c").arg(&script).status(),
            Ok(s) if s.success()
        )
    }

    #[test]
    fn nwb_reader_signal_locked() {
        let dir = tempfile::tempdir().unwrap();
        let fx = dir.path().join("fx.nwb");
        if !make_fixture(&fx) {
            eprintln!("SKIP nwb_reader_signal_locked: h5py unavailable");
            return;
        }
        let bundle = lamquant_core::nwb::read_bundle(&fx).expect("read_bundle");
        let got = sha_signal(&bundle.signal);
        if regen() {
            println!("NWB signal = {got}");
            return;
        }
        assert_eq!(got, GOLDEN_NWB_SIGNAL, "NWB reader signal changed");
    }
}
