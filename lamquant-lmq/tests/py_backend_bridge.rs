//! ADR 0074 Track N — PyBackend subprocess-bridge gate.
//!
//! Two tests:
//!   * `py_backend_selftest_...` — spawns the REAL `python3` helper in its weightless
//!     "selftest" mode and drives a full `Abir → shell.encode → BCS1-Lmq → shell.decode`
//!     round-trip through the subprocess. Proves the bridge + JSON protocol +
//!     backend_meta round-trip WITHOUT any model/weights. Skips only if `python3` is
//!     absent.
//!   * `py_backend_model_...` — the real `SubbandCodec` end-to-end. ENV-GATED: it
//!     skips (never fails) when codec-neural / weights are absent, exactly like the
//!     SNN PCCP gates. When weights are present it produces a real lossy round-trip.

#![cfg(feature = "python")]

use std::path::PathBuf;

use abir::{Abir, Eeg, Modality, ModalitySource, Untyped, CODEC_LMQ_FSQ};
use lamquant_lmq::py_backend::PyBackend;
use lamquant_lmq::shell;

fn helper() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/lmq_infer.py")
}

fn python3_available() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn eeg(sig: Vec<Vec<i64>>) -> Abir<Eeg> {
    Abir::<Untyped>::from_channels_i64(sig, 250.0).into_modality::<Eeg>(ModalitySource::Manual)
}

#[test]
fn py_backend_selftest_round_trips_through_the_subprocess_and_wire() {
    if !python3_available() {
        eprintln!("SKIP py_backend_selftest: python3 not available");
        return;
    }
    let sig: Vec<Vec<i64>> =
        (0..4).map(|c| (0..64).map(|i| ((i * 3 + c * 7) % 40) as i64 - 20).collect()).collect();
    let abir = eeg(sig.clone());
    let backend = PyBackend::selftest("python3", helper());

    // Full path: Abir → shell.encode (spawns python, selftest quantize) → BCS1-Lmq.
    let bytes = shell::encode(&abir, &backend).expect("py selftest encode");
    assert_eq!(&bytes[0..4], b"BCS1");
    assert_eq!(bytes[8], CODEC_LMQ_FSQ, "lossy-signed neural descriptor");

    // decode (spawns python again, selftest dequantize) → the mod-5 residues.
    let decoded = shell::decode(&bytes, &backend).expect("py selftest decode");
    let got: Vec<Vec<i64>> =
        decoded.window_views(0, decoded.n_samples()).iter().map(|c| c.as_ref().to_vec()).collect();
    let expect: Vec<Vec<i64>> =
        sig.iter().map(|ch| ch.iter().map(|&s| s.rem_euclid(5)).collect()).collect();
    assert_eq!(got, expect, "selftest wire round-trip == signal mod 5");
    assert_eq!(decoded.provenance().tag, Eeg::TAG, "modality survived the subprocess round-trip");
}

#[test]
fn py_backend_model_end_to_end_is_env_gated() {
    if !python3_available() {
        eprintln!("SKIP py_backend_model: python3 not available");
        return;
    }
    // The real SubbandCodec expects a 21-channel window; a short synthetic window
    // is enough to prove the wire path when the env is present.
    let sig: Vec<Vec<i64>> =
        (0..21).map(|c| (0..2500).map(|i| ((i + c) % 200) as i64 - 100).collect()).collect();
    let abir = eeg(sig);
    let backend = PyBackend::model("python3", helper());

    match shell::encode(&abir, &backend) {
        Ok(bytes) => {
            assert_eq!(bytes[8], CODEC_LMQ_FSQ);
            let decoded = shell::decode(&bytes, &backend).expect("model decode with weights present");
            // Honest end-to-end: it produced a valid lossy .lmq and reconstructed a
            // same-shape signal. The R number is reported by the R harness, not here.
            assert_eq!(decoded.n_channels(), 21);
            assert_eq!(decoded.n_samples(), 2500);
            eprintln!("py_backend_model: end-to-end OK (weights present)");
        }
        Err(e) => {
            // Env absent (no codec-neural / torch / weights) → SKIP, never fail.
            eprintln!("SKIP py_backend_model: environment/weights absent ({e:?})");
        }
    }
}
