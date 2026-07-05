//! ADR 0051 / #23 — byte-stability goldens for the LOSSY / near-lossless modes.
//!
//! The property gates (`bounded_mae_property`, `target_bps_property`) prove that δ /
//! bits-per-sample are respected; they DON'T pin the exact bytes, so an unintended
//! change in the deadzone-quantize or rate-control path that still satisfies the
//! bound would slip through. This pins the wire: a frozen sha over the encoded
//! output + the header `mode` byte, so any such drift flips the golden. Inputs are
//! deterministic; `LpcMode::default()` is `Anytime{None}` (never reads a clock).
#![cfg(feature = "archive")]

use abir::{Abir, Untyped};
use lamquant_core::abir_container::write_abir_to_vec;
use lamquant_core::lpc::LpcMode;
use sha2::{Digest, Sha256};

fn sig() -> Vec<Vec<i64>> {
    (0..4).map(|c| (0..600).map(|t| ((t * 37 + c * 13) % 4001) - 2000).collect()).collect()
}

fn sha(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    format!("{:x}", h.finalize())
}

#[test]
fn bounded_mae_wire_is_byte_stable_and_mode_tagged() {
    let abir = Abir::<Untyped>::from_channels_i64(sig(), 250.0);
    let a = write_abir_to_vec(&abir, 250.0, 256, 0, "{}", LpcMode::default(), Some(8), None).unwrap();
    let b = write_abir_to_vec(&abir, 250.0, 256, 0, "{}", LpcMode::default(), Some(8), None).unwrap();
    assert_eq!(a, b, "BoundedMae encode must be deterministic");
    assert_eq!(a[9], 1, "header mode byte = BoundedMae(1)");
    assert_eq!(
        sha(&a),
        "16b1951569aff0523eacce8af6c8f22af61c16080f3b31662d9f0f86eba67a06",
        "BoundedMae wire drifted from the frozen golden (regen deliberately if intended)"
    );
}

#[test]
fn target_bps_wire_is_byte_stable_and_mode_tagged() {
    let abir = Abir::<Untyped>::from_channels_i64(sig(), 250.0);
    let a = write_abir_to_vec(&abir, 250.0, 256, 0, "{}", LpcMode::default(), None, Some(4.0)).unwrap();
    let b = write_abir_to_vec(&abir, 250.0, 256, 0, "{}", LpcMode::default(), None, Some(4.0)).unwrap();
    assert_eq!(a, b, "TargetBps encode must be deterministic");
    assert_eq!(a[9], 2, "header mode byte = TargetBps(2)");
    assert_eq!(
        sha(&a),
        "885f75e6bbedef77d545c72adaf2a7ac370984e68665e1884fb3c23afbcde00d",
        "TargetBps wire drifted from the frozen golden (regen deliberately if intended)"
    );
}
