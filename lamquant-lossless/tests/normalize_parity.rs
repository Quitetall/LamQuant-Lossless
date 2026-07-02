//! ADR 0069 S7b — normalization parity gate.
//!
//! Asserts the Rust normalization port (`lml::normalize`) matches the Python
//! `lma_dataset.py::decode_lma_signal` pipeline on the committed golden. The
//! golden (int32 Q31) is produced by `tools/dump_normalize_golden.py`; the
//! input is a deterministic integer µV formula reproduced bit-for-bit here, so
//! only the golden output is committed.
//!
//! Bit-exactness vs scipy is impossible in general, so this gate uses a
//! documented tolerance: the Rust f64 filtfilt matches scipy's f64 filtfilt
//! (scipy upcasts the f32 production input to f64 losslessly), so the only
//! divergence is FP evaluation order — which after the ×(2^31-1) Q31 scaling
//! resolves to at most a few LSB at truncation boundaries. **This gate must be
//! GREEN before `lma_dataset.py` switches to the Rust path.**
//!
//! Regenerate the golden (and bump the manifest scipy version) if the filter
//! operating point or pipeline changes:
//!   python3 tools/dump_normalize_golden.py
#![cfg(feature = "archive")]

use std::path::PathBuf;

const N_CH: usize = 21;

/// MUST match `make_input(t)` in `tools/dump_normalize_golden.py`.
fn synth_input(t: usize) -> Vec<Vec<f64>> {
    (0..N_CH)
        .map(|c| {
            (0..t)
                .map(|tt| (((c * 37 + tt * 5) % 4001) as i64 - 2000) as f64)
                .collect()
        })
        .collect()
}

fn golden(name: &str) -> Vec<i32> {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "fixtures",
        "normalize",
        &format!("{name}.i32"),
    ]
    .iter()
    .collect();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("read golden {}: {e} (run tools/dump_normalize_golden.py)", path.display())
    });
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Compare a flat Rust Q31 output against the named golden; assert max|Δ| ≤ tol.
/// i64 deltas so two near-±i32::MAX Q31 values can't overflow (MiMo review).
fn assert_parity(flat: Vec<i32>, name: &str, tol: i64) {
    let gold = golden(name);
    assert_eq!(flat.len(), gold.len(), "{name}: length");
    let mut max_delta = 0i64;
    let mut n_mismatch = 0usize;
    let mut worst_at = 0usize;
    for (i, (&r, &g)) in flat.iter().zip(gold.iter()).enumerate() {
        let d = (r as i64 - g as i64).abs();
        if d > 0 {
            n_mismatch += 1;
        }
        if d > max_delta {
            max_delta = d;
            worst_at = i;
        }
    }
    assert!(
        max_delta <= tol,
        "{name}: Rust vs Python Q31 parity exceeded tolerance: max|Δ|={max_delta} (>{tol}) at \
         index {worst_at} (rust={}, python={}); {n_mismatch}/{} samples differ",
        flat[worst_at],
        gold[worst_at],
        flat.len()
    );
    eprintln!("{name} parity OK: max|Δ|={max_delta} (tol {tol}), {n_mismatch}/{} differ", flat.len());
}

/// 250 Hz: resample is identity, so this is the bit-exact HP+Q31 chain — expect
/// max|Δ|=0 (tol 2 is FP-order headroom that is currently unused).
#[test]
fn parity_250hz_hp_q31() {
    let out = lamquant_core::normalize::normalize_eeg(&synth_input(128), 250.0)
        .expect("no FFT branch")
        .expect("non-flat input normalizes");
    assert_parity(out.into_iter().flatten().collect(), "eeg_250hz_hp_q31", 2);
}

/// 200 Hz: exercises the polyphase resample branch (up=5, down=4). Now BOTH
/// sides resample in f64 (the S7b fix — production `lma_dataset.py` no longer
/// resamples in float32), so this is bit-exact TODAY (max|Δ|=0). The small
/// tolerance is headroom for cross-libm ULP drift in the transcendental filter
/// design (`sinc`/`i0` call `sin`, which IEEE-754 does not require correctly
/// rounded); the reported actual delta monitors it. Contrast the 250 Hz case,
/// which uses only +,−,× and is bit-exact by IEEE determinism.
#[test]
fn parity_200hz_resample_hp_q31() {
    let out = lamquant_core::normalize::normalize_eeg(&synth_input(160), 200.0)
        .expect("poly branch, no FFT")
        .expect("non-flat input normalizes");
    assert_parity(out.into_iter().flatten().collect(), "eeg_200hz_resample_hp_q31", 4);
}

/// The remaining common poly-branch rates (500/512/1000 Hz). 512 is the
/// poly-branch EDGE (up=125, down=256 — the longest, 5121-tap firwin). All run
/// in f64 and are bit-exact by construction; the small tolerance is
/// cross-libm `sin`-ULP headroom (same as 200 Hz). This is the technical
/// readiness for #37's flip-to-default: every common EEG rate the corpus uses
/// (200/250/256... wait 256 → up=125,down=128 also poly) resamples bit-exactly.
#[test]
fn parity_common_poly_rates() {
    for (name, orig_sr, t_in) in [
        ("eeg_500hz_resample_hp_q31", 500.0f64, 512usize),
        ("eeg_512hz_resample_hp_q31", 512.0, 512),
        ("eeg_1000hz_resample_hp_q31", 1000.0, 1024),
    ] {
        let out = lamquant_core::normalize::normalize_eeg(&synth_input(t_in), orig_sr)
            .expect("poly branch, no FFT")
            .expect("non-flat input normalizes");
        assert_parity(out.into_iter().flatten().collect(), name, 4);
    }
}
