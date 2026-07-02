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
const T: usize = 128;
/// Documented tolerance: max |Δ| between the Rust and scipy int32 Q31 outputs.
/// Both run the identical f64 filtfilt algorithm, so this is FP-order slack at
/// the truncation boundary, not an algorithmic gap. Tighten if it ever drifts.
const TOL: i32 = 2;

/// MUST match `make_input` in `tools/dump_normalize_golden.py`.
fn synth_input() -> Vec<Vec<f64>> {
    (0..N_CH)
        .map(|c| {
            (0..T)
                .map(|t| (((c * 37 + t * 5) % 4001) as i64 - 2000) as f64)
                .collect()
        })
        .collect()
}

fn golden() -> Vec<i32> {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "fixtures",
        "normalize",
        "eeg_250hz_hp_q31.i32",
    ]
    .iter()
    .collect();
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read golden {}: {e} (run tools/dump_normalize_golden.py)", path.display()));
    assert_eq!(bytes.len(), N_CH * T * 4, "golden byte length");
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn rust_normalization_matches_python_golden_within_tolerance() {
    let input = synth_input();
    let out =
        lamquant_core::normalize::normalize_eeg_250hz(&input).expect("non-flat input normalizes");
    assert_eq!(out.len(), N_CH);
    assert_eq!(out[0].len(), T);

    let flat: Vec<i32> = out.into_iter().flatten().collect();
    let gold = golden();
    assert_eq!(flat.len(), gold.len());

    let mut max_delta = 0i32;
    let mut n_mismatch = 0usize;
    let mut worst_at = 0usize;
    for (i, (&r, &g)) in flat.iter().zip(gold.iter()).enumerate() {
        let d = (r - g).abs();
        if d > 0 {
            n_mismatch += 1;
        }
        if d > max_delta {
            max_delta = d;
            worst_at = i;
        }
    }
    assert!(
        max_delta <= TOL,
        "Rust vs Python Q31 parity exceeded tolerance: max|Δ|={max_delta} (>{TOL}) at index {worst_at} \
         (rust={}, python={}); {n_mismatch}/{} samples differ",
        flat[worst_at],
        gold[worst_at],
        flat.len()
    );
    eprintln!(
        "normalize parity OK: max|Δ|={max_delta} (tol {TOL}), {n_mismatch}/{} samples differ",
        flat.len()
    );
}
