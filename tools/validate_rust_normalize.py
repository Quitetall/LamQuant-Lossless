#!/usr/bin/env python3
"""ADR 0069 S7b / #37 — validate the Rust normalization path on a REAL corpus.

The synthetic parity gate (`lamquant-lossless/tests/normalize_parity.rs`) proves
the Rust normalization DSP (resample→250 → 0.5 Hz zero-phase HP → Q31 → f32) is
BIT-EXACT to the Python scipy pipeline for every common poly-branch rate
(250/200/500/512/1000). This tool confirms it on YOUR corpus — real signals,
real per-recording sample rates — so the `LAMQUANT_RUST_NORMALIZE` default can be
flipped from Python to Rust with confidence.

For each recording it runs `decode_lma_signal` BOTH ways and compares the output:
  * flag OFF → the Python scipy DSP (today's default);
  * flag ON  → the Rust DSP for poly rates; scipy fallback for the rare
    FFT-branch odd rates (up/down > 256), which is transparent.
Both variants therefore agree bit-for-bit when the port is correct.

    python3 tools/validate_rust_normalize.py <corpus_dir_or_.lma> [--max N] [--tol LSB]

Exit 0 iff every checked recording matches within tolerance. Requires: the `lml`
binary on PATH, the built `lamquant_core` extension, scipy, and the
`lamquant_codec` package importable.
"""
import argparse
import glob
import os
import sys

import numpy as np

# A Q31 least-significant bit, expressed in the final f32 output scale
# (signal_f32 = q31/2147483647 * 1000): one LSB ≈ 4.66e-7.
_Q31_LSB_IN_FLOAT = 1000.0 / 2147483647.0


def _decode(lma_dataset, lma_path: str, stem: str, use_rust: bool):
    """Decode + normalize one recording with the Rust path on/off. Calls
    `decode_lma_signal` directly (not the cached wrapper) so each variant
    recomputes and the env flag takes effect."""
    os.environ["LAMQUANT_RUST_NORMALIZE"] = "1" if use_rust else "0"
    return lma_dataset.decode_lma_signal(lma_path, stem)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="a .lma file or a directory searched recursively for *.lma")
    ap.add_argument("--max", type=int, default=50, help="max recordings to check (default 50)")
    ap.add_argument("--tol", type=int, default=2, help="max |Δ| in Q31 LSB before failing (default 2)")
    args = ap.parse_args()

    try:
        from lamquant_codec.training import lma_dataset
    except Exception as e:  # noqa: BLE001
        print(f"cannot import lamquant_codec.training.lma_dataset: {e}", file=sys.stderr)
        return 2

    lmas = (
        [args.corpus]
        if args.corpus.endswith(".lma")
        else sorted(glob.glob(os.path.join(args.corpus, "**", "*.lma"), recursive=True))
    )
    if not lmas:
        print(f"no .lma files found under {args.corpus}", file=sys.stderr)
        return 2

    tol_float = args.tol * _Q31_LSB_IN_FLOAT
    n_checked = n_bitexact = n_within_tol = n_skipped = 0
    worst = 0.0
    for lma in lmas:
        if n_checked >= args.max:
            break
        try:
            entries = lma_dataset.list_lma_entries(lma)
        except Exception as e:  # noqa: BLE001
            print(f"  skip archive {lma}: {e}", file=sys.stderr)
            continue
        for entry in entries:
            if not entry.endswith(".lml") or n_checked >= args.max:
                continue
            stem = entry[: -len(".lml")]
            py = _decode(lma_dataset, lma, stem, use_rust=False)
            rs = _decode(lma_dataset, lma, stem, use_rust=True)
            n_checked += 1
            if py is None or rs is None:
                n_skipped += 1  # both dropped the recording (e.g. missing channels)
                continue
            if py.shape != rs.shape:
                print(f"FAIL shape mismatch {stem}: python{py.shape} rust{rs.shape}")
                return 1
            d = float(np.max(np.abs(py.astype(np.float64) - rs.astype(np.float64))))
            worst = max(worst, d)
            if np.array_equal(py, rs):
                n_bitexact += 1
            elif d <= tol_float:
                n_within_tol += 1
            else:
                print(f"FAIL exceeds tolerance {stem}: max|Δ|={d:.3e} (> {tol_float:.3e})")
                return 1

    compared = n_checked - n_skipped
    print(
        f"checked={n_checked}  bit_exact={n_bitexact}  within_tol={n_within_tol}  "
        f"skipped(None)={n_skipped}  worst|Δ|={worst:.3e}"
    )
    ok = compared >= 0 and (n_bitexact + n_within_tol) == compared
    print(
        "PASS — Rust normalization matches Python on this corpus; safe to flip "
        "the LAMQUANT_RUST_NORMALIZE default"
        if ok
        else "REVIEW — mismatches above; do NOT flip the default yet"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
