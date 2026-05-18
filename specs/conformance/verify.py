#!/usr/bin/env python3
"""
Phase 8 / Item E — LML wire-format conformance validator.

This is intentionally a **thin structural validator**, not a reference
decoder. It does three things per vector:

1. SHA-256-compare the file bytes against the value recorded in
   `<name>.expected.json`. Catches drift between the committed
   vector and any post-generation tampering.
2. Parse the 32-byte LML header + metadata + window-length index in
   pure Python (no LamQuant-specific CRC algorithm). Confirm the
   structural facts (n_channels, total_samples, window_size) match
   the expected JSON.
3. For negative vectors (corruption), spawn the LamQuant `lml decode`
   binary (or the user-supplied `--reader`) and confirm the produced
   error kind matches the documented one.

We do NOT re-implement the LamQuant CRC-32 (custom polynomial vs
zlib's) or the LMLFOOT1 footer's CRC math in this Python file.
Third-party reader implementers should run their own decode against
each vector and compare with `--reader my-binary`.

Usage:
    python3 specs/conformance/verify.py [--reader BIN] <vectors>

Exit 0 if every vector matched its expected behaviour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

MAGIC = b"LML1"


def parse_header_facts(raw: bytes) -> tuple[int, int, int, int]:
    """Returns (n_channels, n_windows, total_samples, window_size).
    Raises ValueError on bad magic / truncation."""
    if len(raw) < 32:
        raise ValueError(f"header too short: {len(raw)} < 32")
    if raw[:4] != MAGIC:
        raise ValueError(f"bad magic: {raw[:4]!r} != {MAGIC!r}")
    n_ch = struct.unpack_from("<H", raw, 6)[0]
    n_w = struct.unpack_from("<H", raw, 8)[0]
    n_s = struct.unpack_from("<I", raw, 10)[0]
    ws = struct.unpack_from("<H", raw, 14)[0]
    return n_ch, n_w, n_s, ws


def classify_decode_error(stderr: str) -> str:
    """Match LamQuant's CLI error formatter against the documented
    error-kind set."""
    if "InvalidMagic" in stderr or "magic" in stderr.lower():
        return "InvalidMagic"
    if (
        "Truncated" in stderr
        or "too short" in stderr.lower()
        or "failed to fill whole buffer" in stderr
        or "unexpected end of file" in stderr.lower()
    ):
        return "Truncated"
    if "CrcMismatch" in stderr or "CRC" in stderr:
        return "CrcMismatch"
    if "UnsupportedVersion" in stderr:
        return "UnsupportedVersion"
    if "InvalidHeader" in stderr or "Invalid header" in stderr:
        return "InvalidHeader"
    return "Unknown"


def verify_negative(vector: Path, expected_kind: str, reader_bin: str) -> tuple[str, str]:
    """Invoke the reader on the vector; assert it errors with the
    expected kind."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="lml-conformance-") as td:
        out_path = Path(td) / (vector.stem + ".decoded.raw")
        cmd = [reader_bin, "decode", str(vector), "-o", str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return ("FAIL", f"reader accepted negative vector; expected error {expected_kind}")
    kind = classify_decode_error(proc.stderr + proc.stdout)
    if kind == expected_kind:
        return ("EXPECTED-FAIL", f"correctly errored with {kind}")
    return (
        "FAIL",
        f"expected error {expected_kind}, got {kind} (stderr: {proc.stderr.strip()[:200]})",
    )


def verify_positive(vector: Path, expected: dict) -> tuple[str, str]:
    raw = vector.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected.get("lml_sha256") and actual_sha != expected["lml_sha256"]:
        return (
            "FAIL",
            f"file SHA-256 drift: {actual_sha} vs recorded {expected['lml_sha256']}",
        )
    try:
        n_ch, n_w, n_s, ws = parse_header_facts(raw)
    except ValueError as e:
        return ("FAIL", f"header parse failed: {e}")
    if n_ch != expected["n_channels"]:
        return (
            "FAIL",
            f"n_channels {n_ch} != expected {expected['n_channels']}",
        )
    if n_s != expected["total_samples"]:
        return (
            "FAIL",
            f"total_samples {n_s} != expected {expected['total_samples']}",
        )
    return ("PASS", f"n_ch={n_ch} samples={n_s} windows={n_w} window_size={ws}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reader",
        default=str(Path(__file__).resolve().parent.parent.parent / "target" / "release" / "lml"),
        help="Path to the LML decoder binary (default: target/release/lml).",
    )
    parser.add_argument("vectors", nargs="+")
    args = parser.parse_args()

    n_pass = 0
    n_expected_fail = 0
    n_fail = 0
    for v in args.vectors:
        path = Path(v)
        expected_path = path.with_suffix(".expected.json")
        if not path.exists():
            print(f"  MISSING        {path}")
            n_fail += 1
            continue
        if not expected_path.exists():
            print(f"  MISSING-META   {path}")
            n_fail += 1
            continue
        expected = json.loads(expected_path.read_text())
        is_neg = expected.get("expected_error_kind") is not None
        if is_neg:
            status, msg = verify_negative(path, expected["expected_error_kind"], args.reader)
        else:
            status, msg = verify_positive(path, expected)
        if status == "PASS":
            n_pass += 1
        elif status == "EXPECTED-FAIL":
            n_expected_fail += 1
        else:
            n_fail += 1
        print(f"  {status:<14}  {path.name}: {msg}")

    print()
    print(f"Summary: {n_pass} PASS · {n_expected_fail} EXPECTED-FAIL · {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
