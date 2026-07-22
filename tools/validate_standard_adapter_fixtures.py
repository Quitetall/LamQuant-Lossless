#!/usr/bin/env python3
"""Run independent validators for the bounded DICOM and NWB fixtures.

The command emits a deterministic JSON receipt on stdout.  Tool banners and
absolute host paths are deliberately excluded from the receipt; caller-owned
evidence manifests bind the executable/package hashes and repository revision.

``dciodvfy`` is known to return success even when it prints ``Error -``.  This
runner therefore fails closed on either a non-zero status or error diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_NWB = (
    REPO
    / "crates"
    / "lamquant-standard-adapters"
    / "tests"
    / "fixtures"
    / "single_integer_timeseries.nwb"
)
DEFAULT_DICOM = (
    REPO
    / "lamquant-lossless"
    / "tests"
    / "fixtures"
    / "dicom"
    / "12lead_ecg.dcm"
)
DEFAULT_BIDS = (
    REPO
    / "crates"
    / "lamquant-standard-adapters"
    / "tests"
    / "fixtures"
    / "bids-single-edf-eeg"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for entry in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = entry.relative_to(path).as_posix().encode("utf-8")
        payload = entry.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def run(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout


def validate_nwb(executable: str, path: Path) -> dict[str, object]:
    return_code, diagnostics = run(
        [executable, "--no-cached-namespace", str(path)]
    )
    errors = [
        line.strip()
        for line in diagnostics.splitlines()
        if "error" in line.lower() and "no errors found" not in line.lower()
    ]
    passed = return_code == 0 and not errors and "no errors found" in diagnostics.lower()
    return {
        "fixture_sha256": sha256(path),
        "profile": "nwb.2.10.0.single-integer-timeseries",
        "validator": "pynwb-validate --no-cached-namespace",
        "return_code": return_code,
        "error_count": len(errors),
        "passed": passed,
    }


def validate_bids(executable: str, path: Path) -> dict[str, object]:
    return_code, diagnostics = run([executable, str(path)])
    error_lines = [
        line.strip()
        for line in diagnostics.splitlines()
        if line.lstrip().startswith("[ERROR]")
    ]
    passed = return_code == 0 and not error_lines and "Summary:" in diagnostics
    return {
        "fixture_sha256": tree_sha256(path),
        "profile": "bids.1.11.1.single-edf-eeg",
        "validator": "bids-validator 3.0.1",
        "return_code": return_code,
        "error_count": len(error_lines),
        "passed": passed,
    }


def inspect_nwb(executable: str, path: Path) -> dict[str, object]:
    return_code, diagnostics = run(
        [
            executable,
            str(path),
            "--threshold",
            "CRITICAL",
            "--progress-bar",
            "False",
            "--detailed",
        ]
    )
    passed = return_code == 0 and "No issues found!" in diagnostics
    return {
        "fixture_sha256": sha256(path),
        "profile": "nwb.2.10.0.single-integer-timeseries",
        "validator": "nwbinspector --threshold CRITICAL",
        "return_code": return_code,
        "critical_issue_count": 0 if passed else None,
        "passed": passed,
    }


def validate_dicom(executable: str, path: Path) -> dict[str, object]:
    return_code, diagnostics = run([executable, str(path)])
    errors = [
        line.strip()
        for line in diagnostics.splitlines()
        if line.startswith("Error -") or " - Error - " in line
    ]
    recognized_iod = "TwelveLeadECG" in diagnostics
    passed = return_code == 0 and not errors and recognized_iod
    return {
        "fixture_sha256": sha256(path),
        "profile": "dicom.ps3.2026c.ecg-i16",
        "validator": "dciodvfy",
        "return_code": return_code,
        "error_count": len(errors),
        "recognized_iod": recognized_iod,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids", type=Path, default=DEFAULT_BIDS)
    parser.add_argument("--nwb", type=Path, default=DEFAULT_NWB)
    parser.add_argument("--dicom", type=Path, default=DEFAULT_DICOM)
    parser.add_argument("--pynwb-validate", default="pynwb-validate")
    parser.add_argument("--nwbinspector", default="nwbinspector")
    parser.add_argument("--dciodvfy", default="dciodvfy")
    parser.add_argument("--bids-validator", default="bids-validator-deno")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = {
        "receipt_version": 1,
        "scope": "bounded-standard-adapter-fixtures",
        "results": [
            validate_bids(args.bids_validator, args.bids.resolve()),
            validate_dicom(args.dciodvfy, args.dicom.resolve()),
            validate_nwb(args.pynwb_validate, args.nwb.resolve()),
            inspect_nwb(args.nwbinspector, args.nwb.resolve()),
        ],
    }
    receipt["passed"] = all(result["passed"] for result in receipt["results"])
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
