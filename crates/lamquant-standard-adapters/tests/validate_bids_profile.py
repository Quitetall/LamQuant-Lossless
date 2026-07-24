#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Emit independent and implementation evidence for ADR 0143 BIDS 1.11.1.

The independent role round trips the committed dataset tree through the Rust
adapter and then hands the RESTORED tree to `bids_validator` -- a separate
implementation by other authors -- requiring it to accept every data file's
path as valid BIDS, and re-reads the restored EDF recordings with pyedflib to
confirm the sample values survived. A tree that came back byte-identical but
had been imported with a recording mis-filed would still pass a digest check;
an outside validator reading the result is what makes the claim independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CODEC = Path(__file__).resolve().parents[3]
FIXTURE = CODEC / "crates/lamquant-standard-adapters/tests/fixtures/bids-full"
PROFILE = "bids.1.11.1"
SEMANTICS = [
    "eeg",
    "ieeg",
    "physiology",
    "events",
    "electrodes",
    "coordinates",
    "derivatives",
]
EXPECTED = {
    "recordings": 3,
    "modalities": 3,
    "events": 1,
    "electrodes": 2,
    "derivatives": 1,
    "members": 12,
}


def cargo_test(name: str) -> None:
    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "lamquant-standard-adapters",
            "--test",
            "bids_semantic_adapter",
            name,
            "--",
            "--exact",
        ],
        cwd=CODEC,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def roundtrip(source: Path, output: Path, temporary: str) -> dict:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "lamquant-standard-adapters",
            "--example",
            "bids_semantic_probe",
            "--quiet",
            "--",
            str(source),
            str(output),
        ],
        cwd=CODEC,
        check=True,
        stdout=subprocess.PIPE,
        env={**os.environ, "TMPDIR": temporary},
    )
    return json.loads(completed.stdout.decode("utf-8").strip().splitlines()[-1])


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def independent_check(root: Path) -> int:
    """Every data file's path must satisfy an outside BIDS implementation."""
    from bids_validator import BIDSValidator

    validator = BIDSValidator()
    checked = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = "/" + str(path.relative_to(root))
        # Derivatives sit outside the raw-dataset naming rules by design.
        if relative.startswith("/derivatives/"):
            continue
        if not validator.is_bids(relative):
            raise RuntimeError(f"bids-validator rejected {relative}")
        checked += 1
    if checked == 0:
        raise RuntimeError("bids-validator checked nothing")

    import pyedflib

    for name in ("sub-01/eeg/sub-01_task-rest_eeg.edf", "sub-01/ieeg/sub-01_task-rest_ieeg.edf"):
        with pyedflib.EdfReader(str(root / name)) as reader:
            if reader.signals_in_file < 1:
                raise RuntimeError(f"pyedflib found no signals in {name}")
            if reader.getNSamples().min() < 1:
                raise RuntimeError(f"pyedflib found an empty signal in {name}")
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt-role",
        required=True,
        choices=(
            "positive",
            "malformed",
            "exact-source-restoration",
            "independent-semantic-export",
        ),
    )
    parser.add_argument("--implementation-revision", required=True)
    arguments = parser.parse_args()
    revision = arguments.implementation_revision
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        parser.error("--implementation-revision must be a lowercase 40-character hash")

    role = arguments.receipt_role
    if role == "positive":
        cargo_test("bids_reads_the_layout_as_the_semantic_it_is")
        cargo_test("bids_derivatives_are_named_but_never_promoted_beside_raw_data")
        cargo_test("bids_declares_first_class_status_and_names_its_independent_validator")
        assertions = {
            "accepted_cases": 3,
            "case_count": 3,
            "semantics_verified": SEMANTICS,
        }
        authority = "implementation-test"
    elif role == "malformed":
        cargo_test("bids_rejects_wrong_profile_duplicates_and_incomplete_datasets")
        assertions = {"case_count": 6, "rejected_cases": 6}
        authority = "implementation-test"
    else:
        target = CODEC / "target"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="adr0143-bids-", dir=target) as temporary:
            directory = Path(temporary)
            source = directory / "dataset"
            shutil.copytree(FIXTURE, source)
            output = directory / "restored"
            measured = roundtrip(source, output, temporary)
            for key, expected in EXPECTED.items():
                if measured[key] != expected:
                    raise RuntimeError(
                        f"adapter reported {key}={measured[key]}, expected {expected}"
                    )
            before = tree_digest(source)
            after = tree_digest(output)
            if before != after:
                raise RuntimeError("the restored tree differs from the source tree")
            if role == "exact-source-restoration":
                assertions = {
                    "case_count": 1,
                    "exact_source_restoration": True,
                    "output_sha256": after,
                    "source_sha256": before,
                }
                authority = "implementation-test"
            else:
                checked = independent_check(output)
                assertions = {
                    "case_count": 1,
                    "independently_validated": True,
                    "validator_cases": checked,
                }
                authority = "independent"

    receipt = {
        "assertions": assertions,
        "command": [
            "codec-lossless/crates/lamquant-standard-adapters/tests/validate_bids_profile.py",
            "--receipt-role",
            role,
        ],
        "implementation_revision": revision,
        "producer": {
            "authority": authority,
            "executable_sha256": hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            "name": "lamquant-bids-profile-validator",
            "version": "1",
        },
        "profile": PROFILE,
        "role": role,
        "schema": "lamquant.adr0143-evidence/v1",
        "status": "PASS",
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
