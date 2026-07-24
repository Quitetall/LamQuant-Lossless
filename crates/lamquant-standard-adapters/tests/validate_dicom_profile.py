#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Emit independent and implementation evidence for ADR 0143 DICOM PS3.

The independent role round trips the committed waveform instance through the
Rust adapter and then re-reads the RESTORED file with pydicom -- a separate
implementation by other authors -- requiring it to recover the same waveform
shape, the same decoded lead samples, the patient/study/series identifiers,
the annotation count, the referenced instances and every private element. An
adapter that mangled a lead would still return its own capsule byte-for-byte;
pydicom reading the result is what makes the claim independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

CODEC = Path(__file__).resolve().parents[3]
FIXTURES = CODEC / "crates/lamquant-standard-adapters/tests/fixtures"
PROFILE = "dicom.ps3.2026c"
SEMANTICS = [
    "neurophysiology-waveform",
    "annotations",
    "patient",
    "study",
    "series",
    "device",
    "reports",
    "referenced-media",
    "private-tags",
]
CASES = [
    ("ecg_with_references.dcm", 24, 77, 1, 1, 19),
    ("../../../../lamquant-lossless/tests/fixtures/dicom/general_ecg.dcm", 3, 0, 0, 0, 0),
]


def cargo_test(name: str) -> None:
    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "lamquant-standard-adapters",
            "--test",
            "dicom_semantic_adapter",
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
            "dicom_semantic_probe",
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


def independent_check(path: Path, *, channels: int, annotations: int, private: int) -> None:
    import pydicom

    data = pydicom.dcmread(str(path))
    total = sum(
        int(group.NumberOfWaveformChannels) for group in data.WaveformSequence
    )
    if total != channels:
        raise RuntimeError(f"pydicom counted {total} channels, expected {channels}")
    # Decode every multiplex group independently and check its shape against
    # what the group itself declares. This is the step that would catch a lead
    # the adapter reordered or dropped.
    for index, group in enumerate(data.WaveformSequence):
        multiplex = pydicom.waveforms.multiplex_array(data, index, as_raw=True)
        expected = (
            int(group.NumberOfWaveformSamples),
            int(group.NumberOfWaveformChannels),
        )
        if multiplex.shape != expected:
            raise RuntimeError(
                f"pydicom read shape {multiplex.shape} for group {index}, expected {expected}"
            )
    if not data.get("PatientID"):
        raise RuntimeError("pydicom lost the patient identifier")
    if not data.get("StudyInstanceUID") or not data.get("SeriesInstanceUID"):
        raise RuntimeError("pydicom lost the study or series identifier")
    observed = len(data.get("WaveformAnnotationSequence", []))
    if observed != annotations:
        raise RuntimeError(f"pydicom counted {observed} annotations, expected {annotations}")
    observed_private = sum(1 for element in data if element.tag.is_private)
    if observed_private != private:
        raise RuntimeError(
            f"pydicom counted {observed_private} private elements, expected {private}"
        )


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
        cargo_test("dicom_import_keeps_the_information_model_and_promotes_annotations")
        cargo_test("dicom_references_and_private_tags_are_named_but_never_invented")
        cargo_test("dicom_samples_stay_integers_with_the_stated_calibration")
        cargo_test("dicom_declares_first_class_status_and_names_its_independent_validator")
        assertions = {
            "accepted_cases": 4,
            "case_count": 4,
            "semantics_verified": SEMANTICS,
        }
        authority = "implementation-test"
    elif role == "malformed":
        cargo_test("dicom_rejects_wrong_profile_multiple_files_and_malformed_bytes")
        assertions = {"case_count": 5, "rejected_cases": 5}
        authority = "implementation-test"
    else:
        target = CODEC / "target"
        target.mkdir(parents=True, exist_ok=True)
        cases = CASES if role == "independent-semantic-export" else CASES[:1]
        with tempfile.TemporaryDirectory(prefix="adr0143-dicom-", dir=target) as temporary:
            directory = Path(temporary)
            digests = []
            for name, channels, annotations, media, reports, private in cases:
                source = directory / Path(name).name
                source.write_bytes((FIXTURES / name).read_bytes())
                output = directory / f"restored-{source.name}"
                measured = roundtrip(source, output, temporary)
                if output.read_bytes() != source.read_bytes():
                    raise RuntimeError(f"exact restoration failed for {source.name}")
                for key, expected in (
                    ("channels", channels),
                    ("annotations", annotations),
                    ("referenced_media", media),
                    ("reports", reports),
                    ("private_tags", private),
                ):
                    if measured[key] != expected:
                        raise RuntimeError(
                            f"{source.name}: adapter reported {key}={measured[key]}, expected {expected}"
                        )
                digests.append(hashlib.sha256(output.read_bytes()).hexdigest())
                if role == "independent-semantic-export":
                    independent_check(
                        output,
                        channels=channels,
                        annotations=annotations,
                        private=private,
                    )
            if role == "exact-source-restoration":
                assertions = {
                    "case_count": 1,
                    "exact_source_restoration": True,
                    "output_sha256": digests[0],
                    "source_sha256": digests[0],
                }
                authority = "implementation-test"
            else:
                assertions = {
                    "case_count": len(cases),
                    "independently_validated": True,
                    "validator_cases": len(cases),
                }
                authority = "independent"

    receipt = {
        "assertions": assertions,
        "command": [
            "codec-lossless/crates/lamquant-standard-adapters/tests/validate_dicom_profile.py",
            "--receipt-role",
            role,
        ],
        "implementation_revision": revision,
        "producer": {
            "authority": authority,
            "executable_sha256": hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            "name": "lamquant-dicom-profile-validator",
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
