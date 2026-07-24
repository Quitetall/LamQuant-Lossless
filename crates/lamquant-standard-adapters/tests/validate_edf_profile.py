#!/usr/bin/env python3
"""Emit independent and implementation evidence for ADR 0143 EDF/EDF+/BDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


CODEC = Path(__file__).resolve().parents[3]
ROOT = CODEC.parent
PROFILE = "edfplus.1"
SEMANTICS = [
    "edf-signal",
    "edfplus-continuous",
    "edfplus-discontinuous",
    "bdf-signal",
    "calibration",
    "recording-time",
    "annotations",
    "off-rate-channels",
]


def field(target: bytearray, start: int, width: int, value: str) -> None:
    encoded = value.encode("ascii")
    if len(encoded) > width:
        raise ValueError(f"field too large: {value}")
    target[start : start + width] = encoded.ljust(width, b" ")


def edf_fixture(discontinuous: bool) -> bytes:
    labels = ["EEG Fp1", "AUX", "EDF Annotations"]
    values = [
        (16, labels),
        (80, ["", "", ""]),
        (8, ["uV", "mV", ""]),
        (8, ["-100", "-10", "-1"]),
        (8, ["100", "10", "1"]),
        (8, ["-32768", "-32768", "-32768"]),
        (8, ["32767", "32767", "32767"]),
        (80, ["", "", ""]),
        (8, ["4", "2", "64"]),
        (32, ["", "", ""]),
    ]
    header_len = 256 + len(labels) * 256
    output = bytearray(b" " * header_len)
    field(output, 0, 8, "0")
    field(output, 8, 80, "patient")
    field(output, 88, 80, "recording")
    field(output, 168, 8, "22.07.26")
    field(output, 176, 8, "13.00.00")
    field(output, 184, 8, str(header_len))
    field(output, 192, 44, "EDF+D" if discontinuous else "EDF+C")
    field(output, 236, 8, "2")
    field(output, 244, 8, "1")
    field(output, 252, 4, "3")
    cursor = 256
    for width, entries in values:
        for value in entries:
            field(output, cursor, width, value)
            cursor += width
    for record in range(2):
        eeg = [1, 2, 3, 4] if record == 0 else [5, 6, 7, 8]
        aux = [-3, 4] if record == 0 else [-5, 6]
        for value in eeg + aux:
            output.extend(int(value).to_bytes(2, "little", signed=True))
        if record == 0:
            tal = b"+0\x14\x14\0+0.5\x151\x14event A\x14\0"
        elif discontinuous:
            tal = b"+2\x14\x14\0+2.25\x14event B\x14\0"
        else:
            tal = b"+1\x14\x14\0+1.25\x14event B\x14\0"
        annotation = bytearray(128)
        annotation[: len(tal)] = tal
        output.extend(annotation)
    return bytes(output)


def bdf_fixture() -> bytes:
    output = bytearray(b" " * 512)
    output[0:8] = b"\xffBIOSEMI"
    field(output, 8, 80, "patient")
    field(output, 88, 80, "recording")
    field(output, 168, 8, "22.07.26")
    field(output, 176, 8, "13.00.00")
    field(output, 184, 8, "512")
    field(output, 236, 8, "1")
    field(output, 244, 8, "1")
    field(output, 252, 4, "1")
    cursor = 256
    for width, value in [
        (16, "EEG Cz"),
        (80, ""),
        (8, "uV"),
        (8, "-100"),
        (8, "100"),
        (8, "-8388608"),
        (8, "8388607"),
        (80, ""),
        (8, "4"),
        (32, ""),
    ]:
        field(output, cursor, width, value)
        cursor += width
    for value in [-8_388_608, -1, 0, 8_388_607]:
        output.extend((value & 0xFF_FFFF).to_bytes(3, "little"))
    return bytes(output)


def cargo_test(name: str) -> None:
    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "lamquant-standard-adapters",
            "--test",
            "edf_adapter",
            name,
            "--",
            "--exact",
        ],
        cwd=CODEC,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def roundtrip(source: Path, output: Path, temporary: str) -> None:
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "lamquant-standard-adapters",
            "--example",
            "edf_profile_probe",
            "--quiet",
            "--",
            str(source),
            str(output),
        ],
        cwd=CODEC,
        check=True,
        env={**os.environ, "TMPDIR": temporary},
    )


def independent_cases() -> list[tuple[str, bytes]]:
    return [("continuous.edf", edf_fixture(False)), ("discontinuous.edf", edf_fixture(True)), ("signal.bdf", bdf_fixture())]


def write_independent_cases(directory: Path) -> list[Path]:
    import numpy as np
    import pyedflib
    from pyedflib import highlevel

    header = highlevel.make_header(patientcode="independent", recording_additional="ADR0143")
    header["annotations"] = [(0.5, 1.0, "\u00e9v\u00e9nement A"), (1.25, 0.0, "event B")]
    edf_path = directory / "pyedflib-continuous.edf"
    highlevel.write_edf(
        str(edf_path),
        [np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32), np.array([-3, 4, -5, 6], dtype=np.int32)],
        [
            highlevel.make_signal_header("EEG Fp1", sample_frequency=4, physical_min=-100, physical_max=100),
            highlevel.make_signal_header("AUX", dimension="mV", sample_frequency=2, physical_min=-10, physical_max=10),
        ],
        header=header,
        digital=True,
        file_type=pyedflib.FILETYPE_EDFPLUS,
    )
    bdf_path = directory / "pyedflib-signal.bdf"
    highlevel.write_edf(
        str(bdf_path),
        [np.array([-8_388_608, -1, 0, 8_388_607], dtype=np.int32)],
        [
            highlevel.make_signal_header(
                "EEG Cz",
                sample_frequency=4,
                physical_min=-100,
                physical_max=100,
                digital_min=-8_388_608,
                digital_max=8_388_607,
            )
        ],
        header=highlevel.make_header(patientcode="independent"),
        digital=True,
        file_type=pyedflib.FILETYPE_BDFPLUS,
    )
    return [edf_path, bdf_path]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt-role",
        required=True,
        choices=("positive", "malformed", "exact-source-restoration", "independent-semantic-export"),
    )
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()
    if len(args.implementation_revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.implementation_revision
    ):
        parser.error("--implementation-revision must be a lowercase 40-character hash")

    role = args.receipt_role
    if role == "positive":
        cargo_test("edf_import_maps_samples_and_restores_exact_source")
        cargo_test("edfplus_discontinuous_annotations_off_rate_and_calibration_are_semantic")
        cargo_test("bdf_signed_24_bit_samples_are_promoted_exactly")
        assertions = {"accepted_cases": 3, "case_count": 3, "semantics_verified": SEMANTICS}
        authority = "implementation-test"
    elif role == "malformed":
        cargo_test("edf_rejects_wrong_profile_multiple_files_and_malformed_bytes")
        assertions = {"case_count": 4, "rejected_cases": 4}
        authority = "implementation-test"
    else:
        with tempfile.TemporaryDirectory(prefix="adr0143-edf-", dir=CODEC / "target") as temporary:
            directory = Path(temporary)
            cases = independent_cases()
            selected = []
            if role == "independent-semantic-export":
                selected = [(path.name, path.read_bytes()) for path in write_independent_cases(directory)]
            else:
                selected = [cases[0]]
            for filename, content in selected:
                source = directory / filename
                output = directory / f"restored-{filename}"
                if not source.exists():
                    source.write_bytes(content)
                roundtrip(source, output, temporary)
                if source.read_bytes() != output.read_bytes():
                    raise RuntimeError(f"exact restoration failed for {filename}")
                if role == "independent-semantic-export":
                    import pyedflib

                    with pyedflib.EdfReader(str(output)) as reader:
                        if reader.signals_in_file < 1 or reader.getNSamples().min() < 4:
                            raise RuntimeError(f"pyedflib rejected semantic content in {filename}")
            if role == "exact-source-restoration":
                source_hash = hashlib.sha256(selected[0][1]).hexdigest()
                assertions = {
                    "case_count": 1,
                    "exact_source_restoration": True,
                    "output_sha256": source_hash,
                    "source_sha256": source_hash,
                }
                authority = "implementation-test"
            else:
                assertions = {
                    "case_count": len(selected),
                    "independently_validated": True,
                    "validator_cases": len(selected),
                }
                authority = "independent"

    executable_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = {
        "assertions": assertions,
        "command": [
            "codec-lossless/crates/lamquant-standard-adapters/tests/validate_edf_profile.py",
            "--receipt-role",
            role,
            "--implementation-revision",
            args.implementation_revision,
        ],
        "implementation_revision": args.implementation_revision,
        "producer": {
            "authority": authority,
            "executable_sha256": executable_hash,
            "name": "lamquant-edf-profile-validator",
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
