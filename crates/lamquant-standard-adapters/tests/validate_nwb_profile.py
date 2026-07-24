#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Emit independent and implementation evidence for ADR 0143 NWB 2.10.0.

The independent role writes NWB files with pynwb, round trips them through the
Rust adapter, then re-opens the RESTORED file with pynwb and requires it to
recover the same containers, electrode table, series values and epochs -- and
runs `pynwb.validate` on it. Byte identity alone would not catch an adapter
that mangled a container and then faithfully returned its own capsule; pynwb
reading the result is what makes the claim independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CODEC = Path(__file__).resolve().parents[3]
PROFILE = "nwb.2.10.0"
SEMANTICS = [
    "electrophysiology",
    "time-series",
    "electrodes",
    "intervals",
    "behavior",
    "stimulus",
    "external-assets",
    "derived-data",
]


def write_fixture(path: Path, *, electrodes: int, rows: int) -> None:
    import numpy as np
    from pynwb import NWBFile, NWBHDF5IO, TimeSeries
    from pynwb.behavior import BehavioralTimeSeries
    from pynwb.ecephys import ElectricalSeries
    from pynwb.image import ImageSeries

    nwb = NWBFile(
        session_description="ADR 0143 independent fixture",
        identifier="adr0143-nwb-independent",
        session_start_time=datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc),
    )
    device = nwb.create_device(name="probe")
    group = nwb.create_electrode_group(
        name="shank0", description="fixture", location="cortex", device=device
    )
    for index in range(electrodes):
        nwb.add_electrode(group=group, location=f"site{index}")
    region = nwb.create_electrode_table_region(list(range(electrodes)), "all sites")
    nwb.add_acquisition(
        ElectricalSeries(
            name="ephys",
            data=np.arange(rows * electrodes, dtype=np.int16).reshape(rows, electrodes),
            electrodes=region,
            starting_time=0.0,
            rate=500.0,
        )
    )
    nwb.add_acquisition(
        ImageSeries(
            name="video",
            description="an asset that lives in another file",
            unit="n/a",
            external_file=["session-video.avi"],
            format="external",
            starting_frame=[0],
            starting_time=0.0,
            rate=30.0,
            num_samples=1,
        )
    )
    nwb.add_stimulus(
        TimeSeries(
            name="cue",
            data=np.arange(rows, dtype=np.float32),
            unit="V",
            starting_time=0.0,
            rate=10.0,
        )
    )
    module = nwb.create_processing_module(name="behavior", description="behaviour")
    module.add(
        BehavioralTimeSeries(
            time_series=TimeSeries(
                name="position",
                data=np.linspace(0, 1, rows, dtype=np.float64),
                unit="m",
                timestamps=np.linspace(0.0, 0.1 * (rows - 1), rows),
            )
        )
    )
    derived = nwb.create_processing_module(name="ecephys", description="derived")
    derived.add(
        TimeSeries(
            name="lfp_power",
            data=np.arange(rows, dtype=np.float64),
            unit="V^2",
            starting_time=0.0,
            rate=1.0,
        )
    )
    nwb.add_epoch(0.0, 0.5, ["baseline"])
    nwb.add_epoch(0.5, 1.0, ["task"])
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def cargo_test(name: str) -> None:
    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "lamquant-standard-adapters",
            "--test",
            "nwb_semantic_adapter",
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
            "nwb_semantic_probe",
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


def independent_check(path: Path, *, electrodes: int, rows: int) -> None:
    import numpy as np
    from pynwb import NWBHDF5IO, validate

    with NWBHDF5IO(str(path), "r") as io:
        errors = validate(io=io)
        if errors:
            raise RuntimeError(f"pynwb.validate rejected the restored file: {errors}")
        nwb = io.read()
        ephys = nwb.acquisition["ephys"]
        if ephys.data.shape != (rows, electrodes):
            raise RuntimeError("pynwb disagrees on the acquisition shape")
        expected = np.arange(rows * electrodes, dtype=np.int16).reshape(rows, electrodes)
        if not np.array_equal(np.asarray(ephys.data), expected):
            raise RuntimeError("pynwb read different acquisition samples")
        if len(nwb.electrodes) != electrodes:
            raise RuntimeError("pynwb disagrees on the electrode count")
        if "cue" not in nwb.stimulus:
            raise RuntimeError("pynwb lost the stimulus series")
        if "behavior" not in nwb.processing:
            raise RuntimeError("pynwb lost the behaviour module")
        if len(nwb.epochs) != 2:
            raise RuntimeError("pynwb disagrees on the epoch count")


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
        cargo_test("nwb_import_separates_containers_and_promotes_electrodes_and_intervals")
        cargo_test("nwb_behavior_series_keeps_its_own_timestamps")
        cargo_test("nwb_declares_first_class_status_and_names_its_independent_validator")
        assertions = {
            "accepted_cases": 3,
            "case_count": 3,
            "semantics_verified": SEMANTICS,
        }
        authority = "implementation-test"
    elif role == "malformed":
        cargo_test("nwb_rejects_wrong_profile_multiple_files_and_malformed_bytes")
        assertions = {"case_count": 5, "rejected_cases": 5}
        authority = "implementation-test"
    else:
        target = CODEC / "target"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="adr0143-nwb-", dir=target) as temporary:
            directory = Path(temporary)
            cases = [("wide.nwb", 4, 10), ("narrow.nwb", 2, 6)]
            if role == "exact-source-restoration":
                cases = cases[:1]
            digests = []
            for name, electrodes, rows in cases:
                source = directory / name
                write_fixture(source, electrodes=electrodes, rows=rows)
                output = directory / f"restored-{name}"
                measured = roundtrip(source, output, temporary)
                if output.read_bytes() != source.read_bytes():
                    raise RuntimeError(f"exact restoration failed for {name}")
                if measured["electrodes"] != electrodes:
                    raise RuntimeError("the adapter disagrees on the electrode count")
                if measured["series"] != 4 or measured["streams"] != 4:
                    raise RuntimeError("the adapter did not separate all four containers")
                if measured["external_assets"] != 1:
                    raise RuntimeError("the adapter did not name the external asset")
                if measured["intervals"] != 2:
                    raise RuntimeError("the adapter disagrees on the epoch count")
                digests.append(hashlib.sha256(output.read_bytes()).hexdigest())
                if role == "independent-semantic-export":
                    independent_check(output, electrodes=electrodes, rows=rows)
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
            "codec-lossless/crates/lamquant-standard-adapters/tests/validate_nwb_profile.py",
            "--receipt-role",
            role,
        ],
        "implementation_revision": revision,
        "producer": {
            "authority": authority,
            "executable_sha256": hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            "name": "lamquant-nwb-profile-validator",
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
