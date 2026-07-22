#!/usr/bin/env python3
"""Generate independently valid fixtures for the bounded standards adapters.

This command intentionally covers only the profiles implemented by
``lamquant-standard-adapters`` today: one integer NWB acquisition TimeSeries
and the committed signed-int16 12-lead ECG object.  It does not establish broad
NWB or DICOM conformance.

Create the pinned validation environment from
``tools/standard_adapter_validator_requirements.txt``, then regenerate the
fixtures with its Python interpreter:

    /tmp/abir-validators-20260722/bin/python \
        tools/generate_standard_adapter_fixtures.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydicom
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject


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


def write_nwb(output: Path) -> None:
    """Write the single-series integer fixture accepted by the bounded profile."""

    output.parent.mkdir(parents=True, exist_ok=True)
    nwb = NWBFile(
        session_description="ABIR bounded adapter conformance fixture",
        identifier="abir-adapter-nwb-single-integer-timeseries-v1",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        file_create_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    nwb.subject = Subject(
        subject_id="synthetic-conformance-subject",
        age="P90D",
        sex="U",
        species="Mus musculus",
    )
    nwb.add_acquisition(
        TimeSeries(
            name="ElectricalSeries",
            description="Deterministic integer samples for adapter conformance",
            unit="volts",
            data=np.asarray([[1, 10], [-2, 20], [3, 30], [-4, 40]], dtype="<i2"),
            starting_time=0.0,
            rate=200.0,
        )
    )

    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.stem}.", suffix=".tmp.nwb", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with NWBHDF5IO(temporary_path, "w") as io:
            io.write(nwb)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def repair_dicom(source: Path, output: Path) -> None:
    """Remove two conditional elements rejected by dciodvfy.

    The source object is otherwise kept intact so the existing waveform parser
    goldens continue to exercise the same samples and channel definitions.
    This operation is idempotent and supports an in-place source/output path.
    """

    dataset = pydicom.dcmread(source)
    if "Laterality" in dataset:
        del dataset.Laterality
    for multiplex_group in dataset.WaveformSequence:
        if "MultiplexGroupTimeOffset" in multiplex_group:
            del multiplex_group.MultiplexGroupTimeOffset

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        dataset.save_as(temporary_path, enforce_file_format=True)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb-output", type=Path, default=DEFAULT_NWB)
    parser.add_argument("--dicom-source", type=Path, default=DEFAULT_DICOM)
    parser.add_argument("--dicom-output", type=Path, default=DEFAULT_DICOM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_nwb(args.nwb_output.resolve())
    repair_dicom(args.dicom_source.resolve(), args.dicom_output.resolve())
    print(f"wrote {args.nwb_output}")
    print(f"wrote {args.dicom_output}")


if __name__ == "__main__":
    main()
