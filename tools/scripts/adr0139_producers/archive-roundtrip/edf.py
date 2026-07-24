#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ADR 0139 P2 archive exact-source-restoration producer.

Emits ONE corpus roundtrip receipt as deterministic JSON on stdout. The corpus
is this file's stem; all corpus producers are byte-identical (one reviewed
source, four names). Each builds a deterministic source fixture, drives it
through the real EDF adapter import/export roundtrip, and proves two things:

* exact source restoration -- ``source_sha256 == restored_sha256``;
* independent semantic equivalence -- a dependency-free EDF/BDF reader written
  from the published header layout re-parses BOTH files and compares signal
  inventory and decoded samples, so the check never reuses the implementation
  under test.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PRODUCER_CONTRACT = "archive-roundtrip"

_CORPORA = ("edf", "edfplus-continuous", "edfplus-discontinuous", "bdf")
_SCHEMA = "lamquant.adr0139.archive-roundtrip-receipt/v1"


def _field(target, start, width, value):
    encoded = value.encode("ascii")
    if len(encoded) > width:
        raise SystemExit(f"field too large: {value}")
    target[start : start + width] = encoded.ljust(width, b" ")


def _edf_fixture(reserved):
    """Build a deterministic EDF/EDF+ fixture; `reserved` selects the flavour."""
    labels = ["EEG Fp1", "AUX", "EDF Annotations"]
    columns = [
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
    _field(output, 0, 8, "0")
    _field(output, 8, 80, "patient")
    _field(output, 88, 80, "recording")
    _field(output, 168, 8, "22.07.26")
    _field(output, 176, 8, "13.00.00")
    _field(output, 184, 8, str(header_len))
    _field(output, 192, 44, reserved)
    _field(output, 236, 8, "2")
    _field(output, 244, 8, "1")
    _field(output, 252, 4, str(len(labels)))
    cursor = 256
    for width, entries in columns:
        for value in entries:
            _field(output, cursor, width, value)
            cursor += width
    discontinuous = reserved.startswith("EDF+D")
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


def _bdf_fixture():
    """Build a deterministic 24-bit BioSemi BDF fixture."""
    output = bytearray(b" " * 512)
    output[0:8] = b"\xffBIOSEMI"
    _field(output, 8, 80, "patient")
    _field(output, 88, 80, "recording")
    _field(output, 168, 8, "22.07.26")
    _field(output, 176, 8, "13.00.00")
    _field(output, 184, 8, "512")
    _field(output, 236, 8, "1")
    _field(output, 244, 8, "1")
    _field(output, 252, 4, "1")
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
        _field(output, cursor, width, value)
        cursor += width
    for value in [-8_388_608, -1, 0, 8_388_607]:
        output.extend((value & 0xFF_FFFF).to_bytes(3, "little"))
    return bytes(output)


def _fixture_for(corpus):
    if corpus == "edf":
        return _edf_fixture(" " * 44)
    if corpus == "edfplus-continuous":
        return _edf_fixture("EDF+C")
    if corpus == "edfplus-discontinuous":
        return _edf_fixture("EDF+D")
    if corpus == "bdf":
        return _bdf_fixture()
    raise SystemExit(f"unknown archive corpus: {corpus}")


def _independent_parse(payload):
    """Parse EDF/EDF+/BDF from the published layout, independent of the codec.

    Returns the signal inventory plus every decoded integer sample. Nothing
    here calls the implementation under test.
    """
    if len(payload) < 256:
        raise SystemExit("truncated biosignal header")
    bdf = payload[0:1] == b"\xff"
    width = 3 if bdf else 2
    header_bytes = int(payload[184:192].decode("ascii").strip())
    records = int(payload[236:244].decode("ascii").strip())
    duration = payload[244:252].decode("ascii").strip()
    signals = int(payload[252:256].decode("ascii").strip())
    if header_bytes != 256 + signals * 256 or len(payload) < header_bytes:
        raise SystemExit("inconsistent biosignal header length")
    base = 256

    def column(width_, index):
        start = base + width_ * index
        return [
            payload[start + width_ * position : start + width_ * (position + 1)]
            .decode("ascii")
            .strip()
            for position in range(signals)
        ]

    labels = column(16, 0)
    offsets = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    cursor = base
    parsed = []
    for size in offsets:
        parsed.append(
            [
                payload[cursor + size * position : cursor + size * (position + 1)]
                .decode("ascii")
                .strip()
                for position in range(signals)
            ]
        )
        cursor += size * signals
    per_record = [int(value) for value in parsed[8]]
    samples = []
    position = header_bytes
    for _ in range(records):
        for index in range(signals):
            channel = []
            for _ in range(per_record[index]):
                chunk = payload[position : position + width]
                if len(chunk) != width:
                    raise SystemExit("truncated biosignal sample data")
                channel.append(int.from_bytes(chunk, "little", signed=not bdf))
                position += width
            samples.append(channel)
    summary = {}
    summary["labels"] = labels
    summary["physical_dimension"] = parsed[2]
    summary["digital_minimum"] = parsed[5]
    summary["digital_maximum"] = parsed[6]
    summary["samples_per_record"] = per_record
    summary["records"] = records
    summary["record_duration"] = duration
    summary["samples"] = samples
    return summary


def _roundtrip(source, restored):
    """Drive the real EDF adapter import/export roundtrip."""
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "lamquant-standard-adapters",
            "--example",
            "edf_profile_probe",
            "--",
            str(source),
            str(restored),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode


def produce_evidence():
    """Roundtrip this corpus and return its exact-restoration receipt."""
    corpus = Path(__file__).stem
    if corpus not in _CORPORA:
        raise SystemExit(f"unknown archive corpus: {corpus}")
    payload = _fixture_for(corpus)
    suffix = ".bdf" if corpus == "bdf" else ".edf"
    with tempfile.TemporaryDirectory(prefix="adr0139-archive-") as temporary:
        directory = Path(temporary)
        source = directory / f"{corpus}{suffix}"
        restored = directory / f"restored-{corpus}{suffix}"
        source.write_bytes(payload)
        code = _roundtrip(source, restored)
        if code != 0 or not restored.is_file():
            raise SystemExit(f"archive roundtrip failed for {corpus}")
        restored_payload = restored.read_bytes()
    source_digest = hashlib.sha256(payload).hexdigest()
    restored_digest = hashlib.sha256(restored_payload).hexdigest()
    independent_source = _independent_parse(payload)
    independent_restored = _independent_parse(restored_payload)
    equivalent = independent_source == independent_restored
    receipt = {}
    receipt["schema"] = _SCHEMA
    receipt["case_id"] = corpus
    receipt["corpus"] = corpus
    receipt["status"] = "pass" if source_digest == restored_digest and equivalent else "fail"
    receipt["source_sha256"] = source_digest
    receipt["restored_sha256"] = restored_digest
    receipt["independent_validator_status"] = "pass" if equivalent else "fail"
    receipt["independent_validator"] = "dependency-free EDF/EDF+/BDF reader"
    receipt["signal_count"] = len(independent_source["labels"])
    return receipt


def main():
    rendered = json.dumps(produce_evidence(), indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
