#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ADR 0139 P2 archive malformed-fixture rejection producer.

Emits ONE malformed-fixture receipt as deterministic JSON on stdout. The
fixture family is this file's stem; all malformed producers are byte-identical
(one reviewed source, several names). Each corrupts a known-good EDF fixture in
one specific way and requires the real adapter to REJECT it: a fail-closed
reader must never import a structurally invalid archive, because silently
accepting one would fabricate provenance for data that was never present.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PRODUCER_CONTRACT = "archive-malformed"

_CASES = (
    "truncated-header",
    "truncated-data",
    "signal-count-overflow",
    "non-numeric-header-field",
)
_SCHEMA = "lamquant.adr0139.archive-malformed-receipt/v1"


def _field(target, start, width, value):
    encoded = value.encode("ascii")
    if len(encoded) > width:
        raise SystemExit(f"field too large: {value}")
    target[start : start + width] = encoded.ljust(width, b" ")


def _valid_edf():
    """Build the known-good EDF+C fixture that each case then corrupts."""
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
    _field(output, 192, 44, "EDF+C")
    _field(output, 236, 8, "2")
    _field(output, 244, 8, "1")
    _field(output, 252, 4, str(len(labels)))
    cursor = 256
    for width, entries in columns:
        for value in entries:
            _field(output, cursor, width, value)
            cursor += width
    for record in range(2):
        eeg = [1, 2, 3, 4] if record == 0 else [5, 6, 7, 8]
        aux = [-3, 4] if record == 0 else [-5, 6]
        for value in eeg + aux:
            output.extend(int(value).to_bytes(2, "little", signed=True))
        tal = b"+0\x14\x14\0" if record == 0 else b"+1\x14\x14\0"
        annotation = bytearray(128)
        annotation[: len(tal)] = tal
        output.extend(annotation)
    return bytearray(output)


def _corrupt(case):
    """Return one specifically malformed archive payload."""
    payload = _valid_edf()
    if case == "truncated-header":
        # Cut the signal header table in half; declared length no longer fits.
        return bytes(payload[:400])
    if case == "truncated-data":
        # Keep a valid header but drop most declared data records.
        return bytes(payload[: 256 + 3 * 256 + 8])
    if case == "signal-count-overflow":
        # Claim far more signals than the header table can hold.
        _field(payload, 252, 4, "999")
        return bytes(payload)
    if case == "non-numeric-header-field":
        # Corrupt the record-count field into an unparsable value.
        _field(payload, 236, 8, "not-num")
        return bytes(payload)
    raise SystemExit(f"unknown malformed case: {case}")


def _import_attempt(source, output):
    """Run the real adapter import; a nonzero exit is a correct rejection."""
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
            str(output),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode


def produce_evidence():
    """Corrupt one fixture and return its rejection receipt."""
    case = Path(__file__).stem
    if case not in _CASES:
        raise SystemExit(f"unknown malformed case: {case}")
    payload = _corrupt(case)
    with tempfile.TemporaryDirectory(prefix="adr0139-malformed-") as temporary:
        directory = Path(temporary)
        source = directory / f"{case}.edf"
        output = directory / f"imported-{case}.edf"
        source.write_bytes(payload)
        code = _import_attempt(source, output)
        produced = output.is_file()
    rejected = code != 0 and not produced
    receipt = {}
    receipt["schema"] = _SCHEMA
    receipt["case_id"] = case
    receipt["status"] = "pass" if rejected else "fail"
    receipt["rejected"] = rejected
    receipt["fixture_sha256"] = hashlib.sha256(payload).hexdigest()
    receipt["fixture_bytes"] = len(payload)
    receipt["adapter_exit_code"] = code
    return receipt


def main():
    rendered = json.dumps(produce_evidence(), indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
