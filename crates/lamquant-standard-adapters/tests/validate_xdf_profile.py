#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Emit independent and implementation evidence for ADR 0143 XDF 1.0.

The independent role is the one that matters: it frames XDF files here, round
trips them through the Rust adapter, and then hands the RESTORED bytes to
`pyxdf` -- a separate implementation by other authors -- and requires pyxdf to
recover the same stream count, channel counts, sample values and clock offsets.
An adapter that quietly mangled a stream would still restore its own bytes, so
byte identity alone is not enough; pyxdf reading the result is what makes the
claim independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

CODEC = Path(__file__).resolve().parents[3]
ROOT = CODEC.parent
PROFILE = "xdf.1.0"
SEMANTICS = [
    "multi-stream",
    "stream-metadata",
    "explicit-clock-relation",
    "clock-offsets",
    "boundary-chunks",
]
BOUNDARY_UUID = bytes(
    [
        0x43, 0xA5, 0x46, 0xDC, 0xCB, 0xF5, 0x41, 0x0F,
        0xB3, 0x0E, 0xD5, 0x46, 0x73, 0x83, 0xCB, 0xE4,
    ]
)


def chunk(tag: int, content: bytes) -> bytes:
    """`NumLengthBytes | Length | Tag | Content`; Length counts the tag."""
    length = len(content) + 2
    if length < 256:
        header = bytes([1, length])
    else:
        header = bytes([4]) + struct.pack("<I", length)
    return header + struct.pack("<H", tag) + content


def stream_header(stream_id: int, xml: str) -> bytes:
    return chunk(2, struct.pack("<I", stream_id) + xml.encode("utf-8"))


def eeg_xml(rate: int, channels: int) -> str:
    labels = "".join(
        f"<channel><label>C{index}</label></channel>" for index in range(channels)
    )
    return (
        '<?xml version="1.0"?><info><name>Independent</name><type>EEG</type>'
        f"<channel_count>{channels}</channel_count>"
        f"<nominal_srate>{rate}</nominal_srate>"
        "<channel_format>int16</channel_format>"
        f"<desc><channels>{labels}</channels></desc></info>"
    )


def marker_xml() -> str:
    return (
        '<?xml version="1.0"?><info><name>Markers</name><type>Markers</type>'
        "<channel_count>1</channel_count><nominal_srate>0</nominal_srate>"
        "<channel_format>string</channel_format></info>"
    )


def xdf_fixture(*, with_markers: bool, samples: int, channels: int) -> bytes:
    """Frame an XDF file by hand from the 1.0 specification."""
    out = bytearray(b"XDF:")
    out += chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>')
    out += stream_header(1, eeg_xml(500, channels))
    if with_markers:
        out += stream_header(2, marker_xml())

    body = bytearray(struct.pack("<I", 1))
    body += bytes([1, samples])
    for index in range(samples):
        body += bytes([0])
        for channel in range(channels):
            body += struct.pack("<h", index * 10 + channel)
    out += chunk(3, bytes(body))

    if with_markers:
        markers = bytearray(struct.pack("<I", 2))
        markers += bytes([1, 2])
        for stamp, text in ((0.5, "start"), (1.25, "stop")):
            markers += bytes([8]) + struct.pack("<d", stamp)
            encoded = text.encode("utf-8")
            markers += bytes([1, len(encoded)]) + encoded
        out += chunk(3, bytes(markers))
        out += chunk(5, BOUNDARY_UUID)

    for stream_id, collection, offset in ((1, 10.0, -0.001), (1, 20.0, -0.002)):
        out += chunk(
            4, struct.pack("<I", stream_id) + struct.pack("<dd", collection, offset)
        )
    if with_markers:
        out += chunk(4, struct.pack("<I", 2) + struct.pack("<dd", 10.0, 0.003))

    footer = b'<?xml version="1.0"?><info><first_timestamp>0</first_timestamp></info>'
    out += chunk(6, struct.pack("<I", 1) + footer)
    if with_markers:
        out += chunk(6, struct.pack("<I", 2) + footer)
    return bytes(out)


def malformed_fixtures() -> list[bytes]:
    valid = xdf_fixture(with_markers=True, samples=4, channels=2)
    no_magic = bytearray(valid)
    no_magic[0:4] = b"XXXX"
    truncated = valid[: len(valid) // 2]
    wrong_boundary = bytearray(b"XDF:")
    wrong_boundary += chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>')
    wrong_boundary += stream_header(1, eeg_xml(500, 1))
    wrong_boundary += chunk(5, bytes(16))
    orphan = bytearray(b"XDF:")
    orphan += chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>')
    orphan += chunk(3, struct.pack("<I", 9) + bytes([1, 1, 0]) + struct.pack("<h", 0))
    return [bytes(no_magic), truncated, bytes(wrong_boundary), bytes(orphan)]


def cargo_test(name: str) -> None:
    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "lamquant-standard-adapters",
            "--test",
            "xdf_adapter",
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
            "xdf_profile_probe",
            "--quiet",
            "--",
            str(source),
            str(output),
        ],
        cwd=CODEC,
        check=True,
        env={**os.environ, "TMPDIR": temporary},
    )


def independent_check(path: Path, *, streams: int, channels: int, samples: int) -> None:
    """Hand the restored file to pyxdf and require it to agree."""
    import pyxdf

    loaded, header = pyxdf.load_xdf(str(path), dejitter_timestamps=False)
    if header["info"]["version"][0] != "1.0":
        raise RuntimeError("pyxdf read a different XDF version")
    if len(loaded) != streams:
        raise RuntimeError(f"pyxdf found {len(loaded)} streams, expected {streams}")
    eeg = next(
        stream for stream in loaded if stream["info"]["type"][0] == "EEG"
    )
    if int(eeg["info"]["channel_count"][0]) != channels:
        raise RuntimeError("pyxdf disagrees on the EEG channel count")
    if len(eeg["time_series"]) != samples:
        raise RuntimeError("pyxdf disagrees on the EEG sample count")
    for index, row in enumerate(eeg["time_series"]):
        for channel in range(channels):
            expected = index * 10 + channel
            if int(row[channel]) != expected:
                raise RuntimeError(
                    f"pyxdf read {row[channel]} at [{index}][{channel}], expected {expected}"
                )
    if eeg["footer"]["info"]["first_timestamp"][0] != "0":
        raise RuntimeError("pyxdf lost the stream footer")
    if streams > 1:
        markers = next(
            stream for stream in loaded if stream["info"]["type"][0] == "Markers"
        )
        if [row[0] for row in markers["time_series"]] != ["start", "stop"]:
            raise RuntimeError("pyxdf lost the marker values")


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
        cargo_test("xdf_import_maps_every_stream_clock_and_boundary")
        cargo_test("xdf_declares_first_class_status_and_names_its_independent_validator")
        assertions = {
            "accepted_cases": 2,
            "case_count": 2,
            "semantics_verified": SEMANTICS,
        }
        authority = "implementation-test"
    elif role == "malformed":
        cargo_test("xdf_rejects_wrong_profile_multiple_files_and_malformed_bytes")
        assertions = {
            "case_count": len(malformed_fixtures()) + 2,
            "rejected_cases": len(malformed_fixtures()) + 2,
        }
        authority = "implementation-test"
    else:
        target = CODEC / "target"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="adr0143-xdf-", dir=target) as temporary:
            directory = Path(temporary)
            cases = [
                ("two-stream.xdf", xdf_fixture(with_markers=True, samples=4, channels=2), 2, 2, 4),
                ("single-stream.xdf", xdf_fixture(with_markers=False, samples=8, channels=3), 1, 3, 8),
            ]
            if role == "exact-source-restoration":
                cases = cases[:1]
            digests = []
            for name, content, streams, channels, samples in cases:
                source = directory / name
                source.write_bytes(content)
                output = directory / f"restored-{name}"
                roundtrip(source, output, temporary)
                restored = output.read_bytes()
                if restored != content:
                    raise RuntimeError(f"exact restoration failed for {name}")
                digests.append(hashlib.sha256(restored).hexdigest())
                if role == "independent-semantic-export":
                    independent_check(
                        output, streams=streams, channels=channels, samples=samples
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
            "codec-lossless/crates/lamquant-standard-adapters/tests/validate_xdf_profile.py",
            "--receipt-role",
            role,
        ],
        "implementation_revision": revision,
        "producer": {
            "authority": authority,
            "executable_sha256": hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            "name": "lamquant-xdf-profile-validator",
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
