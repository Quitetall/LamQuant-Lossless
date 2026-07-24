#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ADR 0139 P3 cross-backend byte-equality producer.

Emits ONE conformance-vector receipt as deterministic JSON on stdout. The vector
is this file's stem; all vector producers are byte-identical (one reviewed
source, one name per vector). Each drives the real codec, encoding the vector
with `ComputeBackend::Firmware` and again with `ComputeBackend::Desktop`, and
records both SHA-256 digests. The `.lml` wire format is backend-independent by
contract, so a divergence is a wire-format change rather than an optimisation.

The encode runs in an isolated workspace assembled from this repository's own
backend crates (`lamquant-common`, `lamquant-lml-mcu`, `lamquant-lml-desktop`).
None of those path-depend outside the repository, so the evidence stays
repository-local even though the full host workspace does not build in
isolation.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PRODUCER_CONTRACT = "compute-byte-equality"

_VECTORS = ("1ch_100", "4ch_2500", "32ch_2500")
_SCHEMA = "lamquant.adr0139.compute-byte-equality-receipt/v1"
# Members copied into the isolated workspace, in dependency order.
_MEMBERS = (
    Path("crates/lamquant-common"),
    Path("lamquant-lml-mcu"),
    Path("lamquant-lml-desktop"),
)
_MANIFEST = """[workspace]
resolver = "2"
members = [{members}]
"""


def _assemble_workspace(destination):
    """Copy the self-contained backend crates into a standalone workspace."""
    for member in _MEMBERS:
        source = Path(member)
        if not (source / "Cargo.toml").is_file():
            raise SystemExit(f"missing backend crate: {member}")
        shutil.copytree(source, destination / member)
    members = ", ".join(f'"{member.as_posix()}"' for member in _MEMBERS)
    (destination / "Cargo.toml").write_text(
        _MANIFEST.format(members=members), encoding="utf-8"
    )


def _encode_both_backends(vector, workspace):
    """Run the real codec once per backend and return its reported digests."""
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "lamquant-lml-desktop",
            "--example",
            "backend_byte_equality",
            "--",
            vector,
        ],
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"byte-equality encode failed for {vector}")
    return json.loads(completed.stdout)


def produce_evidence():
    """Encode this vector on both backends and return its equality receipt."""
    vector = Path(__file__).stem
    if vector not in _VECTORS:
        raise SystemExit(f"unknown conformance vector: {vector}")
    with tempfile.TemporaryDirectory(prefix="adr0139-byte-equality-") as temporary:
        workspace = Path(temporary) / "workspace"
        workspace.mkdir()
        _assemble_workspace(workspace)
        measured = _encode_both_backends(vector, workspace)
    firmware = measured["firmware_sha256"]
    desktop = measured["desktop_sha256"]
    receipt = {}
    receipt["schema"] = _SCHEMA
    receipt["case_id"] = vector
    receipt["status"] = "pass" if firmware == desktop else "fail"
    receipt["firmware_sha256"] = firmware
    receipt["desktop_sha256"] = desktop
    receipt["channels"] = measured["channels"]
    receipt["samples"] = measured["samples"]
    receipt["encoded_bytes"] = measured["firmware_bytes"]
    return receipt


def main():
    rendered = json.dumps(produce_evidence(), indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
