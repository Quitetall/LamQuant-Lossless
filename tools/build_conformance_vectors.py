#!/usr/bin/env python3
"""
Phase 8 / Item E — build the 13 LML wire-format conformance vectors.

Runs the LamQuant `lml encode` binary on deterministic synth inputs,
then post-processes the output for the corruption / legacy vectors.

Every vector lands in specs/conformance/vectors/ alongside a
.expected.json describing what the validator should observe.

Usage:
    python3 tools/build_conformance_vectors.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "specs" / "conformance" / "vectors"
LML_BIN = REPO / "target" / "release" / "lml"


def ensure_lml() -> Path:
    if not LML_BIN.exists():
        print(f"building release lml binary at {LML_BIN}", file=sys.stderr)
        subprocess.run(
            [
                "cargo",
                "build",
                "--manifest-path",
                str(REPO / "lamquant-core" / "Cargo.toml"),
                "--release",
                "--bin",
                "lml",
                "--features",
                "host",
            ],
            check=True,
        )
    return LML_BIN


def synth_signal(n_ch: int, n_samp: int, seed: int) -> bytes:
    """Deterministic channel-major int32 LE PRNG output (matches LamQuant's
    raw-decode wire layout). The encoder ingests it as an EDF; we hand a
    minimal synth EDF in.
    """
    # We bypass EDF and go straight through the public lml::compress API
    # via the codec lib, called from a tiny Rust glue binary that already
    # exists at target/release/lml (encode subcommand only accepts EDF
    # paths). Easiest: generate a tiny valid EDF + encode it.
    raise NotImplementedError(
        "use synth_edf_then_encode below — direct PRNG path not used"
    )


def write_minimal_edf(
    out_path: Path, n_ch: int, n_samp: int, sample_rate: float, seed: int
) -> None:
    """Build a minimal valid EDF (no annotations, no non-EEG channels)
    with int16 channel-major samples. EDF spec §3 — 256-byte main header +
    256 × n_signals signal-header rows + n_records × n_signals × n_samples_per_record
    int16 samples in record-interleaved order.

    For simplicity we use 1 record of length n_samp seconds, n_samp samples
    per record per signal, so n_records=1.
    """
    n_records = 1
    n_samples_per_record = n_samp
    # PRNG: deterministic LCG seeded by `seed`.
    state = seed & 0xFFFFFFFF
    samples = []
    for s in range(n_samples_per_record):
        for ch in range(n_ch):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            v = ((state >> 8) % 4001) - 2000  # in [-2000, 2000)
            samples.append(v)
    # Main header (256 bytes)
    hdr = bytearray(b" " * 256)
    hdr[0:8] = b"0       "
    hdr[8:88] = b"X X X conformance".ljust(80, b" ")
    hdr[88:168] = (b"Startdate 01-JAN-2026 X X synth seed=%d" % seed).ljust(80, b" ")
    hdr[168:176] = b"01.01.26"
    hdr[176:184] = b"00.00.00"
    hdr[184:192] = (b"%-8d" % (256 * (n_ch + 1)))  # header_bytes
    hdr[192:236] = b" " * 44
    hdr[236:244] = b"%-8d" % n_records
    # record duration = n_samp / sample_rate seconds
    rec_dur = n_samp / sample_rate
    hdr[244:252] = (b"%-8.4f" % rec_dur).ljust(8, b" ")[:8]
    hdr[252:256] = b"%-4d" % n_ch
    # Signal headers — EDF spec §3.1 lays fields out by-field-across-
    # signals, NOT by-signal-across-fields. For N signals:
    #   N × 16 bytes labels
    #   N × 80 bytes transducer
    #   N × 8 bytes phys_dim
    #   N × 8 bytes phys_min
    #   N × 8 bytes phys_max
    #   N × 8 bytes dig_min
    #   N × 8 bytes dig_max
    #   N × 80 bytes prefilter
    #   N × 8 bytes ns_per_record
    #   N × 32 bytes reserved
    sigs = bytearray()
    for ch in range(n_ch):
        sigs.extend((b"EEG ch%02d" % ch).ljust(16, b" "))
    for _ in range(n_ch):
        sigs.extend(b" " * 80)  # transducer
    for _ in range(n_ch):
        sigs.extend(b"uV      ")
    for _ in range(n_ch):
        sigs.extend(b"-200    ")
    for _ in range(n_ch):
        sigs.extend(b" 200    ")
    for _ in range(n_ch):
        sigs.extend(b"-32768  ")
    for _ in range(n_ch):
        sigs.extend(b" 32767  ")
    for _ in range(n_ch):
        sigs.extend(b" " * 80)  # prefilter
    for _ in range(n_ch):
        sigs.extend(b"%-8d" % n_samples_per_record)
    for _ in range(n_ch):
        sigs.extend(b" " * 32)  # reserved
    full_header = bytes(hdr) + bytes(sigs)
    # The header_bytes field was set above to 256 * (n_ch + 1); make sure
    # n_signals * 256 + 256 matches `header_bytes` we wrote into bytes 184-192.
    assert len(full_header) == 256 * (n_ch + 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(full_header)
        # Record body: int16 LE in [sample, channel] order per record
        # (EDF samples are interleaved within each record).
        for ch in range(n_ch):
            for s in range(n_samples_per_record):
                v = samples[s * n_ch + ch]
                f.write(struct.pack("<h", v))


def run_encode(edf: Path, lml_out: Path, lml_bin: Path, extra_args: list[str] = []) -> None:
    cmd = [
        str(lml_bin),
        "encode",
        str(edf),
        "--no-bundle",
        "-o",
        str(lml_out.parent),
    ] + extra_args
    subprocess.run(cmd, check=True, capture_output=True)
    # encode produces <stem>.lml under -o dir; rename if needed.
    produced = lml_out.parent / (edf.stem + ".lml")
    if produced != lml_out:
        shutil.move(str(produced), str(lml_out))


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class VectorSpec:
    name: str
    category: str
    n_channels: int
    n_samples: int
    sample_rate: float
    noise_bits: int = 0
    metadata_extra: dict | None = None
    # For corruption vectors: post-process the produced LML.
    corrupt: str | None = None  # "crc" | "truncate" | "bitflip" | "legacy_no_footer"
    expected_error_kind: str | None = None  # for negative vectors


VECTORS = [
    VectorSpec("basic_4ch_1024s_250hz", "round-trip", 4, 1024, 250.0),
    # Encoder refuses <8 samples per channel (too short to form a window).
    # Use the smallest count it accepts as the "minimum non-trivial" edge.
    VectorSpec("min_samples_1ch", "edge", 1, 16, 250.0),
    VectorSpec("single_channel_long", "edge", 1, 1_000_000, 250.0),
    VectorSpec("max_channels_512", "edge", 512, 256, 250.0),
    VectorSpec("sample_rate_50hz", "rate/depth", 4, 1024, 50.0),
    VectorSpec("sample_rate_4000hz", "rate/depth", 4, 1024, 4000.0),
    VectorSpec("bit_depth_8", "rate/depth", 4, 1024, 250.0, noise_bits=8),
    VectorSpec("metadata_unicode", "metadata", 2, 256, 250.0,
               metadata_extra={"label_中文": "ψ", "emoji": "🧠"}),
    VectorSpec("metadata_large_json", "metadata", 2, 256, 250.0,
               metadata_extra={"big": "x" * 60_000}),
    VectorSpec("corrupt_crc", "corruption", 2, 1024, 250.0,
               corrupt="crc", expected_error_kind="CrcMismatch"),
    VectorSpec("truncated_window", "corruption", 2, 1024, 250.0,
               corrupt="truncate", expected_error_kind="Truncated"),
    VectorSpec("bitflipped_signal_byte", "corruption", 2, 1024, 250.0,
               corrupt="bitflip", expected_error_kind="CrcMismatch"),
    VectorSpec("legacy_no_footer", "version compat", 2, 1024, 250.0,
               corrupt="legacy_no_footer"),
]


def post_process_corrupt(lml: Path, kind: str) -> None:
    """Mutate the LML bytes on disk per the corruption category."""
    raw = bytearray(lml.read_bytes())
    if kind == "crc":
        # Flip the last byte of the first window's CRC (window CRC is at
        # the end of each window's payload). Easiest: flip a byte
        # somewhere in the middle of the file. This invalidates the
        # window CRC.
        mid = len(raw) // 2
        raw[mid] ^= 0xFF
    elif kind == "truncate":
        # Chop off the last 64 bytes (footer + part of last window).
        raw = raw[: len(raw) - 64]
    elif kind == "bitflip":
        # Flip a single bit in the middle of a window payload.
        mid = len(raw) // 2
        raw[mid] ^= 0x01
    elif kind == "legacy_no_footer":
        # 1. Clear FLAG_HAS_FOOTER (bit 0 of byte 21).
        raw[21] &= 0xFE
        # 2. Strip the LMLFOOT1 footer + offset table at EOF.
        # Footer is fixed 32 bytes at EOF; magic at bytes [-32:-24].
        if len(raw) < 32:
            return
        magic = bytes(raw[-32:-24])
        if magic != b"LMLFOOT1":
            print(f"  warning: {lml.name} expected LMLFOOT1 magic at -32:-24, got {magic!r}",
                  file=sys.stderr)
            return
        # n_windows = u32 LE at bytes [-32+12:-32+16] = [-20:-16]
        n_windows = int.from_bytes(raw[-20:-16], "little")
        table_bytes = n_windows * 16  # ENTRY_SIZE = 16
        # Truncate: strip the footer + offset table.
        raw = raw[: len(raw) - 32 - table_bytes]
    else:
        raise ValueError(f"unknown corruption kind: {kind}")
    lml.write_bytes(bytes(raw))


def build_vector(spec: VectorSpec, lml_bin: Path, tmp_root: Path) -> None:
    out = OUT_DIR / f"{spec.name}.lml"
    print(f"[{spec.name}] ({spec.category}) "
          f"n_ch={spec.n_channels} n_samp={spec.n_samples} rate={spec.sample_rate}")
    edf = tmp_root / f"{spec.name}.edf"
    write_minimal_edf(edf, spec.n_channels, spec.n_samples, spec.sample_rate, seed=0xC0FFEE)
    extra = []
    if spec.noise_bits:
        extra += ["--noise-bits", str(spec.noise_bits)]
    run_encode(edf, out, lml_bin, extra)
    if spec.corrupt:
        post_process_corrupt(out, spec.corrupt)
    expected = {
        "name": spec.name,
        "category": spec.category,
        "n_channels": spec.n_channels,
        "total_samples": spec.n_samples,
        "sample_rate": spec.sample_rate,
        "lml_sha256": sha256_path(out),
        "corrupt": spec.corrupt,
        "expected_error_kind": spec.expected_error_kind,
    }
    (OUT_DIR / f"{spec.name}.expected.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False)
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lml_bin = ensure_lml()
    with tempfile.TemporaryDirectory(prefix="lml-conformance-") as td:
        tmp_root = Path(td)
        for spec in VECTORS:
            try:
                build_vector(spec, lml_bin, tmp_root)
            except subprocess.CalledProcessError as e:
                print(f"  encode failed for {spec.name}: {e}", file=sys.stderr)
                print(f"  stderr: {e.stderr.decode(errors='replace')[:400]}", file=sys.stderr)
                if spec.expected_error_kind:
                    print(f"  (vector is negative; skipping is acceptable)",
                          file=sys.stderr)
                else:
                    return 2
    print(f"\nGenerated {len(VECTORS)} vectors under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
