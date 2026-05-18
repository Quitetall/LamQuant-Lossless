"""CRC-32 over weight buffers — matches `lamquant_core::crc32` and zlib.

Used by the codegen tool to compute `metadata::FIRMWARE_CRC32` over all
weight byte arrays in deterministic enumeration order. Firmware verifies
this at boot.
"""
from __future__ import annotations

import zlib
from typing import Iterable


def crc32_of(byte_buffers: Iterable[bytes]) -> int:
    """Concatenate buffers in order, return CRC-32 (uint32)."""
    crc = 0
    for buf in byte_buffers:
        crc = zlib.crc32(buf, crc)
    return crc & 0xFFFF_FFFF


def crc32_of_file(path) -> int:
    """CRC-32 of a file's bytes. Used for legacy firmware_crc.h compat."""
    with open(path, "rb") as f:
        return zlib.crc32(f.read()) & 0xFFFF_FFFF
