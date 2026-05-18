"""Shared helpers for the C and Rust weight emitters.

Public modules:
    schema      — Load and validate export_schema.toml
    checkpoint  — Load model checkpoint, detect architecture variant
    quantize    — Pure quantization helpers (Q15/Q7/ternary packing/Cayley)
    fsq         — FSQ + rANS frequency table calibration
    crc         — CRC-32 over weight buffers (matches firmware Rust impl)
"""
from . import checkpoint, crc, fsq, quantize, schema  # noqa: F401
