#!/usr/bin/env python3
"""
Phase 8 / Item A — dump a pydicom-derived golden slice for the DICOM
Waveform parser unit test.

Reads `lamquant-core/tests/fixtures/dicom/12lead_ecg.dcm`, extracts the
first 16 samples of channel 0 (raw int16 from the WaveformData stream,
before any per-channel sensitivity scaling — this is what the LamQuant
parser also produces), and prints them as a Rust array literal.

The test asserts the LamQuant parser produces the same 16 samples.
This is a sanity check that we read the bytes in the same order pydicom
does (multiplexed channel-major, little-endian int16).

Re-run when the fixture is regenerated or the parser semantics change.

Usage:
    python3 tools/dump_pydicom_waveform_golden.py
"""

import sys
from pathlib import Path

try:
    import pydicom
except ImportError:
    print(
        "pydicom not installed — `pip install pydicom`. Generator is "
        "human-supervised only; the committed fixture + golden vector "
        "do not require pydicom at CI time.",
        file=sys.stderr,
    )
    sys.exit(2)

FIXTURE = Path("lamquant-core/tests/fixtures/dicom/12lead_ecg.dcm")
if not FIXTURE.exists():
    print(f"missing fixture: {FIXTURE}", file=sys.stderr)
    sys.exit(3)

ds = pydicom.dcmread(FIXTURE)
group = ds.WaveformSequence[0]
n_ch = group.NumberOfWaveformChannels
n_samp = group.NumberOfWaveformSamples
print(f"// Fixture {FIXTURE} group 0: {n_ch} ch × {n_samp} samp")
print(
    f"// SOP class: 1.2.840.10008.5.1.4.1.1.9.1.1 (12-Lead ECG Waveform Storage)"
)
print(f"// Sample interpretation: SS (signed 16-bit two's complement)")

# The WaveformData attribute is the raw OW byte string. pydicom helpfully
# exposes ds.waveform_array(0) which already does the multiplex decoding;
# we use it as the reference, then print the first 16 samples of channel 0.
arr = ds.waveform_array(0)  # shape (n_samp, n_ch) per pydicom convention
print(f"// pydicom waveform_array shape: {arr.shape} dtype: {arr.dtype}")
print()
print("// GOLDEN: first 16 samples of channel 0 (raw int16, post-sensitivity-scaling)")
print("pub const GOLDEN_CH0_FIRST_16: [i64; 16] = [")
for s in arr[:16, 0].tolist():
    print(f"    {int(s)},")
print("];")
print()
# Also dump raw int16 values (pre-sensitivity-scaling) since LamQuant's
# parser keeps the raw int16 path; sensitivity comes through phys_min/
# phys_max metadata, not signal scaling.
import numpy as np

raw_bytes = group.WaveformData
samples = np.frombuffer(raw_bytes, dtype="<i2")  # little-endian int16
samples = samples.reshape(n_samp, n_ch)
print("// GOLDEN_RAW: first 16 samples of channel 0 (raw int16 from WaveformData, multiplexed)")
print("pub const GOLDEN_CH0_RAW_FIRST_16: [i64; 16] = [")
for s in samples[:16, 0].tolist():
    print(f"    {int(s)},")
print("];")
