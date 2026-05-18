#!/usr/bin/env python3
"""
Phase 8 / Item A — synthesise a valid General ECG Waveform Storage SOP
class instance for the second DICOM unit test.

pydicom ships a single 12-Lead ECG example (`pydicom.examples.waveform`).
For coverage of the General ECG SOP class — different SOP class UID but
the same WaveformSequence structure — we build a synthetic fixture
programmatically. Three leads, 5 s @ 500 Hz, deterministic sine-wave
content so the test can compare to a hand-computed slice.

Usage:
    python3 tools/make_general_ecg_fixture.py
"""

import sys
import struct
import math
from pathlib import Path

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
except ImportError:
    print(
        "pydicom not installed — `pip install pydicom`. This generator "
        "is human-supervised only.",
        file=sys.stderr,
    )
    sys.exit(2)

OUTPUT = Path("lamquant-core/tests/fixtures/dicom/general_ecg.dcm")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

N_CHANNELS = 3
N_SAMPLES = 2500            # 5 s × 500 Hz
SAMPLE_RATE = 500.0
GENERAL_ECG_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.9.1.2"

# Synthesise the multiplexed int16 sample stream: each channel is a
# deterministic sine wave with a different frequency. Test assertions
# below pin the first 8 samples per channel.
samples = []
for s in range(N_SAMPLES):
    for ch in range(N_CHANNELS):
        # frequency: ch=0 → 5 Hz, ch=1 → 7 Hz, ch=2 → 11 Hz
        freq = [5.0, 7.0, 11.0][ch]
        val = int(round(1000 * math.sin(2 * math.pi * freq * s / SAMPLE_RATE)))
        samples.append(val)
sample_bytes = struct.pack(f"<{len(samples)}h", *samples)

# Build the FileMeta (DICOM file-format preamble).
file_meta = Dataset()
file_meta.MediaStorageSOPClassUID = GENERAL_ECG_SOP_CLASS_UID
file_meta.MediaStorageSOPInstanceUID = generate_uid()
file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
file_meta.ImplementationClassUID = generate_uid()

ds = FileDataset(str(OUTPUT), {}, file_meta=file_meta, preamble=b"\0" * 128)

# Minimum patient + study metadata so the file passes pydicom's
# basic validation.
ds.PatientName = "Test^Synthetic"
ds.PatientID = "ANON"
ds.StudyInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()
ds.SOPClassUID = GENERAL_ECG_SOP_CLASS_UID
ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
ds.Modality = "ECG"
ds.StudyDate = "20260101"
ds.StudyTime = "120000"

# Build the WaveformSequence with a single multiplex group.
group = Dataset()
group.MultiplexGroupTimeOffset = 0
group.WaveformOriginality = "ORIGINAL"
group.NumberOfWaveformChannels = N_CHANNELS
group.NumberOfWaveformSamples = N_SAMPLES
group.SamplingFrequency = SAMPLE_RATE
group.WaveformBitsAllocated = 16
group.WaveformSampleInterpretation = "SS"
group.WaveformData = sample_bytes

# ChannelDefinitionSequence with labeled leads.
channel_seq = []
for ch in range(N_CHANNELS):
    chan = Dataset()
    chan.ChannelLabel = f"Lead {chr(ord('I') + ch)}"
    chan.WaveformBitsStored = 16
    chan.ChannelSampleSkew = 0
    channel_seq.append(chan)
group.ChannelDefinitionSequence = channel_seq

ds.WaveformSequence = [group]

ds.save_as(OUTPUT, write_like_original=False)
print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
print(f"  SOP class:    General ECG Waveform Storage")
print(f"  n_channels:   {N_CHANNELS}")
print(f"  n_samples:    {N_SAMPLES}")
print(f"  sample_rate:  {SAMPLE_RATE} Hz")
print(f"  channels:     Lead I, Lead J, Lead K (synthetic labels)")
print(f"  ch0[:8] (first 8 samples ch0): "
      f"{[int(round(1000 * math.sin(2*math.pi*5.0*s/SAMPLE_RATE))) for s in range(8)]}")
