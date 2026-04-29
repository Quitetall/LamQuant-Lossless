"""Synthetic EDF/BDF file generator for test fixtures.

Extracted from test_lml_adversarial.py (2026-04-28). Generates EDF/BDF files
with exact byte-level control for adversarial testing of the codec pipeline.

Usage:
    from tests.helpers.edf_factory import create_edf
    create_edf(path, n_channels=21, sample_rate=250, is_bdf=False, ...)
"""
import os
import struct
import numpy as np


def create_edf(path, n_channels=21, n_records=10, sample_rate=250,
               samples=None, patient_id="Test Patient", is_bdf=False,
               annotation_channels=0, extra_trailing=b''):
    """Create a synthetic EDF/BDF file with exact control over every byte.

    Parameters
    ----------
    path : str
        Output file path. Parent directories are created automatically.
    n_channels : int
        Number of EEG signal channels.
    n_records : int
        Number of data records (1 record = 1 second at the given sample_rate).
    sample_rate : int
        Samples per second per channel.
    samples : np.ndarray or None
        Shape (n_channels, n_records * sample_rate), dtype int64.
        If None, generates realistic-looking sinusoidal EEG.
    patient_id : str
        Patient ID field (80-char ASCII, padded with spaces).
    is_bdf : bool
        If True, writes BDF format (24-bit samples, 0xFF version byte).
    annotation_channels : int
        Number of EDF+ annotation channels to append.
    extra_trailing : bytes
        Extra bytes appended after the last data record (tests trailing data handling).
    """
    bps = 3 if is_bdf else 2
    ns_per_rec = sample_rate  # 1-second records
    n_signals = n_channels + annotation_channels

    # Main header (256 bytes)
    hdr = bytearray(256)
    if is_bdf:
        hdr[0:1] = b'\xff'
        hdr[1:8] = b'BIOSEMI'
    else:
        hdr[0:8] = b'0       '
    hdr[8:88] = f'{patient_id:<80s}'.encode('ascii')[:80]
    hdr[88:168] = f'{"Startdate 01-JAN-2024 Test":<80s}'.encode('ascii')[:80]
    hdr[168:176] = b'01.01.24'
    hdr[176:184] = b'00.00.00'
    total_hdr = 256 + 256 * n_signals
    hdr[184:192] = f'{total_hdr:<8d}'.encode('ascii')[:8]
    hdr[192:236] = b'EDF+C' + b' ' * 39
    hdr[236:244] = f'{n_records:<8d}'.encode('ascii')[:8]
    hdr[244:252] = f'{"1":<8s}'.encode('ascii')[:8]
    hdr[252:256] = f'{n_signals:<4d}'.encode('ascii')[:4]

    # Signal headers
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    sig_hdr = bytearray(256 * n_signals)

    for fi, w in enumerate(widths):
        for si in range(n_signals):
            off = sum(widths[:fi]) * n_signals + si * w
            if fi == 0:  # label
                if si >= n_channels:
                    val = 'EDF Annotations'
                else:
                    val = f'EEG Ch{si}'
            elif fi == 2:  # physical dimension
                val = 'uV'
            elif fi == 3:  # physical min
                val = '-3200' if not is_bdf else '-3200000'
            elif fi == 4:  # physical max
                val = '3200' if not is_bdf else '3200000'
            elif fi == 5:  # digital min
                val = '-32768' if not is_bdf else '-8388608'
            elif fi == 6:  # digital max
                val = '32767' if not is_bdf else '8388607'
            elif fi == 8:  # ns_per_rec
                val = str(ns_per_rec) if si < n_channels else str(ns_per_rec // 2)
            else:
                val = ''
            sig_hdr[off:off+w] = f'{val:<{w}s}'.encode('ascii')[:w]

    # Data records
    if samples is None:
        rng = np.random.RandomState(42)
        samples = np.zeros((n_channels, ns_per_rec * n_records), dtype=np.int64)
        for ch in range(n_channels):
            t = np.arange(ns_per_rec * n_records) / sample_rate
            signal = (100 * np.sin(2 * np.pi * 10 * t + ch) +
                     rng.randn(len(t)) * 50).astype(np.int64)
            if is_bdf:
                signal = np.clip(signal, -8388608, 8388607)
            else:
                signal = np.clip(signal, -32768, 32767)
            samples[ch] = signal

    data = bytearray()
    for r in range(n_records):
        for ch in range(n_channels):
            chunk = samples[ch, r*ns_per_rec:(r+1)*ns_per_rec]
            if is_bdf:
                for v in chunk:
                    v = int(v)
                    if v < 0:
                        v += (1 << 24)
                    data.extend(struct.pack('<I', v)[:3])
            else:
                data.extend(chunk.astype(np.int16).tobytes())
        for _ac in range(annotation_channels):
            data.extend(b'\x00' * (ns_per_rec // 2) * bps)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(sig_hdr)
        f.write(data)
        f.write(extra_trailing)
