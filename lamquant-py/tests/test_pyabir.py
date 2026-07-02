"""S7a (ADR 0069): the typed PyAbir handle enforces modality at the PyO3 boundary.

Run after building the extension:

    maturin develop            # (in lamquant-py/)  or:  maturin build && pip install ...
    pytest lamquant-py/tests/test_pyabir.py

Skips cleanly when `lamquant_core` is not importable (extension not built).
"""

import json
import os
import tempfile

import pytest

lc = pytest.importorskip("lamquant_core")
np = pytest.importorskip("numpy")


def _write_container(path, channels, sample_rate):
    """Write a BCS1 container (write_abir emits BCS1) with the given channel
    labels + sample_rate in its metadata, and a deterministic i64 signal."""
    n_ch = len(channels)
    T = 512
    signal = [[((c * 37 + t * 5) % 4001) - 2000 for t in range(T)] for c in range(n_ch)]
    meta = json.dumps({"channels": channels, "sample_rate": sample_rate})
    lc.container_write(path, signal, sample_rate, 128, 0, meta)
    return signal


@pytest.fixture
def eeg_container(tmp_path):
    path = str(tmp_path / "eeg.lml")
    signal = _write_container(path, ["Fp1", "Fp2", "C3", "C4", "O1", "O2"], 256.0)
    return path, signal


@pytest.fixture
def ecg_container(tmp_path):
    path = str(tmp_path / "ecg.lml")
    signal = _write_container(path, ["ECG1", "ECG2", "ECG3"], 500.0)
    return path, signal


def test_eeg_handle_metadata_and_channels(eeg_container):
    path, _ = eeg_container
    a = lc.container_read_abir(path)
    assert a.modality() == "eeg"
    assert a.modality_source() == "channel_label"
    assert a.n_channels() == 6
    assert a.n_samples() == 512
    assert a.sample_rate() == pytest.approx(256.0)
    assert a.channels() == ["Fp1", "Fp2", "C3", "C4", "O1", "O2"]


def test_eeg_accessor_returns_samples(eeg_container):
    path, signal = eeg_container
    a = lc.container_read_abir(path)
    arr = a.eeg()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (6, 512)
    assert arr.dtype == np.int64
    # samples are byte-exact to what was written (lossless round-trip)
    assert int(arr[0, 0]) == signal[0][0]
    assert int(arr[3, 7]) == signal[3][7]


def test_eeg_rejects_wrong_modality_accessor(eeg_container):
    """The trust boundary: an EEG recording cannot be pulled into an ECG path."""
    path, _ = eeg_container
    a = lc.container_read_abir(path)
    with pytest.raises(ValueError):
        a.ecg()


def test_blind_egress_always_available(eeg_container):
    path, _ = eeg_container
    a = lc.container_read_abir(path)
    blind = a.samples_i64()
    assert blind.shape == (6, 512)
    assert blind.dtype == np.int64


def test_ecg_handle_and_cross_modality_block(ecg_container):
    path, _ = ecg_container
    with open(path, "rb") as fh:
        raw = fh.read()
    # write_abir emits the BCS1 wire — this also exercises the #34 read dispatch.
    assert raw[0:4] == b"BCS1"

    b = lc.container_read_bytes_abir(raw)
    assert b.modality() == "ecg"
    carr = b.ecg()
    assert carr.shape == (3, 512)
    with pytest.raises(ValueError):
        b.eeg()
