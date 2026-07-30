"""ABIR dataset-view API tests for lamquant-py."""

import json

import pytest

lc = pytest.importorskip("lamquant_core")
np = pytest.importorskip("numpy")


def _make_fixture_signal():
    samples = 513
    channels = 2
    signal = (
        (np.arange(channels * samples, dtype=np.int64) % 1024)
        .reshape(channels, samples)
    )
    metadata_json = '{"subject":"fixture"}'
    channels = ["C3", "C4"]
    return signal, metadata_json, channels


def test_abir_dataset_view_round_trip_and_zero_copy():
    signal, metadata_json, channel_names = _make_fixture_signal()
    expected_counts = [128, 128, 128, 128, 1]

    view = lc.AbirDatasetView.from_numpy(
        signal,
        sample_rate_hz=256.0,
        window_size=128,
        metadata_json=metadata_json,
        channel_names=channel_names,
    )

    assert view.shape == (2, 513)
    assert view.n_channels == 2
    assert view.total_samples == 513
    assert view.sample_rate_hz == 256.0
    assert view.metadata_json == metadata_json
    assert view.packet_sample_counts == expected_counts
    assert view.n_windows == len(expected_counts)

    samples = view.numpy_view()
    assert samples.shape == signal.shape
    assert np.array_equal(samples, signal)
    assert not samples.flags.writeable
    with pytest.raises(ValueError):
        samples[0, 0] = -1
    with pytest.raises(ValueError):
        samples.setflags(write=True)

    samples_again = view.numpy_view()
    assert samples_again is samples
    assert np.array_equal(samples_again, signal)
    assert samples_again.ctypes.data == samples.ctypes.data
    assert samples.ctypes.data == view.payload_pointer()

    lml_first = view.lml_bytes()
    lml_second = view.lml_bytes()
    assert lml_first is lml_second
    assert isinstance(json.loads(view.canonical_json()), dict)

    by_fn = lc.open_abir(lml_first)
    mutable_lml = bytearray(lml_first)
    by_bytearray = lc.AbirDatasetView.from_lml(mutable_lml)
    mutable_lml[:] = b"\x00" * len(mutable_lml)

    assert by_fn.content_id == view.content_id
    assert by_bytearray.content_id == view.content_id
    assert by_fn.lml_bytes() is lml_first
    assert by_bytearray.lml_bytes() == lml_first
    assert by_fn.canonical_json() == view.canonical_json()
    assert by_bytearray.canonical_json() == view.canonical_json()
    assert np.array_equal(by_fn.numpy_view(), samples)
    assert np.array_equal(by_bytearray.numpy_view(), samples)


def test_abir_dataset_view_negative_errors():
    signal, metadata_json, _ = _make_fixture_signal()

    non_contiguous = signal[:, ::-1]
    with pytest.raises(ValueError, match="C-contiguous"):
        lc.AbirDatasetView.from_numpy(
            non_contiguous,
            sample_rate_hz=256.0,
            window_size=128,
            metadata_json=metadata_json,
            channel_names=["C3", "C4"],
        )

    with pytest.raises(ValueError, match="channel_names length"):
        lc.AbirDatasetView.from_numpy(
            signal,
            sample_rate_hz=256.0,
            window_size=128,
            metadata_json=metadata_json,
            channel_names=["C3"],
        )

    with pytest.raises(ValueError, match="sample_rate_hz"):
        lc.AbirDatasetView.from_numpy(
            signal,
            sample_rate_hz=float("nan"),
        )

    with pytest.raises(ValueError, match="window_size"):
        lc.AbirDatasetView.from_numpy(
            signal,
            sample_rate_hz=256.0,
            window_size=0,
        )

    with pytest.raises(ValueError, match="invalid JSON"):
        lc.AbirDatasetView.from_numpy(
            signal,
            sample_rate_hz=256.0,
            metadata_json="{",
        )

    with pytest.raises(ValueError, match="JSON object"):
        lc.AbirDatasetView.from_numpy(
            signal,
            sample_rate_hz=256.0,
            metadata_json="[]",
        )

    too_many_channels = np.zeros((1025, 1), dtype=np.int64)
    with pytest.raises(ValueError):
        lc.AbirDatasetView.from_numpy(
            too_many_channels,
            sample_rate_hz=256.0,
        )

    corrupted = bytearray(b"not-a-valid-abir-lml")
    with pytest.raises(ValueError):
        lc.AbirDatasetView.from_lml(corrupted)
    with pytest.raises(ValueError):
        lc.open_abir(corrupted)
