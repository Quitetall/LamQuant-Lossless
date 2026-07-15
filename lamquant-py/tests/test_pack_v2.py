"""LQTP2 Python bindings: strict snapshot checks and multi-view access."""

import json

import pytest

lc = pytest.importorskip("lamquant_core")
np = pytest.importorskip("numpy")


MANIFEST_SHA256 = bytes.fromhex("11" * 32)
VIEW_SPEC_SHA256 = bytes.fromhex("22" * 32)


def _write_pack(path):
    metadata = json.dumps(
        {
            "row_count": 2,
            "rows": [
                {"recording_id": "r0", "window_index": 0},
                {"recording_id": "r1", "window_index": 1},
            ],
            "schema": "lamquant.training-window-store/v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    writer = lc.PyPackV2Writer(
        str(path),
        2,
        MANIFEST_SHA256,
        VIEW_SPEC_SHA256,
        metadata,
        [
            ("fullband", "float32", "bfp16", [2, 4], True, "33" * 32),
            ("labels", "int16", "raw", [2], True, "44" * 32),
        ],
    )
    expected = []
    for row in range(2):
        values = np.asarray(
            [[row + 0.125, -2.5, 3.75, 0.0], [8.0, -0.5, row + 1.0, 4.25]],
            dtype=np.float32,
        )
        expected.append(values)
        writer.write_f32_row("fullband", values)
        labels = np.asarray([row, row + 10], dtype="<i2")
        writer.write_raw_row("labels", labels.tobytes())
    writer.finish()
    return metadata, expected


def test_pack_v2_round_trip_and_metadata(tmp_path):
    path = tmp_path / "training.lqtp2"
    metadata, expected = _write_pack(path)

    reader = lc.PyPackV2Reader(
        str(path),
        MANIFEST_SHA256,
        VIEW_SPEC_SHA256,
    )
    assert reader.row_count == 2
    assert reader.view_names == ["fullband", "labels"]
    assert reader.manifest_sha256 == MANIFEST_SHA256
    assert reader.view_spec_sha256 == VIEW_SPEC_SHA256
    assert reader.metadata == metadata

    fullband = reader.view_info("fullband")
    assert fullband["dtype"] == "float32"
    assert fullband["encoding"] == "bfp16"
    assert fullband["row_shape"] == [2, 4]
    assert fullband["required"] is True
    assert fullband["data_offset"] % 8 == 0

    labels = np.frombuffer(reader.row_raw("labels", 1), dtype="<i2")
    assert labels.tolist() == [1, 11]
    restored = reader.dequantize_flat("fullband", 1).reshape(2, 4)
    assert np.allclose(restored, expected[1], rtol=2e-4, atol=2e-4)


def test_pack_v2_rejects_snapshot_mismatch(tmp_path):
    path = tmp_path / "training.lqtp2"
    _write_pack(path)

    with pytest.raises(ValueError, match="manifest"):
        lc.PyPackV2Reader(str(path), bytes.fromhex("99" * 32), VIEW_SPEC_SHA256)
    with pytest.raises(ValueError, match="(?i)view ?spec"):
        lc.PyPackV2Reader(str(path), MANIFEST_SHA256, bytes.fromhex("99" * 32))


def test_pack_v2_rejects_corrupt_view_before_access(tmp_path):
    path = tmp_path / "training.lqtp2"
    _write_pack(path)
    corrupted = bytearray(path.read_bytes())
    corrupted[-1] ^= 0x01
    path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="integrity"):
        lc.PyPackV2Reader(str(path), MANIFEST_SHA256, VIEW_SPEC_SHA256)


def test_pack_v2_missing_view_is_explicit(tmp_path):
    path = tmp_path / "training.lqtp2"
    _write_pack(path)
    reader = lc.PyPackV2Reader(str(path))

    with pytest.raises(KeyError):
        reader.view_info("missing")
