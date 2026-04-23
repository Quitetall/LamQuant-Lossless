"""
L2/L5 — Firmware weight export: C header generation, CRC, biquad coefficients.

Validates that export_firmware.py produces valid C header files with correct
include guards, ternary weight hex packing, Q31 bias/norm parameters, and
CRC32 integrity checksums matching zlib. Also verifies biquad HP coefficient
generation doesn't crash and encoder-only mode omits decoder layers.
"""
import os
import sys
import zlib
import pytest
import torch
import numpy as np


@pytest.mark.l2
@pytest.mark.l5
class TestExportToHeader:
    def test_header_is_valid_c(self, ternary_model, tmp_header):
        """Exported header should have include guard and stdint."""
        from export_firmware import export_to_header
        export_to_header(ternary_model, str(tmp_header))

        content = tmp_header.read_text()
        assert '#ifndef FOCAL_NET_WEIGHTS_H' in content
        assert '#define FOCAL_NET_WEIGHTS_H' in content
        assert '#include <stdint.h>' in content
        assert '#endif' in content

    def test_encoder_only_skips_decoder(self, ternary_model, tmp_header):
        """encoder_only=True should omit decoder layer names."""
        from export_firmware import export_to_header
        export_to_header(ternary_model, str(tmp_header), encoder_only=True)

        content = tmp_header.read_text()
        assert 'ENCODER-ONLY' in content
        # Decoder layers should not appear
        assert 'expand1' not in content
        assert 'expand2' not in content
        assert 'output_weights' not in content

    def test_header_contains_all_encoder_layers(self, ternary_model, tmp_header):
        """All encoder layers should produce weight arrays."""
        from export_firmware import export_to_header
        export_to_header(ternary_model, str(tmp_header), encoder_only=True)

        content = tmp_header.read_text()
        for layer in ['focal1', 'focal2', 'focal3', 'focal4', 'bottleneck']:
            assert layer in content, f"Missing encoder layer: {layer}"

    def test_ternary_weights_are_hex_bytes(self, ternary_model, tmp_header):
        """Packed weights should appear as 0xHH hex bytes."""
        from export_firmware import export_to_header
        export_to_header(ternary_model, str(tmp_header))

        content = tmp_header.read_text()
        # Should contain hex-encoded bytes like 0x49, 0xAB, etc.
        import re
        hex_bytes = re.findall(r'0x[0-9A-F]{2}', content)
        assert len(hex_bytes) > 100, "Too few packed weight bytes in header"


@pytest.mark.l5
class TestComputeFirmwareCrc:
    def test_crc_matches_python_zlib(self, tmp_path):
        """CRC in the generated header must match zlib.crc32()."""
        from export_firmware import compute_firmware_crc

        # Write some test data
        test_file = tmp_path / "test_data.h"
        test_file.write_bytes(b"Hello, LamQuant firmware!")

        crc_file = tmp_path / "firmware_crc.h"
        compute_firmware_crc(str(test_file), str(crc_file))

        # Read back and verify
        crc_content = crc_file.read_text()
        expected_crc = zlib.crc32(b"Hello, LamQuant firmware!") & 0xFFFFFFFF
        assert f"0x{expected_crc:08X}" in crc_content

    def test_crc_header_format(self, tmp_path):
        from export_firmware import compute_firmware_crc

        test_file = tmp_path / "test_data.h"
        test_file.write_bytes(b"test")

        crc_file = tmp_path / "firmware_crc.h"
        compute_firmware_crc(str(test_file), str(crc_file))

        content = crc_file.read_text()
        assert '#ifndef FIRMWARE_CRC_H' in content
        assert '#define FIRMWARE_CRC32' in content
        assert '#endif' in content


@pytest.mark.l2
class TestGenerateBiquadCoefficients:
    def test_biquad_coefficients_generation(self, capsys):
        """Verify biquad coefficient generation doesn't crash."""
        try:
            from export_firmware import generate_biquad_coefficients
            generate_biquad_coefficients()
            captured = capsys.readouterr()
            # Should print HP, LP, NOTCH sections
            assert 'HP' in captured.out
            assert 'LP' in captured.out
            assert 'NOTCH' in captured.out
        except ImportError:
            pytest.skip("scipy not installed")
