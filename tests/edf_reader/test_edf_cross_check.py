"""
L5 cross-implementation tests: LamQuant binary EDF reader vs pyedflib.

These tests read real EDF files shipped with Temple's reference tools and
verify that `edf_to_events._read_edf_binary` produces numerically identical
physical-unit signals on the channels both readers agree to keep.

Both test EDF files are checked into `Reference Software/`:
  - nedc_pyprint_edf/v1.0.0/example.edf    (30 channels, 250 Hz, 5 s)
  - nedc_eeg_resnet_decode_realtime/v1.0.0/test/test.edf  (50 Hz, ERDR input)

If these files disappear or pyedflib is uninstalled, the tests skip cleanly.
"""

import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("ai_models", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import os
import sys
import pytest

pyedflib = pytest.importorskip("pyedflib")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # moved into tests/edf_reader/

from ai_models.validation.edf_cross_check import cross_check_edf  # via conftest sys.path


EXAMPLE_EDF = os.path.join(
    _REPO_ROOT,
    "reference_software",
    "nedc_pyprint_edf",
    "v1.0.0",
    "example.edf",
)
ERDR_TEST_EDF = os.path.join(
    _REPO_ROOT, "reference_software", "nedc_eeg_resnet_decode_realtime", "v1.0.0",
    "test", "test.edf",
)


@pytest.mark.skipif(not os.path.exists(EXAMPLE_EDF),
                    reason="nedc_pyprint_edf example.edf not present")
class TestCrossCheckExampleEdf:

    @pytest.fixture(scope="class")
    def result(self):
        return cross_check_edf(EXAMPLE_EDF)

    def test_sample_rates_agree(self, result):
        assert result.sample_rates_agree, (
            f"sfreq mismatch: ours={result.sfreq_ours} "
            f"pyedflib={result.sfreq_pyedflib}"
        )

    def test_at_least_one_channel_compared(self, result):
        assert len(result.channels_compared) > 0, (
            f"no channels compared. ours={result.channels_ours[:5]} "
            f"pyedflib={result.channels_pyedflib[:5]}"
        )

    def test_no_channels_only_pyedflib(self, result):
        # Our reader applies a mode-rate + ANNOTATION filter; pyedflib reader
        # in edf_cross_check mirrors that filter. If pyedflib kept a channel
        # ours dropped, something is wrong with the filter logic.
        assert len(result.channels_only_pyedflib) == 0, (
            f"pyedflib kept channels we dropped: {result.channels_only_pyedflib}"
        )

    def test_physical_units_bit_equivalent(self, result):
        # Both readers do the same digital→physical affine transform, so the
        # result should be bit-identical on every compared channel.
        assert result.is_bit_equivalent(tol=1e-9), result.summary()


@pytest.mark.skipif(not os.path.exists(ERDR_TEST_EDF),
                    reason="ERDR test.edf not present")
class TestCrossCheckErdrTestEdf:
    """Same test battery against ERDR's shipped 50 Hz test file."""

    @pytest.fixture(scope="class")
    def result(self):
        return cross_check_edf(ERDR_TEST_EDF)

    def test_sample_rates_agree(self, result):
        assert result.sample_rates_agree, result.summary()

    def test_at_least_one_channel_compared(self, result):
        assert len(result.channels_compared) > 0, result.summary()

    def test_physical_units_bit_equivalent(self, result):
        assert result.is_bit_equivalent(tol=1e-9), result.summary()


# ── pyedflib fixture sweep — covers BDF (24-bit) + edge-case EDFs ──
#
# pyedflib ships a small fixture set covering the EDF family corners:
#   .edf  test_generator.edf, test_legacy.edf, test_subsecond.edf,
#         test_utf8.edf  (16-bit, various edge cases)
#   .bdf  test_generator.bdf, test_generator_datarec_generator_0_5.bdf,
#         test_generator_datarec_generator_2.bdf  (24-bit BioSemi)
#
# The fast reader (_read_edf_binary in ai_models/dataset_sim/edf_to_events)
# must produce numerically identical signals to pyedflib on every fixture
# both implementations agree to keep. BDF support was added alongside this
# test sweep — before the fix, fast reader treated every sample as 16-bit
# and silently produced garbage on .bdf inputs.

_PYEDFLIB_FIXTURE_DIR = os.path.join(
    _REPO_ROOT,
    "reference_software",
    "pyedflib-master",
    "pyedflib",
    "tests",
    "data",
)


def _list_pyedflib_fixtures():
    if not os.path.isdir(_PYEDFLIB_FIXTURE_DIR):
        return []
    paths = []
    for name in sorted(os.listdir(_PYEDFLIB_FIXTURE_DIR)):
        if name.lower().endswith(('.edf', '.bdf')):
            paths.append(os.path.join(_PYEDFLIB_FIXTURE_DIR, name))
    return paths


@pytest.mark.parametrize(
    "edf_path",
    _list_pyedflib_fixtures(),
    ids=lambda p: os.path.basename(p),
)
def test_pyedflib_fixture_bit_equivalent(edf_path):
    """Fast reader must agree with pyedflib on every shipped fixture.

    Tolerance: 1e-9 (physical units, after digital→physical scaling).
    Channels are filtered to mode-rate non-annotation per the production
    rule — both readers must drop the same channel set.
    """
    if not os.path.exists(edf_path):
        pytest.skip(f"fixture missing: {edf_path}")
    result = cross_check_edf(edf_path)
    assert result.sample_rates_agree, result.summary()
    assert len(result.channels_compared) > 0, result.summary()
    assert len(result.channels_only_pyedflib) == 0, result.summary()
    assert result.is_bit_equivalent(tol=1e-9), result.summary()
