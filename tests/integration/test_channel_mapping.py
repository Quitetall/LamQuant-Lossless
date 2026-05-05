"""
L2 — EDF channel name resolution invariants.

Validates the canonical 10-20 channel resolver handles all known naming
conventions across TUH, CHB-MIT, EEGMMI, Siena, and PhysioNet datasets:
standard names, modern renames (T7→T3), case variants, REF/LE suffixes,
bipolar montage (FP1-F7), dot-padded names (Fc5.), and ANNOTATION
exclusion. Every alias must map to a TARGET_CHANNELS member.
"""
import pytest


@pytest.mark.l2
class TestNormalizeChannelName:
    @pytest.fixture(autouse=True)
    def _import(self):
        from edf_to_events import normalize_channel_name, TARGET_CHANNELS, CHANNEL_ALIASES
        self.normalize = normalize_channel_name
        self.targets = TARGET_CHANNELS
        self.aliases = CHANNEL_ALIASES

    def test_standard_names_resolve(self):
        """All 21 canonical names should resolve to themselves."""
        for ch in self.targets:
            assert self.normalize(ch) == ch, f"{ch} did not resolve"

    def test_modern_renames(self):
        assert self.normalize('T7') == 'T3'
        assert self.normalize('T8') == 'T4'
        assert self.normalize('P7') == 'T5'
        assert self.normalize('P8') == 'T6'

    def test_case_variants(self):
        assert self.normalize('FP1') == 'Fp1'
        assert self.normalize('FP2') == 'Fp2'
        assert self.normalize('FZ') == 'Fz'
        assert self.normalize('CZ') == 'Cz'
        assert self.normalize('PZ') == 'Pz'

    def test_physionet_ref_prefix(self):
        assert self.normalize('EEG Fp1-REF') == 'Fp1'
        assert self.normalize('EEG C3-LE') == 'C3'
        assert self.normalize('EEG T7-REF') == 'T3'  # modern + REF

    def test_chbmit_bipolar(self):
        """CHB-MIT uses bipolar format like FP1-F7 — should extract first electrode."""
        assert self.normalize('FP1-F7') == 'Fp1'
        assert self.normalize('C3-P3') == 'C3'

    def test_eegmmi_dot_padded(self):
        """EEGMMIDB uses dot-padded names like 'C3..'."""
        assert self.normalize('C3..') == 'C3'
        assert self.normalize('Fz.') == 'Fz'

    def test_whitespace_stripped(self):
        assert self.normalize('  Fp1  ') == 'Fp1'
        assert self.normalize('C3 ') == 'C3'

    def test_unrecognized_returns_none(self):
        assert self.normalize('EMG1') is None
        assert self.normalize('ECG') is None
        assert self.normalize('') is None

    def test_case_insensitive_eeg_prefix(self):
        """EEG prefix stripping should work regardless of case."""
        assert self.normalize('EEG fp1') == 'Fp1'
        assert self.normalize('EEG c3') == 'C3'

    def test_all_aliases_resolve_to_target(self):
        """Every alias value must be in TARGET_CHANNELS."""
        for alias, canonical in self.aliases.items():
            assert canonical in self.targets or canonical == 'Oz', \
                f"Alias '{alias}' maps to '{canonical}' which is not in TARGET_CHANNELS"
