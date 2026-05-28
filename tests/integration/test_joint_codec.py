"""Tests for JointCodec — encoder+decoder as one nn.Module.

Verifies the contract the trainer relies on:
  - construction works with both Tier-1 and Tier-3 decoders
  - forward pass shape matches the production decoder's output
  - gradient flows through BOTH halves (encoder + decoder both update)
  - param-group accessors return distinct sets (encoder/decoder/alphas)
  - freeze_decoder() really freezes (grads don't reach decoder)
  - save_encoder / save_decoder write loadable separate files

These tests use the smallest working tier (1) to keep them fast.
The tier choice doesn't change the JointCodec class behavior.
"""
from __future__ import annotations

import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'ai_models' / 'student'))
sys.path.insert(0, str(REPO / 'ai_models' / 'decoder'))

# Importing torch is mandatory; the codec is torch-native.
torch.manual_seed(0)


@pytest.fixture(scope='module')
def joint():
    """Build a small joint codec for the tests to share."""
    from joint_codec import build_default_joint
    return build_default_joint(latent_dim=32, encoder_width=128,
                                vocos_tier=1, in_channels=21,
                                target_len=313)


def _sample_input():
    return torch.randn(2, 21, 313)


# ============================================================
# Construction
# ============================================================

class TestConstruction:

    def test_can_build_tier1(self):
        from joint_codec import build_default_joint
        codec = build_default_joint(vocos_tier=1)
        assert hasattr(codec, 'encoder')
        assert hasattr(codec, 'decoder')

    def test_can_build_tier3(self):
        """Tier 3 is the production decoder for the joint training run."""
        from joint_codec import build_default_joint
        codec = build_default_joint(vocos_tier=3)
        # Tier 3 should be larger than Tier 1.
        assert sum(p.numel() for p in codec.decoder.parameters()) > 1_000_000

    def test_can_build_tier8_mobile(self):
        """Tier 8 (200M mobile target) — distill destination from Tier 3."""
        from joint_codec import build_default_joint
        codec = build_default_joint(vocos_tier=8)
        n_params = sum(p.numel() for p in codec.decoder.parameters())
        # Should be in the 100M-300M ballpark (designed for 200M).
        assert 50_000_000 < n_params < 500_000_000, \
            f'Tier 8 has {n_params:,} params, expected ~200M'

    def test_default_anchor_is_tier3(self):
        """Joint training anchors at Tier 3 — distill smaller from there."""
        from joint_codec import build_default_joint
        import inspect
        sig = inspect.signature(build_default_joint)
        assert sig.parameters['vocos_tier'].default == 3

    def test_rejects_encoder_without_encode_method(self):
        from joint_codec import JointCodec
        bogus_encoder = torch.nn.Linear(10, 10)   # no .encode method
        bogus_decoder = torch.nn.Linear(10, 10)
        with pytest.raises(TypeError, match='encode'):
            JointCodec(bogus_encoder, bogus_decoder)


# ============================================================
# Forward pass
# ============================================================

class TestForward:

    def test_forward_returns_correct_shape(self, joint):
        x = _sample_input()
        with torch.no_grad():
            y = joint(x)
        assert y.shape[0] == x.shape[0]   # batch
        assert y.shape[1] == x.shape[1]   # channels
        assert y.shape[2] >= 313 - 5      # ~313 samples (might be 312-316)

    def test_forward_quantize_off_works_too(self, joint):
        """During warmup, quantize=False bypasses the ternary path."""
        x = _sample_input()
        with torch.no_grad():
            y_q = joint(x, quantize=True)
            y_f = joint(x, quantize=False)
        # Both must produce sane shapes; values can differ.
        assert y_q.shape == y_f.shape
        assert torch.isfinite(y_q).all()
        assert torch.isfinite(y_f).all()


# ============================================================
# Gradient flow — the whole point of joint training
# ============================================================

class TestGradientFlow:

    def test_loss_backprops_through_both_halves(self):
        """One backward pass must update both encoder AND decoder params."""
        from joint_codec import build_default_joint
        codec = build_default_joint(vocos_tier=1)
        codec.train()

        x = _sample_input()
        y = codec(x, quantize=True)
        target = torch.randn_like(y)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()

        enc_grad = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for p in codec.encoder.parameters() if p.requires_grad
        )
        dec_grad = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for p in codec.decoder.parameters() if p.requires_grad
        )

        assert enc_grad > 0, "encoder received no gradient — joint training broken"
        assert dec_grad > 0, "decoder received no gradient — joint training broken"

    def test_freeze_decoder_zeros_decoder_grads(self):
        from joint_codec import build_default_joint
        codec = build_default_joint(vocos_tier=1)
        codec.freeze_decoder()
        codec.train()

        x = _sample_input()
        y = codec(x, quantize=True)
        loss = y.mean()
        loss.backward()

        for p in codec.decoder.parameters():
            assert p.grad is None, "frozen decoder param accumulated a gradient"

        # Encoder still updates.
        enc_grad_total = sum(
            (p.grad.abs().sum().item() if p.grad is not None else 0.0)
            for p in codec.encoder.parameters() if p.requires_grad
        )
        assert enc_grad_total > 0


# ============================================================
# Optimizer parameter groups
# ============================================================

class TestParamGroups:

    def test_distinct_groups(self, joint):
        enc = joint.encoder_parameters()
        dec = joint.decoder_parameters()
        # No param appears in both.
        enc_ids = {id(p) for p in enc}
        dec_ids = {id(p) for p in dec}
        assert not (enc_ids & dec_ids), "encoder and decoder share parameters"

    def test_alpha_subset_is_inside_encoder(self, joint):
        enc_ids = {id(p) for p in joint.encoder_parameters()}
        for p in joint.encoder_alpha_parameters():
            assert id(p) in enc_ids, "alpha param leaked outside encoder"

    def test_alpha_and_other_partition_encoder(self, joint):
        all_ids = {id(p) for p in joint.encoder_parameters()}
        alpha_ids = {id(p) for p in joint.encoder_alpha_parameters()}
        other_ids = {id(p) for p in joint.encoder_other_parameters()}
        assert alpha_ids | other_ids == all_ids
        assert not (alpha_ids & other_ids), "alpha and other groups overlap"


# ============================================================
# Save / load — encoder and decoder go to separate files
# ============================================================

class TestSaveLoad:

    def test_separate_files(self, joint, tmp_path):
        ep = tmp_path / 'encoder.ckpt'
        dp = tmp_path / 'decoder.ckpt'
        joint.save_encoder(ep)
        joint.save_decoder(dp)
        assert ep.exists() and dp.exists()
        # Two distinct files of non-zero size.
        assert ep.stat().st_size > 0
        assert dp.stat().st_size > 0
        # Sizes typically differ (decoder >> encoder for tier ≥ 1).

    def test_load_recovers_state(self, tmp_path):
        from joint_codec import build_default_joint
        codec_a = build_default_joint(vocos_tier=1)
        codec_a.save_encoder(tmp_path / 'enc.ckpt')
        codec_a.save_decoder(tmp_path / 'dec.ckpt')

        # Build a fresh codec, load both halves.
        codec_b = build_default_joint(vocos_tier=1)
        codec_b.load_encoder(tmp_path / 'enc.ckpt')
        codec_b.load_decoder(tmp_path / 'dec.ckpt')

        # State dicts must now match exactly.
        for ka, va in codec_a.encoder.state_dict().items():
            vb = codec_b.encoder.state_dict()[ka]
            assert torch.equal(va, vb), f'encoder state mismatch on {ka}'
        for ka, va in codec_a.decoder.state_dict().items():
            vb = codec_b.decoder.state_dict()[ka]
            assert torch.equal(va, vb), f'decoder state mismatch on {ka}'

    def test_save_creates_parent_directories(self, joint, tmp_path):
        deep = tmp_path / 'a' / 'b' / 'c' / 'enc.ckpt'
        joint.save_encoder(deep)
        assert deep.exists()


# ============================================================
# Compatibility with CheckpointManager + make_param_groups
# ============================================================

class TestIntegration:

    def test_make_param_groups_works_on_encoder(self, joint):
        from checkpoint_manager import make_param_groups
        groups = make_param_groups(joint.encoder, lr=1e-3,
                                    weight_decay=1e-4,
                                    alpha_weight_decay=1e-3)
        assert len(groups) == 2
        # Group 1 is alphas with the BitNet-style WD.
        assert groups[1]['weight_decay'] == 1e-3

    def test_step_with_optimizer_updates_both(self, joint):
        """One optimizer step changes parameters in both halves."""
        joint.train()
        opt = torch.optim.AdamW([
            {'params': joint.encoder_parameters(), 'lr': 1e-3},
            {'params': joint.decoder_parameters(), 'lr': 1e-3},
        ])
        # Snapshot one param from each half.
        enc_p_before = next(iter(joint.encoder_parameters())).detach().clone()
        dec_p_before = next(iter(joint.decoder_parameters())).detach().clone()

        x = _sample_input()
        y = joint(x, quantize=True)
        loss = y.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        enc_p_after = next(iter(joint.encoder_parameters()))
        dec_p_after = next(iter(joint.decoder_parameters()))
        assert not torch.equal(enc_p_after, enc_p_before), \
            "encoder param did not update"
        assert not torch.equal(dec_p_after, dec_p_before), \
            "decoder param did not update"
