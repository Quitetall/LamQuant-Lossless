"""Encoder model architectures.

Three generations of TNN (Ternary Neural Network) encoders:
  - TernaryMobileNetV5: Gen 7.0, 96-wide stride-8, full-band
  - TernaryMobileNetV5_Subband: Gen 7.1, stride-4, subband (L3 approximation)
  - TernaryMobileNetV5_Subband_V2: Gen 7.6.1, DW-sep, wider
"""
import torch, torch.nn as nn, torch.nn.functional as F

from lamquant_codec.models.blocks import (
    TernaryConv1d,
    INT8Conv1d,
    TernaryConvTranspose1d,
    TernaryFocalBlock,
    TernaryDWSepFocalBlock,
    TernaryUpsampleBlock,
    _build_hadamard_32_cpu,
    _quantize_activation,
)


class TernaryMobileNetV5(nn.Module):
    """
    4-Layer 96-Wide Stride-8 Ternary AutoEncoder.

    Encoder:
      focal1: 21→96, k=7, s=1    T → T         (fine resolution input capture)
      focal2: 96→96, k=5, s=2    T → T/2        (first downsample)
      focal3: 96→96, k=3, s=2    T → T/4        (second downsample)
      focal4: 96→96, k=3, s=2    T/4 → T/8      (third downsample)
      bottleneck: 96→32, k=1     T/8 → T/8

    Decoder (symmetric):
      expand1: 32→96, k=3, s=1   T/8 → T/8
      expand2: 96→96, k=3, s=2   T/8 → T/4      (first upsample)
      expand3: 96→96, k=3, s=2   T/4 → T/2      (second upsample)
      expand4: 96→96, k=5, s=2   T/2 → T        (third upsample)
      output:  96→21, k=1        T → T

    Total stride: 8x. Latent: [B, 32, T/8]
    For T=2500: latent [B, 32, 312]

    Encoder packed: ~42.3KB / 43KB SRAM4 (98.4% utilization)
    """
    def __init__(self, in_ch=21, latent_dim=32):
        super().__init__()
        W = 96

        # --- ENCODER (4 layers + bottleneck) ---
        self.focal1 = TernaryFocalBlock(in_ch, W, kernel_size=7, stride=1)
        self.focal2 = TernaryFocalBlock(W, W, kernel_size=5, stride=2, groups=1)
        self.focal3 = TernaryFocalBlock(W, W, kernel_size=3, stride=2, groups=1)
        self.focal4 = TernaryFocalBlock(W, W, kernel_size=3, stride=2, groups=1)
        self.bottleneck = TernaryConv1d(W, latent_dim, 1, stride=1, bias=True)

        # --- DECODER (symmetric 4 layers + output) ---
        self.expand1 = TernaryFocalBlock(latent_dim, W, kernel_size=3, stride=1)
        self.expand2 = TernaryUpsampleBlock(W, W, kernel_size=3, stride=2, groups=1)
        self.expand3 = TernaryUpsampleBlock(W, W, kernel_size=3, stride=2, groups=1)
        self.expand4 = TernaryUpsampleBlock(W, W, kernel_size=5, stride=2, groups=1)
        self.output = nn.Conv1d(W, in_ch, 1)

    def encode(self, x, quantize=True):
        x = self.focal1(x, quantize=quantize)
        x = self.focal2(x, quantize=quantize)
        x = self.focal3(x, quantize=quantize)
        x = self.focal4(x, quantize=quantize)
        return self.bottleneck(x, quantize=quantize)

    def forward(self, x, quantize=True):
        input_len = x.shape[2]
        lat = self.encode(x, quantize=quantize)
        h = self.expand1(lat, quantize=quantize)
        h = self.expand2(h, quantize=quantize)
        h = self.expand3(h, quantize=quantize)
        h = self.expand4(h, quantize=quantize)
        out = self.output(h)
        return out[:, :, :input_len]


# ============================================================
# Gen 7.1 "Subband" Architecture
# ============================================================
#
# Input: L3 approximation [B, 21, 313] (from 3-level lifting DWT)
# The lifting already did the 8× temporal reduction, so the TNN
# no longer needs 4 strided layers to downsample. Instead it uses
# 3 focal blocks (width 112) with total stride 4:
#
#   premix:    21→21,  k=1, s=1    313 → 313     (spatial decorrelation)
#   focal1:    21→112, k=7, s=2    313 → 157     (first downsample)
#   focal2:   112→112, k=5, s=1    157 → 157     (feature extraction)
#   focal3:   112→112, k=3, s=2    157 → 79      (second downsample)
#   bneck_v:  112→32,  k=1, s=1    79  → 79      (value path)
#   bneck_g:  112→32,  k=1, s=1    79  → 79      (gate path, GLU)
#
# Latent: [B, 32, 79] (was [B, 32, 312])
# Total stride: 2 × 1 × 2 = 4 on 313 samples → 79 (with ceil padding)
#
# Encoder packed: ~38 KB ternary + ~4 KB Q31 = 42 KB / 43 KB SRAM4
#
# Compression math (L=16, 4 bps):
#   Raw:     21 × 2500 × 16 = 840,000 bits
#   Latent:  32 × 79 × 4 = 10,112 bits → before rANS
#   CR ≈ 840,000 / 10,112 = 83x (before entropy coding)
#   With rANS + detail subbands: target 40-80x depending on quality mode


class _ZeroPadShortcut(nn.Module):
    """Channel-padding shortcut: zero-pad narrow input to match wider output.

    Replaces the learned TernaryConv1d(21→112, k=1) projection that costs
    2,352 ternary weights (588 bytes). Zero-padding is free at inference
    and lets focal1's main conv handle the channel mixing instead.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.pad_ch = out_ch - in_ch
        self.stride = stride

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, ::self.stride]
        if self.pad_ch > 0:
            return F.pad(x, (0, 0, 0, self.pad_ch))  # pad channels dim
        return x


class TernaryMobileNetV5_Subband(nn.Module):
    """
    Gen 7.1 Subband TNN: N-layer stride-4 encoder on L3 approximation (width configurable, default 128).

    Architecture changes from analysis of L3 subband properties:

    1. Kernel sizes flipped to "narrow early, wide late" (was 7→5→3):
       L3 is already smooth (lowpass filtered 3×). Wide kernels at the
       start waste capacity; at 79 samples, k=7 covers 9% of the sequence
       which corresponds to ~225ms — one full spike-wave complex duration.

    2. Gated SSM bottleneck: depthwise causal conv before GLU gate gives
       3 timesteps of temporal context at the gating decision point.
       Adds only 336 ternary weights (84 bytes). Every top Mamba/SSM
       gating mechanism includes local temporal context before the gate.

    3. Zero-pad shortcut in focal1: replaces learned TernaryConv1d(21→112)
       projection. Saves 588 bytes and one ternary conv from critical path.

    4. GroupNorm(8) instead of GroupNorm(4): finer normalization matches
       spatial channel groups (frontal/central/temporal/occipital have
       different amplitude scales). Zero memory cost.

    Configurable:
      n_blocks:     number of focal blocks (default 3, minimum 2)
      kernel_sizes: per-block kernel sizes (default (3, 5, 7))
      First block always has stride=2 (with ZeroPadShortcut).
      Last block always has stride=2 (spatial downsampling before bottleneck).
      Middle blocks (if any) have stride=1.

    Default (n_blocks=3, kernel_sizes=(3,5,7)):
      premix:     21→21,   k=1, s=1    313 → 313
      focal1:     21→W,    k=3, s=2    313 → 157  (zero-pad shortcut)
      focal2:     W→W,     k=5, s=1    157 → 157
      focal3:     W→W,     k=7, s=2    157 → 79   (wide late for temporal context)
      dw_gate:    W→W,     k=3, dw     79 → 79    (causal temporal context for gate)
      bneck_v:    W→32,    k=1         79 → 79    (value path)
      bneck_g:    W→32,    k=1         79 → 79    (gate path, after dw_gate)

    Decoder (symmetric, training only):
      expand1:  32→W, k=7, s=1    79 → 79
      expand2:  W→W,  k=5, s=2    79 → 157
      expand3:  W→W,  k=3, s=2    157 → 313
      output:   W→21, k=1         313 → 313

    Total stride: 4x. Latent: [B, 32, 79]
    """
    def __init__(self, in_ch=21, latent_dim=32, width=128, cdf_entries=32,
                 n_blocks=3, kernel_sizes=(3, 5, 7)):
        super().__init__()
        assert len(kernel_sizes) == n_blocks, (
            f"Need {n_blocks} kernel sizes, got {len(kernel_sizes)}")
        assert n_blocks >= 2, "Need at least 2 blocks (first stride-2 + last stride-2)"

        W = width
        N_GROUPS = 8  # finer than 4: matches spatial channel groups
        self.n_blocks = n_blocks
        self.kernel_sizes = tuple(kernel_sizes)

        # --- ENCODER ---
        self.premix = TernaryConv1d(in_ch, in_ch, 1, stride=1)

        # First block: in_ch → W, stride=2, with ZeroPadShortcut
        self.focal1_conv = TernaryConv1d(in_ch, W, kernel_sizes[0], stride=2)
        self.focal1_norm = nn.GroupNorm(N_GROUPS, W)
        self.focal1_shortcut = _ZeroPadShortcut(in_ch, W, stride=2)

        # Middle blocks: W → W, stride=1 (dynamic count via ModuleList)
        self.focal_mid = nn.ModuleList()
        for i in range(1, n_blocks - 1):
            blk = TernaryFocalBlock(W, W, kernel_size=kernel_sizes[i], stride=1)
            blk.norm = nn.GroupNorm(N_GROUPS, W)
            self.focal_mid.append(blk)

        # Last block: W → W, stride=2
        self.focal_last = TernaryFocalBlock(W, W, kernel_size=kernel_sizes[-1], stride=2)
        self.focal_last.norm = nn.GroupNorm(N_GROUPS, W)

        # --- Backward compat aliases (old state_dict keys: focal2, focal3) ---
        # When n_blocks==3, focal_mid[0] IS the old focal2, and focal_last IS
        # the old focal3.  Register property-style aliases so that
        # state_dict keys like "focal2.conv.weight" still resolve during
        # load_state_dict(strict=False) from old checkpoints.
        if n_blocks == 3:
            self.focal2 = self.focal_mid[0]
            self.focal3 = self.focal_last

        # Gated SSM bottleneck: depthwise temporal conv before gate
        # gives 3 timesteps of context at the gating decision.
        # 112 channels × k=3 depthwise = 336 ternary weights = 84 bytes.
        self.dw_gate = TernaryConv1d(W, W, 3, stride=1, groups=W)
        # bneck_v: INT8 projection (the information bottleneck — ternary too coarse)
        # Conv1d(128→32, k=1): 4096 weights. INT8 = 4 KB vs ternary = 0.5 KB.
        # Gives 256-level fine-grained channel mixing instead of {-1,0,+1} selection.
        self.bneck_v = INT8Conv1d(W, latent_dim, 1, stride=1, bias=True)
        self.bneck_g = TernaryConv1d(W, latent_dim, 1, stride=1, bias=True)

        # Learned orthogonal rotation (replaces fixed WHT for FSQ pre-rotation).
        # Cayley parameterization: Q = (I-A)(I+A)^{-1} where A is skew-symmetric.
        # Learns to rotate latent into coordinates where the first K dimensions
        # carry most reconstruction-relevant variance → fewer FSQ dead codes.
        # Tests show 0% dead codes vs 6.2% with fixed WHT.
        # Size: 32×32 skew-symmetric = 496 free parameters (~2 KB at FP32,
        # exported as Q31 int32 = 4 KB, or INT8 = 1 KB).
        self.rotation_A = nn.Parameter(torch.zeros(latent_dim, latent_dim) * 0.01)

        # Empirical CDF lookup table: per-channel quantile breakpoints.
        # 32 breakpoints per channel → binary search (5 comparisons) + lerp.
        # Maps ANY latent distribution to uniform [-1, 1], regardless of shape.
        # Handles wildly varying per-channel kurtosis (0 to 700+) optimally.
        # Computed once from training set latent statistics, frozen forever.
        # Firmware: 32 channels × 32 entries × 2 bytes (INT16) = 2 KB.
        # STE: gradient flows through the linear interpolation segments.
        N_CDF_ENTRIES = cdf_entries
        self.n_cdf_entries = N_CDF_ENTRIES
        # cdf_breakpoints[c, i] = the latent value at quantile i/(N-1)
        # Initialized to linear ramp [-3, 3] (Gaussian assumption, overwritten at init)
        self.register_buffer('cdf_breakpoints',
                             torch.linspace(-3, 3, N_CDF_ENTRIES).unsqueeze(0).expand(latent_dim, -1).clone())

        # --- DECODER (symmetric, training only — not deployed to firmware) ---
        self.expand1 = TernaryFocalBlock(latent_dim, W, kernel_size=7, stride=1)
        self.expand2 = TernaryUpsampleBlock(W, W, kernel_size=5, stride=2)
        self.expand3 = TernaryUpsampleBlock(W, W, kernel_size=3, stride=2)
        self.output = nn.Conv1d(W, in_ch, 1)

        # 32×32 normalized Hadamard matrix for activation quantization.
        # Registered as non-persistent buffer (fixed constant, not saved in state_dict).
        # All child modules get a _hadamard_ref pointing to this buffer.
        # CUDA-graph safe: Dynamo sees a static buffer access, no dict lookup.
        self.register_buffer('_hadamard_32', _build_hadamard_32_cpu(), persistent=False)
        self._refresh_hadamard_refs()

    def _refresh_hadamard_refs(self):
        """Point all child modules' _hadamard_ref to the current buffer.
        Must be called after .to(device) since Python refs don't follow buffer moves."""
        for m in self.modules():
            if isinstance(m, (TernaryConv1d, INT8Conv1d, TernaryConvTranspose1d,
                              TernaryFocalBlock, TernaryUpsampleBlock)):
                m._hadamard_ref = self._hadamard_32

    def _apply(self, fn):
        """Override to re-set Hadamard refs after .to()/.cuda()/.cpu() moves buffers."""
        result = super()._apply(fn)
        self._refresh_hadamard_refs()
        return result

    def _get_rotation(self):
        """Compute orthogonal rotation matrix from skew-symmetric parameters."""
        A = self.rotation_A - self.rotation_A.T  # enforce skew-symmetry
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return torch.linalg.solve(I + A, I - A)

    def _cdf_forward(self, latent):
        """Empirical CDF forward: latent → uniform [-1, 1] via quantile LUT.

        For each channel, binary-search into cdf_breakpoints to find the
        segment, then linearly interpolate to get the output quantile.
        Output range: [-1, 1] (maps breakpoint[0] → -1, breakpoint[N-1] → +1).
        STE: gradient flows through the linear interpolation (piecewise linear).
        Values outside the breakpoint range are clamped (saturated).
        """
        B, C, T = latent.shape
        bp = self.cdf_breakpoints  # [C, N]
        N = bp.shape[1]

        # searchsorted: for each value, find which segment it falls in
        # latent_flat: [C, B*T], bp: [C, N]
        lat_flat = latent.permute(1, 0, 2).reshape(C, -1)  # [C, B*T]
        # torch.searchsorted expects sorted input along last dim
        idx = torch.searchsorted(bp, lat_flat)  # [C, B*T], values in [0, N]
        idx = idx.clamp(1, N - 1)  # clamp to valid segment range [1, N-1]

        # Gather segment endpoints
        lo_idx = idx - 1
        bp_lo = bp.gather(1, lo_idx)  # [C, B*T]
        bp_hi = bp.gather(1, idx)     # [C, B*T]

        # Linear interpolation within segment
        span = (bp_hi - bp_lo).clamp(min=1e-8)
        frac = (lat_flat - bp_lo) / span  # [0, 1] within segment
        frac = frac.clamp(0, 1)

        # Map segment index + frac to uniform [-1, 1]
        # Quantile position: (lo_idx + frac) / (N - 1) → [0, 1] → [-1, 1]
        uniform = (lo_idx.float() + frac) / (N - 1) * 2.0 - 1.0

        # STE: use uniform in forward, but let gradient flow through latent
        # The piecewise-linear interpolation is already differentiable w.r.t. latent
        # (gradient = 1/span per segment, which is the local PDF estimate)
        return uniform.reshape(C, B, T).permute(1, 0, 2)  # [B, C, T]

    def _cdf_inverse(self, uniform):
        """Inverse empirical CDF: uniform [-1, 1] → latent space.

        Reverse lookup: given a uniform value, find the corresponding latent
        value by interpolating the breakpoint table in reverse.
        """
        B, C, T = uniform.shape
        bp = self.cdf_breakpoints  # [C, N]
        N = bp.shape[1]

        u_flat = uniform.permute(1, 0, 2).reshape(C, -1)  # [C, B*T]

        # Map [-1, 1] → [0, N-1] continuous index
        cont_idx = (u_flat + 1.0) / 2.0 * (N - 1)  # [0, N-1]
        cont_idx = cont_idx.clamp(0, N - 1)

        lo_idx = cont_idx.long().clamp(0, N - 2)
        frac = cont_idx - lo_idx.float()

        bp_lo = bp.gather(1, lo_idx)
        bp_hi = bp.gather(1, lo_idx + 1)

        latent = bp_lo + frac * (bp_hi - bp_lo)
        return latent.reshape(C, B, T).permute(1, 0, 2)  # [B, C, T]

    def encode_stage1(self, x, quantize=True):
        """Stage 1: premix → focal1 (313→157). For SMoDi distillation."""
        x = self.premix(x, quantize=quantize)
        identity = self.focal1_shortcut(x)
        out = F.relu(self.focal1_norm(self.focal1_conv(x, quantize=quantize)))
        out = _quantize_activation(out + identity, enabled=quantize, hadamard=self._hadamard_32)
        return out

    def encode_stage2(self, x, quantize=True):
        """Stage 2: middle blocks (all stride-1). For SMoDi distillation.

        For the default 3-block config this is a single block (the old focal2).
        For deeper configs this runs all middle blocks sequentially.
        """
        for block in self.focal_mid:
            x = block(x, quantize=quantize)
        return x

    def encode_stage3(self, x, quantize=True):
        """Stage 3: focal_last → GLU bottleneck → rotation → CDF (157→79→[32,79])."""
        x = self.focal_last(x, quantize=quantize)
        x_gated = self.dw_gate(x, quantize=quantize)
        value = self.bneck_v(x, quantize=quantize)
        gate = torch.sigmoid(self.bneck_g(x_gated, quantize=quantize))
        latent = value * gate
        Q = self._get_rotation()
        latent = torch.einsum('ij,bjt->bit', Q, latent)
        latent = self._cdf_forward(latent)
        if self.training and quantize:
            fsq_levels = [2, 3, 5, 8, 16, 32]
            L = fsq_levels[torch.randint(0, len(fsq_levels), (1,)).item()]
            step = 2.0 / L
            noise = (torch.rand_like(latent) - 0.5) * step
            latent = latent + noise
        return latent

    def encode(self, x, quantize=True):
        x = self.encode_stage1(x, quantize=quantize)
        x = self.encode_stage2(x, quantize=quantize)
        x = self.focal_last(x, quantize=quantize)

        # Gated SSM bottleneck: temporal context before gate
        x_gated = self.dw_gate(x, quantize=quantize)
        value = self.bneck_v(x, quantize=quantize)
        gate = torch.sigmoid(self.bneck_g(x_gated, quantize=quantize))
        latent = value * gate  # [B, 32, 79]

        # Learned orthogonal rotation (replaces fixed WHT pre-rotation for FSQ).
        # Rotate along the channel dimension: [B, 32, 79] → matmul with [32, 32]
        Q = self._get_rotation()
        latent = torch.einsum('ij,bjt->bit', Q, latent)

        # Empirical CDF: map latent → uniform [-1, 1] via quantile LUT.
        # Binary search + linear interpolation. STE through lerp segments.
        latent = self._cdf_forward(latent)

        # StableCodec DitheredFSQ-dropout (ported from reference implementation)
        # Per-sample Bernoulli mask: 50% passthrough, 50% dithered quantization.
        # Randomly varies FSQ level per sample, producing robust adaptive encoding.
        # Source: Reference Software/stable_codec/repo/stable_codec/fsq.py
        if self.training and quantize:
            fsq_levels = [2, 3, 5, 8, 16, 32]
            L = fsq_levels[torch.randint(0, len(fsq_levels), (1,)).item()]
            step = 2.0 / L
            # StableCodec approach: Bernoulli mask per sample
            mask_pass = torch.bernoulli(torch.full(
                (latent.shape[0], 1, 1), 0.5, device=latent.device)).bool()
            mask_noise = torch.bernoulli(torch.full(
                (latent.shape[0], 1, 1), 0.5, device=latent.device)).bool()
            # Where mask_pass: keep unquantized (pass through)
            # Where mask_noise: add uniform dither noise
            # Else: hard quantize with STE
            noise = (torch.rand_like(latent) - 0.5) * step
            q_hard = torch.round(latent / step) * step  # hard FSQ
            q_hard = latent + (q_hard - latent).detach()  # STE
            latent = torch.where(mask_pass.expand_as(latent), latent,
                     torch.where(mask_noise.expand_as(latent),
                                 latent + noise, q_hard))

        return latent

    def decode(self, lat, target_len, quantize=True):
        # Inverse empirical CDF: uniform [-1, 1] → latent space via inverse LUT.
        lat = self._cdf_inverse(lat)

        # Inverse rotation before decoder (Q^{-1} = Q^T for orthogonal)
        Q = self._get_rotation()
        lat = torch.einsum('ij,bjt->bit', Q.T, lat)

        h = self.expand1(lat, quantize=quantize)
        h = self.expand2(h, quantize=quantize)
        h = self.expand3(h, quantize=quantize)
        out = self.output(h)
        return out[:, :, :target_len]

    def ensure_initialized(self):
        """Initialize all quantization parameters. Call before torch.compile."""
        for m in self.modules():
            if hasattr(m, 'ensure_initialized') and m is not self:
                m.ensure_initialized()

    def forward(self, x, quantize=True):
        input_len = x.shape[2]
        lat = self.encode(x, quantize=quantize)
        return self.decode(lat, input_len, quantize=quantize)


    @classmethod
    def from_checkpoint(cls, path, device='cpu', **kwargs):
        """Load a checkpoint, auto-detecting width and architecture variant.

        Handles old checkpoints that used fixed focal2/focal3 attribute names
        (before the configurable n_blocks refactor) by loading with strict=False.
        The backward-compat aliases (focal2 → focal_mid[0], focal3 → focal_last)
        ensure state_dict keys match when n_blocks==3.
        """
        import torch
        try:
            sd = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            sd = torch.load(path, map_location=device, weights_only=False)
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        # Detect V2 (DW-sep): has focal2.dw.weight instead of focal2.conv.weight
        is_v2 = 'focal2.dw.weight' in sd or 'focal4.dw.weight' in sd
        if is_v2:
            # V2: width from pointwise layer
            width = sd.get('focal2.pw.weight', sd.get('focal1_conv.weight', None))
            width = width.shape[0] if width is not None else 216
            model = TernaryMobileNetV5_Subband_V2(width=width, **kwargs).to(device)
        else:
            # V1: width from focal2 conv (or focal_mid.0 for new checkpoints)
            if 'focal2.conv.weight' in sd:
                width = sd['focal2.conv.weight'].shape[0]
            elif 'focal_mid.0.conv.weight' in sd:
                width = sd['focal_mid.0.conv.weight'].shape[0]
            else:
                width = 128
            # Detect n_blocks from checkpoint keys
            n_blocks = kwargs.pop('n_blocks', None)
            kernel_sizes = kwargs.pop('kernel_sizes', None)
            if n_blocks is None:
                # Count focal_mid.N keys to infer block count
                mid_indices = set()
                for k in sd:
                    if k.startswith('focal_mid.'):
                        idx = int(k.split('.')[1])
                        mid_indices.add(idx)
                if mid_indices:
                    n_blocks = len(mid_indices) + 2  # +1 for focal1, +1 for focal_last
                else:
                    n_blocks = 3  # old checkpoint with focal2/focal3
            if kernel_sizes is None:
                kernel_sizes = (3, 5, 7) if n_blocks == 3 else tuple([3] * n_blocks)
            model = cls(width=width, n_blocks=n_blocks,
                        kernel_sizes=kernel_sizes, **kwargs).to(device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"    [from_checkpoint] Skipped {len(unexpected)} old params")
        return model


class TernaryMobileNetV5_Subband_V2(TernaryMobileNetV5_Subband):
    """Gen 7.6.1 Subband TNN: width 144, 4 focal blocks, DW-sep.

    Upgrade from V1 (width 128, 3 focal):
      - Width 144 (+12.5% wider channels)
      - 4th depthwise-separable focal block (more temporal depth)
      - focal2/3/4 use DW-sep (6.6x more efficient per param)
      - focal1 stays full conv (narrow 21->144, DW-sep not beneficial)

    Firmware footprint (encoder only, measured):
      Weights:     ~113 KB (SRAM4 + SRAM5 + partial SRAM6)
      Activations: ~82 KB peak (SRAM6-7, shared with Mode 2 details)
      Total:       ~195 KB / 213 KB budget (18 KB headroom)

    Encoder:
      premix:     21->21,   k=1, s=1        313 -> 313
      focal1:     21->144,  k=3, s=2        313 -> 157  (full conv, zero-pad shortcut)
      focal2:    144->144,  k=5, s=1, DW-sep 157 -> 157
      focal3:    144->144,  k=5, s=2, DW-sep 157 -> 79
      focal4:    144->144,  k=7, s=1, DW-sep  79 -> 79   (wide late, max temporal context)
      dw_gate:   144->144,  k=3, depthwise    79 -> 79
      bneck_v:   144->32,   k=1, INT8         79 -> 79
      bneck_g:   144->32,   k=1, ternary      79 -> 79
      latent = bneck_v(x) * ReLU(bneck_g(dw_gate(x)))   (ReGLU bottleneck)

    Decoder (training only, not deployed):
      expand1:  32->144, k=7, s=1
      expand2: 144->144, k=5, s=2
      expand3: 144->144, k=3, s=2
      output:  144->21,  k=1

    Total stride: 4x. Latent: [B, 32, 79]
    """
    def __init__(self, in_ch=21, latent_dim=32, width=216, cdf_entries=32):
        # Skip parent __init__, build from scratch with DW-sep blocks
        nn.Module.__init__(self)
        W = width
        N_GROUPS = 8 if W % 8 == 0 else 4

        # --- ENCODER ---
        self.premix = TernaryConv1d(in_ch, in_ch, 1, stride=1)

        # focal1: full conv (narrow input, DW-sep not beneficial for 21->144)
        self.focal1_conv = TernaryConv1d(in_ch, W, 3, stride=2)
        self.focal1_norm = nn.GroupNorm(N_GROUPS, W)
        self.focal1_shortcut = _ZeroPadShortcut(in_ch, W, stride=2)

        # focal2-4: depthwise-separable (6.6x more efficient)
        self.focal2 = TernaryDWSepFocalBlock(W, W, kernel_size=5, stride=1)
        self.focal3 = TernaryDWSepFocalBlock(W, W, kernel_size=5, stride=2)
        self.focal4 = TernaryDWSepFocalBlock(W, W, kernel_size=7, stride=1)

        # Gated ReGLU bottleneck
        self.dw_gate = TernaryConv1d(W, W, 3, stride=1, groups=W)
        self.bneck_v = INT8Conv1d(W, latent_dim, 1, stride=1, bias=True)
        self.bneck_g = TernaryConv1d(W, latent_dim, 1, stride=1, bias=True)

        # Learned orthogonal rotation (Cayley)
        self.rotation_A = nn.Parameter(torch.zeros(latent_dim, latent_dim) * 0.01)

        # Empirical CDF-LUT
        N_CDF_ENTRIES = cdf_entries
        self.n_cdf_entries = N_CDF_ENTRIES
        self.register_buffer('cdf_breakpoints',
                             torch.linspace(-3, 3, N_CDF_ENTRIES).unsqueeze(0).expand(latent_dim, -1).clone())

        # --- DECODER (training only) ---
        self.expand1 = TernaryFocalBlock(latent_dim, W, kernel_size=7, stride=1)
        self.expand2 = TernaryUpsampleBlock(W, W, kernel_size=5, stride=2)
        self.expand3 = TernaryUpsampleBlock(W, W, kernel_size=3, stride=2)
        self.output = nn.Conv1d(W, in_ch, 1)

        # Hadamard buffer for activation quantization
        self.register_buffer('_hadamard_32', _build_hadamard_32_cpu(), persistent=False)
        self._refresh_hadamard_refs()

    def encode_stage1(self, x, quantize=True):
        x = self.premix(x, quantize=quantize)
        identity = self.focal1_shortcut(x)
        x = F.relu(self.focal1_norm(self.focal1_conv(x, quantize=quantize)))
        x = x + identity
        x = _quantize_activation(x, enabled=quantize, hadamard=getattr(self, '_hadamard_ref', None))
        return x

    def encode_stage2(self, x, quantize=True):
        x = self.focal2(x, quantize=quantize)
        return x

    def encode_stage3(self, x, quantize=True):
        x = self.focal3(x, quantize=quantize)
        x = self.focal4(x, quantize=quantize)
        return x

    def _encode_bottleneck(self, x, quantize=True):
        x_gated = self.dw_gate(x, quantize=quantize)
        value = self.bneck_v(x, quantize=quantize)
        gate = F.relu(self.bneck_g(x_gated, quantize=quantize))  # ReGLU
        latent = value * gate
        Q = self._get_rotation()
        latent = torch.einsum('ij,bjt->bit', Q, latent)
        latent = self._cdf_forward(latent)
        if self.training and quantize:
            # StableCodec DitheredFSQ dropout
            fsq_levels = [2, 3, 5, 8, 16, 32]
            L = fsq_levels[torch.randint(0, len(fsq_levels), (1,)).item()]
            step = 2.0 / L
            mask_pass = torch.bernoulli(torch.full(
                (latent.shape[0], 1, 1), 0.5, device=latent.device)).bool()
            mask_noise = torch.bernoulli(torch.full(
                (latent.shape[0], 1, 1), 0.5, device=latent.device)).bool()
            noise = (torch.rand_like(latent) - 0.5) * step
            q_hard = torch.round(latent / step) * step
            q_hard = latent + (q_hard - latent).detach()
            latent = torch.where(mask_pass.expand_as(latent), latent,
                     torch.where(mask_noise.expand_as(latent),
                                 latent + noise, q_hard))
        return latent

    def encode(self, x, quantize=True):
        x = self.encode_stage1(x, quantize=quantize)
        x = self.encode_stage2(x, quantize=quantize)
        x = self.encode_stage3(x, quantize=quantize)
        return self._encode_bottleneck(x, quantize=quantize)
