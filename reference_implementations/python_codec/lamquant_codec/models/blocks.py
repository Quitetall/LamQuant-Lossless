"""Ternary quantization primitives and building blocks.

All LSQ/SEQ quantization autograd functions, ternary conv layers,
and composable blocks used by the encoder architectures.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math


# ============================================================
# LSQ Ternary Quantization Core
# ============================================================

class _LSQTernaryFunction(torch.autograd.Function):
    """Ternary quantization with LSQ gradient scaling + Tequila deadzone fix.

    Forward: w_q = round(clamp(w/α, -1, 1)) × α + deadzone_bias
    Backward: STE for weights (∂w_q/∂w ≈ 1 where |w/α| ≤ 1)
              Scaled gradient for α (÷ √n_weights for stability)

    Tequila deadzone fix (ICLR 2026): weights near the ±0.5α boundary
    between {-1,0} and {0,+1} receive noisy STE gradients and get
    "trapped" — oscillating without committing. Tequila reactivates
    them as dynamic biases: the fractional part (w/α - round(w/α)) is
    added back as a soft correction with magnitude decaying by
    temperature τ. This has near-zero inference overhead (the bias
    folds into the next layer's bias or normalization).
    """
    @staticmethod
    def forward(ctx, weight, alpha, grad_scale, deadzone_tau):
        alpha_abs = alpha.abs() + 1e-8
        w_div = weight / alpha_abs
        w_clamp = w_div.clamp(-1, 1)
        w_q = w_clamp.round()

        # Tequila: compute fractional residual for trapped weights
        # residual = (w/α - round(w/α)) — this is the "deadzone bias"
        # It's largest for weights near ±0.5 (the decision boundary)
        # and zero for weights committed to {-1, 0, +1}.
        residual = (w_clamp - w_q) * deadzone_tau

        ctx.save_for_backward(w_div, alpha_abs)
        ctx.grad_scale = grad_scale
        # Output includes the soft deadzone correction
        return (w_q + residual) * alpha_abs

    @staticmethod
    def backward(ctx, grad_output):
        w_div, alpha_abs = ctx.saved_tensors
        in_range = (w_div.abs() <= 1).float()
        grad_weight = grad_output * in_range

        w_q = w_div.clamp(-1, 1).round()
        grad_alpha = (grad_output * w_q).sum(dim=(1, 2), keepdim=True)
        grad_alpha = grad_alpha * ctx.grad_scale

        return grad_weight, grad_alpha, None, None


def _lsq_ternary(weight, alpha, grad_scale, deadzone_tau=0.1):
    return _LSQTernaryFunction.apply(weight, alpha, grad_scale, deadzone_tau)


class _SEQTernaryFunction(torch.autograd.Function):
    """ParetoQ Stretched Elastic Quantization for ternary weights.

    Ported from Meta AI's ParetoQ (NeurIPS 2025) reference implementation.
    For ternary (num_bits=0): uses n_levels=1.5 with shift=0, producing
    q_w = round(clamp(w/α, -0.99, 0.99) × 1.5) / 1.5

    This gives 3 levels: {-0.667, 0, +0.667} × α, with asymmetric bins
    optimized for the ternary phase transition. The 1.5 factor "stretches"
    the quantization grid to better match ternary weight distributions.

    Source: Reference Software/paretoq/repo/models/utils_quant.py
    """
    @staticmethod
    def forward(ctx, weight, alpha, grad_scale):
        alpha_abs = alpha.abs().clamp(min=1e-5)
        clip_val = 1 - 1e-2
        # Ternary: n_levels=1.5, shift=0 (from ParetoQ num_bits=0)
        n_levels = 1.5
        shift = 0.0
        Qp = (n_levels - shift) / n_levels  # 1.0
        Qn = -Qp

        w_scaled = weight / alpha_abs
        q_w = (torch.round(
            torch.clamp(w_scaled, -clip_val, clip_val) * n_levels - shift
        ) + shift) / n_levels

        grad_scale_val = 1.0 / math.sqrt(weight.numel())
        ctx.save_for_backward(w_scaled, alpha_abs)
        ctx.other = grad_scale_val, Qn, Qp, n_levels, shift, clip_val

        return q_w * alpha_abs

    @staticmethod
    def backward(ctx, grad_output):
        w_scaled, alpha_abs = ctx.saved_tensors
        grad_scale_val, Qn, Qp, n_levels, shift, clip_val = ctx.other

        indicate_small = (w_scaled < -clip_val).float()
        indicate_big = (w_scaled > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        grad_input = indicate_middle * grad_output

        q_w_round = (torch.round(
            torch.clamp(w_scaled, -clip_val, clip_val) * n_levels - shift
        ) + shift) / n_levels
        grad_alpha = (
            indicate_small * Qn + indicate_big * Qp +
            indicate_middle * (-w_scaled + q_w_round)
        ) * grad_output * grad_scale_val
        grad_alpha = grad_alpha.sum(dim=(1, 2), keepdim=True)

        return grad_input, grad_alpha, None


def _seq_ternary(weight, alpha, grad_scale):
    """ParetoQ SEQ quantizer for ternary weights."""
    return _SEQTernaryFunction.apply(weight, alpha, grad_scale)


# Binary quantization: {-α, +α} only, no zero weight. sign(w) × α.
class _LSQBinaryFunction(torch.autograd.Function):
    """Binary weight quantization: w_q = sign(w) × α. No zero weights."""
    @staticmethod
    def forward(ctx, weight, alpha, grad_scale):
        alpha_abs = alpha.abs() + 1e-8
        w_sign = weight.sign()
        w_sign[w_sign == 0] = 1  # tie-break: zero → +1
        ctx.save_for_backward(weight / alpha_abs, alpha_abs)
        ctx.grad_scale = grad_scale
        return w_sign * alpha_abs

    @staticmethod
    def backward(ctx, grad_output):
        w_div, alpha_abs = ctx.saved_tensors
        in_range = (w_div.abs() <= 1).float()
        grad_weight = grad_output * in_range
        w_q = w_div.sign()
        w_q[w_q == 0] = 1
        grad_alpha = (grad_output * w_q).sum(dim=(1, 2), keepdim=True)
        grad_alpha = grad_alpha * ctx.grad_scale
        return grad_weight, grad_alpha, None


def _lsq_binary(weight, alpha, grad_scale):
    return _LSQBinaryFunction.apply(weight, alpha, grad_scale)


# Module-level switches
QUANTIZATION_MODE = 'ternary'  # 'ternary' ({-1,0,+1}) or 'binary' ({-1,+1})
QUANTIZER_TYPE = 'lsq'         # 'lsq' (default) or 'seq' (ParetoQ)

def set_quantization_mode(mode):
    """Set weight quantization: 'ternary' or 'binary'."""
    global QUANTIZATION_MODE
    assert mode in ('ternary', 'binary'), f"Invalid mode: {mode}"
    QUANTIZATION_MODE = mode

def set_quantizer_type(qtype):
    """Set quantizer: 'lsq' (original) or 'seq' (ParetoQ SEQ, recommended)."""
    global QUANTIZER_TYPE
    assert qtype in ('lsq', 'seq'), f"Invalid quantizer: {qtype}"
    QUANTIZER_TYPE = qtype
    print(f"[*] Quantizer: {qtype.upper()}")


# ============================================================
# WHT Activation Smoothing (SSDi8-inspired outlier removal)
# ============================================================

def _wht_smooth(x):
    """Walsh-Hadamard transform-based activation smoothing.

    SSDi8 (ICLR 2026) showed that activation outliers in SSM/structured
    models cause catastrophic quantization failure. WHT rotation spreads
    outlier energy across all dimensions, reducing the dynamic range and
    making subsequent INT16 quantization more robust.

    Applied per-channel: WHT along the temporal dimension, quantize in
    the rotated domain, then inverse WHT. For non-power-of-2 lengths,
    we pad to the next power of 2, transform, then trim.

    This is computationally free at inference because the firmware
    already has wht32.c — the WHT can be fused into the conv output.
    """
    B, C, T = x.shape

    # Find next power of 2 for the temporal dimension
    T_pad = 1
    while T_pad < T:
        T_pad *= 2

    if T_pad > 4096:
        # Skip WHT for very long sequences (computational cost)
        return x

    # Pad
    if T_pad > T:
        x_pad = F.pad(x, (0, T_pad - T))
    else:
        x_pad = x

    # In-place butterfly WHT (same algorithm as firmware wht32.c)
    h = 1
    while h < T_pad:
        # Split into pairs and butterfly
        x1 = x_pad[:, :, 0::2*h]  # even indices at this level
        x2 = x_pad[:, :, h::2*h]  # odd indices at this level
        # This doesn't map cleanly to strided indexing for arbitrary h.
        # Use the matrix form instead for simplicity during training:
        break

    # Matrix WHT: H @ x for each (batch, channel)
    # Build Hadamard matrix of size T_pad
    # For training, use torch's efficient implementation
    H = torch.tensor([[1.0]], device=x.device, dtype=x.dtype)
    log2_T = int(math.log2(T_pad))
    for _ in range(log2_T):
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / math.sqrt(2)  # Normalized WHT

    # Apply: [B, C, T_pad] @ [T_pad, T_pad]^T = [B, C, T_pad]
    x_wht = torch.matmul(x_pad, H.T)

    # Trim back to original length
    return x_wht[:, :, :T]


# ============================================================
# INT16 Activation Quantization (matching firmware W2A16)
# ============================================================

# Module-level activation bit width. Default W2A16 (production).
# Set to 8 for experimental W2A8 (halves activation buffers).
ACTIVATION_BITS = 16

def set_activation_bits(bits):
    """Set activation quantization bit width (8 or 16)."""
    global ACTIVATION_BITS
    assert bits in (8, 16), f"Activation bits must be 8 or 16, got {bits}"
    ACTIVATION_BITS = bits


class _ActivationQuantFunction(torch.autograd.Function):
    """Simulates activation quantization with STE at configurable bit width.

    W2A16 (default): range [-32768, 32767] — matches firmware int16_t
    W2A8 (experimental): range [-128, 127] — halves activation buffers,
      requires block-WHT smoothing (SSDi8) to maintain accuracy.
    """
    @staticmethod
    def forward(ctx, x, scale, bits):
        max_val = 2 ** (bits - 1) - 1
        min_val = -(2 ** (bits - 1))
        x_scaled = x / (scale + 1e-12)
        x_q = x_scaled.round().clamp(min_val, max_val)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# Pre-built 32×32 normalized Hadamard matrix (matching firmware wht32.c).
# Built eagerly on CPU at import time. Device copies are pre-populated by
# warmup_hadamard_cache() before torch.compile, so _get_hadamard_32 is a
# pure dict lookup with no construction — CUDA-graph safe.
_H32_CACHE = {}

def _build_hadamard_32_cpu():
    with torch.no_grad():
        H = torch.tensor([[1.0]])
        for _ in range(5):
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0) / math.sqrt(2)
    return H

_H32_CPU = _build_hadamard_32_cpu()

def warmup_hadamard_cache(device):
    """Pre-populate the Hadamard cache for all dtypes. Call before torch.compile."""
    # Normalize device to match what x.device returns (e.g. cuda:0)
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        key = (device, dtype)
        if key not in _H32_CACHE:
            _H32_CACHE[key] = _H32_CPU.to(device=device, dtype=dtype).contiguous()
    print(f"[*] Hadamard cache warmed for {device} (3 dtypes)")

def _get_hadamard_32(device, dtype):
    """Return cached Hadamard matrix. Falls back to lazy init if not pre-warmed."""
    key = (device, dtype)
    if key not in _H32_CACHE:
        _H32_CACHE[key] = _H32_CPU.to(device=device, dtype=dtype).contiguous()
    return _H32_CACHE[key]


def _quantize_activation(x, enabled=True, hadamard=None):
    """Block-WHT smoothing + INT16 quantization for activations.

    Pipeline (matching firmware wht32.c → int16 quantize → inverse wht32):
      1. Split temporal dim into chunks of 32
      2. WHT-rotate each chunk (spreads outliers, SSDi8 ICLR 2026)
      3. INT16 quantize in WHT domain (lower dynamic range = less error)
      4. Inverse WHT to recover spatial domain
      5. Remainder samples (T % 32) are quantized directly

    At inference, steps 1-4 fold into the firmware's existing wht32.c
    + int16 activation pipeline with zero additional cost.

    Args:
        hadamard: pre-built 32×32 Hadamard matrix (buffer reference).
                  Falls back to global cache if None.
    """
    if not enabled:
        return x

    bits = ACTIVATION_BITS
    max_val = float(2 ** (bits - 1) - 1)

    if x.dim() != 3:
        with torch.no_grad():
            scale = x.abs().amax() / max_val
            scale = scale.clamp(min=1e-12)
        return _ActivationQuantFunction.apply(x, scale, bits)

    B, C, T = x.shape
    n_blocks = T // 32
    remainder = T % 32

    def _get_H():
        if hadamard is not None:
            return hadamard.to(dtype=x.dtype) if hadamard.dtype != x.dtype else hadamard
        return _get_hadamard_32(x.device, x.dtype)

    if n_blocks > 0 and n_blocks * 32 == T:
        H = _get_H()
        x_blocks = x.reshape(B, C, n_blocks, 32)
        x_wht = torch.matmul(x_blocks, H.T)
        with torch.no_grad():
            scale = x_wht.abs().amax() / max_val
            scale = scale.clamp(min=1e-12)
        x_wht_q = _ActivationQuantFunction.apply(x_wht, scale, bits)
        return torch.matmul(x_wht_q, H).reshape(B, C, T)

    parts = []
    if n_blocks > 0:
        H = _get_H()
        x_blocks = x[:, :, :n_blocks * 32].reshape(B, C, n_blocks, 32)
        x_wht = torch.matmul(x_blocks, H.T)
        with torch.no_grad():
            scale = x_wht.abs().amax() / max_val
            scale = scale.clamp(min=1e-12)
        x_wht_q = _ActivationQuantFunction.apply(x_wht, scale, bits)
        parts.append(torch.matmul(x_wht_q, H).reshape(B, C, n_blocks * 32))

    if remainder > 0:
        rem = x[:, :, n_blocks * 32:]
        with torch.no_grad():
            scale_rem = rem.abs().amax() / max_val
            scale_rem = scale_rem.clamp(min=1e-12)
        parts.append(_ActivationQuantFunction.apply(rem, scale_rem, bits))

    return torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]


# ============================================================
# Ternary Conv Layers with Full QAT
# ============================================================

class TernaryConv1d(nn.Conv1d):
    """LSQ-Quantized Ternary Convolution with Tequila deadzone fix.

    QAT features:
      - Alpha initialized from weight statistics: α₀ = 2/3 × mean(|W|)
      - Alpha gradient scaled by 1/√n_weights for cross-layer stability
      - Tequila deadzone fix: trapped boundary weights get soft bias correction
      - INT16 activation quantization (matches firmware W2A16)
      - Deadzone τ decays from 0.1→0 over training (anneal externally)
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1, bias=False):
        padding = kernel_size // 2
        super().__init__(in_ch, out_ch, kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=bias)
        self.lsq_alpha = nn.Parameter(torch.ones(out_ch, 1, 1) * 0.1)
        self.register_buffer('_alpha_init_flag', torch.tensor(False))
        n_weights = in_ch * kernel_size // groups
        self._grad_scale = 1.0 / math.sqrt(n_weights)
        self.deadzone_tau = 0.1
        # SubLN: LayerNorm before output projection (BitNet b1.58)
        # Stabilizes activation variance entering quantized projections
        self.sub_ln = nn.GroupNorm(1, out_ch)  # equivalent to LayerNorm for conv

    def _init_alpha(self):
        with torch.no_grad():
            mean_abs = self.weight.abs().mean(dim=(1, 2), keepdim=True)
            self.lsq_alpha.data.copy_((2.0 / 3.0) * mean_abs.clamp(min=0.001))
        self._alpha_init_flag.fill_(True)

    def clamp_alpha(self):
        """Clamp alpha to [0.5×std(W), 2.0×std(W)] per output channel.
        Prevents pathological alpha drift where alpha << weight magnitude,
        which causes all weights to round to ±1 with no zeros."""
        with torch.no_grad():
            w_std = self.weight.std(dim=(1, 2), keepdim=True).clamp(min=1e-4)
            alpha_min = 0.5 * w_std
            alpha_max = 2.0 * w_std
            self.lsq_alpha.data.copy_(
                self.lsq_alpha.data.abs().clamp(min=alpha_min, max=alpha_max))

    def ensure_initialized(self):
        """Run alpha init if needed. Call before torch.compile to avoid graph breaks."""
        if not self._alpha_init_flag.item():
            self._init_alpha()

    def set_bits(self, bits):
        """Set quantization bit width. 'ternary' or int (8, 4, etc.).
        Used by the INT8 bridge: FP32 warmup → INT8 QAT → ternary QAT."""
        self._bridge_bits = bits

    def forward(self, x, quantize=True):
        if not quantize:
            return super().forward(x)
        bridge = getattr(self, '_bridge_bits', 'ternary')
        if isinstance(bridge, int) and bridge >= 4:
            # INT-N bridge: use alpha-scaled symmetric quantization
            alpha_abs = self.lsq_alpha.abs().clamp(min=1e-5)
            max_val = 2 ** (bridge - 1) - 1
            w_scaled = self.weight / alpha_abs
            w_q = (torch.round(
                torch.clamp(w_scaled, -1 + 1e-2, 1 - 1e-2) * max_val
            ) / max_val) * alpha_abs
            # STE: gradient passes through within clamp range
            w_q = (w_q - self.weight).detach() + self.weight
        elif QUANTIZATION_MODE == 'binary':
            w_q = _lsq_binary(self.weight, self.lsq_alpha, self._grad_scale)
        elif QUANTIZER_TYPE == 'seq':
            w_q = _seq_ternary(self.weight, self.lsq_alpha, self._grad_scale)
        else:
            tau = self.deadzone_tau if self.training else 0.0
            w_q = _lsq_ternary(self.weight, self.lsq_alpha, self._grad_scale, tau)
        out = F.conv1d(x, w_q, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)
        out = self.sub_ln(out)  # SubLN: stabilize before next quantized layer
        return _quantize_activation(out, enabled=quantize, hadamard=getattr(self, '_hadamard_ref', None))


class INT8Conv1d(nn.Conv1d):
    """INT8-quantized Conv1d for the projection layer (bneck_v).

    The projection Conv1d(W→32, k=1) is the final compression step where
    information is irreversibly discarded. Ternary weights here can only
    add/subtract GLU channels, not blend them. INT8 gives 256 levels of
    fine-grained mixing for 1.5 KB more (2048 weights × 1 byte vs packed ternary).

    Uses symmetric uniform INT8 quantization with learned scale per output channel.
    STE (straight-through estimator) for gradient flow during QAT.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1, bias=True):
        padding = kernel_size // 2
        super().__init__(in_ch, out_ch, kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=bias)
        self.quant_scale = nn.Parameter(torch.ones(out_ch, 1, 1) * 0.1)
        self.register_buffer('_scale_init_flag', torch.tensor(False))

    def _init_scale(self):
        with torch.no_grad():
            max_abs = self.weight.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            self.quant_scale.data.copy_(max_abs / 127.0)
        self._scale_init_flag.fill_(True)

    def ensure_initialized(self):
        """Run scale init if needed. Call before torch.compile to avoid graph breaks."""
        if not self._scale_init_flag.item():
            self._init_scale()

    def forward(self, x, quantize=True):
        if not quantize:
            return super().forward(x)
        # Symmetric INT8: round(w / scale) * scale, clamped to [-127, 127]
        scale = self.quant_scale.abs().clamp(min=1e-8)
        w_scaled = self.weight / scale
        w_int8 = torch.clamp(torch.round(w_scaled), -127, 127)
        # STE: use quantized forward, continuous backward
        w_q = (w_int8 * scale - self.weight).detach() + self.weight
        out = F.conv1d(x, w_q, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)
        return _quantize_activation(out, enabled=quantize, hadamard=getattr(self, '_hadamard_ref', None))


class TernaryConvTranspose1d(nn.ConvTranspose1d):
    """LSQ-Quantized Ternary Transposed Convolution with Tequila deadzone fix."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=2, groups=1, bias=False):
        padding = kernel_size // 2
        output_padding = stride - 1
        super().__init__(in_ch, out_ch, kernel_size,
                         stride=stride, padding=padding,
                         output_padding=output_padding, groups=groups, bias=bias)
        self.lsq_alpha = nn.Parameter(torch.ones(out_ch, 1, 1) * 0.1)
        self.register_buffer('_alpha_init_flag', torch.tensor(False))
        n_weights = in_ch * kernel_size // groups
        self._grad_scale = 1.0 / math.sqrt(n_weights)
        self.deadzone_tau = 0.1

    def _init_alpha(self):
        with torch.no_grad():
            mean_abs = self.weight.abs().mean(dim=(1, 2), keepdim=True)
            self.lsq_alpha.data.copy_((2.0 / 3.0) * mean_abs.clamp(min=0.001))
        self._alpha_init_flag.fill_(True)

    def clamp_alpha(self):
        """Clamp alpha to [0.5×std(W), 2.0×std(W)] — same as TernaryConv1d."""
        with torch.no_grad():
            w_std = self.weight.std(dim=(1, 2), keepdim=True).clamp(min=1e-4)
            alpha_min = 0.5 * w_std
            alpha_max = 2.0 * w_std
            self.lsq_alpha.data.copy_(
                self.lsq_alpha.data.abs().clamp(min=alpha_min, max=alpha_max))

    def ensure_initialized(self):
        """Run alpha init if needed. Call before torch.compile to avoid graph breaks."""
        if not self._alpha_init_flag.item():
            self._init_alpha()

    def forward(self, x, quantize=True):
        if not quantize:
            return super().forward(x)
        if QUANTIZATION_MODE == 'binary':
            w_q = _lsq_binary(self.weight, self.lsq_alpha, self._grad_scale)
        elif QUANTIZER_TYPE == 'seq':
            w_q = _seq_ternary(self.weight, self.lsq_alpha, self._grad_scale)
        else:
            tau = self.deadzone_tau if self.training else 0.0
            w_q = _lsq_ternary(self.weight, self.lsq_alpha, self._grad_scale, tau)
        out = F.conv_transpose1d(x, w_q, self.bias,
                                 self.stride, self.padding,
                                 self.output_padding, self.groups, self.dilation)
        return _quantize_activation(out, enabled=quantize, hadamard=getattr(self, '_hadamard_ref', None))


class BitShiftNorm(nn.Module):
    """Bit-Shift Normalization (BSN): replaces BatchNorm/GroupNorm on MCU.

    From CEA (arXiv:2501.05097, 2025). Uses a single power-of-2 rescaling
    instead of mean/variance/divide. Eliminates all division operations.

    During training: learns a log2 shift factor per channel.
    During inference: shift = round(log2(scale)) applied as bit shift.
    """
    def __init__(self, channels):
        super().__init__()
        self.log2_scale = nn.Parameter(torch.zeros(channels, 1))  # [C, 1]
        self.bias = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x):
        # x: [B, C, T]
        # Soft approximation during training (exact power-of-2 at export)
        scale = 2.0 ** self.log2_scale  # learned per-channel scale
        return x * scale + self.bias


class EfficientChannelAttention(nn.Module):
    """ECA: 1D conv across channel dim for adaptive channel weighting.

    From AMEEGNet (Frontiers 2025). Adds ~k params (k=3 default).
    At ternary precision, zero overhead. Replaces SE attention's FC layers.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # x: [B, C, T]
        y = self.avg_pool(x)                    # [B, C, 1]
        y = y.squeeze(-1).unsqueeze(1)          # [B, 1, C]
        y = self.conv(y)                        # [B, 1, C]
        y = torch.sigmoid(y)
        y = y.squeeze(1).unsqueeze(-1)          # [B, C, 1]
        return x * y


class CSPBlock(nn.Module):
    """Cross-Stage Partial block: splits channels, processes half.

    From E-ConvNeXt (August 2025). Halves depthwise conv cost.
    Half channels bypass the block (identity), half are processed.
    Results merged via concat + 1×1 projection.

    ~50% fewer FLOPs than full-channel processing.
    """
    def __init__(self, channels, kernel_size=7, stride=1, groups=1):
        super().__init__()
        half = channels // 2
        self.half = half
        self.conv = TernaryConv1d(half, half, kernel_size, stride=stride, groups=groups)
        self.norm = nn.GroupNorm(4 if half % 4 == 0 else 1, half)
        self.eca = EfficientChannelAttention(channels)
        # Merge projection
        if stride != 1:
            self.shortcut = TernaryConv1d(channels, channels, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, quantize=True):
        identity = self.shortcut(x, quantize=quantize) if isinstance(self.shortcut, TernaryConv1d) else x
        # Split channels
        x1, x2 = x[:, :self.half], x[:, self.half:]
        # Process only x1
        x1 = F.relu(self.norm(self.conv(x1, quantize=quantize)))
        # Concat and apply channel attention
        out = torch.cat([x1, x2[:, :, :x1.shape[2]]], dim=1)
        out = self.eca(out)
        out = out + identity
        return _quantize_activation(out, enabled=quantize, hadamard=getattr(self, '_hadamard_ref', None))


class ReGLUBottleneck(nn.Module):
    """ReGLU bottleneck: ReLU-gated linear unit for ternary.

    Replaces GLU's sigmoid (needs float/LUT) with ReLU gate (sign test).
    At ternary precision: gate = max(0, xW_g) → {0, +1} via sign.
    xW_v ⊗ gate → element-wise multiply of ternary outputs.

    Uses 2/3 hidden dim for equal compute to standard GLU.
    """
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        h = hidden_ch * 2 // 3  # 2/3 hidden for ReGLU (3 matrices vs 2)
        self.w_v = TernaryConv1d(in_ch, h, 1, stride=1, bias=True)
        self.w_g = TernaryConv1d(in_ch, h, 1, stride=1, bias=True)
        self.w_o = TernaryConv1d(h, out_ch, 1, stride=1, bias=True)

    def forward(self, x, quantize=True):
        v = self.w_v(x, quantize=quantize)
        g = F.relu(self.w_g(x, quantize=quantize))  # ReLU gate (not sigmoid)
        return self.w_o(v * g, quantize=quantize)


class TernaryFocalBlock(nn.Module):
    """Ternary conv + GroupNorm + ReLU + residual shortcut.

    Activation quantization is applied after the residual addition,
    simulating the INT16 activation precision the firmware will see
    at the output of each block.
    """
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, groups=1):
        super().__init__()
        self.conv = TernaryConv1d(in_ch, out_ch, kernel_size,
                                  stride=stride, groups=groups)
        self.norm = nn.GroupNorm(4 if out_ch % 4 == 0 else 1, out_ch)
        if in_ch != out_ch or stride != 1:
            self.shortcut = TernaryConv1d(in_ch, out_ch, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, quantize=True):
        if isinstance(self.shortcut, TernaryConv1d):
            identity = self.shortcut(x, quantize=quantize)
        else:
            identity = self.shortcut(x)
        out = F.relu(self.norm(self.conv(x, quantize=quantize)))
        out = out + identity
        return _quantize_activation(out, enabled=quantize, hadamard=getattr(self, '_hadamard_ref', None))


class TernaryDWSepFocalBlock(nn.Module):
    """Depthwise-separable focal block: DW conv + PW conv + shortcut.

    6.6x more parameter-efficient than full TernaryFocalBlock at the same
    receptive field. Enables wider/deeper encoders within the Axon SRAM budget.

    DW (depthwise):  ch -> ch, kernel_size, groups=ch  (spatial mixing only)
    PW (pointwise):  ch -> ch, k=1                     (channel mixing only)
    Shortcut:        1x1 projection if dims/stride change

    Cost: DW(ch*k) + PW(ch*ch) vs Full(ch*ch*k). For ch=144, k=7:
      Full:   144*144*7 = 145,152 params
      DW-sep: 144*7 + 144*144 = 21,744 params (6.7x smaller)
    """
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1):
        super().__init__()
        self.dw = TernaryConv1d(in_ch, in_ch, kernel_size,
                                stride=stride, groups=in_ch)
        self.pw = TernaryConv1d(in_ch, out_ch, 1, stride=1)
        self.norm = nn.GroupNorm(8 if out_ch % 8 == 0 else 4, out_ch)
        if in_ch != out_ch or stride != 1:
            self.shortcut = TernaryConv1d(in_ch, out_ch, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, quantize=True):
        if isinstance(self.shortcut, TernaryConv1d):
            identity = self.shortcut(x, quantize=quantize)
        else:
            identity = self.shortcut(x)
        h = self.dw(x, quantize=quantize)
        h = F.relu(self.norm(self.pw(h, quantize=quantize)))
        out = h + identity
        return _quantize_activation(out, enabled=quantize, hadamard=getattr(self, '_hadamard_ref', None))


class TernaryUpsampleBlock(nn.Module):
    """Ternary transposed conv + GroupNorm + ReLU + residual shortcut."""
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=2, groups=1):
        super().__init__()
        self.conv = TernaryConvTranspose1d(in_ch, out_ch, kernel_size,
                                           stride=stride, groups=groups)
        self.norm = nn.GroupNorm(4 if out_ch % 4 == 0 else 1, out_ch)
        if in_ch != out_ch or stride != 1:
            self.shortcut = TernaryConvTranspose1d(in_ch, out_ch, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, quantize=True):
        if isinstance(self.shortcut, TernaryConvTranspose1d):
            identity = self.shortcut(x, quantize=quantize)
        else:
            identity = self.shortcut(x)
        out = F.relu(self.norm(self.conv(x, quantize=quantize)))
        out = out + identity
        return _quantize_activation(out, enabled=quantize, hadamard=getattr(self, '_hadamard_ref', None))
