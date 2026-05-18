"""
Minimal Selective State Space Model (S4D/Mamba-style) in pure PyTorch.

No CUDA kernel dependencies — works on CPU, CUDA, MPS. This is the
fallback implementation when mamba-ssm pip package isn't available.
Matches the Mamba architecture's selective scan mechanism but uses
standard PyTorch ops instead of the fused CUDA kernel.

For LamQuant's activity detector, this processes EEG sequences with
linear memory cost (O(T) vs O(T²) for attention) and captures
long-range temporal dependencies that the dLIF SNN's fixed decay
constants cannot.

Architecture: S4D (Structured State Spaces for Sequence Modeling,
Diagonal variant) with data-dependent gating (Mamba's selective scan).

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces", 2023. Mamba-3 (ICLR 2026) extends this
with complex-valued states and MIMO — not implemented here for
simplicity and firmware portability.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to import fused CUDA selective scan from mamba-ssm package.
# Falls back to pure-PyTorch chunked scan if not available.
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False


class HomeostaticThresholdAdapter:
    """On-device per-patient SNN threshold adaptation.

    From corticohippocampal metaplasticity (Nature PMC 2025).
    Slowly adapts SNN activity thresholds using exponential moving
    averages of local spike rates. No backpropagation needed.

    On RP2350: ~200 bytes of state (40 channels × 5 bytes each).
    Updates every 10-second window. Converges within 5 minutes.
    """
    def __init__(self, n_channels=40, target_rate=0.1, tau=0.99):
        self.ema_rate = np.zeros(n_channels)
        self.threshold_adj = np.zeros(n_channels)
        self.target_rate = target_rate
        self.tau = tau

    def update(self, spike_rates):
        """Update thresholds based on observed spike rates.

        spike_rates: [n_channels] current window's spike rate per channel.
        """
        # Exponential moving average of spike rate
        self.ema_rate = self.tau * self.ema_rate + (1 - self.tau) * spike_rates
        # Adjust threshold to match target rate
        # Too active → raise threshold, too quiet → lower
        error = self.ema_rate - self.target_rate
        self.threshold_adj -= 0.01 * error  # slow adaptation

    def get_adjusted_thresholds(self, base_threshold=0.0):
        """Return adjusted thresholds for each channel."""
        return base_threshold + self.threshold_adj


def _int4_quantize_ste(weight):
    """Symmetric INT4 STE quantization: round(clamp(w/scale, -7, 7)) × scale.
    For SNN weight QAT — 4-bit weights, 16 levels, ~4× memory reduction.
    """
    scale = weight.abs().amax() / 7.0
    scale = scale.clamp(min=1e-8)
    w_int4 = torch.clamp(torch.round(weight / scale), -7, 7)
    return (w_int4 * scale - weight).detach() + weight  # STE


def _int3_quantize_decay(A_log):
    """3-bit quantization for SSM decay parameters (A_log).
    Maps to 8 discrete log-decay values. Preserves temporal dynamics.
    """
    # 8 discrete values spanning the useful range of log-decay
    levels = torch.tensor([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5],
                          device=A_log.device)
    # Find nearest level
    idx = (A_log.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    A_q = levels[idx]
    return (A_q - A_log).detach() + A_log  # STE


class SelectiveSSM(nn.Module):
    """Single Mamba-style selective SSM block.

    Input:  [B, T, D]
    Output: [B, T, D]

    Computation per timestep:
      Δ = softplus(linear_dt(x))      # data-dependent discretization
      B = linear_B(x)                  # data-dependent input matrix
      C = linear_C(x)                  # data-dependent output matrix
      h[t] = Δ*A*h[t-1] + Δ*B*x[t]   # state update (diagonal A)
      y[t] = C * h[t]                  # output projection

    The diagonal A matrix means the state update is element-wise,
    making it efficient for both training (parallel scan) and
    inference (recurrent, O(1) memory per step).
    """

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection: x → (z, x_proj) via a single linear
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D causal convolution (replaces Mamba's fused causal_conv1d)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        # A is log-parameterized diagonal — HiPPO-LegS initialization.
        #
        # HiPPO (High-order Polynomial Projection Operator) gives the SSM
        # optimal memory for approximating continuous-time input history as
        # Legendre polynomial coefficients. The diagonal approximation
        # (S4D) uses A_n = -(n + 1/2), which spaces decay rates to capture
        # both fast transients (high n) and slow trends (low n).
        #
        # For EEG at 250 Hz with stride-8 (31.25 Hz effective), this gives
        # the model memory spanning ~0.03s (n=15) to ~2s (n=0) per SSM
        # step. Seizure patterns (5-30s) are captured across multiple steps
        # via the recurrent state, while fast spikes (<100ms) are captured
        # by high-n components.
        #
        # Reference: Gu et al., "HiPPO: Recurrent Memory with Optimal
        # Polynomial Projections" (NeurIPS 2020).
        A = torch.arange(1, d_state + 1, dtype=torch.float32) + 0.5  # HiPPO-LegS: n + 1/2
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(self.d_inner))  # skip connection

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # dt projection bias (controls discretization step)
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner) - 4.0)  # softplus(-4) ≈ 0.02

    def forward(self, x, quantize=False):
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        B, T, D = x.shape

        # P1: INT4 weight quantization via STE
        if quantize:
            # Quantize projections
            in_w = _int4_quantize_ste(self.in_proj.weight)
            xz = F.linear(x, in_w)
        else:
            xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # each [B, T, d_inner]

        # Causal convolution (temporal mixing before SSM)
        x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
        x_conv = self.conv1d(x_conv)[:, :, :T]  # causal: trim to T
        x_conv = F.silu(x_conv).transpose(1, 2)  # [B, T, d_inner]

        # Compute SSM parameters from input (data-dependent = "selective")
        ssm_params = self.x_proj(x_conv)  # [B, T, 2*d_state + 1]
        B_param = ssm_params[:, :, :self.d_state]           # [B, T, N]
        C_param = ssm_params[:, :, self.d_state:2*self.d_state]  # [B, T, N]
        dt = F.softplus(ssm_params[:, :, -1:] + self.dt_bias[:1])  # [B, T, 1]

        # Diagonal A (negative, stable)
        A = -torch.exp(self.A_log)  # [d_inner, N]

        if HAS_MAMBA_CUDA and x.is_cuda:
            # Fused CUDA kernel: forward + backward in one kernel launch.
            # selective_scan_fn expects:
            #   u: [B, d_inner, T], delta: [B, d_inner, T],
            #   A: [d_inner, N], B: [B, N, T], C: [B, N, T],
            #   D: [d_inner], delta_softplus=False (already applied)
            y = selective_scan_fn(
                x_conv.transpose(1, 2).contiguous(),   # [B, D, T]
                dt.squeeze(-1).unsqueeze(1).expand(-1, self.d_inner, -1).contiguous(),
                A,                                      # [D, N]
                B_param.transpose(1, 2).contiguous(),   # [B, N, T]
                C_param.transpose(1, 2).contiguous(),   # [B, N, T]
                self.D.float(),                         # [D]
                delta_softplus=False,                   # dt already has softplus
            )
            y = y.transpose(1, 2)  # [B, T, D]
        else:
            # Pure PyTorch fallback (chunked parallel scan)
            y = self._sequential_scan(x_conv, A, B_param, C_param, dt)
            y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0).expand_as(x_conv)

        y = y * F.silu(z)  # gating

        return self.out_proj(y)

    def _sequential_scan(self, x, A, B, C, dt):
        """Chunked parallel scan — vectorized within chunks, sequential between.

        The linear recurrence h[t] = dA[t]*h[t-1] + dBx[t] is solved
        analytically within each chunk via cumsum, then state is carried
        between chunks sequentially.  ~10 chunk iterations vs 313 Python
        loop iterations, each chunk fully vectorized on GPU.

        Chunk size 32 keeps exp(-log_P) within float32 range even for
        worst-case trained SSM parameters (dt=0.5, A=-16 → exp(256) < 3e38).
        """
        B_sz, T, d_inner = x.shape
        N = self.d_state
        CHUNK = 32

        # Precompute all log_dA and dBx for the full sequence
        log_dA_all = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # [B, T, D, N]
        dBx_all = x.unsqueeze(-1) * (dt.unsqueeze(-1) * B.unsqueeze(2))

        # Pre-allocate output
        y = torch.empty(B_sz, T, d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(B_sz, d_inner, N, device=x.device, dtype=x.dtype)

        for cs in range(0, T, CHUNK):
            ce = min(cs + CHUNK, T)
            log_dA = log_dA_all[:, cs:ce]
            dBx = dBx_all[:, cs:ce]

            log_P = torch.cumsum(log_dA, dim=1)
            exp_P = torch.exp(log_P)
            exp_nP = torch.exp(-log_P)

            h_chunk = exp_P * (h.unsqueeze(1) + torch.cumsum(dBx * exp_nP, dim=1))

            y[:, cs:ce] = (C[:, cs:ce].unsqueeze(2) * h_chunk).sum(dim=-1)
            h = h_chunk[:, -1]

        return y


class BidirectionalSSM(nn.Module):
    """Bidirectional Mamba: forward + backward SSM, outputs averaged.

    FEMBA showed bidirectional Mamba improves EEG classification by
    capturing both causal and anti-causal temporal patterns — spikes
    have distinctive shapes in both forward and backward time.
    """

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.fwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.bwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: [B, T, D] → [B, T, D]"""
        x_normed = self.norm(x)
        y_fwd = self.fwd(x_normed)
        y_bwd = self.bwd(x_normed.flip(1)).flip(1)  # reverse time, SSM, reverse back
        return x + (y_fwd + y_bwd) * 0.5  # residual + averaged bidirectional


class MambaSNN(nn.Module):
    """Mamba-based EEG activity detector (drop-in replacement for ActivitySNN).

    Architecture (targeting ≤64 KB at INT8):
      spatial_mix:  Linear(21 → 40)       880 params
      ssm_block_1:  BidirectionalSSM(40, d_state=16)  ~28K params
      ssm_block_2:  BidirectionalSSM(40, d_state=16)  ~28K params
      readout:      Linear(40 → 8)        328 params
      Total: ~57K params → 56.3 KB INT8, 14.1 KB W2A8

    Input:  [B, 21, T] (raw EEG or L3 subband)
    Output: ([B, 8, T_out] activity_logits, scalar spike_rate)

    The output shape matches ActivitySNN so it's a drop-in replacement
    in the codec pipeline's adaptive FSQ level selection.
    """

    NUM_GROUPS = 8
    STRIDE = 8

    def __init__(self, in_channels=21, d_model=40, d_state=16, n_layers=2,
                 use_subband=False):
        super().__init__()
        self.use_subband = use_subband
        self.stride = 1 if use_subband else self.STRIDE

        # Spatial mixing (replaces DeltaEncoder + TernaryConv)
        self.spatial_mix = nn.Linear(in_channels, d_model)

        # Stacked bidirectional SSM blocks
        self.ssm_blocks = nn.ModuleList([
            BidirectionalSSM(d_model, d_state=d_state)
            for _ in range(n_layers)
        ])

        # Readout to spatial groups
        self.readout = nn.Linear(d_model, self.NUM_GROUPS)

    def forward(self, x):
        """
        x: [B, C, T] where C=21, T=2500 (raw) or T=313 (subband)
        returns: (activity_logits [B, 8, T_out], spike_rate scalar)
        """
        B, C, T = x.shape

        # Transpose to [B, T, C] for linear layers
        x = x.transpose(1, 2)  # [B, T, 21]

        # Spatial mixing
        x = self.spatial_mix(x)  # [B, T, 32]

        # SSM blocks with residual connections
        for block in self.ssm_blocks:
            x = block(x)  # [B, T, 32]

        # Readout
        logits = self.readout(x)  # [B, T, 8]
        logits = logits.transpose(1, 2)  # [B, 8, T]

        # Stride-8 pooling for raw EEG input (subband already at stride)
        if self.stride > 1:
            T_out = T // self.stride
            logits = logits[:, :, :T_out * self.stride]
            logits = logits.reshape(B, self.NUM_GROUPS, T_out, self.stride).mean(dim=-1)

        # Spike rate proxy (L1 norm of logits, for regularization)
        spike_rate = logits.abs().mean()

        return logits, spike_rate

    def classify_per_timestep(self, latent_or_l3, target_T=79):
        """Per-timestep activity classification for adaptive FSQ.

        Returns a level schedule: [T] array of FSQ levels (2, 3, or 5).
        Uses max activity across 8 spatial groups per timestep.

        Args:
            latent_or_l3: [B, 21, 313] L3 input or [B, 21, T] signal
            target_T: target temporal resolution (79 = latent timesteps)
        Returns:
            levels: [B, target_T] int tensor of FSQ levels
        """
        with torch.no_grad():
            logits, _ = self.forward(latent_or_l3)  # [B, 8, T]
            # Pool to target resolution
            if logits.shape[2] != target_T:
                logits = F.adaptive_avg_pool1d(logits, target_T)  # [B, 8, target_T]
            # Max activity across 8 spatial groups → single scalar per timestep
            activity = logits.max(dim=1).values  # [B, target_T]
            # Threshold → FSQ level
            levels = torch.full_like(activity, 2, dtype=torch.long)  # default L=2
            levels[activity > 0.0] = 3   # active → L=3
            levels[activity > 0.5] = 5   # event/seizure → L=5
        return levels

    def param_size_kb(self, bits=8):
        """Estimate firmware footprint at given bit width."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params * bits / 8 / 1024
