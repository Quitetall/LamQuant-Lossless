"""Production SIREN module for Mode 4 INR codec.

SIREN (Sinusoidal Representation Network) maps continuous time coordinates
to multi-channel EEG signal values. Used for the smooth L4 subband component
in Mode 4 compression.

Architecture (fixed for Mode 4 v0):
    Input:  t ∈ [-1, 1]  (1 dim)
    Hidden: 3 layers × 44 units, sin(ω₀ · Wx + b) activation
    Output: 21 channels (EEG)
    Total:  4993 parameters

Encode: per-signal gradient descent fit (~0.5s on GPU, 1500 epochs)
Decode: forward pass evaluation at T timesteps (~30ms on MCU)

References:
    Sitzmann et al. 2020, "Implicit Neural Representations with
    Periodic Activation Functions"
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Mode 4 v0 architecture constants
SIREN_HIDDEN_DIM = 44
SIREN_N_LAYERS = 3
SIREN_N_CHANNELS = 21
SIREN_OMEGA_0 = 60.0
SIREN_N_PARAMS = 4993  # 44+44 + 1936+44 + 1936+44 + 924+21


class SirenLayer(nn.Module):
    """SIREN layer: Linear → sin(ω₀ · x)."""

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 60.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(is_first)

    def _init_weights(self, is_first: bool):
        with torch.no_grad():
            if is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """Production SIREN for Mode 4: time → 21-channel EEG.

    Fixed architecture (4993 params):
        SirenLayer(1 → 44)  × sin(ω₀·)
        SirenLayer(44 → 44) × sin(ω₀·)
        SirenLayer(44 → 44) × sin(ω₀·)
        Linear(44 → 21)     (no activation)
    """

    def __init__(self, hidden_dim: int = SIREN_HIDDEN_DIM,
                 n_layers: int = SIREN_N_LAYERS,
                 n_channels: int = SIREN_N_CHANNELS,
                 omega_0: float = SIREN_OMEGA_0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.omega_0 = omega_0

        layers = [SirenLayer(1, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.net = nn.Sequential(*layers)

        self.output = nn.Linear(hidden_dim, n_channels)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.output.weight.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate SIREN at time coordinates.

        Args:
            coords: [T, 1] time coordinates in [-1, 1]

        Returns:
            [T, n_channels] signal values (normalized to [-1, 1])
        """
        return self.output(self.net(coords))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flatten_weights(self) -> np.ndarray:
        """Flatten all parameters to a single vector (for quantization)."""
        return torch.cat([p.detach().cpu().flatten()
                          for p in self.parameters()]).numpy()

    def load_flat_weights(self, weights: np.ndarray):
        """Load weights from a flat vector (inverse of flatten_weights)."""
        w_tensor = torch.from_numpy(weights).float()
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(w_tensor[offset:offset + n].reshape(p.shape))
            offset += n


def make_coords(T: int, device: str = 'cpu') -> torch.Tensor:
    """Create normalized time coordinates [-1, 1] for T timesteps."""
    return torch.linspace(-1, 1, T, device=device).unsqueeze(1)


# ============================================================
# Per-signal SIREN fitting (encode path)
# ============================================================

def fit_siren(
    signal: np.ndarray,
    epochs: int = 1500,
    lr: float = 5e-4,
    device: str = 'cuda',
) -> Tuple[SIREN, float]:
    """Fit a SIREN to a single L4 subband signal.

    Args:
        signal: [21, T] float32 array (L4 subband, unnormalized)
        epochs: optimization epochs (default 1500)
        lr: learning rate (default 5e-4)
        device: 'cuda' or 'cpu'

    Returns:
        (fitted_siren, pearson_r) — the fitted model and reconstruction quality
    """
    C, T = signal.shape

    # Normalize to [-1, 1]
    vmin, vmax = signal.min(), signal.max()
    vrange = max(vmax - vmin, 1e-8)
    sig_norm = (signal - vmin) / vrange * 2.0 - 1.0

    # Build model
    model = SIREN(n_channels=C).to(device)
    coords = make_coords(T, device=device)
    target = torch.from_numpy(sig_norm).float().to(device).T  # [T, C]

    # Optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=lr * 0.01)

    for _ in range(epochs):
        pred = model(coords)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Measure quality
    with torch.no_grad():
        pred_final = model(coords).cpu().numpy().T  # [C, T]
    pred_denorm = (pred_final + 1.0) / 2.0 * vrange + vmin

    # Pearson R
    p = pred_denorm.flatten()
    t = signal.flatten()
    pc, tc = p - p.mean(), t - t.mean()
    r = float(np.sum(pc * tc) / max(np.sqrt(np.sum(pc**2) * np.sum(tc**2)), 1e-12))

    return model, r


# ============================================================
# INT4 Quantization
# ============================================================

def quantize_int4(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """Symmetric INT4 quantization of SIREN weights.

    Args:
        weights: [N] float32 weight vector

    Returns:
        (quantized_int8, scale) — quantized values in [-7, 7] stored as int8,
        plus the scale factor for dequantization.
    """
    scale = float(np.abs(weights).max()) / 7.0
    if scale < 1e-12:
        scale = 1e-12
    quantized = np.clip(np.round(weights / scale), -7, 7).astype(np.int8)
    return quantized, scale


def dequantize_int4(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize INT4 weights back to float32."""
    return quantized.astype(np.float32) * scale


# ============================================================
# SIREN evaluation from flat weights (decode path)
# ============================================================

def eval_siren_from_weights(
    weights: np.ndarray,
    T: int,
    n_channels: int = SIREN_N_CHANNELS,
    hidden_dim: int = SIREN_HIDDEN_DIM,
    n_layers: int = SIREN_N_LAYERS,
    omega_0: float = SIREN_OMEGA_0,
) -> np.ndarray:
    """Evaluate SIREN from a flat weight vector (decode path).

    This is the decode-only path — no gradient, no model object needed.
    Suitable for porting to C for MCU decode.

    Args:
        weights: [N_PARAMS] float32 weight vector
        T: number of timesteps to evaluate
        n_channels: output channels (21)

    Returns:
        [n_channels, T] float32 signal in [-1, 1]
    """
    model = SIREN(hidden_dim=hidden_dim, n_layers=n_layers,
                  n_channels=n_channels, omega_0=omega_0)
    model.load_flat_weights(weights)
    model.eval()

    coords = make_coords(T)
    with torch.no_grad():
        out = model(coords).numpy().T  # [C, T]
    return out
