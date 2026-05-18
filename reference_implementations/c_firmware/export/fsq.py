"""FSQ + rANS frequency table calibration.

Runs N forward passes of the model on random clamped EEG, collects latent
samples, and fits a frequency table with `total_freq` budget. Output is the
same shape used by the legacy C exporter — both emitters consume this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FsqCalibration:
    num_levels: int
    total_freq: int
    freq: list[int]            # length = num_levels
    start: list[int]           # cumulative starts; length = num_levels
    vmin_q31: int              # vmin * 1000 (matches training scale)
    vmax_q31: int
    inv_range_q31: int
    entropy_bps: float


def calibrate(
    model: torch.nn.Module,
    n_samples: int = 50,
    input_shape: tuple[int, int, int] = (4, 21, 2500),
    input_clamp: int = 50,
    num_levels: int = 16,
    total_freq: int = 4096,
) -> FsqCalibration:
    """Run forward passes on random EEG, fit FSQ + rANS frequency table.

    Returns:
        FsqCalibration with per-bin frequencies, cumulative starts, and the
        latent value range. Identical math to the legacy C exporter.

    Raises:
        RuntimeError: if the model fails to produce any latent samples.
    """
    model.eval()
    device = next(model.parameters()).device

    latents: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_samples):
            x = torch.clamp(
                torch.randn(*input_shape, device=device) * 20,
                -input_clamp,
                input_clamp,
            )
            lat = model.encode(x, quantize=True)
            latents.append(lat.cpu().numpy())

    if not latents:
        raise RuntimeError("FSQ calibration produced no latent samples.")

    flat = np.concatenate(latents, axis=0).flatten()

    vmin = float(flat.min())
    vmax = float(flat.max())
    span = vmax - vmin + 1e-8

    # Quantize to bins.
    normalized = (flat - vmin) / span
    bins = np.clip((normalized * num_levels).astype(np.int32), 0, num_levels - 1)

    # Frequency table with min-1 floor.
    counts = np.bincount(bins, minlength=num_levels)
    total = int(counts.sum())
    freq = np.maximum(1, (counts / total * total_freq).astype(np.int32))
    # Adjust to hit exact total.
    diff = total_freq - int(freq.sum())
    freq[int(np.argmax(freq))] += diff

    start = np.zeros(num_levels, dtype=np.int32)
    for i in range(1, num_levels):
        start[i] = start[i - 1] + freq[i - 1]

    # Q31 range (×1000 scale, matching training normalization).
    vmin_q31 = int(vmin * 1000)
    vmax_q31 = int(vmax * 1000)
    range_q31 = max(1, vmax_q31 - vmin_q31)
    inv_range_q31 = int((num_levels * (1 << 30)) / range_q31)

    p = freq / freq.sum()
    entropy_bps = float(-np.sum(p * np.log2(p + 1e-12)))

    return FsqCalibration(
        num_levels=num_levels,
        total_freq=total_freq,
        freq=freq.tolist(),
        start=start.tolist(),
        vmin_q31=vmin_q31,
        vmax_q31=vmax_q31,
        inv_range_q31=inv_range_q31,
        entropy_bps=entropy_bps,
    )
