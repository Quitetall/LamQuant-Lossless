"""Model architectures — canonical location for inference-time definitions.

Encoder classes (TernaryMobileNetV5_Subband, etc.) and building blocks
(TernaryConv1d, etc.) live here. Training code imports FROM here,
not the other way around.
"""
from lamquant_codec.models.encoder import (
    TernaryMobileNetV5,
    TernaryMobileNetV5_Subband,
    TernaryMobileNetV5_Subband_V2,
)
from lamquant_codec.models.blocks import (
    TernaryConv1d,
    TernaryConvTranspose1d,
    INT8Conv1d,
)
from lamquant_codec.models.snn import (
    load_mamba_snn,
    resolve_production_snn,
)

__all__ = [
    'TernaryMobileNetV5',
    'TernaryMobileNetV5_Subband',
    'TernaryMobileNetV5_Subband_V2',
    'TernaryConv1d',
    'TernaryConvTranspose1d',
    'INT8Conv1d',
    'load_mamba_snn',
    'resolve_production_snn',
]
