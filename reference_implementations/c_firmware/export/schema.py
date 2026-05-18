"""Load and validate `firmware/export_schema.toml`.

The schema is the single source of truth for layer layout, dtypes, and
Q-formats. Both the C and Rust emitters consume the parsed schema.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore


# ────────────────────────────────────────────────────────────────────
# Schema dataclasses (typed, validated)
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArchSpec:
    """One architecture variant: encoder class + width + checkpoint globs."""

    name: str                   # subband_v1, subband_v2, legacy_v7_0
    display_name: str
    encoder_class: str
    encoder_width: int
    n_focal_blocks: int
    latent_dims: int
    latent_timesteps: int
    checkpoint_globs: list[str]


@dataclass(frozen=True)
class LayerSpec:
    """One layer in the encoder. `out_channels="_arch_width"` is resolved
    against the active architecture's `encoder_width` at export time."""

    name: str
    in_channels: int | str
    out_channels: int | str
    kernel_size: int
    stride: int
    weight_kind: str            # "ternary_2bit_packed" | "int8"
    has_alphas: bool
    has_norm: bool
    has_shortcut: bool
    groups: int | str | None = None
    has_quant_scale: bool = False

    def resolve(self, arch: ArchSpec) -> ResolvedLayer:
        """Substitute `_arch_width` placeholders, return concrete shape."""
        return ResolvedLayer(
            name=self.name,
            in_channels=_resolve_dim(self.in_channels, arch),
            out_channels=_resolve_dim(self.out_channels, arch),
            kernel_size=self.kernel_size,
            stride=self.stride,
            weight_kind=self.weight_kind,
            has_alphas=self.has_alphas,
            has_norm=self.has_norm,
            has_shortcut=self.has_shortcut,
            groups=_resolve_dim(self.groups, arch) if self.groups is not None else None,
            has_quant_scale=self.has_quant_scale,
        )


def _resolve_dim(value: int | str, arch: ArchSpec) -> int:
    if isinstance(value, int):
        return value
    if value == "_arch_width":
        return arch.encoder_width
    raise ValueError(f"Unknown dimension placeholder: {value!r}")


@dataclass(frozen=True)
class ResolvedLayer:
    """Layer with all dimensions concrete (no placeholders)."""

    name: str
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    weight_kind: str
    has_alphas: bool
    has_norm: bool
    has_shortcut: bool
    groups: int | None
    has_quant_scale: bool

    @property
    def n_weights(self) -> int:
        eff_in = self.in_channels // (self.groups or 1) * (self.groups or 1) // self.in_channels * self.in_channels
        # depthwise: in_channels = out_channels = groups
        if self.groups == self.in_channels == self.out_channels:
            return self.out_channels * self.kernel_size
        return self.in_channels * self.out_channels * self.kernel_size

    @property
    def n_packed_bytes(self) -> int:
        """Length of the 2-bit-packed weight array. Ternary only."""
        if self.weight_kind != "ternary_2bit_packed":
            return 0
        return (self.n_weights + 3) // 4


@dataclass(frozen=True)
class FsqSpec:
    levels: list[int]
    quant_scale_q31: int
    n_freq_bins: int
    rans_total_freq: int
    calibration_n_samples: int
    calibration_input_shape: list[int]
    calibration_input_clamp: int


@dataclass(frozen=True)
class RotationSpec:
    source_param: str
    dim: int
    storage_dtype: str
    storage_q_format: str


@dataclass(frozen=True)
class ToeplitzSpec:
    n_channels: int
    seed_dtype: str
    seeds: list[int]


@dataclass(frozen=True)
class SnnTensorSpec:
    name: str
    storage: str
    has_scale: bool = False
    storage_q_format: str | None = None


@dataclass(frozen=True)
class SnnSpec:
    encoder_class: str
    in_channels: int
    d_model: int
    n_layers: int
    n_groups: int
    stride: int
    spatial_mix_hidden: int
    ssm_state_dim: int
    checkpoint_globs: list[str]
    tensors: list[SnnTensorSpec]


@dataclass(frozen=True)
class Schema:
    schema_version: str
    target_crate: str
    output_subdir: str
    architectures: dict[str, ArchSpec]
    layers: list[LayerSpec]
    rotation: RotationSpec
    fsq: FsqSpec
    toeplitz: ToeplitzSpec
    snn: SnnSpec

    def get_arch(self, name: str) -> ArchSpec:
        if name not in self.architectures:
            raise KeyError(
                f"Unknown architecture {name!r}. "
                f"Known: {sorted(self.architectures)}"
            )
        return self.architectures[name]

    def resolved_layers(self, arch_name: str) -> list[ResolvedLayer]:
        arch = self.get_arch(arch_name)
        return [layer.resolve(arch) for layer in self.layers]


# ────────────────────────────────────────────────────────────────────
# Loader
# ────────────────────────────────────────────────────────────────────


def load_schema(path: str | os.PathLike) -> Schema:
    """Parse `export_schema.toml` into typed dataclasses. Validates at load."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Schema not found: {p}")

    with p.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)

    meta = data.get("meta", {})
    schema_version = str(meta.get("schema_version", "?"))
    if schema_version != "1.0":
        raise ValueError(
            f"Unsupported schema_version {schema_version!r}; this loader handles 1.0."
        )

    archs: dict[str, ArchSpec] = {}
    for arch_name, arch_data in data.get("architectures", {}).items():
        archs[arch_name] = ArchSpec(name=arch_name, **arch_data)

    layers = [LayerSpec(**ld) for ld in data.get("layers", [])]

    rotation = RotationSpec(**data["rotation"])
    fsq = FsqSpec(**data["fsq"])
    toeplitz = ToeplitzSpec(**data["toeplitz"])

    snn_data = dict(data["snn"])
    snn_tensors = [SnnTensorSpec(**td) for td in snn_data.pop("tensors", [])]
    snn = SnnSpec(tensors=snn_tensors, **snn_data)

    return Schema(
        schema_version=schema_version,
        target_crate=str(meta.get("target_crate", "lamquant-weights")),
        output_subdir=str(meta.get("output_subdir", "src/generated")),
        architectures=archs,
        layers=layers,
        rotation=rotation,
        fsq=fsq,
        toeplitz=toeplitz,
        snn=snn,
    )


def validate_schema(schema_path: str | os.PathLike) -> None:
    """CLI entry point: --validate-schema. Raises on any error."""
    schema = load_schema(schema_path)
    if not schema.architectures:
        raise ValueError("No architectures declared.")
    if not schema.layers:
        raise ValueError("No layers declared.")
    for arch_name in schema.architectures:
        layers = schema.resolved_layers(arch_name)
        for layer in layers:
            if layer.weight_kind not in ("ternary_2bit_packed", "int8"):
                raise ValueError(
                    f"{arch_name}/{layer.name}: unknown weight_kind "
                    f"{layer.weight_kind!r}"
                )
    print(f"[OK] schema {schema_path} valid; "
          f"{len(schema.architectures)} archs, {len(schema.layers)} layers")
