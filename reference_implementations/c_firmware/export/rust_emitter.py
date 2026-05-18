"""Rust emitter — generates the `lamquant-weights` crate from a checkpoint.

Output:
    {crate_root}/src/generated/mod.rs
    {crate_root}/src/generated/focal/{layer}.rs
    {crate_root}/src/generated/rotation.rs
    {crate_root}/src/generated/fsq.rs
    {crate_root}/src/generated/toeplitz.rs
    {crate_root}/src/generated/crc.rs
    {crate_root}/src/metadata.rs
    {crate_root}/.exportlock.json

The codegen is data-only: every layer iteration is driven by the schema, and
quantization helpers come from `firmware.export.quantize`. No layer logic
lives in the emitter itself.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import torch
import torch.nn as nn

from .checkpoint import LoadedCheckpoint
from .crc import crc32_of
from .fsq import FsqCalibration, calibrate as fsq_calibrate
from .snn_emitter import emit_snn_crate
from .quantize import (
    cayley_rotation_q15,
    clamp_int8_weight,
    lsq_ternarize,
    pack_ternary_2bit,
    to_q15,
    to_q7,
    validate_native_ternary,
)
from .schema import ResolvedLayer, Schema


EXPORTER_VERSION = "1.0.0"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "rust"


# ────────────────────────────────────────────────────────────────────
# Per-layer extraction
# ────────────────────────────────────────────────────────────────────


@dataclass
class TernaryLayerData:
    """Quantized data for one ternary layer, ready for templating."""
    spec: ResolvedLayer
    packed: list[int]               # uint8 packed ternary
    alphas_q15: list[int]           # int16 per-output alpha
    norm_weight_q7: list[int] | None
    norm_bias_q15: list[int] | None
    n_weights: int

    @property
    def packed_bytes(self) -> bytes:
        return bytes(self.packed)


@dataclass
class Int8LayerData:
    """Quantized data for one INT8 layer (bneck_v output bottleneck)."""
    spec: ResolvedLayer
    weights: list[int]              # int8 raw
    scales_q15: list[int]


def _module_at(model: nn.Module, dotted_name: str) -> nn.Module | None:
    """Walk dotted path. Returns None if any segment is missing."""
    cur: Any = model
    for part in dotted_name.split("."):
        if hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return None
    return cur if isinstance(cur, nn.Module) else None


def _find_conv(model: nn.Module, layer_name: str) -> nn.Module | None:
    """Locate the conv module for a schema layer name. Tries common patterns."""
    candidates = [
        layer_name,                # e.g. "premix"
        f"{layer_name}.conv",      # e.g. "focal2.conv"
        f"{layer_name}_conv",      # e.g. "focal1_conv" (legacy)
    ]
    for name in candidates:
        m = _module_at(model, name)
        if m is not None and (hasattr(m, "weight") or hasattr(m, "weights")):
            return m
    return None


def _find_norm(model: nn.Module, layer_name: str) -> nn.GroupNorm | None:
    """Locate the GroupNorm/LayerNorm under a layer.

    Tries (in order):
      `{layer}.norm`         — sub-module convention
      `{layer}_norm`         — flat naming
      `{base}_norm` where base strips `_conv` suffix
                             — Gen 7.1 pattern: focal1_conv + focal1_norm
    """
    candidates = [f"{layer_name}.norm", f"{layer_name}_norm"]
    if layer_name.endswith("_conv"):
        base = layer_name[: -len("_conv")]
        candidates.extend([f"{base}.norm", f"{base}_norm"])
    for candidate in candidates:
        m = _module_at(model, candidate)
        if isinstance(m, nn.GroupNorm):
            return m
    return None


def extract_ternary_layer(
    model: nn.Module,
    spec: ResolvedLayer,
) -> TernaryLayerData:
    """Pull weights, alphas, and (optionally) norm parameters for a ternary layer."""
    conv = _find_conv(model, spec.name)
    if conv is None:
        raise KeyError(f"Layer {spec.name!r} not found on model.")

    if not hasattr(conv, "lsq_alpha"):
        raise ValueError(f"Layer {spec.name!r} has no lsq_alpha — expected ternary.")

    weight_t = conv.weight.detach().cpu().numpy().astype(np.float64)
    alpha_t = conv.lsq_alpha.detach().cpu().numpy().astype(np.float64)

    # Quantize weight to {-1, 0, +1} using LSQ alpha.
    w_ternary = lsq_ternarize(weight_t, alpha_t)
    packed = pack_ternary_2bit(w_ternary.flatten()).tolist()
    validate_native_ternary(packed)

    alphas_q15 = to_q15(np.abs(alpha_t).flatten()).tolist()

    norm_weight_q7: list[int] | None = None
    norm_bias_q15: list[int] | None = None
    if spec.has_norm:
        norm = _find_norm(model, spec.name)
        if norm is None:
            raise KeyError(f"Layer {spec.name!r}: norm declared but not found.")
        norm_weight_q7 = to_q7(norm.weight.detach().cpu().numpy()).tolist()
        norm_bias_q15 = to_q15(norm.bias.detach().cpu().numpy()).tolist()

    n_weights = int(np.prod(weight_t.shape))
    return TernaryLayerData(
        spec=spec,
        packed=packed,
        alphas_q15=alphas_q15,
        norm_weight_q7=norm_weight_q7,
        norm_bias_q15=norm_bias_q15,
        n_weights=n_weights,
    )


def extract_int8_layer(model: nn.Module, spec: ResolvedLayer) -> Int8LayerData:
    """Extract INT8 conv (e.g. bneck_v output bottleneck)."""
    conv = _find_conv(model, spec.name)
    if conv is None:
        raise KeyError(f"Layer {spec.name!r} not found on model.")

    weight_t = conv.weight.detach().cpu().numpy().astype(np.float64)

    # INT8 path: clamp + scale to int8.
    w_int8 = clamp_int8_weight(weight_t * 127.0).flatten().astype(np.int8)
    weights = w_int8.tolist()

    # Per-output scale: 1/127 if no quant_scale, else from module attr.
    if hasattr(conv, "quant_scale"):
        qs = conv.quant_scale.detach().cpu().numpy().astype(np.float64).flatten()
        scales_q15 = to_q15(qs).tolist()
    else:
        scales_q15 = [int(round((1.0 / 127.0) * 32767.0))] * spec.out_channels

    return Int8LayerData(spec=spec, weights=weights, scales_q15=scales_q15)


# ────────────────────────────────────────────────────────────────────
# Emitter
# ────────────────────────────────────────────────────────────────────


@dataclass
class EmitContext:
    """Scratch state passed to all template renders."""
    schema: Schema
    arch_name: str
    ckpt: LoadedCheckpoint
    crate_root: Path
    git_commit: str
    timestamp: str
    export_timestamp_unix: int


class RustEmitter:
    def __init__(
        self,
        schema: Schema,
        ckpt: LoadedCheckpoint,
        crate_root: Path,
        arch_name: str | None = None,
    ):
        self.schema = schema
        self.ckpt = ckpt
        self.crate_root = Path(crate_root)
        self.arch_name = arch_name or ckpt.arch_name
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    # ── Public entry point ──────────────────────────────────────────

    def emit(
        self,
        model: nn.Module,
        fsq_cal: FsqCalibration | None = None,
        snn_ckpt: Path | None = None,
    ) -> Path:
        """Generate every .rs file under `{crate_root}/src/generated/` plus
        `metadata.rs` and `.exportlock.json`. Returns the crate root.

        If `snn_ckpt` is given, also emits `generated/snn/*.rs` from the
        Mamba bidirectional SNN checkpoint and folds those bytes into the
        firmware CRC.
        """
        gen_dir = self.crate_root / "src" / "generated"
        focal_dir = gen_dir / "focal"
        gen_dir.mkdir(parents=True, exist_ok=True)
        focal_dir.mkdir(parents=True, exist_ok=True)

        ctx = EmitContext(
            schema=self.schema,
            arch_name=self.arch_name,
            ckpt=self.ckpt,
            crate_root=self.crate_root,
            git_commit=self._git_commit(),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            export_timestamp_unix=int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
        )
        arch = self.schema.get_arch(self.arch_name)
        layers = self.schema.resolved_layers(self.arch_name)

        # Pass 1: extract per-layer data.
        ternary_layers: list[TernaryLayerData] = []
        int8_layers: list[Int8LayerData] = []
        for spec in layers:
            try:
                if spec.weight_kind == "ternary_2bit_packed":
                    ternary_layers.append(extract_ternary_layer(model, spec))
                elif spec.weight_kind == "int8":
                    int8_layers.append(extract_int8_layer(model, spec))
                else:
                    raise ValueError(f"Unknown weight_kind: {spec.weight_kind}")
            except KeyError as e:
                # Missing layer — skip with a warning. Architecture variants
                # may not have every layer (e.g. legacy_v7_0 lacks dw_gate).
                print(f"  [skip] {spec.name}: {e}", file=sys.stderr)

        # Pass 2: rotation matrix (optional).
        rotation_q15: list[int] | None = None
        if self.schema.rotation.source_param in self.ckpt.state_dict:
            a = self.ckpt.state_dict[self.schema.rotation.source_param].cpu().numpy()
            rotation_q15 = cayley_rotation_q15(a).tolist()

        # Pass 3: render templates + collect CRC inputs.
        crc_inputs: list[bytes] = []
        crc_buffer_order: list[str] = []

        # Per-layer files.
        for layer_data in ternary_layers:
            self._emit_ternary_layer(focal_dir / f"{layer_data.spec.name}.rs",
                                     layer_data, ctx)
            crc_inputs.append(layer_data.packed_bytes)
            crc_buffer_order.append(f"focal::{layer_data.spec.name}::PACKED_WEIGHTS")

        for int8_data in int8_layers:
            self._emit_int8_layer(focal_dir / f"{int8_data.spec.name}.rs",
                                  int8_data, ctx)
            crc_inputs.append(bytes(np.asarray(int8_data.weights, dtype=np.int8).tobytes()))
            crc_buffer_order.append(f"focal::{int8_data.spec.name}::WEIGHTS_RAW")

        if rotation_q15 is not None:
            self._emit_rotation(gen_dir / "rotation.rs", rotation_q15, ctx)
            crc_inputs.append(np.asarray(rotation_q15, dtype=np.int16).tobytes())
            crc_buffer_order.append("rotation::ROTATION_Q_Q15")

        # FSQ + rANS table.
        if fsq_cal is None:
            print("  [warn] No FSQ calibration provided; skipping fsq.rs.",
                  file=sys.stderr)
        else:
            self._emit_fsq(gen_dir / "fsq.rs", fsq_cal, ctx)

        # Toeplitz seeds (constant from schema).
        self._emit_toeplitz(gen_dir / "toeplitz.rs", ctx)

        # SNN (Mamba) — optional, gated by a separate checkpoint.
        has_snn = False
        if snn_ckpt is not None:
            try:
                snn_crc_inputs, snn_crc_order = emit_snn_crate(snn_ckpt, gen_dir)
                crc_inputs.extend(snn_crc_inputs)
                crc_buffer_order.extend(snn_crc_order)
                has_snn = True
                print(f"  SNN: {len(snn_crc_inputs)} buffers from {snn_ckpt.name}")
            except Exception as e:
                print(f"  [warn] SNN emission failed: {e}", file=sys.stderr)

        # CRC over all weight bytes (encoder + SNN if present).
        firmware_crc = crc32_of(crc_inputs)

        # crc.rs.
        self._emit_crc(gen_dir / "crc.rs", firmware_crc, crc_buffer_order, ctx)

        # mod.rs (re-export tree).
        self._emit_generated_mod(
            gen_dir / "mod.rs",
            ternary_layers=[ld.spec for ld in ternary_layers],
            int8_layers=[ld.spec for ld in int8_layers],
            has_snn=has_snn,
            ctx=ctx,
        )

        # Top-level metadata.rs (sibling of generated/).
        self._emit_metadata(self.crate_root / "src" / "metadata.rs",
                            arch_name=self.arch_name,
                            firmware_crc=firmware_crc,
                            ctx=ctx)

        # .exportlock.json.
        self._write_exportlock(arch_name=self.arch_name,
                               firmware_crc=firmware_crc,
                               ctx=ctx)

        return self.crate_root

    # ── Per-template render helpers ─────────────────────────────────

    def _common_ctx(self, ctx: EmitContext) -> dict:
        return {
            "schema_version": ctx.schema.schema_version,
            "exporter_version": EXPORTER_VERSION,
            "arch_name": ctx.arch_name,
            "encoder_class": ctx.schema.get_arch(ctx.arch_name).encoder_class,
            "ckpt_path": str(ctx.ckpt.path),
            "ckpt_basename": ctx.ckpt.path.name,
            "ckpt_sha256": ctx.ckpt.sha256,
            "timestamp": ctx.timestamp,
            "git_commit": ctx.git_commit,
            "export_timestamp_unix": ctx.export_timestamp_unix,
        }

    def _emit_ternary_layer(self, dest: Path, data: TernaryLayerData, ctx: EmitContext) -> None:
        tmpl = self.env.get_template("layer_ternary.rs.j2")
        rendered = tmpl.render(
            **self._common_ctx(ctx),
            layer=data.spec,
            packed=data.packed,
            alphas=data.alphas_q15,
            norm_weight=data.norm_weight_q7 or [],
            norm_bias=data.norm_bias_q15 or [],
            n_weights=data.n_weights,
        )
        dest.write_text(rendered)

    def _emit_int8_layer(self, dest: Path, data: Int8LayerData, ctx: EmitContext) -> None:
        tmpl = self.env.get_template("layer_int8.rs.j2")
        rendered = tmpl.render(
            **self._common_ctx(ctx),
            layer=data.spec,
            weights=data.weights,
            scales=data.scales_q15,
        )
        dest.write_text(rendered)

    def _emit_rotation(self, dest: Path, q_q15: list[int], ctx: EmitContext) -> None:
        tmpl = self.env.get_template("rotation.rs.j2")
        rendered = tmpl.render(
            **self._common_ctx(ctx),
            dim=self.schema.rotation.dim,
            q_q15=q_q15,
        )
        dest.write_text(rendered)

    def _emit_fsq(self, dest: Path, cal: FsqCalibration, ctx: EmitContext) -> None:
        tmpl = self.env.get_template("fsq.rs.j2")
        rendered = tmpl.render(
            **self._common_ctx(ctx),
            num_levels=cal.num_levels,
            total_freq=cal.total_freq,
            levels=self.schema.fsq.levels,
            quant_scale_q31=self.schema.fsq.quant_scale_q31,
            freq=cal.freq,
            start=cal.start,
            vmin_q31=cal.vmin_q31,
            vmax_q31=cal.vmax_q31,
            inv_range_q31=cal.inv_range_q31,
            vmin_float=cal.vmin_q31 / 1000.0,
            vmax_float=cal.vmax_q31 / 1000.0,
            entropy_bps=cal.entropy_bps,
            calibration_n=self.schema.fsq.calibration_n_samples,
        )
        dest.write_text(rendered)

    def _emit_toeplitz(self, dest: Path, ctx: EmitContext) -> None:
        tmpl = self.env.get_template("toeplitz.rs.j2")
        rendered = tmpl.render(
            **self._common_ctx(ctx),
            seeds=self.schema.toeplitz.seeds,
        )
        dest.write_text(rendered)

    def _emit_crc(
        self,
        dest: Path,
        firmware_crc: int,
        buffer_order: list[str],
        ctx: EmitContext,
    ) -> None:
        tmpl = self.env.get_template("crc.rs.j2")
        rendered = tmpl.render(
            **self._common_ctx(ctx),
            firmware_crc32=firmware_crc,
            buffer_order=buffer_order,
        )
        dest.write_text(rendered)

    def _emit_generated_mod(
        self,
        dest: Path,
        ternary_layers: list[ResolvedLayer],
        int8_layers: list[ResolvedLayer],
        has_snn: bool,
        ctx: EmitContext,
    ) -> None:
        tmpl = self.env.get_template("generated_mod.rs.j2")
        rendered = tmpl.render(
            **self._common_ctx(ctx),
            ternary_layers=ternary_layers,
            int8_layers=int8_layers,
            has_snn=has_snn,
        )
        dest.write_text(rendered)

    def _emit_metadata(
        self,
        dest: Path,
        arch_name: str,
        firmware_crc: int,
        ctx: EmitContext,
    ) -> None:
        arch = self.schema.get_arch(arch_name)
        tmpl = self.env.get_template("metadata.rs.j2")
        sha_bytes = bytes.fromhex(ctx.ckpt.sha256)
        rendered = tmpl.render(
            **self._common_ctx(ctx),
            model_version="7.7.0",  # TODO: from pyproject or git tag
            encoder_width=arch.encoder_width,
            n_focal_blocks=arch.n_focal_blocks,
            latent_dims=arch.latent_dims,
            latent_timesteps=arch.latent_timesteps,
            ckpt_sha256_bytes=list(sha_bytes),
            firmware_crc32=firmware_crc,
        )
        dest.write_text(rendered)

    # ── Auxiliary ───────────────────────────────────────────────────

    def _git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.crate_root.parent),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def _write_exportlock(self, arch_name: str, firmware_crc: int, ctx: EmitContext) -> None:
        lock = {
            "schema_version": self.schema.schema_version,
            "exporter_version": EXPORTER_VERSION,
            "is_initialized": True,
            "model_arch": arch_name,
            "checkpoint": str(ctx.ckpt.path),
            "checkpoint_sha256": ctx.ckpt.sha256,
            "git_commit": ctx.git_commit,
            "timestamp": ctx.timestamp,
            "firmware_crc32": f"0x{firmware_crc:08X}",
        }
        (self.crate_root / ".exportlock.json").write_text(
            json.dumps(lock, indent=2) + "\n"
        )
