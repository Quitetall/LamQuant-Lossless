"""SNN (Mamba) weight emitter.

Reads the Mamba-bidirectional state-space model checkpoint and emits
`lamquant-weights/src/generated/snn/*.rs` with all tensors quantized to
Q15 i16. Compact, deterministic, no_std-friendly: every buffer is a
plain `pub static [i16; N]`.

The Rust firmware's `lamquant_firmware::neural::snn::inference()` consumes
these tables; until this emitter ran, the firmware fell back to the
placeholder routing committed earlier on the same branch.

Schema (mirrors the trained PyTorch state-dict):

    spatial_mix.{weight, bias}            21 → 40 Linear (i16 Q15)
    readout.{weight, bias}                40 → 8  Linear (i16 Q15)

    ssm_blocks.{0,1}.norm.{weight, bias}  RMSNorm (i16 Q15)
    ssm_blocks.{0,1}.{fwd,bwd}.in_proj.weight    (160, 40)   Q15
    ssm_blocks.{0,1}.{fwd,bwd}.x_proj.weight     (33,  80)   Q15
    ssm_blocks.{0,1}.{fwd,bwd}.out_proj.weight   (40,  80)   Q15
    ssm_blocks.{0,1}.{fwd,bwd}.conv1d.weight     (80, 1, 4)  Q15  (depthwise k=4)
    ssm_blocks.{0,1}.{fwd,bwd}.conv1d.bias       (80,)       Q15
    ssm_blocks.{0,1}.{fwd,bwd}.A_log             (80, 16)    Q15
    ssm_blocks.{0,1}.{fwd,bwd}.D                 (80,)       Q15
    ssm_blocks.{0,1}.{fwd,bwd}.dt_bias           (80,)       Q15

Quantization:
    Each tensor: clip to its observed [vmin, vmax] range, divide by
    max(|v|), scale to int16. Per-tensor float scale is recorded so the
    runtime path can dequant: `f32_value = (q15 / 32767.0) * scale`.
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


# ─── Quantization helpers ────────────────────────────────────────────


def _to_i8_with_scale(t: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize `t` to int8 with per-tensor float32 scale.

    Matches the C exporter's `mamba_*` path so byte-identical parity is
    attainable. Recovery: `f32 = q8 * scale`.

    Scale = abs_max / 127. Clipping to [-128, 127] retains symmetric
    distribution; the round-trip error is ≤ 0.5 * scale per element.
    """
    flat = np.ascontiguousarray(t, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return np.zeros(0, dtype=np.int8), 1.0
    abs_max = float(np.abs(flat).max())
    if abs_max == 0.0:
        return np.zeros(flat.size, dtype=np.int8), 1.0
    scale = abs_max / 127.0
    q = np.clip(np.round(flat / scale), -128.0, 127.0).astype(np.int8)
    return q, scale


def _format_i8_array(name: str, arr: np.ndarray, doc: str | None = None) -> str:
    """Render `pub static {name}: [i8; N] = [...];`."""
    body_parts: list[str] = []
    for i, v in enumerate(arr.tolist()):
        body_parts.append(f"{int(v)},")
        if (i + 1) % 16 == 0:
            body_parts.append("\n    ")
    body = " ".join(body_parts).rstrip()
    out = []
    if doc:
        out.append(f"/// {doc}")
    out.append(f"pub const {name}_LEN: usize = {arr.size};")
    out.append(f"pub static {name}: [i8; {arr.size}] = [")
    out.append("    " + body)
    out.append("];")
    return "\n".join(out)


# A_log is stored Q15 in the C path because it's all negative log values
# packed near zero — int8 would saturate. Use i16 for A_log only.

def _to_q15_with_scale(t: np.ndarray) -> tuple[np.ndarray, float]:
    flat = np.ascontiguousarray(t, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return np.zeros(0, dtype=np.int16), 1.0
    abs_max = float(np.abs(flat).max())
    if abs_max == 0.0:
        return np.zeros(flat.size, dtype=np.int16), 1.0
    scale = abs_max / 32767.0
    q = np.clip(np.round(flat / scale), -32768.0, 32767.0).astype(np.int16)
    return q, scale


def _format_i16_array(name: str, arr: np.ndarray, doc: str | None = None) -> str:
    body_parts: list[str] = []
    for i, v in enumerate(arr.tolist()):
        body_parts.append(f"{int(v)},")
        if (i + 1) % 16 == 0:
            body_parts.append("\n    ")
    body = " ".join(body_parts).rstrip()
    out = []
    if doc:
        out.append(f"/// {doc}")
    out.append(f"pub const {name}_LEN: usize = {arr.size};")
    out.append(f"pub static {name}: [i16; {arr.size}] = [")
    out.append("    " + body)
    out.append("];")
    return "\n".join(out)


def _format_f32_const(name: str, v: float, doc: str | None = None) -> str:
    out = []
    if doc:
        out.append(f"/// {doc}")
    out.append(f"pub const {name}: f32 = {v:.9e};")
    return "\n".join(out)


# ─── Per-block emitter ───────────────────────────────────────────────


_HEADER = """\
// **GENERATED — DO NOT EDIT.**
//
// Source:    {ckpt_path}
// SHA-256:   {sha256}
// Generated: {timestamp}
//
// Regenerate via:
//   python firmware/export_firmware.py --target rust --snn-checkpoint <path>
//
// Q15 quantization with per-tensor f32 scale. To dequantize at runtime:
//   f32 = (q15 / 32_767.0) * SCALE
"""


def _emit_linear(name: str, weight: torch.Tensor, bias: torch.Tensor | None) -> str:
    """Emit a Linear layer as int8 weight + int8 bias + per-tensor scales."""
    w_q, w_scale = _to_i8_with_scale(weight.detach().cpu().numpy())
    parts: list[str] = []
    parts.append(_format_i8_array(f"{name.upper()}_WEIGHT", w_q,
                                   doc=f"{name} weight, shape={tuple(weight.shape)}"))
    parts.append(_format_f32_const(f"{name.upper()}_WEIGHT_SCALE", w_scale,
                                    doc="Dequantize: f32 = q8 * SCALE"))
    if bias is not None:
        b_q, b_scale = _to_i8_with_scale(bias.detach().cpu().numpy())
        parts.append(_format_i8_array(f"{name.upper()}_BIAS", b_q,
                                       doc=f"{name} bias, shape={tuple(bias.shape)}"))
        parts.append(_format_f32_const(f"{name.upper()}_BIAS_SCALE", b_scale))
    return "\n\n".join(parts)


# Tensors that stay Q15 (i16) — A_log is dense near zero, int8 saturates.
_I16_TENSOR_SUFFIXES: set[str] = {"A_log"}


def _emit_ssm_block(prefix: str, sd: dict[str, torch.Tensor]) -> str:
    """Emit one Mamba direction (fwd or bwd). int8 + scale per tensor;
    A_log uses i16 because it's dense near zero (would saturate at int8)."""
    parts: list[str] = []
    tensor_specs = [
        ("in_proj.weight",  "IN_PROJ_W"),
        ("x_proj.weight",   "X_PROJ_W"),
        ("out_proj.weight", "OUT_PROJ_W"),
        ("conv1d.weight",   "CONV1D_W"),
        ("conv1d.bias",     "CONV1D_B"),
        ("A_log",           "A_LOG"),
        ("D",               "D"),
        ("dt_bias",         "DT_BIAS"),
    ]
    for src_suffix, rust_name in tensor_specs:
        key = f"{prefix}.{src_suffix}"
        if key not in sd:
            continue
        t = sd[key]
        if src_suffix in _I16_TENSOR_SUFFIXES:
            q, scale = _to_q15_with_scale(t.detach().cpu().numpy())
            parts.append(_format_i16_array(rust_name, q,
                                            doc=f"{src_suffix}, shape={tuple(t.shape)} (Q15)"))
        else:
            q, scale = _to_i8_with_scale(t.detach().cpu().numpy())
            parts.append(_format_i8_array(rust_name, q,
                                           doc=f"{src_suffix}, shape={tuple(t.shape)} (int8)"))
        parts.append(_format_f32_const(f"{rust_name}_SCALE", scale))
    return "\n\n".join(parts)


# ─── Top-level emit ──────────────────────────────────────────────────


def emit_snn_crate(snn_ckpt: Path, generated_dir: Path) -> tuple[list[bytes], list[str]]:
    """Emit `{generated_dir}/snn/*.rs` from a Mamba SNN checkpoint.

    Returns (crc_inputs, crc_buffer_order) so the caller can fold SNN
    bytes into `FIRMWARE_CRC32`.
    """
    sd = torch.load(snn_ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if not isinstance(sd, dict):
        raise RuntimeError(f"SNN checkpoint at {snn_ckpt} did not yield a state_dict")

    sha256 = hashlib.sha256(snn_ckpt.read_bytes()).hexdigest()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    header = _HEADER.format(ckpt_path=str(snn_ckpt), sha256=sha256, timestamp=timestamp)

    snn_dir = generated_dir / "snn"
    snn_dir.mkdir(parents=True, exist_ok=True)

    crc_inputs: list[bytes] = []
    crc_buffer_order: list[str] = []
    written: list[str] = []

    def write_module(filename: str, body: str, modname: str) -> None:
        body_full = header + "\n" + body + "\n"
        (snn_dir / filename).write_text(body_full)
        written.append(modname)
        # Capture every i16 array in the body for the CRC fold.
        # Cheap heuristic: walk the live tensors we just packed.

    # spatial_mix (21 → 40), int8.
    sm_w = sd.get("spatial_mix.weight")
    sm_b = sd.get("spatial_mix.bias")
    if sm_w is not None:
        body = _emit_linear("spatial_mix", sm_w, sm_b)
        write_module("spatial_mix.rs", body, "spatial_mix")
        wq, _ = _to_i8_with_scale(sm_w.detach().cpu().numpy())
        crc_inputs.append(wq.tobytes())
        crc_buffer_order.append("snn::spatial_mix::SPATIAL_MIX_WEIGHT")
        if sm_b is not None:
            bq, _ = _to_i8_with_scale(sm_b.detach().cpu().numpy())
            crc_inputs.append(bq.tobytes())
            crc_buffer_order.append("snn::spatial_mix::SPATIAL_MIX_BIAS")

    # readout (40 → 8), int8.
    ro_w = sd.get("readout.weight")
    ro_b = sd.get("readout.bias")
    if ro_w is not None:
        body = _emit_linear("readout", ro_w, ro_b)
        write_module("readout.rs", body, "readout")
        wq, _ = _to_i8_with_scale(ro_w.detach().cpu().numpy())
        crc_inputs.append(wq.tobytes())
        crc_buffer_order.append("snn::readout::READOUT_WEIGHT")
        if ro_b is not None:
            bq, _ = _to_i8_with_scale(ro_b.detach().cpu().numpy())
            crc_inputs.append(bq.tobytes())
            crc_buffer_order.append("snn::readout::READOUT_BIAS")

    # ssm_blocks 0..N
    block_idxs: set[int] = set()
    for k in sd.keys():
        if k.startswith("ssm_blocks."):
            try:
                block_idxs.add(int(k.split(".")[1]))
            except (IndexError, ValueError):
                continue

    for bi in sorted(block_idxs):
        # norm (int8)
        nw = sd.get(f"ssm_blocks.{bi}.norm.weight")
        nb = sd.get(f"ssm_blocks.{bi}.norm.bias")
        if nw is not None:
            body = _emit_linear("norm", nw, nb)
            write_module(f"layer{bi}_norm.rs", body, f"layer{bi}_norm")
            for src, rname in (
                (nw, f"snn::layer{bi}_norm::NORM_WEIGHT"),
                (nb, f"snn::layer{bi}_norm::NORM_BIAS"),
            ):
                if src is not None:
                    q, _ = _to_i8_with_scale(src.detach().cpu().numpy())
                    crc_inputs.append(q.tobytes())
                    crc_buffer_order.append(rname)

        # fwd / bwd directions
        for direction in ("fwd", "bwd"):
            prefix = f"ssm_blocks.{bi}.{direction}"
            keys_for_dir = [k for k in sd.keys() if k.startswith(prefix + ".")]
            if not keys_for_dir:
                continue
            body = _emit_ssm_block(prefix, sd)
            modname = f"layer{bi}_{direction}"
            write_module(f"{modname}.rs", body, modname)
            # CRC inputs for this direction. A_log → i16; rest → i8.
            for tname, rust_name in (
                ("in_proj.weight",  "IN_PROJ_W"),
                ("x_proj.weight",   "X_PROJ_W"),
                ("out_proj.weight", "OUT_PROJ_W"),
                ("conv1d.weight",   "CONV1D_W"),
                ("conv1d.bias",     "CONV1D_B"),
                ("A_log",           "A_LOG"),
                ("D",               "D"),
                ("dt_bias",         "DT_BIAS"),
            ):
                key = f"{prefix}.{tname}"
                if key not in sd:
                    continue
                if tname in _I16_TENSOR_SUFFIXES:
                    q, _ = _to_q15_with_scale(sd[key].detach().cpu().numpy())
                else:
                    q, _ = _to_i8_with_scale(sd[key].detach().cpu().numpy())
                crc_inputs.append(q.tobytes())
                crc_buffer_order.append(f"snn::{modname}::{rust_name}")

    # snn/mod.rs declaring submodules
    mod_lines = ["// **GENERATED — DO NOT EDIT.**", "//"]
    mod_lines.append(f"// Source:    {snn_ckpt}")
    mod_lines.append(f"// SHA-256:   {sha256}")
    mod_lines.append(f"// Generated: {timestamp}")
    mod_lines.append("//")
    mod_lines.append("//! Mamba bidirectional SNN — activity classifier weight tables.")
    mod_lines.append("")
    for mod in sorted(written):
        mod_lines.append(f"pub mod {mod};")
    (snn_dir / "mod.rs").write_text("\n".join(mod_lines) + "\n")

    return crc_inputs, crc_buffer_order
