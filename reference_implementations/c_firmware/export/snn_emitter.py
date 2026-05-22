"""SNN (Mamba) weight emitter — pure Q-format firmware contract.

Reads the Mamba-bidirectional state-space model checkpoint and emits
`lamquant-weights/src/generated/snn/*.rs` with **no f32 constants**.
Every dequant scale is pre-baked into Q15 (`i32`); the per-block
`|A|` table is pre-baked into Q10 (`[[i16; D_STATE]; D_INNER]`); the
`D` skip table into Q15 (`[i16; D_INNER]`); the scalar `dt_bias` into
Q15 (`i32`). The firmware's `lamquant_firmware::neural::snn::inference()`
reads these tables verbatim — no `f32`, `libm`, or `core::f32::*` on
the MCU init path.

Pre-bake formulas mirror the prior firmware init code exactly:
  scale_to_q15(s_f32) :=  round_half_up_clamp_i32(s * 32768)
  d_q15(d_i8, s)      :=  round_half_up_clamp_i16(d_i8 * s * 32768)
  a_abs_q10(a, s)     :=  round_half_up_clamp_i16(exp(a_i16 * s) * 1024)
  dt_bias_q15(b, s)   :=  round_half_up_clamp_i32(b_i8 * s * 32768)

Schema (mirrors the trained PyTorch state-dict):

    spatial_mix.{weight, bias}            21 → 40 Linear (i8 + Q15 scale)
    readout.{weight, bias}                40 → 8  Linear (i8 + Q15 scale)

    ssm_blocks.{0,1}.norm.{weight, bias}  RMSNorm (i8 + Q15 scale)
    ssm_blocks.{0,1}.{fwd,bwd}.in_proj.weight    (160, 40)   i8 + Q15 scale
    ssm_blocks.{0,1}.{fwd,bwd}.x_proj.weight     (33,  80)   i8 + Q15 scale
    ssm_blocks.{0,1}.{fwd,bwd}.out_proj.weight   (40,  80)   i8 + Q15 scale
    ssm_blocks.{0,1}.{fwd,bwd}.conv1d.weight     (80, 1, 4)  i8 + Q15 scale
    ssm_blocks.{0,1}.{fwd,bwd}.conv1d.bias       (80,)       i8 + Q15 scale
    ssm_blocks.{0,1}.{fwd,bwd}.A_ABS_Q10         (80, 16)    pre-baked
    ssm_blocks.{0,1}.{fwd,bwd}.D_Q15             (80,)       pre-baked
    ssm_blocks.{0,1}.{fwd,bwd}.DT_BIAS_Q15       scalar      pre-baked
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
    """LEGACY — retained only for documentation. Pure-Q codegen no
    longer emits any `pub const ... : f32` line; use
    `_format_q15_scale_const` instead."""
    out = []
    if doc:
        out.append(f"/// {doc}")
    out.append(f"pub const {name}: f32 = {v:.9e};")
    return "\n".join(out)


# ─── Pure Q-format codegen helpers (firmware bake-out) ───────────────


def _round_half_up_clamp(x: float, lo: int, hi: int) -> int:
    """Round-half-away-from-zero with saturating clamp.

    Mirrors the Rust firmware's pattern (`if x>=0 {x+0.5} else {x-0.5}`
    + i32::MIN/MAX bounds) used in `scale_to_q15`, `dt_bias_q15`,
    `round_f32_to_i16`. The clamp uses `<= hi`/`>= lo` to match the
    `>=` / `<=` semantics from the V4 Pro 7ce5a488 review fix.
    """
    if x >= 0.0:
        r = int(x + 0.5)
    else:
        r = int(x - 0.5)
    if r > hi:
        return hi
    if r < lo:
        return lo
    return r


def _q15_scale(s_f32: float) -> int:
    """Pre-bake `scale_to_q15(s)`. Returns Q15 i32."""
    return _round_half_up_clamp(s_f32 * 32768.0, -(1 << 31), (1 << 31) - 1)


def _format_q15_scale_const(name: str, v_f32: float, doc: str | None = None) -> str:
    """Emit `pub const NAME_Q15: i32 = <pre-baked>`."""
    q = _q15_scale(v_f32)
    out = []
    if doc:
        out.append(f"/// {doc} (Q15, pre-baked from f32={v_f32:.9e})")
    out.append(f"pub const {name}_Q15: i32 = {q};")
    return "\n".join(out)


def _format_i32_const(name: str, v_i32: int, doc: str | None = None) -> str:
    out = []
    if doc:
        out.append(f"/// {doc}")
    out.append(f"pub const {name}: i32 = {v_i32};")
    return "\n".join(out)


def _bake_a_abs_q10(a_log_q15: np.ndarray, a_log_scale: float,
                    d_inner: int, d_state: int) -> np.ndarray:
    """Pre-compute |A|_q10 table: shape (D_INNER, D_STATE) i16.

    Mirrors firmware `build_a_abs_q10`: A_real = exp(A_LOG[i] * scale),
    Q10 = round_clamp_i16(A_real * 1024). Uses `np.exp` (f64) at codegen
    time — strictly more precise than the firmware's prior Padé f32
    approximant; downstream SNN robustness handles ≤ ±1 LSB drift.
    """
    a_real = np.exp(a_log_q15.astype(np.float64).reshape(d_inner, d_state) * a_log_scale)
    a_q10 = np.clip(np.round(a_real * 1024.0), -32768, 32767).astype(np.int16)
    return a_q10


def _bake_d_q15(d_table_i8: np.ndarray, d_scale: float) -> np.ndarray:
    """Pre-compute D_q15 table: shape (D_INNER,) i16."""
    d_real = d_table_i8.astype(np.float64) * d_scale
    return np.clip(np.round(d_real * 32768.0), -32768, 32767).astype(np.int16)


def _bake_dt_bias_q15(dt_bias_i8_0: int, dt_bias_scale: float) -> int:
    """Pre-compute scalar DT_BIAS_Q15 i32 (firmware uses [0] only)."""
    real = float(dt_bias_i8_0) * dt_bias_scale
    return _round_half_up_clamp(real * 32768.0, -(1 << 31), (1 << 31) - 1)


def _format_i16_2d_array(name: str, arr: np.ndarray, doc: str | None = None) -> str:
    """Render `pub static {name}: [[i16; N]; M] = [...]`."""
    m, n = arr.shape
    parts: list[str] = []
    parts.append(f"pub static {name}: [[i16; {n}]; {m}] = [")
    for row in arr:
        cells = " ".join(f"{int(v)}," for v in row)
        parts.append(f"    [{cells.rstrip(',')}],")
    parts.append("];")
    out = []
    if doc:
        out.append(f"/// {doc}")
    out.append(f"pub const {name}_INNER: usize = {m};")
    out.append(f"pub const {name}_STATE: usize = {n};")
    out.append("\n".join(parts))
    return "\n".join(out)


# Mamba SSM shape constants — must match `lamquant_firmware::neural::snn`.
_D_INNER = 80
_D_STATE = 16


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
    """Emit a Linear layer as int8 weight + int8 bias + **Q15 i32 scales**
    (pre-baked from the per-tensor f32 abs_max/127 scale). Pure Q-format
    contract — no `pub const ... : f32` lines in the generated file."""
    w_q, w_scale = _to_i8_with_scale(weight.detach().cpu().numpy())
    parts: list[str] = []
    parts.append(_format_i8_array(f"{name.upper()}_WEIGHT", w_q,
                                   doc=f"{name} weight, shape={tuple(weight.shape)}"))
    parts.append(_format_q15_scale_const(f"{name.upper()}_WEIGHT_SCALE", w_scale,
                                          doc="Dequantize: real = q8 * (Q15 / 32768)"))
    if bias is not None:
        b_q, b_scale = _to_i8_with_scale(bias.detach().cpu().numpy())
        parts.append(_format_i8_array(f"{name.upper()}_BIAS", b_q,
                                       doc=f"{name} bias, shape={tuple(bias.shape)}"))
        parts.append(_format_q15_scale_const(f"{name.upper()}_BIAS_SCALE", b_scale))
    return "\n\n".join(parts)


# Tensors that stay Q15 (i16) — A_log is dense near zero, int8 saturates.
_I16_TENSOR_SUFFIXES: set[str] = {"A_log"}


def _emit_ssm_block(prefix: str, sd: dict[str, torch.Tensor]) -> str:
    """Emit one Mamba direction (fwd or bwd) — **pure Q-format**.

    Linear-projection weights (in_proj / x_proj / out_proj / conv1d_w /
    conv1d_b) stay int8 + per-tensor Q15 scale. `A_log + scale` collapse
    into a pre-baked `A_ABS_Q10` 2-D table. `D + scale` collapse into a
    pre-baked `D_Q15` 1-D table. `dt_bias + scale` collapse into a single
    scalar `DT_BIAS_Q15` (firmware uses `[0]` only). No `f32` constants
    survive in the generated file.
    """
    parts: list[str] = []
    # Linear-style tensors — int8 array + pre-baked Q15 i32 scale.
    linear_specs = [
        ("in_proj.weight",  "IN_PROJ_W"),
        ("x_proj.weight",   "X_PROJ_W"),
        ("out_proj.weight", "OUT_PROJ_W"),
        ("conv1d.weight",   "CONV1D_W"),
        ("conv1d.bias",     "CONV1D_B"),
    ]
    for src_suffix, rust_name in linear_specs:
        key = f"{prefix}.{src_suffix}"
        if key not in sd:
            continue
        t = sd[key]
        q, scale = _to_i8_with_scale(t.detach().cpu().numpy())
        parts.append(_format_i8_array(rust_name, q,
                                       doc=f"{src_suffix}, shape={tuple(t.shape)} (int8)"))
        parts.append(_format_q15_scale_const(f"{rust_name}_SCALE", scale))

    # A_log → pre-baked |A|_q10 table.
    a_log_key = f"{prefix}.A_log"
    if a_log_key in sd:
        a_log_np = sd[a_log_key].detach().cpu().numpy()
        a_log_q15, a_log_scale = _to_q15_with_scale(a_log_np)
        a_abs_q10 = _bake_a_abs_q10(a_log_q15, a_log_scale, _D_INNER, _D_STATE)
        parts.append(_format_i16_2d_array(
            "A_ABS_Q10", a_abs_q10,
            doc=f"A_log → |A|_q10 (Q10 pre-baked from shape={tuple(a_log_np.shape)})",
        ))

    # D → pre-baked D_q15 table.
    d_key = f"{prefix}.D"
    if d_key in sd:
        d_np = sd[d_key].detach().cpu().numpy()
        d_i8, d_scale = _to_i8_with_scale(d_np)
        d_q15 = _bake_d_q15(d_i8, d_scale)
        parts.append(_format_i16_array(
            "D_Q15", d_q15,
            doc=f"D skip table, Q15 pre-baked from shape={tuple(d_np.shape)}",
        ))

    # dt_bias → scalar DT_BIAS_Q15 (firmware only uses index 0).
    dt_key = f"{prefix}.dt_bias"
    if dt_key in sd:
        dt_np = sd[dt_key].detach().cpu().numpy()
        dt_i8, dt_scale = _to_i8_with_scale(dt_np)
        dt_q15 = _bake_dt_bias_q15(int(dt_i8[0]), dt_scale)
        parts.append(_format_i32_const(
            "DT_BIAS_Q15", dt_q15,
            doc=f"dt_bias[0] in Q15 — pre-baked from f32 scale {dt_scale:.6e}",
        ))

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
            # CRC inputs for this direction — pure Q-format:
            # int8 linear weights + pre-baked A_ABS_Q10 + D_Q15 + DT_BIAS_Q15.
            for tname, rust_name in (
                ("in_proj.weight",  "IN_PROJ_W"),
                ("x_proj.weight",   "X_PROJ_W"),
                ("out_proj.weight", "OUT_PROJ_W"),
                ("conv1d.weight",   "CONV1D_W"),
                ("conv1d.bias",     "CONV1D_B"),
            ):
                key = f"{prefix}.{tname}"
                if key not in sd:
                    continue
                q, _ = _to_i8_with_scale(sd[key].detach().cpu().numpy())
                crc_inputs.append(q.tobytes())
                crc_buffer_order.append(f"snn::{modname}::{rust_name}")
            # Pre-baked tables — they replace raw A_log, D, dt_bias bytes.
            a_log_key = f"{prefix}.A_log"
            if a_log_key in sd:
                a_log_q15, a_log_scale = _to_q15_with_scale(sd[a_log_key].detach().cpu().numpy())
                a_abs_q10 = _bake_a_abs_q10(a_log_q15, a_log_scale, _D_INNER, _D_STATE)
                crc_inputs.append(a_abs_q10.tobytes())
                crc_buffer_order.append(f"snn::{modname}::A_ABS_Q10")
            d_key = f"{prefix}.D"
            if d_key in sd:
                d_i8, d_scale = _to_i8_with_scale(sd[d_key].detach().cpu().numpy())
                d_q15 = _bake_d_q15(d_i8, d_scale)
                crc_inputs.append(d_q15.tobytes())
                crc_buffer_order.append(f"snn::{modname}::D_Q15")
            dt_key = f"{prefix}.dt_bias"
            if dt_key in sd:
                dt_i8, dt_scale = _to_i8_with_scale(sd[dt_key].detach().cpu().numpy())
                dt_q15 = _bake_dt_bias_q15(int(dt_i8[0]), dt_scale)
                crc_inputs.append(int(dt_q15).to_bytes(4, "little", signed=True))
                crc_buffer_order.append(f"snn::{modname}::DT_BIAS_Q15")

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
