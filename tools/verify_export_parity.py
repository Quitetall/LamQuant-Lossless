#!/usr/bin/env python3
"""
verify_export_parity.py — byte-equal parity check between C and Rust weight exports.

Reads the generated C headers under firmware/firmware_export/*.h and the generated
Rust crate under lamquant-weights/src/generated/, parses the static array literals
in each, and compares element-by-element.

Both exports must originate from the same checkpoint or the comparison is meaningless.

Exit code:
    0 — full parity (every mapped buffer is byte-identical)
    1 — at least one mismatch, missing buffer, or parse failure
    2 — environment problem (export tree not found)

Usage:
    python tools/verify_export_parity.py
    python tools/verify_export_parity.py --strict
    python tools/verify_export_parity.py --c-dir firmware/firmware_export \\
        --rust-dir lamquant-weights/src/generated --arch subband_v1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------- mapping

# C-symbol-name → Rust-module-path-and-symbol.
# A trailing wildcard `{N}` over `range_kind` expands the entry.
NAME_MAPPING: dict[str, str] = {
    # premix
    "premix_alphas_q15":           "focal::premix::ALPHAS_Q15",
    "premix_packed":               "focal::premix::PACKED_WEIGHTS",
    "premix_weights":              "focal::premix::PACKED_WEIGHTS",

    # focal blocks 1..3 (focal1 may be missing on the Rust side; that's flagged)
    "focal1_conv_alphas_q15":      "focal::focal1::ALPHAS_Q15",
    "focal1_conv_packed":          "focal::focal1::PACKED_WEIGHTS",
    "focal1_conv_weights":         "focal::focal1::PACKED_WEIGHTS",
    "focal1_norm_weight_q7":       "focal::focal1::NORM_WEIGHT_Q7",
    "focal1_norm_bias_q15":        "focal::focal1::NORM_BIAS_Q15",

    "focal2_conv_alphas_q15":      "focal::focal2::ALPHAS_Q15",
    "focal2_conv_packed":          "focal::focal2::PACKED_WEIGHTS",
    "focal2_conv_weights":         "focal::focal2::PACKED_WEIGHTS",
    "focal2_norm_weight_q7":       "focal::focal2::NORM_WEIGHT_Q7",
    "focal2_norm_bias_q15":        "focal::focal2::NORM_BIAS_Q15",

    "focal3_conv_alphas_q15":      "focal::focal3::ALPHAS_Q15",
    "focal3_conv_packed":          "focal::focal3::PACKED_WEIGHTS",
    "focal3_conv_weights":         "focal::focal3::PACKED_WEIGHTS",
    "focal3_norm_weight_q7":       "focal::focal3::NORM_WEIGHT_Q7",
    "focal3_norm_bias_q15":        "focal::focal3::NORM_BIAS_Q15",

    # depthwise gate
    "dw_gate_alphas_q15":          "focal::dw_gate::ALPHAS_Q15",
    "dw_gate_packed":              "focal::dw_gate::PACKED_WEIGHTS",
    "dw_gate_weights":             "focal::dw_gate::PACKED_WEIGHTS",

    # bneck_g (ternary)
    "bneck_g_alphas_q15":          "focal::bneck_g::ALPHAS_Q15",
    "bneck_g_packed":              "focal::bneck_g::PACKED_WEIGHTS",
    "bneck_g_weights":             "focal::bneck_g::PACKED_WEIGHTS",

    # bneck_v (INT8)
    "bneck_v_weights":             "focal::bneck_v::WEIGHTS_RAW",
    "bneck_v_weights_int8":        "focal::bneck_v::WEIGHTS_RAW",
    "bneck_v_alphas_q15":          "focal::bneck_v::SCALES_Q15",
    "bneck_v_scales_q15":          "focal::bneck_v::SCALES_Q15",

    # rotation, FSQ, toeplitz
    "rotation_Q_q15":              "rotation::ROTATION_Q_Q15",
    "rotation_q_q15":              "rotation::ROTATION_Q_Q15",
    "fsq_levels":                  "fsq::LEVELS",
    "FSQ_LEVELS":                  "fsq::LEVELS",
    "fsq_rans_freq":               "fsq::RANS_FREQ",
    "fsq_rans_start":              "fsq::RANS_START",
    "toeplitz_seeds":              "toeplitz::SEEDS",
    "TOEP_SEEDS":                  "toeplitz::SEEDS",

    # SNN (Mamba) — current C export emits these; Rust export will populate
    # lamquant_weights::generated::snn::* once W4 covers SNN. Until then,
    # these will appear as C-only mismatches (real parity gap).
    "mamba_l0_fwd_in_proj_w":      "snn::layer0_fwd::IN_PROJ_W",
    "mamba_l0_fwd_x_proj_w":       "snn::layer0_fwd::X_PROJ_W",
    "mamba_l0_fwd_out_proj_w":     "snn::layer0_fwd::OUT_PROJ_W",
    "mamba_l0_fwd_conv1d_w":       "snn::layer0_fwd::CONV1D_W",
    "mamba_l0_fwd_conv1d_b":       "snn::layer0_fwd::CONV1D_B",
    "mamba_l0_fwd_a_log_q15":      "snn::layer0_fwd::A_LOG_Q15",
    "mamba_l0_fwd_D":              "snn::layer0_fwd::D",
    "mamba_l0_fwd_dt_bias":        "snn::layer0_fwd::DT_BIAS",

    "mamba_l0_bwd_in_proj_w":      "snn::layer0_bwd::IN_PROJ_W",
    "mamba_l0_bwd_x_proj_w":       "snn::layer0_bwd::X_PROJ_W",
    "mamba_l0_bwd_out_proj_w":     "snn::layer0_bwd::OUT_PROJ_W",
    "mamba_l0_bwd_conv1d_w":       "snn::layer0_bwd::CONV1D_W",
    "mamba_l0_bwd_conv1d_b":       "snn::layer0_bwd::CONV1D_B",
    "mamba_l0_bwd_a_log_q15":      "snn::layer0_bwd::A_LOG_Q15",
    "mamba_l0_bwd_D":              "snn::layer0_bwd::D",
    "mamba_l0_bwd_dt_bias":        "snn::layer0_bwd::DT_BIAS",

    "mamba_l1_fwd_in_proj_w":      "snn::layer1_fwd::IN_PROJ_W",
    "mamba_l1_fwd_x_proj_w":       "snn::layer1_fwd::X_PROJ_W",
    "mamba_l1_fwd_out_proj_w":     "snn::layer1_fwd::OUT_PROJ_W",
    "mamba_l1_fwd_conv1d_w":       "snn::layer1_fwd::CONV1D_W",
    "mamba_l1_fwd_conv1d_b":       "snn::layer1_fwd::CONV1D_B",
    "mamba_l1_fwd_a_log_q15":      "snn::layer1_fwd::A_LOG_Q15",
    "mamba_l1_fwd_D":              "snn::layer1_fwd::D",
    "mamba_l1_fwd_dt_bias":        "snn::layer1_fwd::DT_BIAS",

    "mamba_l1_bwd_in_proj_w":      "snn::layer1_bwd::IN_PROJ_W",
    "mamba_l1_bwd_x_proj_w":       "snn::layer1_bwd::X_PROJ_W",
    "mamba_l1_bwd_out_proj_w":     "snn::layer1_bwd::OUT_PROJ_W",
    "mamba_l1_bwd_conv1d_w":       "snn::layer1_bwd::CONV1D_W",
    "mamba_l1_bwd_conv1d_b":       "snn::layer1_bwd::CONV1D_B",
    "mamba_l1_bwd_a_log_q15":      "snn::layer1_bwd::A_LOG_Q15",
    "mamba_l1_bwd_D":              "snn::layer1_bwd::D",
    "mamba_l1_bwd_dt_bias":        "snn::layer1_bwd::DT_BIAS",

    "mamba_spatial_mix_w":         "snn::spatial_mix::W",
    "mamba_spatial_mix_b":         "snn::spatial_mix::B",
    "mamba_readout_w":             "snn::readout::W",
    "mamba_readout_b":             "snn::readout::B",
}


# --------------------------------------------------------------------- parsing

# C: optional macros/attrs, then `static const TYPE NAME[N] = { values };`
_C_ARRAY_RE = re.compile(
    r"static\s+const\s+(?P<type>u?int\d+_t|int8_t|uint8_t)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*\[\s*\d*\s*\]\s*"
    r"(?:[A-Z_][A-Za-z0-9_]*\s*)*"            # macro tags (TNN_DATA, __attribute__((..)))
    r"(?:__attribute__\s*\(\([^)]*\)\)\s*)*"
    r"=\s*\{(?P<values>[^}]*)\};",
    re.DOTALL,
)

# Rust: `pub static NAME: [TYPE; N] = [ values ];` (tolerates `#[link_section]`).
_RUST_ARRAY_RE = re.compile(
    r"(?:#\[[^\]]*\]\s*)*"
    r"pub\s+static\s+(?P<name>[A-Z_][A-Z0-9_]*)\s*:\s*"
    r"\[(?P<type>[ui]\d+)\s*;\s*[A-Za-z_0-9 +*\-/()]+\]\s*=\s*\[(?P<values>[^\]]*)\];",
    re.DOTALL,
)


def _parse_int_list(raw: str) -> np.ndarray:
    """Parse a comma-separated list of decimal or 0x-hex ints."""
    nums: list[int] = []
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue
        token = token.rstrip(",")
        if not token:
            continue
        if token.endswith(("u", "U", "L")):
            token = token.rstrip("uUL")
        try:
            nums.append(int(token, 0))
        except ValueError:
            # Skip malformed token; will manifest as length mismatch later.
            continue
    return np.asarray(nums, dtype=np.int64)


def _normalise(arr: np.ndarray, type_str: str) -> np.ndarray:
    """Clip / sign-cast the array values to the target dtype's representable range."""
    type_str = type_str.lower()
    if type_str in ("u8", "uint8_t"):
        return (arr & 0xFF).astype(np.uint8)
    if type_str in ("i8", "int8_t"):
        a = arr.astype(np.int64) & 0xFF
        a = np.where(a >= 0x80, a - 0x100, a)
        return a.astype(np.int8)
    if type_str in ("u16", "uint16_t"):
        return (arr & 0xFFFF).astype(np.uint16)
    if type_str in ("i16", "int16_t"):
        a = arr.astype(np.int64) & 0xFFFF
        a = np.where(a >= 0x8000, a - 0x10000, a)
        return a.astype(np.int16)
    if type_str in ("u32", "uint32_t"):
        return (arr & 0xFFFFFFFF).astype(np.uint32)
    if type_str in ("i32", "int32_t"):
        return arr.astype(np.int32)
    return arr.astype(np.int64)


def parse_c_dir(c_dir: Path) -> dict[str, np.ndarray]:
    """Walk every .h file under c_dir and extract `static const ARRAY` literals."""
    out: dict[str, np.ndarray] = {}
    for h in sorted(c_dir.rglob("*.h")):
        text = h.read_text()
        for m in _C_ARRAY_RE.finditer(text):
            name = m.group("name")
            arr = _normalise(_parse_int_list(m.group("values")), m.group("type"))
            out[name] = arr
    return out


def parse_rust_dir(rust_dir: Path) -> dict[str, np.ndarray]:
    """Walk every .rs file under rust_dir, extract `pub static NAME: [T; N] = [...]`."""
    out: dict[str, np.ndarray] = {}
    for rs in sorted(rust_dir.rglob("*.rs")):
        # The Rust path-key is `<relative_module_path>::<NAME>`.
        rel = rs.relative_to(rust_dir).with_suffix("")
        # parts e.g. ('focal', 'premix') for focal/premix.rs
        parts = list(rel.parts)
        if parts and parts[-1] == "mod":
            parts = parts[:-1]
        module_path = "::".join(parts)

        text = rs.read_text()
        for m in _RUST_ARRAY_RE.finditer(text):
            name = m.group("name")
            arr = _normalise(_parse_int_list(m.group("values")), m.group("type"))
            full = f"{module_path}::{name}" if module_path else name
            out[full] = arr
    return out


# --------------------------------------------------------------------- diff

def diff_arrays(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    """Return (equal, detail). Detail describes first divergence if any."""
    if a.shape != b.shape:
        return False, f"shape mismatch: C has {a.shape}, Rust has {b.shape}"
    # Cast both to int64 so signed/unsigned of same bytes compare numerically.
    ai = a.astype(np.int64)
    bi = b.astype(np.int64)
    eq = np.array_equal(ai, bi)
    if eq:
        return True, ""
    diffs = np.where(ai != bi)[0]
    first = int(diffs[0])
    return False, (
        f"first diff at index {first}: C={int(ai[first])} Rust={int(bi[first])}; "
        f"{len(diffs)} of {ai.size} elements differ"
    )


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c-dir",     default=str(repo / "firmware" / "firmware_export"))
    p.add_argument("--rust-dir",  default=str(repo / "lamquant-weights" / "src" / "generated"))
    p.add_argument("--strict",    action="store_true",
                   help="Exit on first mismatch instead of reporting all.")
    args = p.parse_args()

    c_dir = Path(args.c_dir)
    rust_dir = Path(args.rust_dir)
    if not c_dir.is_dir():
        print(f"error: C export dir not found: {c_dir}", file=sys.stderr)
        return 2
    if not rust_dir.is_dir():
        print(f"error: Rust export dir not found: {rust_dir}", file=sys.stderr)
        return 2

    print(f"[*] Parsing C export   : {c_dir}")
    c_arrays = parse_c_dir(c_dir)
    print(f"    found {len(c_arrays)} arrays")

    print(f"[*] Parsing Rust export: {rust_dir}")
    rust_arrays = parse_rust_dir(rust_dir)
    print(f"    found {len(rust_arrays)} arrays")

    equal:     list[str] = []
    mismatch:  list[tuple[str, str, str]] = []   # (c_name, rust_name, detail)
    c_only:    list[str] = []
    rust_only: list[str] = []
    unmapped:  list[str] = []

    # Walk C names; for each mapped name pair, compare.
    rust_seen: set[str] = set()
    for c_name, arr_c in sorted(c_arrays.items()):
        rust_path = NAME_MAPPING.get(c_name)
        if rust_path is None:
            unmapped.append(c_name)
            continue
        arr_r = rust_arrays.get(rust_path)
        if arr_r is None:
            c_only.append(f"{c_name}  → expected {rust_path}")
            continue
        rust_seen.add(rust_path)
        ok, detail = diff_arrays(arr_c, arr_r)
        if ok:
            equal.append(c_name)
        else:
            mismatch.append((c_name, rust_path, detail))
            if args.strict:
                break

    # Anything in Rust not mapped from C is reported as rust_only.
    rust_unmapped = set(rust_arrays.keys()) - rust_seen
    rust_only = sorted(rust_unmapped)

    # ---- summary ----
    print()
    print(f"=== Parity Report ===")
    print(f"  equal:        {len(equal):4d}")
    print(f"  mismatch:     {len(mismatch):4d}")
    print(f"  C-only:       {len(c_only):4d}  (mapped but absent in Rust)")
    print(f"  Rust-only:    {len(rust_only):4d}  (no C mapping; usually generated metadata)")
    print(f"  unmapped C:   {len(unmapped):4d}  (no entry in NAME_MAPPING)")

    if mismatch:
        print()
        print("=== Mismatches (first 5) ===")
        for c_name, rust_path, detail in mismatch[:5]:
            print(f"  {c_name} ↔ {rust_path}")
            print(f"      {detail}")

    if c_only:
        print()
        print("=== C-only (mapped but Rust missing) ===")
        for line in c_only[:10]:
            print(f"  {line}")
        if len(c_only) > 10:
            print(f"  ... and {len(c_only) - 10} more")

    if rust_only:
        print()
        print("=== Rust-only (≤10 shown) ===")
        for n in rust_only[:10]:
            print(f"  {n}")
        if len(rust_only) > 10:
            print(f"  ... and {len(rust_only) - 10} more")

    if unmapped:
        print()
        print("=== Unmapped C symbols (≤10 shown) ===")
        for n in unmapped[:10]:
            print(f"  {n}")
        if len(unmapped) > 10:
            print(f"  ... and {len(unmapped) - 10} more")

    print()
    if mismatch or c_only:
        print("RESULT: PARITY FAILED")
        return 1
    print("RESULT: parity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
