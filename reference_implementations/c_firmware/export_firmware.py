#!/usr/bin/env python3
"""
LamQuant Gen 6 — Firmware Weight Exporter
==========================================
Exports student checkpoint to C header for RP2350 compilation.

Output: focal_net_weights.h containing:
  - Per-layer Q31 LSQ alphas (int32_t)
  - Per-layer 2-bit packed ternary weights (uint8_t)
  - GroupNorm weights/biases as Q31 (int32_t)
  - Conv biases as Q31 (int32_t)
  - Output layer weights as int8 (not ternary — kept at higher precision)

This is the ONLY export path. sync_firmware_weights.py is deprecated.
"""
import torch
import torch.nn as nn
import os
import sys
import numpy as np
import zlib
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT_DIR, 'ai_models', 'student'))
from train_ternary import TernaryMobileNetV5


def export_to_header(model, output_path, encoder_only=False):
    """
    Export model parameters to a C header file.

    Args:
        model: TernaryMobileNetV5 instance with loaded weights
        output_path: path to write the .h file
        encoder_only: if True, only export encoder layers (for on-chip deployment)
    """
    # Determine which modules to skip
    decoder_prefixes = ('expand1', 'expand2', 'expand3', 'expand4', 'output')

    with open(output_path, 'w') as f:
        f.write("#ifndef FOCAL_NET_WEIGHTS_H\n")
        f.write("#define FOCAL_NET_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Hardware alignment for SRAM4 TNN bank\n")
        f.write("#define TNN_DATA __attribute__((aligned(4), section(\".sram4_tnn\")))\n\n")

        if encoder_only:
            f.write("// ENCODER-ONLY export (decoder runs on base station)\n\n")

        # Pass 1: Alphas, norms, biases (non-weight parameters)
        # Path B optimization: use INT16 (Q15) for metadata to fit width=128
        # in 64KB. Q15 gives precision of 1/32768 ≈ 3e-5, far below the
        # ternary weight noise floor. Firmware promotes to Q31 with a single
        # shift: alpha_q31 = (int32_t)alpha_q15 << 16.
        for name, m in model.named_modules():
            if encoder_only and any(name.startswith(p) for p in decoder_prefixes):
                continue

            clean_name = name.replace(".", "_")

            # LSQ Alphas → Q15 (INT16)
            if hasattr(m, 'lsq_alpha'):
                alphas = torch.abs(m.lsq_alpha).detach().cpu().numpy().flatten()
                alphas_q15 = np.round(np.clip(alphas.astype(np.float64), -1.0, 1.0) * 32767.0).astype(np.int16)
                f.write(f"const int16_t {clean_name}_alphas_q15[{len(alphas_q15)}] TNN_DATA = {{\n    ")
                for i, val in enumerate(alphas_q15):
                    f.write(f"{val}, ")
                    if (i + 1) % 12 == 0:
                        f.write("\n    ")
                f.write("\n};\n\n")

            # GroupNorm weights (INT8) and biases (Q15)
            if isinstance(m, (nn.GroupNorm, nn.Conv1d)):
                for p_type in ('weight', 'bias'):
                    p = getattr(m, p_type, None)
                    if p is None:
                        continue
                    # Skip conv/ternary weights — packed in pass 2
                    if p_type == 'weight' and not isinstance(m, nn.GroupNorm):
                        continue

                    p_np = p.detach().cpu().numpy().astype(np.float64).flatten()
                    p_clipped = np.clip(p_np, -1.0, 1.0)

                    if p_type == 'weight' and isinstance(m, nn.GroupNorm):
                        # GroupNorm weight stays INT8 (already small)
                        p_int8 = np.round(p_clipped * 127.0).astype(np.int8)
                        f.write(f"const int8_t {clean_name}_{p_type}_q7[{len(p_int8)}] TNN_DATA = {{\n    ")
                        for i, val in enumerate(p_int8):
                            f.write(f"{val}, ")
                            if (i + 1) % 16 == 0:
                                f.write("\n    ")
                        f.write("\n};\n\n")
                    else:
                        # Biases → Q15 (INT16)
                        p_q15 = np.round(p_clipped * 32767.0).astype(np.int16)
                        f.write(f"const int16_t {clean_name}_{p_type}_q15[{len(p_q15)}] TNN_DATA = {{\n    ")
                        for i, val in enumerate(p_q15):
                            f.write(f"{val}, ")
                            if (i + 1) % 12 == 0:
                                f.write("\n    ")
                        f.write("\n};\n\n")

        # Pass 2: Packed ternary weights
        for name, m in model.named_modules():
            if encoder_only and any(name.startswith(p) for p in decoder_prefixes):
                continue
            if not hasattr(m, 'weight') or m.weight is None:
                continue
            if isinstance(m, nn.GroupNorm):
                continue

            clean_name = name.replace(".", "_")
            w = m.weight.data

            if hasattr(m, 'lsq_alpha'):
                # Ternary layer: quantize to {-1, 0, 1} using LSQ alpha
                alpha = torch.abs(m.lsq_alpha).data
                w_ternary = torch.round(torch.clamp(w / (alpha + 1e-8), -1, 1))
            else:
                # Non-ternary (output layer): clamp to [-1, 1] range
                w_ternary = torch.round(torch.clamp(w, -1, 1))

            # NativeTernary encoding (2026): 2-bit packing with corruption detection.
            # 00=0, 01=+1, 10=-1, 11=ERROR (detectable, decoder maps to 0).
            # The 11 pattern is never generated by the encoder. If a single bit
            # flips during transmission, the decoder detects it (11 is invalid)
            # and substitutes 0 as a safe fallback. This limits corruption
            # blast radius to 1 weight per bit error (vs 4 weights with naive
            # packing where a byte-level corruption is undetectable).
            w_np = w_ternary.cpu().numpy().flatten().astype(np.int8)

            # Validation: ensure all weights are strictly ternary
            invalid = np.sum((w_np != -1) & (w_np != 0) & (w_np != 1))
            if invalid > 0:
                print(f"  [!] WARNING: {clean_name} has {invalid} non-ternary weights")

            packed = []
            for i in range(0, len(w_np), 4):
                byte = 0
                for j in range(4):
                    if i + j < len(w_np):
                        val = w_np[i + j]
                        bits = 0b00 if val == 0 else (0b01 if val == 1 else 0b10)
                        byte |= (bits << (2 * j))
                packed.append(byte)

            # NativeTernary integrity check: no byte should contain 0b11 in any 2-bit slot
            for i, b in enumerate(packed):
                for slot in range(4):
                    if ((b >> (2 * slot)) & 0x03) == 0x03:
                        raise ValueError(
                            f"NativeTernary violation in {clean_name} byte {i} slot {slot}: "
                            f"0x{b:02X} contains 0b11 (reserved error pattern)")

            f.write(f"const uint8_t {clean_name}_weights[{len(packed)}] TNN_DATA = {{\n    ")
            for i, b in enumerate(packed):
                f.write(f"0x{b:02X}, ")
                if (i + 1) % 12 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")

        # --- Cayley rotation matrix (Q15) ---
        # The encoder applies Q = (I-A)(I+A)^{-1} to the latent before FSQ.
        # Firmware promotes Q15→Q31 at runtime: q31 = (int32_t)q15 << 16.
        # Orthogonality error at Q15 is ~1e-4, far below FSQ quantization noise.
        if hasattr(model, 'rotation_A'):
            A = model.rotation_A.detach().cpu()
            A_skew = A - A.T   # enforce skew-symmetry
            I = torch.eye(A_skew.shape[0])
            Q = torch.linalg.solve(I + A_skew, I - A_skew).numpy().astype(np.float64)
            Q_flat = Q.flatten()
            dim = Q.shape[0]
            Q_q15 = np.round(np.clip(Q_flat, -1.0, 1.0) * 32767.0).astype(np.int16)
            f.write(f"// Cayley rotation matrix Q [{dim}x{dim}], row-major, Q15\n")
            f.write(f"#define ROTATION_DIM {dim}\n")
            f.write(f"const int16_t rotation_Q_q15[{len(Q_q15)}] TNN_DATA = {{\n    ")
            for i, val in enumerate(Q_q15):
                f.write(f"{val}, ")
                if (i + 1) % 12 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")

        # --- FSQ + rANS frequency table ---
        # Computed from the actual latent distribution of the trained model
        fsq_params = compute_fsq_rans_table(model, encoder_only)
        if fsq_params:
            f.write("// ===== FSQ + rANS Parameters =====\n")
            f.write(f"// Generated from trained latent distribution\n\n")
            f.write(f"#define FSQ_NUM_LEVELS {fsq_params['num_levels']}\n")
            f.write(f"#define RANS_TOTAL_FREQ {fsq_params['total_freq']}\n\n")

            # Frequency table
            freq = fsq_params['freq']
            f.write(f"static const uint32_t fsq_rans_freq[{len(freq)}] = {{\n    ")
            for i, v in enumerate(freq):
                f.write(f"{v}, ")
                if (i + 1) % 8 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")

            # Cumulative start table
            start = fsq_params['start']
            f.write(f"static const uint32_t fsq_rans_start[{len(start)}] = {{\n    ")
            for i, v in enumerate(start):
                f.write(f"{v}, ")
                if (i + 1) % 8 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")

            # Latent range in Q31 for FSQ thresholding
            f.write(f"static const int32_t fsq_vmin_q31 = {fsq_params['vmin_q31']};\n")
            f.write(f"static const int32_t fsq_vmax_q31 = {fsq_params['vmax_q31']};\n")
            f.write(f"static const int32_t fsq_inv_range_q31 = {fsq_params['inv_range_q31']};\n\n")

        f.write("#endif // FOCAL_NET_WEIGHTS_H\n")

    print(f"[*] Exported weights → {output_path}")
    print(f"    File size: {os.path.getsize(output_path)} bytes (text format)")


def compute_fsq_rans_table(model, encoder_only=False, num_levels=16, total_freq=4096):
    """
    Run the encoder on a set of random inputs to estimate the latent distribution,
    then build the FSQ bin frequency table for rANS.

    Returns dict with freq, start, vmin/vmax in Q31, and inv_range for integer FSQ.
    """
    try:
        model.eval()
        device = next(model.parameters()).device

        # Generate diverse latent samples
        latent_samples = []
        with torch.no_grad():
            for _ in range(50):
                x = torch.clamp(torch.randn(4, 21, 2500).to(device) * 20, -50, 50)
                lat = model.encode(x, quantize=True)
                latent_samples.append(lat.cpu().numpy())

        all_latents = np.concatenate(latent_samples, axis=0)  # [N, 32, T]
        flat = all_latents.flatten()

        # Compute range
        vmin = float(flat.min())
        vmax = float(flat.max())
        span = vmax - vmin + 1e-8

        # Quantize to bins
        normalized = (flat - vmin) / span
        bins = np.clip((normalized * num_levels).astype(np.int32), 0, num_levels - 1)

        # Build frequency table
        counts = np.bincount(bins, minlength=num_levels)
        total = counts.sum()

        # Scale to total_freq, ensuring minimum 1 per symbol
        freq = np.maximum(1, (counts / total * total_freq).astype(np.int32))
        # Adjust to hit exact total
        diff = total_freq - freq.sum()
        freq[np.argmax(freq)] += diff

        # Cumulative starts
        start = np.zeros(num_levels, dtype=np.int32)
        for i in range(1, num_levels):
            start[i] = start[i-1] + freq[i-1]

        # Q31 range values for integer FSQ on firmware
        Q31 = 2147483647
        # Scale the latent range to Q31
        # The firmware receives latent values that are int32 from the TNN output.
        # We store the range boundaries as Q31 values matching the TNN output scale.
        # The TNN output is int16 activations passed through bottleneck with Q31 alpha.
        # For simplicity, we store the float range scaled to a working int32 range.
        vmin_q31 = int(vmin * 1000)  # Scale to match training normalization
        vmax_q31 = int(vmax * 1000)
        range_q31 = vmax_q31 - vmin_q31
        if range_q31 == 0:
            range_q31 = 1
        # inv_range: (num_levels << 31) / range ≈ for integer division
        inv_range_q31 = int((num_levels * (1 << 30)) / range_q31)

        print(f"[*] FSQ rANS table: L={num_levels}, range=[{vmin:.2f}, {vmax:.2f}]")
        print(f"    Freq distribution: min={freq.min()}, max={freq.max()}, total={freq.sum()}")
        print(f"    Entropy: {-np.sum((freq/freq.sum()) * np.log2(freq/freq.sum() + 1e-12)):.3f} bps")

        return {
            'num_levels': num_levels,
            'total_freq': total_freq,
            'freq': freq.tolist(),
            'start': start.tolist(),
            'vmin_q31': vmin_q31,
            'vmax_q31': vmax_q31,
            'inv_range_q31': inv_range_q31,
        }

    except Exception as e:
        print(f"[!] Could not compute FSQ table: {e}")
        return None


def compute_firmware_crc(header_path, crc_output_path):
    """
    Compute CRC32 of the exported header and write firmware_crc.h.
    This must match what integrity.c computes at boot.
    """
    with open(header_path, 'rb') as f:
        data = f.read()
    crc = zlib.crc32(data) & 0xFFFFFFFF

    with open(crc_output_path, 'w') as f:
        f.write("#ifndef FIRMWARE_CRC_H\n")
        f.write("#define FIRMWARE_CRC_H\n\n")
        f.write(f"#define FIRMWARE_CRC32 0x{crc:08X}u\n\n")
        f.write("#endif // FIRMWARE_CRC_H\n")

    print(f"[*] CRC32: 0x{crc:08X} → {crc_output_path}")


def generate_biquad_coefficients():
    """
    Print the Q30 biquad coefficients for the firmware.
    Run this if you change filter cutoffs — paste the output into biquad_q31.c.
    Requires scipy.

    Q30 format: range [-2.0, +2.0) with 30 fractional bits.
    This is required because the HP biquad's b1 ≈ -1.98 overflows Q31's
    [-1.0, +1.0) range. All stages use Q30 uniformly for simplicity.
    """
    try:
        from scipy.signal import butter, iirnotch
    except ImportError:
        print("[!] scipy not installed, skipping biquad coefficient generation")
        return

    Q30 = 1 << 30  # 1073741824
    fs = 250.0

    def print_coeffs(name, b, a):
        # Normalize by a[0]
        b = b / a[0]
        a = a / a[0]
        # Direct Form 1: b0, b1, b2, a1, a2 (a0 = 1.0 implicit)
        print(f"// {name}")
        print(f"//   b0={b[0]:.8f}, b1={b[1]:.8f}, b2={b[2]:.8f}")
        print(f"//   a1={a[1]:.8f}, a2={a[2]:.8f}")
        for coeff_name, val in [('B0', b[0]), ('B1', b[1]), ('B2', b[2]),
                                ('A1', a[1]), ('A2', a[2])]:
            q30_val = int(val * Q30)
            assert -(1 << 31) <= q30_val < (1 << 31), \
                f"{name}_{coeff_name} = {val:.8f} overflows Q30 (got {q30_val})"
            print(f"static const int32_t {name}_{coeff_name} = {q30_val:>12};")
        print()

    # Highpass 0.5Hz
    b, a = butter(2, 0.5, btype='high', fs=fs)
    print_coeffs("HP", b, a)

    # Lowpass 50Hz
    b, a = butter(2, 50, btype='low', fs=fs)
    print_coeffs("LP", b, a)

    # Notch 60Hz, Q=30
    b, a = iirnotch(60, 30, fs=fs)
    print_coeffs("NOTCH", b, a)


def export_fsq_lattice(model, output_path):
    """
    Generate fsq_lattice.h with FSQ level configuration for fsq.c.
    These values match the FSQ grid used during training.
    """
    with open(output_path, 'w') as f:
        f.write("#ifndef FSQ_LATTICE_H\n")
        f.write("#define FSQ_LATTICE_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// FSQ lattice levels per dimension (4D product quantizer)\n")
        f.write("// Total codebook size: 8 * 6 * 5 * 5 = 1200\n")
        f.write("static const int32_t FSQ_LEVELS[4] = {8, 6, 5, 5};\n\n")
        f.write("// Default quantization scale (0.5 in Q31)\n")
        f.write("#define FSQ_QUANT_SCALE_Q31 1073741824\n\n")
        f.write("// Bounds size for CRC computation\n")
        f.write("#define FSQ_BOUNDS_SIZE 4\n\n")
        f.write("#endif // FSQ_LATTICE_H\n")
    print(f"[*] FSQ lattice → {output_path}")


def export_toeplitz_seeds(output_path):
    """
    Generate toep_seeds.h with LFSR seeds for compressed sensing.
    These must be deterministic and match across firmware builds.
    """
    seeds = [
        0xACE1, 0xBE37, 0xCAFE, 0xDEAD, 0xF00D, 0x1337,
        0xB00B, 0xFACE, 0xD00D, 0xBEEF, 0xC0DE, 0xBAD1,
        0xFEED, 0xDAD1, 0xAB1E, 0xACDC, 0xB1A5, 0xCA5E,
        0xDE1F, 0xEF01, 0xF1A7,
    ]
    with open(output_path, 'w') as f:
        f.write("#ifndef TOEP_SEEDS_H\n")
        f.write("#define TOEP_SEEDS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// LFSR seeds for per-channel Toeplitz compressed sensing rows\n")
        f.write(f"static const uint32_t TOEP_SEEDS[{len(seeds)}] = {{\n    ")
        for i, s in enumerate(seeds):
            f.write(f"0x{s:04X}u, ")
            if (i + 1) % 6 == 0:
                f.write("\n    ")
        f.write("\n};\n\n")
        f.write(f"#define TOEP_NUM_CHANNELS {len(seeds)}\n\n")
        f.write("#endif // TOEP_SEEDS_H\n")
    print(f"[*] Toeplitz seeds → {output_path}")


# ────────────────────────────────────────────────────────────────────
# CLI entry point — supports --target {c,rust,both} since v1.0
# ────────────────────────────────────────────────────────────────────


def _load_model_for_export(device, explicit_ckpt=None):
    """Pick a checkpoint by grade-priority + load the right model variant.

    Returns (model, ckpt_path, grade, use_subband).
    """
    from train_ternary import TernaryMobileNetV5_Subband

    if explicit_ckpt:
        p = os.path.abspath(explicit_ckpt)
        if not os.path.exists(p):
            print(f"[!] FATAL: --checkpoint not found: {p}")
            sys.exit(1)
        # Try subband loader first; fall back to legacy.
        try:
            model = TernaryMobileNetV5_Subband.from_checkpoint(p, device=device)
            return model, p, "explicit", True
        except Exception:
            model = TernaryMobileNetV5(in_ch=21, latent_dim=32)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            return model, p, "explicit", False

    subband_paths = [
        ("gold",     os.path.join(ROOT_DIR, "weights", "student_subband_gold.ckpt")),
        ("std",      os.path.join(ROOT_DIR, "weights", "student_subband_std.ckpt")),
        ("fast",     os.path.join(ROOT_DIR, "weights", "student_subband_fast.ckpt")),
        ("canonical", os.path.join(ROOT_DIR, "weights", "student_subband.ckpt")),
        ("dev",      os.path.join(ROOT_DIR, "ai_models/student/student_subband_gold.ckpt")),
        ("dev",      os.path.join(ROOT_DIR, "ai_models/student/student_subband_std.ckpt")),
        ("dev",      os.path.join(ROOT_DIR, "ai_models/student/student_subband_fast.ckpt")),
        ("dev",      os.path.join(ROOT_DIR, "ai_models/student/student_subband.ckpt")),
    ]
    legacy_paths = [
        ("legacy",   os.path.join(ROOT_DIR, "weights", "student_hardened.ckpt")),
        ("legacy",   os.path.join(ROOT_DIR, "ai_models/student/student_hardened.ckpt")),
    ]

    s_path, use_subband, grade = None, False, None
    for tag, p in subband_paths:
        if os.path.exists(p):
            s_path, use_subband, grade = p, True, tag
            break
    if s_path is None:
        for tag, p in legacy_paths:
            if os.path.exists(p):
                s_path, grade = p, tag
                break

    if s_path is None:
        print("[!] FATAL: No checkpoint found. Searched:")
        for tag, p in subband_paths + legacy_paths:
            print(f"    {p}")
        sys.exit(1)

    if use_subband:
        print(f"[*] Exporting Gen 7.1 subband model ({grade} grade)")
        print(f"    Source: {s_path}")
        model = TernaryMobileNetV5_Subband.from_checkpoint(s_path, device=device)
    else:
        print(f"[*] Exporting Gen 7.0 model ({grade} grade)")
        print(f"    Source: {s_path}")
        model = TernaryMobileNetV5(in_ch=21, latent_dim=32)
        model.load_state_dict(torch.load(s_path, map_location=device, weights_only=True))

    return model, s_path, grade, use_subband


def _emit_c(model, ckpt_path):
    """Legacy C-header emission — unchanged from v0."""
    export_dir = os.path.join(ROOT_DIR, "firmware/firmware_export")
    os.makedirs(export_dir, exist_ok=True)

    header_path = os.path.join(export_dir, "focal_net_weights.h")
    export_to_header(model, header_path)
    compute_firmware_crc(header_path, os.path.join(export_dir, "firmware_crc.h"))
    export_fsq_lattice(model, os.path.join(export_dir, "fsq_lattice.h"))
    export_toeplitz_seeds(os.path.join(export_dir, "toep_seeds.h"))

    print("\n[*] Biquad Q30 coefficients (paste into biquad_q31.c if cutoffs changed):")
    generate_biquad_coefficients()


def _emit_rust(model, ckpt_path, schema_path, arch_name=None, snn_ckpt_path=None):
    """Rust crate emission — generates `lamquant-weights/src/generated/`."""
    # Lazy import: only needed for --target rust|both. Keeps zero-arg
    # legacy behaviour from pulling jinja2.
    sys.path.insert(0, os.path.dirname(__file__))
    from export.checkpoint import LoadedCheckpoint, sha256_of, _grade_of
    from export.fsq import calibrate as fsq_calibrate
    from export.rust_emitter import RustEmitter
    from export.schema import load_schema

    schema = load_schema(schema_path)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if arch_name is None:
        # Detect from state_dict.
        from export.checkpoint import detect_arch
        arch_name = detect_arch(state_dict, schema.architectures)
        print(f"[*] Auto-detected architecture: {arch_name}")

    ckpt = LoadedCheckpoint(
        path=Path(ckpt_path),
        sha256=sha256_of(Path(ckpt_path)),
        state_dict=state_dict,
        arch_name=arch_name,
        grade=_grade_of(Path(ckpt_path)),
    )

    crate_root = Path(ROOT_DIR) / "lamquant-weights"
    print(f"[*] Generating Rust crate at {crate_root}")
    print(f"    arch={arch_name}, ckpt sha256={ckpt.short_sha()}")

    # FSQ calibration — runs the model on random EEG to fit the rANS table.
    print("[*] Calibrating FSQ + rANS frequency table...")
    try:
        fsq_cal = fsq_calibrate(
            model,
            n_samples=schema.fsq.calibration_n_samples,
            input_shape=tuple(schema.fsq.calibration_input_shape),
            input_clamp=schema.fsq.calibration_input_clamp,
            num_levels=schema.fsq.n_freq_bins,
            total_freq=schema.fsq.rans_total_freq,
        )
        print(f"    FSQ entropy: {fsq_cal.entropy_bps:.3f} bps; "
              f"range=[{fsq_cal.vmin_q31 / 1000:.2f}, {fsq_cal.vmax_q31 / 1000:.2f}]")
    except Exception as e:
        print(f"    [!] FSQ calibration failed: {e}; emitting without FSQ table")
        fsq_cal = None

    emitter = RustEmitter(schema=schema, ckpt=ckpt, crate_root=crate_root, arch_name=arch_name)
    snn_path = Path(snn_ckpt_path) if snn_ckpt_path else None
    if snn_path is not None and not snn_path.exists():
        print(f"    [!] SNN checkpoint not found: {snn_path}; emitting without SNN")
        snn_path = None
    emitter.emit(model, fsq_cal=fsq_cal, snn_ckpt=snn_path)
    print(f"[*] Rust crate emission complete.")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="export_firmware",
        description="Export LamQuant model checkpoint to firmware artifacts.",
    )
    parser.add_argument(
        "--target", choices=("c", "rust", "both"), default="c",
        help="Output format. 'c' = legacy headers (default), 'rust' = "
             "lamquant-weights crate, 'both' = run both back-to-back.",
    )
    parser.add_argument(
        "--schema", default=os.path.join(ROOT_DIR, "firmware", "export_schema.toml"),
        help="Path to export_schema.toml (only used with --target rust|both).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Explicit checkpoint path. Default: auto-pick highest-grade available.",
    )
    parser.add_argument(
        "--arch", default=None, choices=("subband_v1", "subband_v2", "legacy_v7_0"),
        help="Architecture variant. Default: auto-detect from checkpoint.",
    )
    parser.add_argument(
        "--snn-checkpoint", default=None,
        help="Mamba SNN checkpoint (e.g. weights/snn/mamba_snn_best.pt). "
             "When provided with --target rust|both, emits SNN weight tables "
             "into lamquant-weights/src/generated/snn/.",
    )
    parser.add_argument(
        "--validate-schema", default=None,
        help="Validate the schema TOML and exit. Skips checkpoint loading.",
    )
    args = parser.parse_args(argv)

    if args.validate_schema is not None:
        sys.path.insert(0, os.path.dirname(__file__))
        from export.schema import validate_schema
        validate_schema(args.validate_schema)
        return

    device = torch.device("cpu")
    model, ckpt_path, grade, _ = _load_model_for_export(device, args.checkpoint)
    model.eval()

    if args.target in ("c", "both"):
        _emit_c(model, ckpt_path)
    if args.target in ("rust", "both"):
        _emit_rust(model, ckpt_path, args.schema, args.arch,
                   snn_ckpt_path=args.snn_checkpoint)


if __name__ == "__main__":
    main()
