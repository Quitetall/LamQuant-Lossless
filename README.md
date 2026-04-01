# LamQuant Gen 6: Embedded Clinical Neural Codec

LamQuant Gen 6 is a full-stack, clinical-grade EEG neural codec engineered for the **RP2350 (Hazard3 RISC-V)** microcontroller. This repository houses the bare-metal C firmware, the PyTorch distillation loops, and the validation harness required to train and deploy Ternary Quantized neural networks directly onto strict hardware limits enforcing < 4ms latencies.

## Core Architectural Guardrails

- **Zero Dynamic Allocation**: The firmware does not use `malloc()`. Memory bounds are perfectly statically assigned.
- **TCM Memory Pinning**: Critical PyTorch weight arrays and execution loops are pinned to the Cortex/Hazard3 32KB Tightly Coupled Memory (`.tcm_data`) bypassing SRAM execution cache misses completely.
- **Hazard3 Zbb Exploitation**: Ternary neural MAC operations skip FP32 multiplication natively, unrolling the network across pure integer bit-masks (`cpop` via `__builtin_popcount`) executed in single ALU cycles natively in `ternary_mac.c`.
- **Hardware PMP Enforcements**: Stack overflows map identically against native Physical Memory Protection blocks at `0x20007800` safely halting execution over normal software canaries.

## Directory Structure

- **/firmware/**
  - `core/` - Tickless RTOS scheduler, hardware PMP exception trapping, Safe-Mode fallback mapping, and IEEE 802.3 CRC32 firmware integrity checks.
  - `dsp/` - O(1) Q31 Biquad bounds, Lifting 2D mappings, and Toeplitz Compressed Sensing wrappers.
  - `neural/` - Bare-metal `ternary_mac.c` executions extracting native tensors and Finite Scalar Quantization (FSQ) targets.
  - `transport/` - Adaptive EMC Channel Coding over Bluetooth LE bounds safely guaranteeing throughput scaling.
- **/ai_models/**
  - `dataset_sim/` - Extractors natively mapping CHB-MIT PhysioNet datasets applying strict clinical `seizure_mask` validation natively to generated Q31 `.npz` records.
  - `oracle/` - The `MobileNetV5Focal` FP32 configuration mapping heavily penalized `EventWeightedMSELoss` tracking structural neurological morphologies (Spike-waves).
  - `student/` - PyTorch `TernaryQuantizeSTE` implementations tracking identical native Knowledge Distillation bridging `export_firmware.py` headers tightly wrapping `<16KB` output traces natively into your `.tcm_data` layout natively.
- **/testing/**
  - JSON-native Hardware-In-The-Loop simulation scripts explicitly stress testing Stack faults, EMC bounds, and CRC Flash corruption natively verifying Safety Constraints.

## Compilation

Compilation is mapped via CMake directly integrating `pico_sdk` libraries utilizing GCC warning rules bounded mathematically:

```bash
# Requires $PICO_SDK_PATH
mkdir build && cd build
cmake ..
make lamquant
```

_GCC limits ensure `lamquant` fails compilation aggressively if any localized memory structure violates the 2,400-byte stack boundary limit structurally._

## Training Pipeline

1. Synchronize `chbmit/1.0.0/` datasets utilizing `s5cmd --no-sign-request cp`.
2. Map `edf_to_events.py` converting `.edf` limits against `.seizure` summary files into `.npz`.
3. Train `train_teacher.py` (FP32) applying strictly Patient-Wise 20% validation mapping native morphological validation constraints safely avoiding data leakage.
4. Execute `train_ternary.py` Distillating the Teacher down to exactly 15,800 `-1, 0, 1` Ternary markers.
5. `export_firmware.py` binds the checkpoint identically extracting native `C` array maps wrapped perfectly into your embedded core bounds.
