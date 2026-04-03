# LamQuant Gen 6: Hardware Interoperability

This document defines the interface between the LamQuant algorithms and the **Raspberry Pi RP2350** (Cortex-M33 / Hazard3 RISC-V) hardware.

## 1. Hazard3 RISC-V Optimization
LamQuant targets the RP2350's dual **Hazard3 RISC-V cores**. By utilizing the RISC-V `Zbb` (Bit-manipulation) instruction set, the firmware executes branchless ternary MAC operations. This approach avoids the clock-cycle overhead of floating-point emulations and enables the high-speed inference engine to meet real-time constraints.

## 2. Memory Architecture & Latency
To meet the 4.0ms real-time deadline and avoid XIP (eXecute In Place) Flash stalls, all neural weights are pinned to high-speed internal RAM.
- **SRAM4 Bank Isolation**: The 64KB SRAM4 bank is dedicated exclusively to the TNN (Ternary Neural Network) manifold.
- **Memory Pinning**: Weights and Q31 alphas are explicitly placed in the `.sram4_tnn` section using `__attribute__((section(".sram4_tnn")))`, ensuring 0-waitstate access.
- **Alignment**: All data arrays are 32-bit word-aligned (`__attribute__((aligned(4)))`) to prevent alignment faults and optimize cache pipeline efficiency.

## 3. PMP Sandboxing & Stack Security
The firmware implements Physical Memory Protection (PMP) to ensure system stability.
- **NAPOT Stack Guard**: `stack_guard.c` uses **NAPOT (Naturally Aligned Power-Of-Two)** mode to lock a specific 2KB stack canary region. This prevents stack overflows from corrupting adjacent memory.
- **Hardware Traps**: Any illegal memory access or bit-corruption triggers a hardware-level Exception, forcing the system into an isolated `Safe Mode` to prevent erroneous clinical telemetry.
