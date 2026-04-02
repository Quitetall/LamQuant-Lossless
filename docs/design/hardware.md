# LamQuant Gen 6: Hardware Interoperability

This document details the exact hardware coupling between the LamQuant Gen 6 `C` algorithms and the physical constraints of the **Raspberry Pi RP2350 Microcontroller**.

## 1. The Hazard3 RISC-V Core
LamQuant deviates critically from older ARM M0+ architectures by actively targeting the RP2350's **Hazard3 RISC-V Dual Cores**.
By capitalizing directly on the RISC-V `Zbb` (Bitmanip) extension, we avoid floating-point math entirely during neural inference. 
Our Ternary (`-1, 0, 1`) arrays are bit-packed, reducing 32 full-precision MAC (Multiply-Accumulate) operations into a single unrolled `__builtin_popcount` execution natively.

## 2. TCM (Tightly Coupled Memory) Execution
Latency is universally fatal for biological telemetry streams. If the FocalNet inference retrieves weights traversing through the RP2350's external QSPI flash, cache-misses will violently blow past the `4.0 ms` computation deadline.

*   **TCM Limits:** The RP2350 inherently grants us only `32 KB` natively of guaranteed 0-wait-state Tightly Coupled Memory.
*   **The Gen 6 Strategy:** The PyTorch pipeline violently distills the focal networks down into exactly a subset of matrices weighing identically `~28 KB`. 
*   **Memory Pinning:** We explicitly lock this C header into the TCM bounds utilizing `#pragma GCC section` and `__attribute__((section(".tcm_data")))`.

## 3. PMP Sandboxing & Failsafes
As detailed natively in `COMPLIANCE.md`, the Firmware stack explicitly protects against logic errors crashing the silicon limits.

*   **Stack Bounds Check:** `stack_guard.c` writes canary values natively into the limits of the software stack boundary continuously verified natively.
*   **Physical Traps:** We manually manipulate the RP2350 `PMP` (Physical Memory Protection) trap hooks. If a buffer overflow physically attempts to push pointers outside of the 32KB limit footprint natively, the processor violently traps the exception dropping into a safe `.brownout` reset physically bypassing arbitrary execution bugs.
