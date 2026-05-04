# System Architecture

This document describes the complete LamQuant Gen 7.6.1 system: how EEG data flows from electrodes through on-chip compression to base station reconstruction.

---

## Overview

LamQuant is a three-phase pipeline:

1. **Training** (offline, Python/PyTorch) — Distill a ternary-quantized student autoencoder from a full-precision teacher, with subband preprocessing.
2. **On-chip encoding** (real-time, C on RP2350) — Acquire, HP filter, LPC analyze, lifting DWT decompose, compress, and transmit EEG every 4 ms.
3. **Base station decoding** (real-time, Python) — Decompress and reconstruct the original 21-channel EEG signal.

---

## Phase 1: Training Pipeline

### Teacher

A full-precision (FP32) Transformer autoencoder is trained on the CHB-MIT Scalp EEG Database. It learns to compress and reconstruct 21-channel EEG at 250 Hz with high fidelity. The teacher produces:

- Distillation targets (reconstructed waveforms) for the student
- Seizure masks (from CHB-MIT annotations) for event-weighted loss
- A strided teacher variant (for Route B decoding)

### Student: TernaryMobileNetV5_Subband (Gen 7.6.1)

The student is a 128-wide ternary autoencoder with 3 focal blocks plus a ReGLU bottleneck. The TNN operates on the L3 approximation subband output by the lifting DWT, not the raw 2500-sample input:

```
Encoder:
  focal1: TernaryConv1d(21 -> 112, k=7, s=1)  + GN(4,112) + ReLU + shortcut(21->112, k=1, s=1)
  focal2: TernaryConv1d(112 -> 112, k=5, s=2) + GN(4,112) + ReLU + shortcut(112->112, k=1, s=2)
  focal3: TernaryConv1d(112 -> 112, k=3, s=2) + GN(4,112) + ReLU + shortcut(112->112, k=1, s=2)
  glu_bottleneck: TernaryConv1d(112 -> 64, k=1, s=1) * sigmoid(TernaryConv1d(112 -> 64, k=1, s=1))
  project: TernaryConv1d(64 -> 32, k=1, s=1) + bias

Decoder (symmetric):
  expand1: TernaryFocalBlock(32 -> 112, k=3, s=1)
  expand2: TernaryUpsampleBlock(112 -> 112, k=3, s=2)
  expand3: TernaryUpsampleBlock(112 -> 112, k=5, s=2)
  output:  Conv1d(112 -> 21, k=1)
```

- **Total temporal stride**: 1 x 2 x 2 = **4x**
- **Input**: `[B, 21, 313]` (L3 approximation subband from 3-level lifting DWT)
- **Latent**: `[B, 32, 79]` (32 channels, 79 time steps)
- **Weight precision**: 2-bit ternary {-1, 0, +1} via LSQ quantization
- **Activation precision**: 16-bit integer (W2A16)
- **Encoder packed size**: ~75 KB (V1 w=128, spans SRAM4+SRAM5)
- **GLU bottleneck**: Gated Linear Unit replaces the old linear bottleneck for better information gating

> **Legacy (Gen 7.0)**: 96-wide, 4 focal blocks, stride 8, input [21,2500], latent [32,312], encoder ~42.3 KB.

### Training schedule

1. **Subband preprocessing** (`subband_preprocess.py`): Applies HP biquad, LPC order-8 analysis, and 3-level Le Gall 5/3 lifting DWT to training data offline. Produces L3 approximation subband [21,313] inputs and detail coefficient arrays for the student.
2. **Joint training** (`train_joint.py`): Warm phase (FP32) + QAT phase (ternary) + EMA checkpoint selection. Trains encoder + decoder jointly with multi-scale loss (MSE + Pearson-R + PRD + spectral).
3. **Validation** (`validate_subband.py`): Validates subband pipeline fidelity (round-trip reconstruction, per-quality-mode metrics).
4. **Hardening** (`harden_artifacts.py`): Aligns student latent space with teacher using distillation loss.
5. **Export** (`export_firmware.py`): Serializes ternary weights (4 per byte), Q31 alphas, GroupNorm params, rANS frequency tables, and detail encoding config into C headers.

---

## Phase 2: On-Chip Encoding (RP2350 Firmware)

### Boot Sequence (`main.c`)

```
Phase 1: Hardware init
  └── stack_setup_hardware_trap()  — PMP stack guard at 0x20007800
  └── watchdog_enable(500ms)

Phase 2: Self-test
  └── boot_ternary_parity_kat()    — Known-Answer Test for ternary LUT
  └── verify_firmware_crc()        — CRC32 over weights + seeds + FSQ config
  └── On failure: enter_safe_mode() — BLE flush, WFI halt

Phase 3: Peripheral init
  └── ble_spi_init()               — SPI1 at 8 MHz to nRF52840
  └── ADC backend:
      ├── ADS1299: ads1299_init() + ads1299_start_continuous()
      └── LC-ADC:  PIO program + DMA init + dma_adc_rearm()

Phase 4: Main loop
  └── while(1) { lamquant_scheduler_run(); watchdog_update(); }
```

### Scheduler FSM (`scheduler_v7_1.c`)

The scheduler is event-driven with a multi-core pipeline. It sleeps (WFI) until a DMA interrupt signals that a 2500-sample ADC window is ready, then processes one window across both RP2350 cores:

```
State Machine:
  SLEEP ──(DMA IRQ)──> HP_FILTER ──> LPC_ANALYZE ──> LIFTING_DWT
                                                        │
                                                        ├──> SNN on L3 (Core 0) ──> dispatch Core 1
                                                        │                           │
                                                        │                    TNN + WHT + FSQ + rANS
                                                        │                           │
                                                        ├──> LIGHTNING_DSP          │
                                                        │                           │
                                                        └──> DETAIL_ENCODE ─────────┘
                                                                                    │
                                                                              BLE_TX ──> SLEEP
```

**Core 0 stages**: HP biquad -> LPC order-8 analysis -> 3-level lifting DWT -> SNN on L3 approximation -> path select -> dispatch Core 1 -> detail encoding -> BLE TX.

**Core 1 stages** (golden path): TNN encoder on [21,313] -> Walsh-Hadamard pre-rotation -> FSQ quantize -> rANS entropy code.

**Path selection** runs after the SNN classifies the L3 subband. It computes a variance estimate from the first 6 channels x 32 samples (O(192), negligible latency):

| Condition | Path | Reason |
|-----------|------|--------|
| Variance > 5,000,000 (seizure) | Lightning | Seizure morphology needs guaranteed deadline |
| Elapsed > 3.5 ms | Lightning | Golden path would breach 4 ms budget |
| Otherwise | Golden | Full TNN quality |

> **Legacy (Gen 7.0)**: Single-core scheduler in `scheduler.c`. HP+LP+notch biquad directly followed by path select. No LPC, lifting, or WHT stages.

### Golden Path

```
HP biquad(2500)
  → LPC order-8 analysis (256-sample autocorrelation, per-channel)
    → 3-level Le Gall 5/3 lifting DWT: [21][2500] → L3 approx[21][313] + detail subbands
      → SNN classify L3 (Core 0)
        → dispatch Core 1:
          → run_tnn_encoder_inference([21][313]) → latent[32][79]
            → WHT pre-rotation on latent
              → FSQ quantize (L=2/3/5/32, quality-dependent) → rANS entropy code
        → detail_encode (Core 0, quality-dependent)
          → LAMQ v2 packet → SPI1 DMA TX to nRF52840
```

1. **HP biquad prefilter** (`biquad_q31.c`): Single-stage highpass 0.5 Hz biquad per channel. Q30 fixed-point. Runs in-place on `raw_adc_buffer[21][2500]`. The former lowpass and notch stages are replaced by the lifting DWT's inherent subband decomposition plus detail thresholding.

2. **LPC analysis** (`lpc_delta.c`): Order-8 LPC per channel using 256-sample autocorrelation windows. Levinson-Durbin recursion produces prediction coefficients. LPC coefficients are delta-encoded: keyframe (672B) -> Q15 deltas (336B) -> Q8 deltas (168B).

3. **Lifting DWT** (`detail_threshold.c`): 3-level Le Gall 5/3 integer lifting transform decomposes [21,2500] into L3 approximation [21,313] plus three detail subbands (L1: [21,1250], L2: [21,625], L3 detail: [21,313]). In-place, zero allocation.

4. **SNN** (`snn.c`): Spiking neural network classifier on L3 approximation. Input [21,313] stride 1. Topology 21->64->8. Weights ~8 KB. Classifies signal content for quality mode and path selection.

5. **TNN encoder** (`focal_modulation.c`): 3 focal blocks + GLU bottleneck + projection. Input [21][313] (L3 subband), Q31 truncated to int16 (Q15). Each block: ternary conv -> GroupNorm(4 groups) -> ReLU -> add shortcut. Double-buffered in SRAM5. Output: `latent_output[32][79]`.

6. **Walsh-Hadamard Transform** (`wht32.c`): 32-point WHT pre-rotation applied to the 32-channel latent dimension at each time step. Decorrelates latent channels for improved FSQ codebook utilization. In-place butterfly decomposition.

7. **FSQ** (`fsq.c`): Uniform scalar quantization on latent [32][79]. Quality-dependent levels: L=2/3/5 (ALERTING, ~150:1), L=2/3/5/16 (MONITORING, ~80:1), L=2/3/5/32 (CLINICAL, ~40:1). MAX_SYMBOLS=32. Adaptive gain based on rolling RMS prevents dead codes and clipping.

8. **Detail encoding** (`detail_threshold.c`): SNN-driven thresholding of detail coefficients. Sparse non-zero coefficients are encoded using Golomb-Rice with run-length coding. ALERTING sends L3 only; MONITORING sends L3+L2; CLINICAL sends all detail subbands.

9. **rANS** (`hybrid_entropy.c`): 32-bit asymmetric numeral systems encoder. Frequency tables are calibrated from the trained latent distribution and baked into `focal_net_weights.h`. Encodes in reverse (LIFO), flushes state as trailing 4 bytes.

10. **BLE TX** (`ble_spi_host.c`): DMA-paced SPI1 transfer to nRF52840. Non-blocking — DMA runs while scheduler re-arms ADC.

> **Legacy (Gen 7.0)**: 3-stage biquad (HP+LP+notch) -> TNN on [21][2500] -> FSQ (16 levels) -> rANS. No LPC, lifting, WHT, detail encoding, or quality modes.

### Lightning Path

```
biquad_prefilter(2500)
  → Toeplitz CS: [21][2500] → [6][32]
    → 2D lifting wavelet (Le Gall 5/3)
      → LPC order-4 prediction residuals
        → Golomb-Rice entropy code
          → 240-byte BLE packet → SPI1 DMA TX
```

1. **Toeplitz compressed sensing** (`toeplitz_cs.c`): Projects each of 6 channels from 2500 samples down to 32 measurements using LFSR-generated binary Toeplitz matrices. Branchless ±1 accumulation: `mask = bit - 1; acc += (sample ^ mask) - mask`. Batch LFSR generates 32 bits at once.

2. **2D lifting wavelet** (`lifting_2d.c`): Le Gall 5/3 integer lifting transform — temporal pass (32 samples per channel) then spatial pass (6 channels cross-linked). In-place, zero allocation.

3. **LPC predictor** (`lpc_predictor.c`): Order-4 Linear Predictive Coding per channel. Autocorrelation -> Levinson-Durbin recursion -> forward prediction residuals. All Q31 arithmetic. Processes backwards to avoid overwriting needed samples.

4. **Golomb-Rice** (`hybrid_entropy.c`): Zigzag encoding of signed residuals, then unary quotient + binary remainder with k=4 Rice parameter.

### Safety Systems

- **PMP stack guard** (`stack_guard.c`): NAPOT-mode PMP entry at 0x20007800 poisons a 2 KB region. Lock bit set — immutable after boot. Stack overflow triggers hardware exception -> `enter_safe_mode()`.

- **Safe mode** (`power_states.c`): BLE emergency flush -> scheduler abort -> DSP pipeline reset -> infinite WFI (or soft recovery if watchdog reset disabled).

- **CRC32 integrity** (`integrity.c`): Boot-time check covers Toeplitz seeds, neural network weights, and FSQ lattice config. IEEE 802.3 polynomial, compatible with Python's `zlib.crc32()`.

- **Watchdog**: 500 ms timeout, pet in main loop. Failure -> hardware reset -> warm boot (skip KAT, SRAM4 weights survive soft reset).

- **EMC adaptation** (`ble_spi_host.c`): PSRR monitoring triggers adaptive channel coding (rate 1/1 -> 2/3 -> 1/2) and FSQ level reduction under electromagnetic interference.

---

## Phase 3: Base Station Decoding

Two reconstruction routes:

### Route A: Student Ternary Decoder
- Uses the student's symmetric decoder (expand1-3 + output layer)
- Always available, lightweight (runs on CPU)
- Latent `[32, 79]` -> reconstructed L3 approx `[21, 313]`
- Inverse lifting DWT synthesizes full `[21, 2500]` from L3 approx + received detail subbands

### Route B: Teacher FP32 Decoder
- Uses the teacher's FP32 decoder for higher quality
- Requires latent upsampling: T/4 (79) -> T (313) via linear interpolation, then inverse lifting
- Higher compute cost, optional

Both routes are implemented in `lamquant_codec/codec.py` and accessible via the `lamquant-decode` CLI.

---

## ADC Backend Selection

Two ADC paths are available, selected at compile time:

| Backend | Flag | Hardware | Use Case |
|---------|------|----------|----------|
| ADS1299 | `-DADC_BACKEND_ADS1299=ON` | TI ADS1299 8ch 24-bit delta-sigma via SPI0 | Production |
| LC-ADC | `-DADC_BACKEND_LC_ADC=ON` (default) | Level-crossing comparator via PIO + DMA | Development/hobby |

The ADS1299 provides 8 physical channels mapped to positions 0-7 of `raw_adc_buffer[21][2500]`. Channels 8-20 are zero-filled. The LC-ADC uses a PIO state machine to monitor a comparator output and trigger DMA transfers.

Both backends fill the same buffer and fire `on_adc_dma_complete()` when a 2500-sample window is ready.

---

## Memory Layout

| Region | Size | Contents | Access |
|--------|------|----------|--------|
| SRAM0 | 64 KB | `raw_adc_buffer[21][2500]` (ADC + HP biquad workspace) | DMA write, CPU read/write |
| SRAM4 | 64 KB | TNN weights (~75 KB packed, SRAM4+SRAM5), SNN weights (~28 KB INT4), FSQ tables, rANS freq, WHT twiddles | CPU read-only at runtime |
| SRAM5 | 64 KB | Activation double-buffers (~140 KB peak via double-buffering), latent output [32][79], lifting subbands, LPC state, detail encoding workspace | CPU read/write |
| Stack | 2.4 KB | Function call frames | Enforced by `-Wstack-usage=2400 -Werror` |

**Peak SRAM utilization**: 58% (down from 71.5% in Gen 7.0, thanks to the reduced activation buffer from `act_buf` shrinking 480 KB -> 140 KB with the smaller [21,313] TNN input).

Weight placement: `__attribute__((section(".sram4_tnn"), aligned(4)))`.
Workspace placement: `__attribute__((section(".workspace_sram5")))`.

---

## Data Flow Summary

```
Electrodes (21 ch, 250 Hz)
    │
    ▼
ADC (ADS1299 SPI0 or LC-ADC PIO) ──> raw_adc_buffer[21][2500] (SRAM0)
    │
    ▼
HP biquad prefilter (HP 0.5Hz only, Q30, in-place)
    │
    ▼
LPC order-8 analysis (256-sample autocorrelation, per-channel)
    │
    ▼
3-level Le Gall 5/3 lifting DWT
    [21][2500] → L3 approx[21][313] + detail subbands (L1/L2/L3d)
    │
    ├── SNN on L3 (Core 0, topology 21→64→8, ~8KB)
    │   classifies signal → quality mode select
    │
    ├── Golden (Core 1) ────────────────────────────┐
    │   TNN Encoder (focal1→focal2→focal3           │
    │   →GLU bottleneck→project)                    │
    │   act_buf_a/b[112][313] (SRAM5)               │
    │   latent_output[32][79] (SRAM5)               │
    │       │                                       │
    │       ▼                                       │
    │   WHT-32 pre-rotation on latent               │
    │       │                                       │
    │       ▼                                       │
    │   FSQ quantize (L=2/3/5/32, quality-dep.)     │
    │       │                                       │
    │       ▼                                       │
    │   rANS entropy code                           │
    │       │                                       │
    ├── Lightning ──────────────────────────┐       │
    │   Toeplitz CS: [21][2500] → [6][32]   │       │
    │   2D Lifting wavelet                  │       │
    │   LPC order-4 residuals               │       │
    │   Golomb-Rice code                    │       │
    │                                       │       │
    ├── Detail encoding (Core 0) ──────────┐│       │
    │   SNN-driven thresholding             ││      │
    │   Golomb-Rice + run-length coding     ││      │
    │   (quality mode determines subbands)  ││      │
    │                                       ││      │
    ▼                                       ▼▼      ▼
LAMQ v2 packet (sync 'LMQ2', 30-byte header)
    │
    ▼
SPI1 DMA → nRF52840 → BLE → Base Station
    │
    ▼
Student decoder (Route A) or Teacher decoder (Route B)
  + inverse lifting DWT (recombine L3 approx + detail subbands)
    │
    ▼
Reconstructed EEG [21][2500]
```
