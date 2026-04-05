# System Architecture

This document describes the complete LamQuant system: how EEG data flows from electrodes through on-chip compression to base station reconstruction.

---

## Overview

LamQuant is a three-phase pipeline:

1. **Training** (offline, Python/PyTorch) — Distill a ternary-quantized student autoencoder from a full-precision teacher.
2. **On-chip encoding** (real-time, C on RP2350) — Acquire, filter, compress, and transmit EEG every 4 ms.
3. **Base station decoding** (real-time, Python) — Decompress and reconstruct the original 21-channel EEG signal.

---

## Phase 1: Training Pipeline

### Teacher

A full-precision (FP32) Transformer autoencoder is trained on the CHB-MIT Scalp EEG Database. It learns to compress and reconstruct 21-channel EEG at 250 Hz with high fidelity. The teacher produces:

- Distillation targets (reconstructed waveforms) for the student
- Seizure masks (from CHB-MIT annotations) for event-weighted loss
- A strided teacher variant (for Route B decoding)

### Student: TernaryMobileNetV5

The student is a 96-wide ternary autoencoder with 4 encoder layers:

```
Encoder:
  focal1: TernaryConv1d(21 -> 96, k=7, s=1)  + GN(4,96) + ReLU + shortcut(21->96, k=1, s=1)
  focal2: TernaryConv1d(96 -> 96, k=5, s=2)  + GN(4,96) + ReLU + shortcut(96->96, k=1, s=2)
  focal3: TernaryConv1d(96 -> 96, k=3, s=2)  + GN(4,96) + ReLU + shortcut(96->96, k=1, s=2)
  focal4: TernaryConv1d(96 -> 96, k=3, s=2)  + GN(4,96) + ReLU + shortcut(96->96, k=1, s=2)
  bottleneck: TernaryConv1d(96 -> 32, k=1, s=1) + bias

Decoder (symmetric):
  expand1: TernaryFocalBlock(32 -> 96, k=3, s=1)
  expand2: TernaryUpsampleBlock(96 -> 96, k=3, s=2)
  expand3: TernaryUpsampleBlock(96 -> 96, k=3, s=2)
  expand4: TernaryUpsampleBlock(96 -> 96, k=5, s=2)
  output:  Conv1d(96 -> 21, k=1)
```

- **Total temporal stride**: 1 x 2 x 2 x 2 = **8x**
- **Input**: `[B, 21, 2500]` (10 seconds at 250 Hz)
- **Latent**: `[B, 32, 312]` (32 channels, 312 time steps)
- **Weight precision**: 2-bit ternary {-1, 0, +1} via LSQ quantization
- **Activation precision**: 16-bit integer (W2A16)
- **Encoder packed size**: ~42.3 KB / 43 KB SRAM4 budget (98.4% utilization)

### Training schedule

1. **Standalone training** (`train_student.py`): 500 epochs in 3 phases — FP32 warm-up (50 ep), quantization-aware (200 ep), fine-tune with spectral loss (250 ep).
2. **Hardening** (`harden_artifacts.py`): Aligns student latent space with teacher using distillation loss.
3. **Export** (`export_firmware.py`): Serializes ternary weights (4 per byte), Q31 alphas, GroupNorm params, and rANS frequency tables into C headers.

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

### Scheduler FSM (`scheduler.c`)

The scheduler is event-driven. It sleeps (WFI) until a DMA interrupt signals that a 2500-sample ADC window is ready, then processes one window:

```
State Machine:
  SLEEP ──(DMA IRQ)──> PREFILTER ──> [path select] ──> TX_READY ──> SLEEP
                                      │
                                      ├──> INFERENCE (golden) ──> FSQ_ENCODE
                                      └──> LIGHTNING_DSP
```

**Path selection** runs after the biquad prefilter. It computes a variance estimate from the first 6 channels x 32 samples (O(192), negligible latency):

| Condition | Path | Reason |
|-----------|------|--------|
| Variance > 5,000,000 (seizure) | Lightning | Seizure morphology needs guaranteed deadline |
| Elapsed > 3.5 ms | Lightning | Golden path would breach 4 ms budget |
| Otherwise | Golden | Full TNN quality |

### Golden Path

```
biquad_prefilter(2500)
  → run_tnn_encoder_inference([21][2500]) → latent[32][312]
    → FSQ quantize (16 levels) → rANS entropy code
      → 240-byte BLE packet → SPI1 DMA TX to nRF52840
```

1. **Biquad prefilter** (`biquad_q31.c`): 3-stage cascade per channel — highpass 0.5 Hz, lowpass 50 Hz, notch 60 Hz. All Q30 fixed-point. Runs in-place on `raw_adc_buffer[21][2500]`.

2. **TNN encoder** (`focal_modulation.c`): 4 focal blocks + bottleneck. Input Q31 is truncated to int16 (Q15). Each block: ternary conv -> GroupNorm(4 groups) -> ReLU -> add shortcut. Double-buffered in SRAM5.

3. **FSQ** (`fsq.c`): Uniform scalar quantization to 16 bins per latent dimension. Adaptive gain based on rolling RMS prevents dead codes and clipping.

4. **rANS** (`hybrid_entropy.c`): 32-bit asymmetric numeral systems encoder. Frequency tables are calibrated from the trained latent distribution and baked into `focal_net_weights.h`. Encodes in reverse (LIFO), flushes state as trailing 4 bytes.

5. **BLE TX** (`ble_spi_host.c`): DMA-paced SPI1 transfer to nRF52840. Non-blocking — DMA runs while scheduler re-arms ADC.

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
- Uses the student's symmetric decoder (expand1-4 + output layer)
- Always available, lightweight (runs on CPU)
- Latent `[32, 312]` -> reconstructed `[21, 2500]`

### Route B: Teacher FP32 Decoder
- Uses the teacher's FP32 decoder for higher quality
- Requires latent upsampling: T/8 (312) -> T (2500) via linear interpolation
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
| SRAM0 | 64 KB | `raw_adc_buffer[21][2500]` (ADC + biquad workspace) | DMA write, CPU read/write |
| SRAM4 | 64 KB | TNN weights (~42.3 KB packed), FSQ tables, rANS freq | CPU read-only at runtime |
| SRAM5 | 64 KB | Activation double-buffers, latent output, lifting tile | CPU read/write |
| Stack | 2.4 KB | Function call frames | Enforced by `-Wstack-usage=2400 -Werror` |

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
Biquad prefilter (HP 0.5Hz + LP 50Hz + Notch 60Hz, Q30, in-place)
    │
    ├── Golden ─────────────────────────────┐
    │   TNN Encoder (focal1→focal2→focal3   │
    │   →focal4→bottleneck)                 │
    │   act_buf_a/b[96][2500] (SRAM5)       │
    │   latent_output[32][312] (SRAM5)       │
    │       │                                │
    │       ▼                                │
    │   FSQ quantize (16 levels)             │
    │       │                                │
    │       ▼                                │
    │   rANS entropy code                    │
    │       │                                │
    ├── Lightning ──────────────────────────┐│
    │   Toeplitz CS: [21][2500] → [6][32]   ││
    │   2D Lifting wavelet                   ││
    │   LPC order-4 residuals                ││
    │   Golomb-Rice code                     ││
    │                                        ││
    ▼                                        ▼▼
bit_buffer[240] (hybrid_entropy.c)
    │
    ▼
SPI1 DMA → nRF52840 → BLE → Base Station
    │
    ▼
Student decoder (Route A) or Teacher decoder (Route B)
    │
    ▼
Reconstructed EEG [21][2500]
```
