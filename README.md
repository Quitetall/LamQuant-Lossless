# LamQuant Gen 6

An on-chip EEG neural codec that compresses 21-channel electroencephalography data in real time using ternary-quantized neural networks. Runs on the RP2350 microcontroller (dual Hazard3 RISC-V cores, 150 MHz) under a hard 4 ms latency ceiling with 64 KB SRAM.

> **Research and educational use only.** Not a cleared medical device. See [SAFETY.md](SAFETY.md).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What This Project Does](#what-this-project-does)
3. [System Architecture](#system-architecture)
4. [Directory Layout](#directory-layout)
5. [Installation](#installation)
6. [Training Pipeline](#training-pipeline)
7. [Firmware Build](#firmware-build)
8. [Desktop GUI](#desktop-gui)
9. [Testing](#testing)
10. [Documentation Index](#documentation-index)
11. [Licensing](#licensing)

---

## Quick Start

```bash
# Run the Python test suite (no GPU, no data required)
pip install -e ".[test]"
pytest tests/ -v

# Host-compile and run the C firmware tests
gcc -I firmware tests/c_host/test_c_firmware.c -o test_fw -lm && ./test_fw
```

---

## What This Project Does

LamQuant compresses continuous EEG signals on a microcontroller in real time, transmits the compressed stream over BLE, and reconstructs the signals on a base station. The compression pipeline has three stages:

1. **Training (Python/PyTorch)** — A full-precision teacher autoencoder is trained on clinical EEG data (CHB-MIT). A ternary-quantized student (weights in {-1, 0, +1}) is distilled from the teacher using Learned Step Size (LSQ) quantization. The student encoder fits in 42.3 KB.

2. **On-chip encoding (C, bare metal)** — The RP2350 firmware acquires 21-channel EEG at 250 Hz, filters it through a 3-stage biquad cascade, and encodes it via one of two paths:
   - **Golden path**: Ternary neural network encoder -> Finite Scalar Quantization (FSQ) -> rANS entropy coding. Full quality, ~25-35x compression.
   - **Lightning path**: Toeplitz compressed sensing -> 2D lifting wavelet -> LPC prediction -> Golomb-Rice coding. Guaranteed deadline, used during seizures or budget overruns.

3. **Base station decoding (Python)** — The compressed stream is decoded using either the student's ternary decoder (lightweight) or the teacher's FP32 decoder (higher quality).

### Key numbers

| Metric | Value |
|--------|-------|
| Input | 21 channels x 2500 samples (10 s at 250 Hz) |
| Latent | 32 dims x 312 time steps (8x temporal stride) |
| Encoder size | 42.3 KB (98.4% of 43 KB SRAM4 budget) |
| Compression | 25-35x (golden path, rANS) |
| Latency | < 4 ms per window (ADC complete -> BLE TX start) |
| Fidelity | Pearson R > 0.85 on held-out patients |
| Stack usage | < 2.4 KB (enforced at compile time) |
| BLE packet | max 240 bytes per window |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RP2350 (On-Chip)                      │
│                                                         │
│  ADC ──> Biquad ──┬──> TNN Encoder ──> FSQ ──> rANS ──>│──> BLE ──> Base Station
│  (21ch    (HP+LP   │     (Golden Path)                   │
│  250Hz)   +Notch)  │                                     │
│                    └──> Toeplitz CS ──> Lifting ──>      │
│                          LPC ──> Golomb-Rice ──>────────>│──> BLE ──> Base Station
│                          (Lightning Path)                │
└─────────────────────────────────────────────────────────┘
```

The scheduler selects the golden path for normal EEG and switches to the lightning path when seizure activity is detected (variance threshold) or when the golden path would breach its 3.5 ms budget.

For a complete architecture description, see [docs/design/architecture.md](docs/design/architecture.md).

---

## Directory Layout

```
lamquant_gen6/
├── ai_models/                  Python training pipeline
│   ├── dataset_sim/            EDF -> Q31 dataset conversion
│   │   ├── edf_to_events.py   CHB-MIT parser, channel aliasing, Q31 scaling
│   │   └── audit_dataset.py   Dataset integrity checker
│   ├── oracle/                 FP32 teacher models
│   │   ├── train_teacher.py   Standard teacher autoencoder
│   │   └── train_teacher_strided.py  Strided teacher for Route B
│   └── student/                Ternary student distillation
│       ├── train_ternary.py   TernaryMobileNetV5 architecture + LSQ
│       ├── train_student.py   Standalone student training (3-phase)
│       ├── harden_artifacts.py Latent alignment with teacher
│       ├── finetune_student.py Optional fine-tuning
│       └── train_route_b_decoder.py  Route B (teacher decoder) training
├── firmware/                   Bare-metal C for RP2350
│   ├── core/                   Boot, scheduler, safety
│   │   ├── main.c             Entry point, boot sequence, ADC init
│   │   ├── scheduler.c        Pipeline FSM (golden/lightning selection)
│   │   ├── power_states.c     Safe mode, dormant state
│   │   ├── stack_guard.c      PMP hardware stack protection
│   │   ├── integrity.c        CRC32 firmware verification
│   │   └── math_utils.h       Q31/Q30 fixed-point primitives
│   ├── dsp/                    Signal processing
│   │   ├── biquad_q31.c       3-stage IIR filter cascade (Q30)
│   │   ├── toeplitz_cs.c      LFSR-based compressed sensing
│   │   ├── lifting_2d.c       Le Gall 5/3 wavelet transform
│   │   └── lpc_predictor.c    Order-4 LPC (Levinson-Durbin)
│   ├── neural/                 Neural network inference
│   │   ├── focal_modulation.c 4-layer TNN encoder (stride 8x)
│   │   ├── ternary_mac.c      Branchless ternary MAC + KAT
│   │   ├── ternary_manifold.c Grouped conv helper (Q31 scaling)
│   │   └── fsq.c              Finite scalar quantization
│   ├── transport/              Output encoding + BLE
│   │   ├── hybrid_entropy.c   rANS + Golomb-Rice encoder
│   │   └── ble_spi_host.c     SPI1 DMA to nRF52840, EMC adaptation
│   ├── afe/                    Analog front-end drivers
│   │   ├── ads1299_driver.c   ADS1299 SPI0 driver (production)
│   │   └── ads1299_driver.h   Public API
│   ├── pio/                    PIO programs
│   │   └── lc_adc_trigger.pio LC comparator trigger (development)
│   ├── firmware_export/        Generated headers (from export_firmware.py)
│   │   ├── focal_net_weights.h Packed ternary weights + Q31 alphas
│   │   ├── fsq_lattice.h      FSQ level configuration
│   │   ├── toep_seeds.h       LFSR seeds for compressed sensing
│   │   └── firmware_crc.h     Expected CRC32 checksum
│   └── export_firmware.py      PyTorch -> C header converter
├── gui/                        Tauri v2 desktop application
│   ├── src-tauri/              Rust backend (serial, BLE, export)
│   └── src/                    SvelteKit frontend
├── lamquant_codec/             Pip-installable Python codec package
│   ├── codec.py               TernaryCodec wrapper class
│   ├── decode.py              CLI decoder entry point
│   └── export.py              CLI export entry point
├── tests/
│   ├── test_*.py              Python unit tests (pytest)
│   ├── c_host/                Host-compiled C firmware tests
│   └── benchmarks/            Integration benchmarks
├── tools/
│   └── visualize_pipeline.py  4-panel pipeline visualization
├── docs/                       Technical documentation
├── CMakeLists.txt             Firmware build (Pico SDK)
├── pyproject.toml             Python package configuration
├── install.sh                 Linux/macOS build script
└── install.ps1                Windows build script
```

For a file-by-file description, see [docs/directory_structure.md](docs/directory_structure.md).

---

## Installation

### Python (training + codec)

Requires Python 3.10+.

```bash
git clone https://github.com/Quitetall/lamquant_gen6.git
cd lamquant_gen6

pip install -e .            # Core: torch, numpy, scipy, mne, tqdm
pip install -e ".[test]"    # + pytest, coverage
pip install -e ".[dev]"     # + wandb, mlflow, matplotlib
```

### Firmware

Requires the [Pico SDK v2.x](https://github.com/raspberrypi/pico-sdk) and a RISC-V cross-compiler. See [docs/BUILDING.md](docs/BUILDING.md).

### Desktop GUI

Requires Node.js 18+ and Rust 1.77+. See [docs/gui_guide.md](docs/gui_guide.md).

```bash
bash install.sh   # Linux/macOS: installs Python package, builds GUI
```

---

## Training Pipeline

Run these steps in order. Each script accepts `--help`.

### 1. Prepare dataset

```bash
# Download CHB-MIT from PhysioNet
s5cmd --no-sign-request cp "s3://physionet-open/chb-mit-scalp-eeg-database/1.0.0/*" ./chbmit/

# Convert EDF files to Q31 tensors
python ai_models/dataset_sim/edf_to_events.py --input ./chbmit --output ./q31_events
```

### 2. Train teacher (FP32 reference)

```bash
python ai_models/oracle/train_teacher.py            # ~1-2 hours on RTX 4090
python ai_models/oracle/train_teacher_strided.py     # Strided variant for Route B decoder
```

### 3. Train student (ternary quantized)

```bash
python ai_models/student/train_student.py            # ~8 min on RTX 4090
```

Three-phase schedule:
- Phase 1 (50 epochs): FP32 warm-up, no quantization
- Phase 2 (200 epochs): LSQ quantization enabled (STE gradients)
- Phase 3 (250 epochs): Fine-tune with spectral loss

### 4. Harden for deployment

```bash
python ai_models/student/harden_artifacts.py         # Aligns student latents with teacher
```

### 5. Export to firmware headers

```bash
python firmware/export_firmware.py
```

Generates four files in `firmware/firmware_export/`:
- `focal_net_weights.h` — Packed ternary weights (4 per byte), Q31 alphas, GroupNorm params, rANS frequency tables
- `fsq_lattice.h` — FSQ level configuration
- `toep_seeds.h` — LFSR seeds for compressed sensing
- `firmware_crc.h` — CRC32 checksum covering all exported data

For a detailed walkthrough, see [docs/training_pipeline.md](docs/training_pipeline.md).

---

## Firmware Build

```bash
export PICO_SDK_PATH=/path/to/pico-sdk
mkdir build && cd build
cmake .. -DADC_BACKEND_ADS1299=ON   # Production: ADS1299 AFE
# cmake ..                           # Development: LC-ADC comparator
make lamquant
```

Output: `lamquant.uf2` — flash to RP2350 via USB boot mode (hold BOOTSEL, plug in, copy file).

See [docs/BUILDING.md](docs/BUILDING.md) for full instructions.

---

## Desktop GUI

The Tauri v2 desktop application provides impedance checking, live signal visualization, recording, and EDF/CSV export.

```bash
cd gui
npm install
npx tauri dev    # Development mode (mock data, no hardware needed)
```

Three main screens:
1. **Impedance Check** — SVG head diagram with 10-20 electrode positions, color-coded by impedance (green < 5 kOhm, amber 5-20, red > 20)
2. **Live Signals** — 8-channel scrolling Canvas2D waveforms at 250 Hz, configurable gain/filter/time window
3. **Export** — Record sessions and export as EDF or CSV

See [docs/gui_guide.md](docs/gui_guide.md) for architecture details.

---

## Testing

### Python tests

```bash
pytest tests/ -v                    # 51 unit tests
pytest tests/ -v --cov=ai_models    # With coverage report
```

Covers: weight packing round-trip, LSQ quantization shape/gradient, channel mapping aliases, export firmware integrity, Q31 arithmetic, utility functions.

### C host tests

```bash
gcc -I firmware tests/c_host/test_c_firmware.c -o test_fw -lm && ./test_fw
```

48 assertions covering: ternary MAC KAT, biquad Q30 filter, LFSR period/batch, CRC32 cross-platform parity, LPC predictor residuals, lifting wavelet, Golomb-Rice round-trip.

### Integration benchmarks

```bash
python tests/benchmarks/test_golden_path_e2e.py    # Requires trained model + dataset
python tests/benchmarks/benchmark_decoder_e2e.py
```

Pass criteria: R >= 0.85, PRD <= 40%, CR >= 5.0x.

See [docs/design/validation.md](docs/design/validation.md) for the full test strategy.

---

## Documentation Index

| Document | Contents |
|----------|----------|
| [docs/design/architecture.md](docs/design/architecture.md) | Full system architecture, data flow, dual-path scheduler |
| [docs/design/hardware.md](docs/design/hardware.md) | RP2350 memory map, PMP stack guard, SPI bus allocation, pin map |
| [docs/design/mathematics.md](docs/design/mathematics.md) | LSQ quantization, Q31/Q30 arithmetic, FSQ+rANS, distillation loss |
| [docs/design/validation.md](docs/design/validation.md) | Test strategy, stress profiles, graduation thresholds |
| [docs/BUILDING.md](docs/BUILDING.md) | Firmware and GUI build instructions |
| [docs/directory_structure.md](docs/directory_structure.md) | File-by-file codebase reference |
| [docs/firmware_reference.md](docs/firmware_reference.md) | Every C function: signature, behavior, callers |
| [docs/training_pipeline.md](docs/training_pipeline.md) | Python training walkthrough, model architecture |
| [docs/gui_guide.md](docs/gui_guide.md) | Tauri GUI architecture, screens, wire protocol |
| [docs/wire_protocol.md](docs/wire_protocol.md) | Device-host packet format and command reference |
| [docs/hardware_bom.md](docs/hardware_bom.md) | Component list, PCB specs, power budget |
| [SAFETY.md](SAFETY.md) | Regulatory disclaimer, F1 fragility defense |
| [COMPLIANCE.md](COMPLIANCE.md) | IEC 60601-1, ISO 13485, IEC 62304 traceability |

---

## Licensing

- **Code** (firmware + Python): [AGPL-3.0](LICENSE.md)
- **Trained weights** (`focal_net_weights.h`): [CC BY-NC 4.0](LICENSE.md)

## Compliance

See [COMPLIANCE.md](COMPLIANCE.md) for IEC 60601-1, ISO 13485, and IEC 62304 traceability.
