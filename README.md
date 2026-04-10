<p align="center">
  <img src="assets/banner.svg" alt="LamQuant — Clinical-Grade EEG Neural Codec" width="100%">
</p>

<p align="center">
  <a href="https://github.com/quitetall/lamquant/actions/workflows/ci.yml"><img src="https://github.com/quitetall/lamquant/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/version-7.0.0-blue" alt="Version 7.0.0">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="License AGPL-3.0">
  <img src="https://img.shields.io/badge/platform-RP2350-red" alt="RP2350">
</p>

# LamQuant

An on-chip EEG neural codec that compresses 21-channel electroencephalography data in real time using ternary-quantized neural networks with subband decomposition. Runs on the RP2350 microcontroller (dual Hazard3 RISC-V cores, 150 MHz) under a hard 4 ms latency ceiling with 64 KB SRAM.

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

### One-Click Setup (no terminal required)

| Platform | What to do |
|----------|-----------|
| **Windows** | Double-click `run.bat`. On first run it installs everything automatically. |
| **macOS** | Double-click `LamQuant.command`. Right-click → Open if macOS blocks it. |
| **Linux** | Run `./install.sh` once, then find **OpenHuman Vision** in your app menu. |

The app opens in **mock mode** by default — no hardware needed to explore every feature.

### For developers

```bash
# Full install (Python + Node + Rust + GUI build)
./install.sh              # Linux/macOS
.\install.ps1             # Windows (PowerShell)

# Launch
./run.sh                  # Linux/macOS
.\run.bat                 # Windows (double-click)

# Or run tests directly
pip install -e ".[test]"
pytest tests/ -v

# Host-compile and run the C firmware tests
gcc -I firmware tests/c_host/test_c_firmware.c -o test_fw -lm && ./test_fw
```

---

## What This Project Does

LamQuant compresses continuous EEG signals on a microcontroller in real time, transmits the compressed stream over BLE, and reconstructs the signals on a base station. The compression pipeline has three stages:

1. **Training (Python/PyTorch)** — A full-precision teacher autoencoder is trained on clinical EEG data (CHB-MIT). A ternary-quantized student (weights in {-1, 0, +1}) is distilled from the teacher using Learned Step Size (LSQ) quantization. The student encoder fits in 40.3 KB.

2. **On-chip encoding (C, bare metal)** — The RP2350 firmware acquires 21-channel EEG at 250 Hz and encodes it through the Gen 7.1 subband pipeline:
   - **Golden path**: HP biquad -> LPC order-8 -> 3-level lifting DWT -> TNN on L3 approx -> WHT -> FSQ -> rANS + detail Golomb-Rice + LPC delta encoding. Three quality modes: Alerting (~150:1), Monitoring (~80:1), Clinical (~40:1).
   - **Lightning path**: Toeplitz compressed sensing -> 2D lifting wavelet -> LPC prediction -> Golomb-Rice coding. Guaranteed deadline, used during seizures or budget overruns.

3. **Base station decoding (Python)** — The compressed stream is decoded using either the student's ternary decoder (lightweight) or the teacher's FP32 decoder (higher quality). Gen 7.1 adds `SubbandCodec` for full subband reconstruction.

### Key numbers

| Metric | Value |
|--------|-------|
| Input | 21 channels x 2500 samples (10 s at 250 Hz) |
| TNN input | 21 channels x 313 samples (L3 approximation subband) |
| Latent | 32 dims x 79 time steps (4x temporal stride on L3) |
| Encoder size | 40.3 KB (93.8% of 43 KB SRAM4 budget) |
| Compression | 40-150x depending on quality mode (golden path) |
| Quality modes | Alerting (~150:1, L3 only), Monitoring (~80:1, L3+L2), Clinical (~40:1, all details) |
| Fidelity | Pearson R = 0.96-0.98 at clinical mode |
| Peak SRAM | ~450 KB (58%), activation buffers 140 KB |
| Latency | < 4 ms per window (ADC complete -> BLE TX start) |
| Stack usage | < 2.4 KB (enforced at compile time) |
| BLE packet | max 240 bytes per window |
| Recording duration | Streaming mode: limited only by disk (~38 MB/hour, 300 MB/8h). See [docs/RECORDING.md](docs/RECORDING.md). |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         RP2350 (On-Chip) — Gen 7.1 Subband              │
│                                                                          │
│  ADC ──> HP Biquad ──> LPC-8 ──> 3-level Lifting DWT ──┐               │
│  (21ch                                                   │               │
│  250Hz)                    ┌──────────────────────────────┘               │
│                            │                                             │
│              L3 approx ────┼──> SNN (Core 0, activity detect)            │
│              [21,313]      │                                             │
│                            ├──> TNN ──> WHT ──> FSQ ──> rANS ──>│       │
│                            │    (Core 1, Golden Path)            │       │
│                            │                                     ├─> BLE │
│              L1/L2 detail ─┼──> Golomb-Rice detail encoding ────>│       │
│              LPC residual ─┼──> LPC delta encoding ─────────────>│       │
│                            │                                             │
│                            └──> Toeplitz CS ──> Lifting ──>              │
│                                  LPC ──> Golomb-Rice ──>────────>│─> BLE │
│                                  (Lightning Path)                        │
└──────────────────────────────────────────────────────────────────────────┘
```

The Gen 7.1 scheduler (`scheduler_v7_1.c`) orchestrates the subband pipeline: HP biquad -> LPC analysis -> 3-level lifting -> SNN on L3 -> dispatch Core 1 (TNN+WHT+FSQ+rANS) -> Lightning path -> detail encoding. Three quality modes (Alerting, Monitoring, Clinical) control which subbands are transmitted. The scheduler switches to the lightning path when seizure activity is detected (variance threshold) or when the golden path would breach its 3.5 ms budget.

For a complete architecture description, see [docs/design/architecture.md](docs/design/architecture.md).

---

## Directory Layout

```
lamquant/
├── assets/                     Branding and icons
│   ├── banner.svg             GitHub README banner
│   ├── icon.svg               App icon (scalable source)
│   └── icons/                 Generated app icons (PNG/ICO/ICNS)
├── ai_models/                  Python training pipeline
│   ├── dataset_sim/            EDF -> Q31 dataset conversion
│   ├── oracle/                 FP32 teacher models
│   └── student/                Ternary student distillation (LSQ, subband)
├── firmware/                   Bare-metal C for RP2350
│   ├── core/                   Boot, scheduler, safety
│   ├── dsp/                    Signal processing (biquad, DWT, LPC, WHT)
│   ├── neural/                 TNN inference (focal modulation, FSQ)
│   ├── transport/              rANS + Golomb-Rice entropy coding, BLE
│   ├── afe/                    ADS1299 analog front-end driver
│   ├── pio/                    PIO programs
│   └── firmware_export/        Generated C headers (weights, CRC)
├── gui/                        Tauri v2 desktop app — OpenHuman Vision
│   ├── src-tauri/              Rust backend (serial, recording, export)
│   └── src/                    SvelteKit frontend
├── installer/                  Tauri v2 setup wizard — OpenHuman Portal
├── lamquant_codec/             Pip-installable Python codec package
├── tests/                      Python unit tests + C host tests + benchmarks
├── tools/                      Visualization utilities
├── docs/                       Technical documentation and session reports
├── weights/                    Pre-trained model checkpoints
├── CMakeLists.txt             Firmware build (Pico SDK)
├── pyproject.toml             Python package configuration
├── install.sh                 Linux/macOS build script
└── install.ps1                Windows build script
```

For a file-by-file description, see [docs/directory_structure.md](docs/directory_structure.md).

---

## Installation

### Automated (recommended) — OpenHuman Portal

The **OpenHuman Portal** install scripts detect your OS, install all system dependencies, create a Python virtual environment, build the GUI, and set up a desktop launcher. Run without arguments for an interactive component menu, or pass `--quiet` to install everything non-interactively.

```bash
# Linux / macOS
./install.sh                       # interactive menu
./install.sh --quiet               # install everything, no prompts
./install.sh --components=gui      # only the OpenHuman Vision GUI
./install.sh --components=python   # only the Python pipeline

# Windows (PowerShell)
.\install.ps1                      # interactive menu
.\install.ps1 -Quiet               # install everything
.\install.ps1 -Components gui      # only the OpenHuman Vision GUI
```

You can also install via pip and use the unified `lamquant` CLI:

```bash
pip install -e .
lamquant setup --yes               # wraps install.sh / install.ps1 --quiet
lamquant gui                       # launch OpenHuman Vision
lamquant validate                  # run installation validator
```

After install, launch from your app menu (Linux), Desktop shortcut (Windows), or by double-clicking `LamQuant.command` (macOS).

**What gets installed automatically**: Python 3.10+, Node.js 18+, Rust 1.77+, CMake, system libraries (GTK/WebKitGTK on Linux, Xcode CLI on macOS). The scripts are idempotent — safe to re-run.

### Manual (per component)

#### Python (training + codec)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # Core: torch, numpy, scipy, mne, tqdm
pip install -e ".[test]"    # + pytest, coverage
pip install -e ".[dev]"     # + wandb, mlflow, matplotlib
```

#### Firmware

Requires the [Pico SDK v2.x](https://github.com/raspberrypi/pico-sdk) and a RISC-V cross-compiler. See [docs/BUILDING.md](docs/BUILDING.md).

#### Desktop GUI

```bash
cd gui && npm install && npx tauri build
```

See [docs/gui_guide.md](docs/gui_guide.md) for development mode and architecture details.

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
# Gen 7.0 (TernaryMobileNetV5, stride 8, [21,2500] -> [32,312])
python ai_models/student/train_student.py            # ~8 min on RTX 4090

# Gen 7.1 Subband (TernaryMobileNetV5_Subband, stride 4, [21,313] -> [32,79])
python ai_models/student/train_student_subband.py    # Requires subband_preprocess.py
```

Three-phase schedule (both variants):
- Phase 1 (50 epochs): FP32 warm-up, no quantization
- Phase 2 (200 epochs): LSQ quantization enabled (STE gradients)
- Phase 3 (250 epochs): Fine-tune with spectral loss

Gen 7.1 uses `subband_preprocess.py` to apply LPC order-8 + 3-level lifting DWT before training. The TNN operates on L3 approximation coefficients (width 112, 3 focal blocks + GLU bottleneck).

### 4. Harden for deployment

```bash
python ai_models/student/harden_artifacts.py         # Aligns student latents with teacher
```

### 5. Export to firmware headers

```bash
python firmware/export_firmware.py
```

Generates header files in `firmware/firmware_export/`:
- `focal_net_weights.h` — Packed ternary weights (4 per byte), Q31 alphas, GroupNorm params, rANS frequency tables (Gen 7.1: V5_Subband weights, WHT coefficients)
- `fsq_lattice.h` — FSQ level configuration (Gen 7.1: L=2/3/5/32, L=32 for clinical mode)
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

## Desktop GUI — OpenHuman Vision

The Tauri v2 desktop application **OpenHuman Vision** provides impedance checking, live 21-channel signal visualization, recording, data export, firmware flashing, and hardware preflight testing — all without a terminal.

Two operating modes, toggled from the connection bar:

### OpenHuman Vision Mode (clinical use)

| Page | What it does |
|------|-------------|
| **Impedance** | SVG head diagram with 21 electrode positions (10-20 system), color-coded by impedance |
| **Signals** | 21-channel scrolling Canvas2D waveforms at 250 Hz. Configurable gain, filter, time window, montage (Referential / Bipolar Banana / Bipolar Transverse). Output mode selector (Raw USB / Compressed BLE / Dual) |
| **Export** | Save As dialog for EDF, CSV, or NPZ export. Event marker support |
| **Firmware** | Detect RP2350 in bootloader mode, browse for `.uf2` file, one-click flash |

### OpenHuman Eagle Mode (developer preflight)

| Page | What it does |
|------|-------------|
| **Dashboard** | Summary of hardware, software, and benchmark status. Run All button |
| **Hardware** | 8 preflight checks (KAT, CRC, PMP, ADC, BLE, SNN, Dual-Core, Mailbox). Memory usage bars |
| **Software** | Run pytest + C host tests from the GUI. Expandable pass/fail results |
| **Benchmarks** | Run benchmark suite from the GUI. Pass/fail grid |

See [docs/gui_guide.md](docs/gui_guide.md) for architecture details and [docs/user_manual.md](docs/user_manual.md) for the full user manual.

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

48 assertions covering: ternary MAC KAT, biquad Q30 filter, LFSR period/batch, CRC32 cross-platform parity, LPC predictor residuals (order-8), lifting wavelet (3-level DWT), Golomb-Rice round-trip, WHT round-trip, detail thresholding.

### Integration benchmarks

```bash
python tests/benchmarks/test_golden_path_e2e.py    # Requires trained model + dataset
python tests/benchmarks/benchmark_decoder_e2e.py
```

Pass criteria: R >= 0.96 (clinical mode), PRD <= 40%, CR >= 40x (clinical), CR >= 150x (alerting). Gen 7.0 legacy: R >= 0.85, CR >= 5.0x.

See [docs/design/validation.md](docs/design/validation.md) for the full test strategy.

---

## Documentation Index

| Document | Contents |
|----------|----------|
| [docs/user_manual.md](docs/user_manual.md) | **User Manual** — setup, every screen, firmware flashing, troubleshooting |
| [docs/gui_guide.md](docs/gui_guide.md) | GUI developer guide — architecture, Tauri commands, stores, wire protocol |
| [docs/design/architecture.md](docs/design/architecture.md) | Full system architecture, data flow, dual-path scheduler |
| [docs/design/hardware.md](docs/design/hardware.md) | RP2350 memory map, PMP stack guard, SPI bus allocation, pin map |
| [docs/design/mathematics.md](docs/design/mathematics.md) | LSQ quantization, Q31/Q30 arithmetic, FSQ+rANS, distillation loss |
| [docs/design/validation.md](docs/design/validation.md) | Test strategy, stress profiles, graduation thresholds |
| [docs/BUILDING.md](docs/BUILDING.md) | Firmware and GUI build instructions |
| [docs/directory_structure.md](docs/directory_structure.md) | File-by-file codebase reference |
| [docs/firmware_reference.md](docs/firmware_reference.md) | Every C function: signature, behavior, callers |
| [docs/training_pipeline.md](docs/training_pipeline.md) | Python training walkthrough, model architecture |
| [docs/wire_protocol.md](docs/wire_protocol.md) | Device-host packet format and command reference |
| [docs/hardware_bom.md](docs/hardware_bom.md) | Component list, PCB specs, power budget |
| [docs/RECORDING.md](docs/RECORDING.md) | Recording duration, RAM/disk costs, streaming mode |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common bring-up problems and fixes |
| [SUPPORT.md](SUPPORT.md) | How to file bugs, ask questions, report security issues |
| [CITATION.cff](CITATION.cff) | Cite this work |
| [SAFETY.md](SAFETY.md) | Regulatory disclaimer, F1 fragility defense |
| [COMPLIANCE.md](COMPLIANCE.md) | IEC 60601-1, ISO 13485, IEC 62304 traceability |

---

## Licensing

- **Code** (firmware + Python): [AGPL-3.0](LICENSE.md)
- **Trained weights** (`focal_net_weights.h`): [CC BY-NC 4.0](LICENSE.md)

## Compliance

See [COMPLIANCE.md](COMPLIANCE.md) for IEC 60601-1, ISO 13485, and IEC 62304 traceability.
