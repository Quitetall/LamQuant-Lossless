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
├── ai_models/                  Python training pipeline
│   ├── dataset_sim/            EDF -> Q31 dataset conversion
│   │   ├── edf_to_events.py   CHB-MIT parser, channel aliasing, Q31 scaling
│   │   └── audit_dataset.py   Dataset integrity checker
│   ├── oracle/                 FP32 teacher models
│   │   ├── train_teacher.py   Standard teacher autoencoder
│   │   └── train_teacher_strided.py  Strided teacher for Route B
│   └── student/                Ternary student distillation
│       ├── train_ternary.py   TernaryMobileNetV5 / V5_Subband architecture + LSQ
│       ├── train_student.py   Standalone student training (3-phase)
│       ├── train_student_subband.py  Gen 7.1 subband student training
│       ├── subband_preprocess.py     LPC + lifting DWT preprocessing
│       ├── validate_subband.py       Subband pipeline validation
│       ├── harden_artifacts.py Latent alignment with teacher
│       ├── finetune_student.py Optional fine-tuning
│       └── train_route_b_decoder.py  Route B (teacher decoder) training
├── firmware/                   Bare-metal C for RP2350
│   ├── core/                   Boot, scheduler, safety
│   │   ├── main.c             Entry point, boot sequence, ADC init
│   │   ├── scheduler.c        Pipeline FSM (legacy Gen 7.0)
│   │   ├── scheduler_v7_1.c   Gen 7.1 subband pipeline scheduler
│   │   ├── power_states.c     Safe mode, dormant state
│   │   ├── stack_guard.c      PMP hardware stack protection
│   │   ├── integrity.c        CRC32 firmware verification
│   │   └── math_utils.h       Q31/Q30 fixed-point primitives
│   ├── dsp/                    Signal processing
│   │   ├── biquad_q31.c       HP-only biquad filter (Q30)
│   │   ├── toeplitz_cs.c      LFSR-based compressed sensing
│   │   ├── lifting_2d.c       Le Gall 5/3 wavelet transform (3-level lifting DWT)
│   │   ├── lpc_predictor.c    Order-8 LPC (Levinson-Durbin)
│   │   ├── lpc_delta.c        LPC delta encoding for residuals
│   │   ├── detail_threshold.c Detail coefficient thresholding
│   │   └── wht32.c            Walsh-Hadamard Transform (32-point)
│   ├── neural/                 Neural network inference
│   │   ├── focal_modulation.c 3 focal blocks + GLU bottleneck (stride 4x on L3)
│   │   ├── ternary_mac.c      Branchless ternary MAC + KAT
│   │   ├── ternary_manifold.c Grouped conv helper (Q31 scaling)
│   │   └── fsq.c              Finite scalar quantization (L=2/3/5/32)
│   ├── transport/              Output encoding + BLE
│   │   ├── hybrid_entropy.c   rANS + Golomb-Rice + LPC delta encoder (LAMQ v1/v2)
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
│   ├── codec.py               TernaryCodec + SubbandCodec wrapper classes
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

**341 tests total** (155 Python + 186 C firmware), organized by 7 paranoia levels.

### Python tests (155 tests)

```bash
pytest tests/ -v                      # Full suite
pytest tests/ -m l2 -v                # L2 property-based only (79 tests)
pytest tests/ -m l5 -v                # L5 cross-implementation (54 tests)
pytest tests/ -m l7 -v                # L7 adversarial (41 tests)
pytest tests/ -v --cov=ai_models      # With coverage report
```

Covers: Q31 math, lifting DWT invertibility, LPC stability, preprocessing idempotence, ternary quantization, model shapes, channel mapping, firmware weight export, NEDC clinical metrics, EDF reader cross-validation (pyedflib parity), ERDR subprocess wrapper, canonical split enforcement, adversarial saturation/dead-electrode handling, C↔Python firmware parity (mul_q31, WHT32, lifting).

### C firmware tests (186 tests)

```bash
gcc -O2 -lm tests/c_host/test_c_firmware.c -o test_fw && ./test_fw
```

Covers: Q31/Q30 math, ternary MAC (KAT + exhaustive 256-combo + Q31 alpha), CRC32, LFSR, biquad IIR (DC rejection, impulse response, state carryover, cascade stability), lifting DWT (roundtrip even/odd/2500, constant, boundary, negative rounding), WHT32 (roundtrip, delta function), LPC Levinson-Durbin (flat signal, overflow guard), LPC delta codec (keyframe/Q8 roundtrip, no-prev guard), FSQ quantization (range, monotonicity).

### Training cockpit

```bash
python training_cockpit.py            # Interactive menu with 9 options
```

Live 6-line dashboard with epoch progress, loss, PRD, gradient norm, weight sparsity, GPU utilization, VRAM, temperature, ETA. Supports resume from checkpoint, factory reset, and edit queue.

### Integration benchmarks

```bash
python tests/benchmarks/test_golden_path_e2e.py    # Requires trained model + dataset
python tests/benchmarks/benchmark_decoder_e2e.py
```

Pass criteria: R >= 0.96 (clinical mode), PRD <= 2%, CR >= 40x (clinical), CR >= 150x (alerting).

See [PARANOID_TEST_GUIDE.md](PARANOID_TEST_GUIDE.md) and [docs/design/validation.md](docs/design/validation.md) for the full test strategy.

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
