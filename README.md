# LamQuant Gen 6

A clinical-grade EEG neural codec that compresses 21-channel EEG data on-chip using ternary-quantized neural networks, targeting the RP2350 (Hazard3 RISC-V) microcontroller under strict real-time constraints (< 4ms latency, 64KB SRAM).

> **Research and educational use only.** This is not a cleared medical device. See [SAFETY.md](SAFETY.md) for details.

## Quick Start

**Run the test suite** (no GPU, no data required):
```bash
pip install -e ".[test]"
pytest tests/ -v
```

**Train a model** (GPU recommended, requires EEG dataset):
```bash
pip install -e .
# See "Training Pipeline" below
```

**Build firmware** (requires Pico SDK):
```bash
# See docs/BUILDING.md
```

## Installation

**Prerequisites:** Python 3.10+, pip

```bash
git clone https://github.com/Quitetall/lamquant_gen6.git
cd lamquant_gen6
pip install -e .            # runtime deps (torch, numpy, scipy, mne, tqdm)
pip install -e ".[test]"    # + pytest, coverage
pip install -e ".[dev]"     # + wandb, mlflow (optional)
```

For firmware compilation, see [docs/BUILDING.md](docs/BUILDING.md).

## Training Pipeline

Run these steps in order. Each script has `--help` for options.

### 1. Download and convert EEG data

```bash
# Download CHB-MIT dataset from PhysioNet
s5cmd --no-sign-request cp "s3://physionet-open/chb-mit-scalp-eeg-database/1.0.0/*" ./chbmit/

# Convert EDF files to Q31 format
python ai_models/dataset_sim/edf_to_events.py --input ./chbmit --output ./q31_events
```

### 2. Train teacher (FP32 reference model)

```bash
python ai_models/oracle/train_teacher.py            # ~1-2 hours on RTX 4090
python ai_models/oracle/train_teacher_strided.py     # parallel: strided variant for Route B
```

### 3. Train student (ternary quantized)

```bash
python ai_models/student/train_student.py            # ~8 minutes on RTX 4090
```

### 4. Harden for deployment

```bash
python ai_models/student/harden_artifacts.py         # aligns latents with strided teacher
```

See [ai_models/student/README.md](ai_models/student/README.md) for optional fine-tuning and Route B decoder steps.

### 5. Export to firmware

```bash
python firmware/export_firmware.py                   # generates focal_net_weights.h
```

### 6. Build firmware

```bash
export PICO_SDK_PATH=/path/to/pico-sdk
mkdir build && cd build && cmake .. && make lamquant
```

## Directory Structure

```
ai_models/
  dataset_sim/    EDF→Q31 dataset conversion and audit
  oracle/         FP32 teacher model training
  student/        Ternary student distillation and hardening
  training_cockpit.py   ANSI training dashboard
firmware/
  core/           Boot, scheduler, CRC integrity, stack guard
  dsp/            Q30 biquad filters, lifting wavelets, Toeplitz CS
  neural/         Ternary MAC, focal modulation, FSQ
  transport/      BLE/SPI host, hybrid entropy coding
  export_firmware.py    PyTorch→C header exporter
tests/
  test_*.py       Unit tests (pytest)
  c_host/         Host-compiled C firmware tests
  benchmarks/     Integration benchmarks (fidelity, parity, compression)
docs/
  design/         Architecture, hardware, mathematics, validation specs
```

## Testing

```bash
# Unit tests (51 Python + 44 C assertions)
pytest tests/ -v

# Integration benchmarks (requires trained model + dataset)
python tests/benchmarks/run_all_benchmarks.py
```

## Architecture

The system uses knowledge distillation to compress a full-precision teacher into a ternary-quantized student (W2A6) that fits in 43KB SRAM. The encoder runs on-chip at < 4ms; the decoder runs on the base station.

For technical deep-dives, see [docs/design/](docs/design/).

## Licensing

- **Firmware & code**: [GPLv3](LICENSE.md)
- **Transport layer** (`ternary_mac.c`, `hybrid_entropy.c`): [AGPLv3](LICENSE.md)
- **Trained weights** (`focal_net_weights.h`): [CC BY-NC 4.0](LICENSE.md)

## Compliance

See [COMPLIANCE.md](COMPLIANCE.md) for IEC 60601-1, ISO 13485, and IEC 62304 traceability.
