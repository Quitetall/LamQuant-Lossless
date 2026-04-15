<p align="center">
  <img src="assets/banner.svg" alt="LamQuant — Clinical-Grade EEG Neural Codec" width="100%">
</p>

<p align="center">
  <a href="https://github.com/quitetall/lamquant/actions/workflows/ci.yml"><img src="https://github.com/quitetall/lamquant/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/version-7.2.0-blue" alt="Version 7.2.0">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="License AGPL-3.0">
  <img src="https://img.shields.io/badge/platform-RP2350-red" alt="RP2350">
</p>

# LamQuant

An asymmetric EEG neural codec: ternary-quantized encoder on the RP2350 microcontroller (dual Hazard3 RISC-V, 150 MHz, 64 KB SRAM), paired with GPU-side decoders (100M-837M params). Two production modes:

- **Mode 1 — Neural**: 274:1 compression, adaptive FSQ driven by on-device SNN activity classification
- **Mode 2 — Lossless**: 3.76:1 compression, PRD=0%, integer-exact lifting + LPC + Golomb-Rice

> **Research and educational use only.** Not a cleared medical device. See [SAFETY.md](SAFETY.md).

---

## Quick Start

```bash
# Install
./install.sh              # Linux/macOS
.\install.ps1             # Windows

# Launch GUI
lamquant gui              # Opens OpenHuman Vision (mock mode, no hardware needed)

# Run tests
pip install -e ".[test]"
pytest tests/ -v          # 29 end-to-end codec tests

# Encode/decode via CLI
lamquant decode -c weights/student_subband.ckpt -i eeg.npy --encode --subband -o compressed.bin
lamquant decode -c weights/student_subband.ckpt -i compressed.bin --subband -o recon.npy
lamquant decode -i eeg.npy --lossless --encode -o lossless.bin
```

---

## Architecture

```
MCU (RP2350)                                              GPU (Base Station)
┌─────────────────────────────────────┐    BLE    ┌──────────────────────────┐
│ ADC → HP → LPC → 3-level Lifting    │───────────│ rANS decode              │
│          ↓                          │           │ FSQ dequantize           │
│   L3 [21,313] → SNN (activity)      │           │ WHT inverse              │
│          ↓           ↓              │           │ Student decode [21,313]  │
│   TNN encode → WHT → FSQ → rANS     │           │ Inverse lifting          │
│   (213K ternary DW-sep, W2A16)      │           │ LPC synthesis            │
│                                     │           │          ↓               │
│   Detail subbands → Golomb-Rice     │           │ Vocos decoder [21,2500]  │
│                                     │           │ (100M/400M/837M iSTFT)   │
│   OR: Lossless (lifting+LPC+GR)     │           │          ↓               │
└─────────────────────────────────────┘           │ CFM postfilter (opt.)    │
                                                  └──────────────────────────┘
```

### Encoder (MCU-side, 213K params — V2)

TernaryMobileNetV5_Subband_V2: width 216, 4 depthwise-separable focal blocks, ReGLU bottleneck, Bit-Shift Normalization. Weights in {-1, 0, +1} with ParetoQ SEQ quantizer and SubLN. DW-sep blocks are 6.8x more parameter-efficient than full convolutions. Trained with unified QAT (350 epochs).

### Decoders (GPU-side)

| Tier | Params | Architecture | Use Case |
|------|--------|-------------|----------|
| 5 | ~100M | Vocos ConvNeXt, dim=896, 20 blocks, iSTFT n_fft=32 | Real-time monitoring |
| 6 | ~400M | Vocos ConvNeXt, dim=1792, 20 blocks, iSTFT n_fft=64 | Clinical review |
| 7 | ~837M | Vocos ConvNeXt, dim=1792, 32 blocks, iSTFT n_fft=128 | Research/archival |

All tiers use Snake activation (`x + (1/a)sin^2(ax)`) for periodic inductive bias, anti-wrapping phase loss, and band-weighted reconstruction.

### SNN Activity Classifier

MambaSNN (bidirectional SSM) classifies each L3 timestep as quiescent/active/seizure. Drives adaptive FSQ: L=2 (quiet), L=3 (active), L=5 (event). INT4 weight quantization for MCU deployment.

---

## Standard Packet API

All codec modes produce a standard `EEGPacket`. All quality benchmarks consume packets. This decouples codec internals from evaluation.

```python
from lamquant_codec import EEGPacket, Benchmark

# Any codec produces a packet
packet = EEGPacket.from_reconstruction(
    signal=recon_signal,        # [21, T] numpy array
    compressed_bytes=384,
    mode='neural',
)

# Architecture-agnostic benchmarks
report = Benchmark.full_report(original, packet)
# {'prd': 12.3, 'r': 0.89, 'cr': 274.0, 'snr_db': 18.2, ...}

# Per-band frequency analysis
bands = Benchmark.per_band_prd(original, packet)
# {'delta': 8.1, 'theta': 11.2, 'alpha': 9.5, 'beta': 15.3, 'gamma': 22.1}
```

The GUI, CLI, and all official tools use EEGPacket as their interchange format. The `--json-out` flag emits the EEGPacket schema for IPC with the Tauri desktop app.

---

## Codec Modes

### Mode 1: Neural (adaptive, 175:1 to 525:1)

```
EEG [21,2500] → HP → LPC → lifting → L3 [21,313]
    → TNN encode → latent [32,79] → WHT → adaptive FSQ → rANS → packet
```

Sub-window adaptive FSQ assigns different quantization levels per timestep based on SNN classification. Quiet segments get L=2 (1 bit/dim), events get L=5 (2.3 bits/dim).

```python
from lamquant_codec import SubbandCodec
codec = SubbandCodec.from_checkpoint("weights/student_subband.ckpt")
packet = codec.compress_to_packet(latent, l3_signal, quality_mode=2)
```

### Mode 2: Lossless (3.76:1, PRD=0%)

```
EEG [21,2500] → integer lifting (4-level) → LPC per subband → Golomb-Rice → packet
```

Pure DSP, no neural network. Bit-exact integer arithmetic throughout. Optional integer-exact KLT via lifting-based Givens rotations.

```python
from lamquant_codec import LosslessCodec
codec = LosslessCodec(klt_matrix=None, n_levels=3)
packet = codec.compress_to_packet(signal)
```

---

## Training Pipeline

### Training cockpit (recommended)

```bash
python training_cockpit.py    # Interactive dashboard with live metrics
```

### Manual steps

```bash
# 1. Convert EDF datasets to Q31 windows
python ai_models/dataset_sim/edf_to_events.py --input ./datasets/chbmit --output ./q31_events

# 2. Precompute L3 approximations
python ai_models/student/precompute_l3_fast.py --input ./q31_events --workers 8

# 3. Train teacher (FP32 autoencoder)
python ai_models/oracle/train_teacher.py --headless --resume

# 4. Train student (ternary encoder, unified QAT)
python ai_models/student/train_student_subband.py

# 5. Train SNN activity classifier
python ai_models/snn/train_mamba_snn.py --data ai_models/snn/labels --epochs 500

# 6. Train GPU decoder (Vocos + iSTFT)
python run_decoder_tier.py --tier 5 --epochs 200

# 7. Export encoder to firmware C headers
python firmware/export_firmware.py
```

### QAT features

- **ParetoQ SEQ** (StretchedElasticQuant): improved ternary quantizer from Meta's ParetoQ (ICLR 2025)
- **SubLN**: GroupNorm after every quantized projection (from BitNet b1.58)
- **Unified QAT**: single 350-epoch phase with cosine tau anneal, no phase transitions
- **StableCodec FSQ-dropout**: per-sample Bernoulli masks (passthrough/noise/hard-quant)
- **Two-stage weight decay**: normal WD for 2/3, remove for final 1/3

### Perceptual loss teachers

GPU decoder training uses multi-teacher perceptual loss from EEG foundation models:
- **LaBraM** (ICLR 2024): 2,500 hours pretrained, VQ-NSP tokenizer features
- **FEMBA** (arXiv 2502.06438): 21,000 hours pretrained, bidirectional Mamba encoder
- **DAC**: Descript Audio Codec features (audio-pretrained, transfers to EEG)

---

## Desktop GUI — OpenHuman Vision

Tauri v2 desktop app (Rust + SvelteKit). Two modes:

**Vision Mode** (clinical): impedance check, 21-channel live visualization at 250 Hz, EDF/CSV/NPZ export, firmware flash, configurable montages (referential/bipolar banana/bipolar transverse).

**Eagle Mode** (developer preflight): hardware checks (KAT, CRC, PMP, ADC, BLE, SNN), software tests, benchmarks.

```bash
lamquant gui              # Launch (mock mode by default)
cd gui && npx tauri dev   # Development mode with hot reload
```

See [docs/gui_guide.md](docs/gui_guide.md) for architecture and [docs/user_manual.md](docs/user_manual.md) for the full manual.

---

## Firmware

```bash
export PICO_SDK_PATH=/path/to/pico-sdk
mkdir build && cd build
cmake .. -DADC_BACKEND_ADS1299=ON
make lamquant
```

Output: `lamquant.uf2` — flash via USB boot mode (hold BOOTSEL, plug in, copy file).

See [docs/BUILDING.md](docs/BUILDING.md) and [docs/firmware_reference.md](docs/firmware_reference.md).

---

## Testing

```bash
# Python end-to-end tests (29 tests)
pytest tests/ -v

# C firmware host tests
gcc -O2 -lm tests/c_host/test_c_firmware.c -o test_fw && ./test_fw
```

Test coverage: lossless roundtrip, CR thresholds, adaptive FSQ, L3 error correction, rANS entropy coding, integer lifting invertibility, CDF-LUT, model architecture shapes, SNN classification, EEGPacket API, per-band PRD.

---

## Directory Layout

```
lamquant/
├── ai_models/                  Python training pipeline
│   ├── dataset_sim/            EDF → Q31 dataset conversion
│   ├── oracle/                 FP32 teacher + combined decoder training
│   ├── student/                Ternary student (QAT, diagnostics, distillation)
│   ├── decoder/                Vocos decoder, perceptual losses, CFM postfilter
│   └── snn/                    Mamba SNN activity classifier
├── firmware/                   Bare-metal C for RP2350
├── gui/                        Tauri v2 desktop app (Rust + SvelteKit)
├── lamquant_codec/             Python codec package (pip-installable)
│   ├── packet.py               EEGPacket standard format
│   ├── benchmark.py            Architecture-agnostic quality metrics
│   ├── codec.py                TernaryCodec, SubbandCodec, LosslessCodec
│   ├── decode.py               CLI encoder/decoder
│   └── export.py               Firmware header export
├── tests/                      Python + C host tests
├── tools/                      Visualization utilities
├── docs/                       Technical documentation
├── weights/                    Model checkpoints
└── Reference Software/         Downloaded research repos (15+)
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/user_manual.md](docs/user_manual.md) | User manual |
| [docs/gui_guide.md](docs/gui_guide.md) | GUI developer guide |
| [docs/design/architecture.md](docs/design/architecture.md) | System architecture |
| [docs/design/hardware.md](docs/design/hardware.md) | RP2350 memory map, pin map |
| [docs/design/mathematics.md](docs/design/mathematics.md) | Quantization, arithmetic, codec math |
| [docs/design/validation.md](docs/design/validation.md) | Test strategy, graduation thresholds |
| [docs/BUILDING.md](docs/BUILDING.md) | Firmware and GUI build instructions |
| [docs/firmware_reference.md](docs/firmware_reference.md) | C function reference |
| [docs/training_pipeline.md](docs/training_pipeline.md) | Training walkthrough |
| [docs/wire_protocol.md](docs/wire_protocol.md) | Device-host packet format |
| [docs/hardware_bom.md](docs/hardware_bom.md) | BOM, PCB specs, power budget |
| [SAFETY.md](SAFETY.md) | Regulatory disclaimer |
| [COMPLIANCE.md](COMPLIANCE.md) | IEC 60601-1, ISO 13485, IEC 62304 |

---

## License

**Code** (firmware + Python): [AGPL-3.0](LICENSE.md) | **Trained weights**: [CC BY-NC 4.0](LICENSE.md) | **Compliance**: [IEC 60601-1, ISO 13485, IEC 62304](COMPLIANCE.md)
