# LamQuant

An EEG neural codec for real-time compression on microcontrollers. Ternary encoder on the RP2350 (150 MHz RISC-V), GPU decoders (100M-800M params) on the base station.

**Mode 1 — Neural** (.lmq): 63-525:1 compression, adaptive FSQ driven by on-device SNN
**Mode 2 — Lossless** (.lml): 3.76:1 compression, PRD=0%, integer-exact

> Research and educational use only. Not a cleared medical device. See [docs/SAFETY.md](docs/SAFETY.md).

---

## Quick Start

```bash
# Clone
git clone https://github.com/Quitetall/LamQuant.git && cd LamQuant

# Install
./installer/install.sh         # Linux/macOS
.\installer\install.ps1        # Windows

# Run
./lamquant.py                  # interactive menu
./lamquant.py gui              # desktop app
```

## Two Ways to Use

**Terminal** — `./lamquant.py` is the all-in-one CLI:

```bash
./lamquant.py train            # train encoder, decoder, SNN
./lamquant.py encode -i eeg.npy -o compressed.lmq
./lamquant.py decode -i compressed.lmq -o reconstructed.npy
./lamquant.py validate         # LQS compliance check
./lamquant.py test             # run 43 e2e tests
```

**Desktop** — `./lamquant.py gui` opens OpenHuman Vision:

| Mode | Features |
|------|----------|
| Vision | Live 21-channel EEG, impedance, recording, firmware flash |
| Eagle | Hardware diagnostics, software tests, benchmarks |

---

## Documentation

| Document | What you'll learn |
|----------|-------------------|
| [**CLI Guide**](docs/cli_guide.md) | Every command, every flag, examples |
| [**Master Spec**](docs/SPEC.md) | Complete system architecture (18 sections) |
| [**API Reference**](docs/api_reference.md) | Python API for EEGPacket, Benchmark, file I/O |
| [**Pipeline Architecture**](docs/pipeline_architecture.md) | Unix pipeline design, typed dataclass boundaries |
| [**File Format Spec**](docs/file_format_spec.md) | .lmq/.lml binary format |
| [**Training Pipeline**](docs/training_pipeline.md) | Step-by-step training walkthrough |
| [**LQS Standard**](docs/SPEC.md#12-lamquant-quality-standard-lqs-v10) | Quality levels (L/C/M/A) |
| [**GUI Guide**](docs/gui_guide.md) | Desktop app architecture |
| [**Firmware Reference**](docs/firmware_reference.md) | C function reference |
| [**Production Config**](ai_models/PRODUCTION_CONFIG.md) | Current best training recipe |

---

## Architecture

```
RP2350 (Encoder)                    GPU (Decoder)
┌──────────────────────┐    BLE    ┌──────────────────────┐
│ HP → LPC → Lifting   │─────────▶│ rANS decode          │
│ → TNN (226K ternary)  │          │ → Vocos ConvNeXt     │
│ → FSQ → rANS          │          │ → iSTFT → [21,2500]  │
│ (4ms, 520 KB SRAM)   │          │ (100M-800M params)   │
└──────────────────────┘          └──────────────────────┘
```

Two modes, same hardware: **Mode 1** uses the neural encoder (63-525:1). **Mode 2** bypasses it for bit-exact lossless (3.76:1).

---

## Quality Standard (LQS v1.0)

| Level | Use Case | PRD | R | CR |
|-------|----------|-----|---|-----|
| LQS-L | Lossless archival | 0% | 1.0 | 3.76:1 |
| LQS-C | Clinical diagnosis | <9% | >0.95 | >20:1 |
| LQS-M | Seizure detection | <20% | >0.85 | >100:1 |
| LQS-A | ICU alerting | <40% | >0.70 | >200:1 |

---

## License

**Code**: [AGPL-3.0](LICENSE.md) | **Weights**: CC BY-NC 4.0 | **Compliance**: [IEC 60601-1, ISO 13485](docs/COMPLIANCE.md)
