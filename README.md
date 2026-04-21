<p align="center">
  <img src="assets/banner.svg" alt="LamQuant — Open-Source Neural EEG Codec" width="100%">
</p>

<p align="center">
  <a href="https://github.com/quitetall/lamquant/actions/workflows/ci.yml"><img src="https://github.com/quitetall/lamquant/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/version-7.7.0-blue" alt="Version 7.7.0">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="License AGPL-3.0">
  <img src="https://img.shields.io/badge/MCU-RP2350%20%7C%20ESP32--S3-red" alt="Multi-MCU">
</p>

# LamQuant

A lossless + neural EEG codec. Ternary encoder on the RP2350 (150 MHz, 520 KB SRAM), GPU decoders (3.8M-844M params) on the base station.

**Lossless** (.lml) — 2.26:1 CR, PRD=0%, CRC-32 + SHA-256 verified, 94% of Shannon limit
**Neural** (.lmq) — 63-525:1 CR, adaptive quality driven by on-device SNN

> Research and educational use only. Not a cleared medical device. See [docs/SAFETY.md](docs/SAFETY.md).

---

## Quick Start

```bash
git clone https://github.com/Quitetall/LamQuant.git && cd LamQuant
pip install -e .

# Interactive menu
./lamquant.py

# Or directly
./lamquant.py compress /data/eeg/ -o /data/lml/
```

## What It Does

```
EDF files (16-bit, any SR, any channel count)
    │
    ▼
┌─ LML Lossless Codec ─────────────────────────────┐
│  lifting DWT → LPC → bias cancel → Golomb-Rice   │
│  CRC-32 per window · SHA-256 per file             │
│  verify-after-write · noise-aware mode            │
└───────────────────────────────────────────────────┘
    │
    ▼
.lml files (2.26:1 CR, bit-exact, human-readable header)
```

Every `.lml` file starts with a readable ASCII line:
```
LML v1 | 21ch | lossless | CRC-32
```
Then binary. `head -1 file.lml` tells you what it is without any tools.

---

## Lossless Codec (LML v1)

State of the art for 16-bit clinical EEG. No published method beats it on comparable data.

| Metric | Value |
|--------|-------|
| Compression ratio | 2.26 : 1 on TUEG (69,672 files) |
| Shannon efficiency | 94% of theoretical limit |
| Gap to floor | 0.54 bits/sample |
| Encode speed | 15 MiB/s (JIT) |
| Decode speed | 49 MiB/s (JIT) |
| MCU decode | 35 ms/window (3.5% of real-time) |
| Integrity | CRC-32 per window, SHA-256 per file |
| Format overhead | 0.01% (34-byte readable header) |

### vs. Baselines

| Codec | CR on 16-bit EEG | Type |
|-------|-------------------|------|
| gzip -6 | 1.61 : 1 | General |
| zstd -3 | 1.64 : 1 | General |
| FLAC | ~2.0 : 1 | Audio |
| 2D-DWT VLSI (2024) | 1.95 : 1 | Academic |
| **LML v1** | **2.26 : 1** | **EEG-specific** |
| Chen/Wu (2018) | 2.35 : 1 | Academic (CHB-MIT only) |
| Shannon limit | 2.41 : 1 | Theoretical |

### Safety

- CRC-32 catches every single-bit error (tested: 800 bit-flip positions, 0 misses)
- SHA-256 per file with mandatory verify-after-write
- Double-strip prevention: encoder refuses to strip already-stripped data
- Truncation detection: header-declared length verified before decode
- 23 conformance tests, 100-file adversarial verification, 0 silent corruption

### Noise-Aware Mode

Strip ADC noise floor bits for better CR without losing signal:

```python
# Archive (default): fully lossless
compressed = _compress_bytes(signal, noise_bits=0)   # 2.26:1

# Monitoring (strip 3 noise bits, ENOB=13):
compressed = _compress_bytes(signal, noise_bits=3)   # 2.85:1
```

Set `noise_bits = bit_depth - ENOB` for your hardware.

---

## CLI

```bash
# Compress an entire corpus (auto-detects workers, resumes on interrupt)
./lamquant.py compress /data/tueg/ -o /data/lml/

# Interactive guided workflow
./lamquant.py

# Inspect a compressed file
./lamquant.py info recording.lml

# System profile + recommended settings
./lamquant.py syscheck

# Run conformance tests
./lamquant.py test
```

The CLI provides:
- Live dashboard with progress bar, throughput, ETA
- Per-file SHA-256 verification
- Audit log (crash-safe, line-buffered)
- JSON manifest with full reproducibility data
- Ctrl-C clean shutdown with partial manifest
- `--skip-existing` for resumable runs (default ON)
- `--verify-outliers` checks best/worst files post-compression (default ON)

---

## Neural Codec (LMQ)

| Tier | Params | Use Case | Target |
|------|--------|----------|--------|
| Tier 3 | 3.8M | Research / archival | R > 0.94 |
| Tier 6 | 400M | Clinical workstation | R > 0.92 |
| Tier 7 | 844M | Joint training anchor | R > 0.94 |
| Tier 8 | 200M | Mobile (INT8 NPU) | R > 0.88 |

```bash
# Train the full pipeline
./lamquant.py train

# Or directly
python ai_models/student/train_joint.py --config production --deployment research
```

Training features: SOAP optimizer (+0.0135 R), V2-fixed SE decoder, ParetoQ ternary quantization, GAN, clinical balanced sampling, WSD infinite LR schedule.

---

## Architecture

```
RP2350 (Encoder, 435K ternary)          Base Station (Decoder, 3.8M-844M FP32)
┌──────────────────────────────┐  BLE  ┌─────────────────────────────────┐
│ 21ch EEG @ 250 Hz            │──────▶│ FSQ decode                      │
│ → HP filter → subband lifting│       │ → Vocos ConvNeXt (iSTFT)        │
│ → TernaryMobileNet encoder   │       │ → 21ch × 2500 fullband          │
│ → FSQ quantization           │       │                                 │
│ (4 ms, 520 KB SRAM)         │       │ Lossless: inverse GR+LPC+DWT    │
└──────────────────────────────┘       └─────────────────────────────────┘
```

---

## Project Structure

```
lamquant_codec/          Codec implementation
  lossless.py              LML v1 lossless codec (compress/decompress/peek_header)
  ops/golomb.py            Golomb-Rice entropy coder (JIT'd)
  ops/rans.py              rANS coder (experimental)
  edf_to_lml.py            EDF → LML converter (pyedflib)
  cli/                     CLI system
    compress.py              Production compress command
    readout.py               Dashboard, banner, summary
    config.py                TOML config system
    state.py                 Crash-safe state file
    syscheck.py              System profiler

ai_models/               Neural codec
  student/                 Ternary encoder + training
  decoder/                 Vocos iSTFT decoder
  oracle/                  Data loading + streaming

tests/                   Test suites
  test_lml_conformance.py  23 conformance tests
  test_lml_paranoid.py     45 roundtrip + adversarial tests
  test_lml_fuzz.py         Hypothesis property-based fuzzing

decisions/               Architecture decision records
  0010-lml-format-spec.md  Wire format specification
  0011-lml-entropy-coder-final.md  Entropy coder analysis
  0012-lml-production-config.md    Production configuration
```

---

## Documentation

| Document | Content |
|----------|---------|
| [Format Spec](decisions/0010-lml-format-spec.md) | LMQ5 wire format, byte layout, conformance requirements |
| [Entropy Analysis](decisions/0011-lml-entropy-coder-final.md) | GR vs rANS vs arithmetic, Shannon gap, competitive analysis |
| [Production Config](decisions/0012-lml-production-config.md) | All modes, safety systems, performance numbers |
| [CLI Output Spec](decisions/0013-cli-output-spec.md) | Dashboard, audit log, manifest format |
| [Config Spec](decisions/0014-config-and-resumability-spec.md) | TOML config, state file, crash recovery |
| [Latent Scaling](decisions/0009-latent-scaling.md) | Empirical R vs latent dimension curve |
| [Engineering Audit](decisions/0007-v763-engineering-audit.md) | 14 bugs fixed, A/B matrix, production stack |

---

## License

**Code**: [AGPL-3.0](LICENSE.md) | **Weights**: CC BY-NC 4.0 | **Compliance**: [IEC 60601-1, ISO 13485](docs/COMPLIANCE.md)

---

<p align="center">
  <sub>OpenHuman Technologies</sub>
</p>
