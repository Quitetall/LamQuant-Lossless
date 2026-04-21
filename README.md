<p align="center">
  <img src="assets/banner.svg" alt="LamQuant — Open-Source Neural EEG Codec" width="100%">
</p>

<p align="center">
  <a href="https://github.com/quitetall/lamquant/actions/workflows/ci.yml"><img src="https://github.com/quitetall/lamquant/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/version-7.7.0-blue" alt="Version 7.7.0">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/rust-1.70%2B-dea584" alt="Rust 1.70+">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="License AGPL-3.0">
  <img src="https://img.shields.io/badge/MCU-RP2350%20Hazard3-red" alt="RP2350">
</p>

<p align="center">
  <strong>Clinical-grade lossless EEG compression + ternary neural codec</strong><br>
  435K-param encoder on a 150 MHz RISC-V MCU. 3.8M-844M-param decoders on the base station.
</p>

---

## What is LamQuant?

LamQuant compresses brain signals. Two codecs, one pipeline:

**LML (Lossless)** -- bit-exact reconstruction, 2.26:1 compression ratio, verified on 370,000+ windows across 2,212 clinical EEG recordings. Every bit comes back.

**LMQ (Neural)** -- 63:1 to 525:1 adaptive compression driven by an on-device spiking neural network. The ternary encoder runs entirely on the RP2350 Hazard3 RISC-V core at 150 MHz with 520 KB SRAM.

Both formats have CRC-32 per window, SHA-256 per file, and human-readable ASCII headers you can inspect with `head -1`.

> Research and educational use only. Not a cleared medical device. See [docs/SAFETY.md](docs/SAFETY.md).

---

## Install

```bash
pip install lamquant-codec              # lossless codec (numpy only, no GPU needed)
pip install lamquant-codec[fast]        # + numba JIT (3-4x faster)
pip install lamquant-codec[neural]      # + torch (neural codec + training)
pip install lamquant-codec[all]         # everything
```

Or from source:

```bash
git clone https://github.com/Quitetall/LamQuant.git && cd LamQuant
pip install -e ".[all]"
```

Rust CLI (single binary, no dependencies):

```bash
cargo install --path lamquant-core
lml encode recording.edf -o recording.lml --verify
```

---

## Usage

### Compress EDF to LML

```bash
lamquant compress /data/eeg/ -o /data/lml/ --recursive --verify
```

### Python API

```python
from lamquant_codec.edf_to_lml import write_lml_file, read_lml_file
import numpy as np

# Compress
signal = np.random.randint(-5000, 5000, (21, 2500), dtype=np.int64)
write_lml_file("recording.lml", signal, {"sample_rate": 250, "channels": ["FP1", "FP2", ...]})

# Decompress (bit-exact)
recovered, metadata = read_lml_file("recording.lml")
assert np.array_equal(signal, recovered)
```

### Rust CLI

```bash
lml encode recording.edf -o recording.lml --verify     # compress + verify
lml decode recording.lml -o recording.raw               # decompress
lml info recording.lml                                   # inspect metadata
lml verify recording.lml                                 # CRC integrity check
lml bench recording.edf                                  # throughput benchmark
lml recover damaged.lml -o fixed.lml                     # salvage valid windows
```

---

## Lossless Codec (LML v1)

State of the art for clinical EEG compression.

```
signal[21ch][2500]
  -> 3-level Le Gall 5/3 integer lifting DWT
  -> per-subband LPC prediction (order 1/1/2/3)
  -> bias cancellation (running mean, ctx=32, floor division)
  -> Golomb-Rice entropy coding (adaptive k)
  -> CRC-32 per window, SHA-256 per file
```

Every `.lml` file starts with a readable ASCII line:
```
LML | 21ch | lossless | CRC-32
```

### Performance

| Metric | Value |
|--------|-------|
| Compression ratio | **2.26 : 1** on TUEG (69,672 clinical recordings) |
| Shannon efficiency | 94% of theoretical limit |
| Rust encode | **198 MB/s** (1,035 us/window, AVX2) |
| Rust decode | **197 MB/s** (1,039 us/window) |
| Python+numba | 15 MB/s encode, 49 MB/s decode |
| Integrity | CRC-32 per window + SHA-256 per file |
| Roundtrip | Bit-exact on 370,823 windows, 0 failures |

### Competitive Analysis

| Codec | CR | Type |
|-------|-----|------|
| gzip -6 | 1.61 : 1 | General |
| zstd -3 | 1.64 : 1 | General |
| FLAC | ~2.0 : 1 | Audio lossless |
| **LML v1** | **2.26 : 1** | **EEG-specific** |
| Shannon limit | 2.41 : 1 | Theoretical floor |

---

## Neural Codec (LMQ)

The ternary encoder runs on the MCU. GPU decoders reconstruct the signal on the base station.

| Component | Params | Precision | Where |
|-----------|--------|-----------|-------|
| Encoder (TernaryMobileNetV5_Subband) | 435K | W2A16 ternary | RP2350 (flash via XIP) |
| SNN activity detector (dLIF) | 57K | INT4 | RP2350 |
| Decoder (Vocos ConvNeXt) | 3.8M-844M | FP32 | Base station GPU |

```
RP2350 Hazard3 (150 MHz RISC-V)         Base Station
 21ch EEG @ 250 Hz                       FSQ decode
 -> HP filter -> LPC -> lifting DWT      -> Vocos ConvNeXt decoder
 -> SNN activity detection               -> 21ch x 2500 fullband
 -> TNN encoder (w=128, stride 2)        
 -> WHT -> FSQ -> rANS                   Lossless: inverse GR+LPC+DWT
 -> BLE transmit (.lmq or .lml)          
```

### Firmware (linker-verified)

| | Value |
|---|---|
| Target | RP2350B Hazard3, RV32IMAC + Zba/Zbb/Zbs, 150 MHz |
| SRAM | 509.8 KB / 520 KB (98.0%) |
| Flash | 95.3 KB code + 104.8 KB weights (XIP) |
| Headroom | 11.1 KB (7.6 KB main + 1.5 KB SCRATCH_X + 2.0 KB SCRATCH_Y) |
| Ternary MAC | ~1.2 cyc/MAC (3-path: SW pipeline + XNOR+CPOP + Zbb sat) |
| Soft float | Zero. Pure integer firmware. |
| Compiler | Max pedantic warnings, zero warnings |
| Safety | 8 FDA safety subsystems (BLE retry, impedance monitoring, pre-ictal buffer, seizure diary, event log, battery monitor, fault log, channel quality) |

---

## Project Structure

```
lamquant_codec/                 Codec package (pip install lamquant-codec)
  ops/                            DSP primitives (single source of truth)
    lifting.py                      Le Gall 5/3 integer DWT (float + int + JIT)
    lpc.py                          LPC analysis/synthesis (float + int + JIT)
    golomb.py                       Golomb-Rice entropy coder (Rust/numba/Python)
    rans.py                         rANS entropy coder (Rust/numba/Python)
    bias.py                         Context-adaptive bias cancellation
    wht.py                          Walsh-Hadamard transform
    pipeline.py                     Preprocessing orchestrators
    noise.py                        ADC noise floor estimation
    constants.py                    Wire-format constants (single source of truth)
  lossless.py                     LML codec (compress/decompress)
  compress.py                     Neural packet encoder (LMQ1)
  decompress.py                   Neural packet decoder (LMQ1)
  codec.py                        SubbandCodec (TNN + lossless)
  edf_to_lml.py                   EDF -> LML converter + container I/O
  batch.py                        Batch compress/decompress/validate
  cli/                            CLI system (compress, syscheck, dashboard)

lamquant-core/                  Rust crate (cargo install)
  src/lml.rs                      LML codec (3-4x faster than Python)
  src/lifting.rs                  Split-buffer DWT (auto-vectorized)
  src/lpc.rs                      LPC + bias (floor division)
  src/golomb.rs                   Golomb-Rice encoder/decoder
  src/container.rs                LML file container
  src/bin/lml.rs                  CLI binary
  src/ffi.rs                      C FFI (cbindgen -> include/lml.h)
  src/wasm.rs                     WebAssembly bindings
  fuzz/                           cargo-fuzz targets (decompress + roundtrip)

ai_models/                      Training code
  architectures/                  Model definitions (extracted, not monolithic)
    encoder.py                      TernaryMobileNetV5, _Subband, _V2
    teacher.py                      L3Teacher (FP32 oracle)
    snn.py                          ActivitySNN, ActivitySNN_Subband
    blocks.py                       TernaryConv1d, LSQ, focal blocks
  recipes.py                      15 named experiment configs
  experiment_runner.py            Unified runner (run/ab/sweep/leaderboard)
  student/training_config.py      Frozen dataclass config with hash provenance

firmware/                       RP2350 Hazard3 firmware (C, builds with Pico SDK 2.1)
  core/                           Scheduler, mailbox, boot, watchdog
  dsp/                            Biquad, lifting, LPC, WHT
  neural/                         Focal modulation, ternary MAC, FSQ
  snn/                            Spiking neural network (dLIF)
  codec/                          rANS, Golomb-Rice, detail threshold

scripts/                        Tools
  audit_format_consistency.py     CI: magic bytes, headers, constants check
  verify_roundtrip.py             FDA: per-window SHA-256 roundtrip verification
  estimate_noise_floor.py         Per-dataset ADC noise floor estimation
  benchmark_improvements.py       A/B codec improvement testing

docs/                           Specifications and design docs
  lml-format-v1.md                Frozen wire format specification
  SPEC.md                         System specification
```

---

## Testing

```bash
# Python (1853 tests, ~3.5 min)
pytest tests/ -q

# Rust (32 tests: 16 unit + 16 conformance)
cargo test --manifest-path lamquant-core/Cargo.toml

# Format consistency audit
python scripts/audit_format_consistency.py

# Lossless roundtrip verification (FDA-grade)
python scripts/verify_roundtrip.py /path/to/edfs/ --recursive --workers 4

# Fuzzing (continuous)
cd lamquant-core && cargo +nightly fuzz run decompress
```

---

## Training

```bash
# Single experiment
python -m ai_models.experiment_runner run --tier 3 --preset fast

# Named recipe
python -m ai_models.experiment_runner recipe snac_balanced --tier 3

# A/B comparison
python -m ai_models.experiment_runner sweep --tier 3 --grid '{"prd_weight": [0.05, 0.1, 0.2]}'

# Leaderboard
python -m ai_models.experiment_runner leaderboard
```

Training uses SOAP optimizer (+0.0135 R over AdamW), ParetoQ ternary quantization, WSD infinite LR schedule, GAN adversarial phase, and clinical weighted sampling.

---

## Key Design Decisions

- **Golomb-Rice over rANS** for entropy coding: zero-overhead headers (3 bytes vs 512+ byte frequency tables). At 84 subbands per window, rANS tables would consume 43 KB -- destroying the compression ratio.
- **Integer-only wire format**: no floating-point in encoded data. Bit-exact across all platforms (x86, ARM, RISC-V, WASM).
- **Ternary quantization** (W2) for the MCU encoder: XNOR+popcount achieves 16 MACs in ~5 instructions on Hazard3's Zbb extension. No multiply instruction needed.
- **Weights in flash via XIP**: model size is bounded by 16 MB flash, not 520 KB SRAM. Activation memory (210 KB) is the real constraint.
- **BIAS_CTX_LEN=32**: validated on 370,823 windows (100.00% win rate vs ctx=16). +0.2% CR, 128 bytes live buffer on MCU.

---

## Documentation

| Document | Content |
|----------|---------|
| [LML Format Spec](docs/lml-format-v1.md) | Frozen wire format -- implementable without reference code |
| [System Spec](docs/SPEC.md) | Full system architecture, memory map, timing |
| [Hardware Design](docs/design/hardware.md) | RP2350 memory layout, ADS1299 AFE, BLE transport |
| [Training Pipeline](docs/training_pipeline.md) | Data prep, training configs, evaluation |
| [Safety](docs/SAFETY.md) | Risk analysis, regulatory status |

---

## License

**Code**: [AGPL-3.0](LICENSE.md) | **Weights**: CC BY-NC 4.0 | **Spec**: CC BY 4.0 | **Compliance**: [IEC 60601-1, ISO 13485](docs/COMPLIANCE.md)

---

<p align="center">
  <sub>OpenHuman Technologies -- open-source brain-computer interfaces for everyone</sub>
</p>
