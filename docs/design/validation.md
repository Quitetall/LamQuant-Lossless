# Validation Strategy

How LamQuant is tested, from unit tests through firmware stress profiles.

---

## Test Layers

### Layer 1: Python Unit Tests (pytest)

**Location**: `tests/test_*.py`
**Count**: 51 tests
**Run**: `pytest tests/ -v`
**Requires**: Python 3.10+, no GPU, no dataset

| Test File | What It Covers |
|-----------|---------------|
| `test_weight_packing.py` | Round-trip ternary packing (00=0, 01=+1, 10=-1, 11=0). Edge cases: all-zero, all-one, all-negative-one, non-multiple-of-4 lengths |
| `test_lsq_quantization.py` | `TernaryMobileNetV5` shape correctness: input `[B,21,2500]` -> latent `[B,32,312]` -> output `[B,21,2500]`. STE gradient flow. Quantized weights are exactly ternary |
| `test_export_firmware.py` | `export_firmware.py` output: valid C header syntax, Q31 alpha range, CRC32 parity with `zlib.crc32()`, rANS frequency table monotonicity |
| `test_channel_mapping.py` | EDF channel aliasing (T7->T3, T8->T4, P7->T5, P8->T6). Missing-channel rejection. Q31 scaling within int32 range |
| `test_utils.py` | `percent_zero_weights()`: all-zero=100%, no-zero=0%, empty model=0% (no division by zero) |
| `test_c_host.py` | Invokes host-compiled C test binary, asserts exit code 0 |

### Layer 2: C Host Tests

**Location**: `tests/c_host/test_c_firmware.c`
**Count**: 48 assertions
**Run**: `gcc -I firmware tests/c_host/test_c_firmware.c -o test_fw -lm && ./test_fw`
**Requires**: Any C compiler (x86/ARM/RISC-V), no RP2350 hardware

These tests compile the firmware's pure-math functions for the host platform using stub headers that replace RP2350 intrinsics.

| Test Group | Assertions | What It Verifies |
|------------|-----------|------------------|
| Ternary MAC | 4 | LUT correctness, byte unpacking, KAT result = 300 |
| Biquad Q30 | 6 | `mul_q30` range, DC rejection (HP stage), impulse response shape |
| LFSR | 5 | Period = 65535, `lfsr_batch32()` matches 32 sequential `lfsr_advance()` calls |
| CRC32 | 4 | Known-data CRC matches Python `zlib.crc32()`, single-bit corruption detection |
| Lifting wavelet | 4 | In-place transform/inverse round-trip, energy compaction |
| Golomb-Rice | 3 | Encode/decode round-trip for signed residuals, k=4 |
| LPC predictor | 6 | Residuals smaller than input (correlation removed), flat signal unchanged, zero signal unchanged |
| Saturating arithmetic | 4 | `add_sat_q31(INT32_MAX, 1) = INT32_MAX`, underflow clamp |
| mul_q31 | 4 | Known products, boundary values, zero multiplication |
| FSQ quantize | 4 | Grid symmetry, clamp at bounds, mid-range correctness |
| Integer sqrt | 4 | `isqrt32(0)=0`, `isqrt32(1)=1`, `isqrt32(100)=10`, large values |

### Layer 3: Integration Benchmarks

**Location**: `tests/benchmarks/`
**Run**: `python tests/benchmarks/test_golden_path_e2e.py`
**Requires**: Trained model checkpoint + EEG dataset

| Benchmark | What It Measures | Pass Criteria |
|-----------|-----------------|---------------|
| `test_golden_path_e2e.py` | Full encode -> FSQ -> rANS -> decode -> quality | R >= 0.85, PRD <= 40%, CR >= 5.0x |
| `benchmark_decoder_e2e.py` | Route A/B decoder latency, memory, throughput | Latency < 50ms (Route A), < 200ms (Route B) |

### Layer 4: Firmware Stress Profiles

These are design-time validation targets. They require RP2350 hardware and test fixtures.

| Profile | Condition | Pass Criteria |
|---------|-----------|---------------|
| `baseline_nominal` | Normal EEG rhythms (alpha, beta, theta) | PRD < 2.0% |
| `seizure_burst` | High-frequency ictal discharge (from CHB-MIT seizure segments) | PRD < 2.5% |
| `emc_burst` | IEC 61000-4-4 transient injection (fast transient burst test) | < 15% packet loss |
| `thermal_derating` | 85 C junction temperature, clock throttle to 120 MHz | Latency < 4.0 ms |
| `flash_corruption` | Induced byte-flips in weight storage (SRAM4) | Safe mode entry, no corrupted TX |
| `electrode_pop` | DC offset jump + drift (electrode displacement artifact) | PRD < 40%, R >= 0.85 |

---

## Graduation Thresholds

A firmware build is ready to ship when all of the following hold:

| Metric | Threshold | How Measured |
|--------|-----------|-------------|
| Fidelity | Pearson R > 0.85 | Held-out patients (chb15-chb20), golden path |
| Latency | < 4.0 ms per window | ADC DMA complete -> BLE TX start, measured via GPIO toggle |
| Stack usage | < 2.4 KB | Compile-time: `-Werror=stack-usage=2400` |
| Numerical parity | 0.000 drift | Python encoder vs C encoder on identical input |
| Compression ratio | >= 5.0x | At R >= 0.85, averaged over held-out set |
| Memory footprint | Encoder <= 43 KB | Linker map section sizes for `.sram4_tnn` |
| Boot self-test | 100% pass | KAT + CRC32 on every cold boot |
| Seizure preservation | PRD < 2.5% on ictal segments | Event-weighted evaluation on CHB-MIT annotations |

---

## Coverage Status

Current coverage by module:

| Module | Unit Tests | Integration | Notes |
|--------|-----------|-------------|-------|
| `train_ternary.py` (TernaryMobileNetV5) | Shape, gradient, ternary constraint | E2E benchmark | Core architecture |
| `export_firmware.py` | Packing, CRC, alpha range | C parity benchmark | Critical path |
| `edf_to_events.py` | Channel aliases, Q31 scaling | Dataset audit | Data pipeline |
| `ternary_mac.c` | KAT, LUT correctness | C parity benchmark | Ternary MAC core |
| `biquad_q31.c` | mul_q30, DC rejection, impulse | - | Filter cascade |
| `toeplitz_cs.c` | LFSR period, batch, seeds | - | Lightning path |
| `lifting_2d.c` | Round-trip, energy compaction | - | Lightning path |
| `lpc_predictor.c` | Residual reduction, edge cases | - | Lightning path |
| `hybrid_entropy.c` | Golomb-Rice round-trip | E2E golden path | Entropy coding |
| `integrity.c` | CRC32 cross-platform parity | Boot self-test | Safety-critical |
| `focal_modulation.c` | - | C parity benchmark | Needs unit tests |
| `scheduler.c` | - | - | Needs integration tests |
| `fsq.c` | Grid symmetry, bounds | - | Adaptive quantizer |

---

## Running All Tests

```bash
# Python unit tests
pip install -e ".[test]"
pytest tests/ -v --tb=short

# C host tests
gcc -I firmware tests/c_host/test_c_firmware.c -o test_fw -lm && ./test_fw

# Integration benchmarks (requires trained model + dataset)
python tests/benchmarks/test_golden_path_e2e.py
python tests/benchmarks/benchmark_decoder_e2e.py

# Coverage report
pytest tests/ --cov=ai_models --cov=firmware --cov-report=term-missing
```
