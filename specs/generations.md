# LamQuant Version History: Gen 1-6

A comprehensive technical record of the LamQuant neural EEG codec across six
generations of development, from a pure wavelet compressor to a production
ternary neural codec targeting the RP2350 Hazard3 RISC-V microcontroller.

*Generated 2026-04-24 from backup archives on /mnt/2tb*

---

## Table of Contents

- [Gen 1 -- Pure Wavelet Codec](#gen-1----pure-wavelet-codec)
- [Gen 2 -- Embedded Neural Semantic Encoder](#gen-2----embedded-neural-semantic-encoder)
- [Gen 3 -- Neural Predictor + Spatial Decorrelation](#gen-3----neural-predictor--spatial-decorrelation)
- [Gen 4 -- Teacher-Student Knowledge Distillation](#gen-4----teacher-student-knowledge-distillation)
- [Gen 5 -- Quantization-Aware Training](#gen-5----quantization-aware-training-qat)
- [Gen 6 -- Ternary Neural Codec with Entropy Coding](#gen-6----ternary-neural-codec-with-entropy-coding)
- [Gen 7 -- Production Lossless + Neural Codec System](#gen-7----production-lossless--neural-codec-system)
- [Generation Comparison](#generation-comparison)

---

## Gen 1 -- Pure Wavelet Codec

**Location:** `/mnt/2tb/Gen1 Archive/LamQuant Gen 1/`
**Date:** ~March 2026
**Language:** C (ISO C99)
**Target:** Desktop simulation / embedded prototype

### 1.1 Overview

The first generation was a time-budgeted wavelet codec for real-time EEG
compression. The key idea: try progressively more expensive wavelet transforms
within a hard deadline, always keeping a cheaper fallback ready.

### 1.2 Architecture

Three-tier cascade, each independent (not refinements of each other):

| Tier | Wavelet | Filter Taps | Quality | Relative Cost |
|------|---------|-------------|---------|---------------|
| 0    | Haar (db1) | 2        | Blocky  | 1x            |
| 1    | Daubechies-4 | 8     | Good    | ~4x           |
| 2    | Daubechies-8 | 16    | Smooth  | ~8x           |

If the deadline expires before a tier finishes, the system ships the previous
tier's result. If no tier completes, raw quantized ADC samples are sent.

### 1.3 Data Structures

```c
typedef enum {
    LAMQUANT_RAW = -1,   // no transform, raw ADC output
    LAMQUANT_HAAR = 0,   // tier 0: Haar
    LAMQUANT_DB4  = 1,   // tier 1: db4
    LAMQUANT_DB8  = 2,   // tier 2: db8
} LamQuantTier;

typedef struct {
    LamQuantTier tier_achieved;   // which tier completed
    int          levels;          // decomposition levels applied
    int          zeroed;          // coefficients zeroed by thresholding
    long long    time_ns;         // total processing time in nanoseconds
    double       time_budget_pct; // percentage of budget consumed
} LamQuantResult;

typedef struct {
    const char   *name;
    const double *lo;     // low-pass decomposition filter
    const double *hi;     // high-pass decomposition filter
    int           filt_len;
} WaveletTier;

typedef struct {
    double *data;
    size_t  capacity;
} Scratch;
```

### 1.4 Filter Coefficients

**Haar (2-tap):**
```
lo = [0.7071067811865476,  0.7071067811865476]    // 1/sqrt(2)
hi = [-0.7071067811865476, 0.7071067811865476]
```

**Daubechies-4 (8-tap):**
```
lo = [-0.010597, 0.032883, 0.030841, -0.187035,
      -0.027984, 0.630881, 0.714847,  0.230378]
hi = [-0.230378, 0.714847, -0.630881, -0.027984,
       0.187035, 0.030841, -0.032883, -0.010597]
```

**Daubechies-8 (16-tap):**
```
lo = [-0.000117,  0.000677, -0.000392, -0.004870,
       0.008746,  0.013981, -0.044088, -0.017369,
       0.128747,  0.000472, -0.284016, -0.015829,
       0.585355,  0.675631,  0.312872,  0.054416]
hi = [-0.054416,  0.312872, -0.675631,  0.585355,
       0.015829, -0.284016, -0.000472,  0.128747,
       0.017369, -0.044088, -0.013981,  0.008746,
       0.004870, -0.000392, -0.000677, -0.000117]
```

### 1.5 Core Functions

**`lamquant_encode(buf, len, deadline_ns, rel_thresh, scratch)`**
- Entry point. Takes ADC buffer, deadline, threshold, scratch.
- Saves backup of raw buffer in scratch.
- For each tier (Haar -> db4 -> db8):
  - Checks remaining budget. If tier > 0, estimates cost =
    `haar_measured_time * (filt_len / 2) * 1.5` (safety margin).
  - Restores raw data from backup.
  - Runs `fwt_forward` with this tier's filters.
  - Applies `threshold_coeffs` (relative thresholding).
  - Records tier, levels, zeroed count.
- Each tier transforms the ORIGINAL raw data (not prior tier output).
- Returns `LamQuantResult`.

**`fwt_forward(buf, len, levels, scratch, lo, hi, filt_len)`**
- Generic Mallat decomposition with periodic boundary extension.
- Per level: convolve with lo/hi filters, downsample by 2.
- Boundary wrap via bitmask: `(2k + j) & (w - 1)` (no modulo).
- Output layout: `[approx | detail_L | ... | detail_1]`.
- Returns number of levels applied.

**`fwt_inverse(buf, len, levels, scratch, lo, hi, filt_len)`**
- Upsample + convolve + accumulate (transpose of forward).
- Zero-clears scratch, accumulates `scratch[pos] += a*lo[j] + d*hi[j]`.

**`threshold_coeffs(buf, len, levels, rel_thresh)`**
- Scans detail region `[approx_len, len)` for peak absolute value.
- Cutoff = `peak * rel_thresh`.
- Zeros coefficients below cutoff. Approximation never touched.

**`compute_prd(orig, recon, len)`**
- PRD = `100 * sqrt(sum((orig-recon)^2) / sum(orig^2))`.

**`get_time_ns()`** / **`elapsed_ns(start)`**
- `CLOCK_MONOTONIC` via `clock_gettime`. Nanosecond precision.

### 1.6 Haar Module (`haar_fwt.c`)

Standalone Haar-only implementation with identical API but using the simpler
sum/difference formula:
```
Forward:  approx[k] = (even + odd) * 0.5
          detail[k] = (even - odd) * 0.5
Inverse:  even = approx + detail
          odd  = approx - detail
```
Sequential memory access, no multiplies (0.5 is free in IEEE 754).

### 1.7 Quantize Module (`quantize.c`)

ADC simulation for benchmarking:

```c
typedef struct {
    double *samples;
    size_t  num_samples;
    double  v_min, v_max;
    double  step_size;
    int     bit_depth;
} QuantizedSignal;

QuantizedSignal uniform_quantize(const double *analog_in,
                                  size_t analog_len,
                                  size_t num_points,
                                  int bit_depth);
```

Two passes: (1) find voltage range, (2) downsample and quantize via
`level = round(normalized * (2^bit_depth - 1))`.

### 1.8 Benchmark (`bench_eeg.c`)

Self-contained benchmark with all FWT code inlined. Loads real PhysioNet EEG
data from CSV, runs all three wavelets at multiple thresholds, reports PRD, SNR,
max error, and compression ratio. Finds optimal threshold for PRD < 1%
(clinical near-lossless).

`run_benchmark.py` automates: download PhysioNet `s10_ex08.csv` (Cz channel) ->
extract column -> `gcc -DLAMQUANT_TEST -O2` -> run.

### 1.9 Key Design Decisions

- **No allocation in hot path:** Scratch buffer pre-allocated, reused.
- **Relative thresholding:** Scales with signal amplitude automatically.
- **Power-of-2 only:** Enables bitmask wrapping instead of modulo.
- **Predictive cost estimation:** Measures Haar, extrapolates db4/db8 cost.
- **Perfect reconstruction:** Zero loss before thresholding (PRD ~1e-10%).

### 1.10 Performance

~2-5:1 compression at PRD < 1% (clinical near-lossless).

### 1.11 File Index

| File | Lines | Purpose |
|------|-------|---------|
| `lamquant.c` | ~650 | Pipeline orchestrator, tiered cascade, time budget |
| `haar_fwt.c` | ~400 | Standalone Haar forward/inverse/threshold |
| `daub_fwt.c` | ~600 | db4 + db8 via generic Mallat engine |
| `quantize.c` | ~170 | ADC simulation (uniform quantization) |
| `bench_eeg.c` | ~370 | PhysioNet EEG benchmark harness |
| `run_benchmark.py` | ~55 | Automation: download, compile, run |

---

## Gen 2 -- Embedded Neural Semantic Encoder

**Location:** `/mnt/2tb/Gen2 Archive/lamquant-v2 backup/`
**Date:** ~March 2026
**Language:** C (embedded), Python (host tools)
**Target:** RP2350 Hazard3 RISC-V + ADS1299 ADC + BLE 5.3
**License:** GPLv3 (Cognitive Liberty clause)

### 2.1 Overview

Gen 2 brought the wavelet codec onto real embedded hardware with a full signal
acquisition pipeline, BLE streaming, and a learned spike dictionary for semantic
compression. Complete bare-metal firmware, no RTOS.

### 2.2 Pipeline

```
ADS1299 (SPI, 4MHz) -> Triple Buffer (DMA/CPU/BLE)
  -> Haar/db4/db8 FWT -> MAD Threshold -> Polymorphic Packing -> BLE 5.3 PDU
```

### 2.3 Wire Protocol

**32-bit Packet Structure:**

```
Bit 31:      META flag (1=metadata, 0=data)
Bits 30-28:  MODE (3 bits: 0=Sniper, 1=DoubleTap, 2=Shotgun, 3=Atom)
Bits 27-25:  CHANNEL (3 bits: 0-7)
Bits 24-12:  DELTA (13 bits: time since last packet, 0-8191)
Bits 11-0:   PAYLOAD (12 bits: polymorphic interpretation)
```

**Masks:**
```c
#define LQ_META_MASK    0x01U     // 1 bit
#define LQ_MODE_MASK    0x07U     // 3 bits
#define LQ_CHANNEL_MASK 0x07U     // 3 bits
#define LQ_DELTA_MASK   0x1FFFU   // 13 bits
#define LQ_PAYLOAD_MASK 0x0FFFU   // 12 bits
```

**Encoding Modes:**

| Mode | Name | Payload | Q-Format | Scale | Coefficients |
|------|------|---------|----------|-------|--------------|
| 0 | Sniper | 12-bit signed | Q1.11 | 2048.0 | 1 large |
| 1 | DoubleTap | 2x 6-bit signed | Q1.5 | 32.0 | 2 medium |
| 2 | Shotgun | 3x 4-bit signed | Q2.2 | 4.0 | 3 small |
| 3 | Atom | 8-bit ID + 4-bit gain | N/A | N/A | 64-sample template |

**Coefficient Classification (normalized to max_abs):**
- CLS_ZERO: `< zero_thresh` -> skip via run-length encoding
- CLS_SMALL: `< 0.03125` (1/32) -> Shotgun candidate
- CLS_MEDIUM: `0.03125 - 0.5` -> DoubleTap candidate
- CLS_LARGE: `>= 0.5` -> Sniper or Atom candidate

**Batch Header (8 bytes, big-endian):**
```
Byte 0:   Sequence number
Bytes 1-4: Absolute timestamp (32-bit)
Byte 5:   Number of packets
Bytes 6-7: Fletcher-16 checksum
```

**BLE PDU:** Header (8B) + up to 58 x 4B packets = 240 bytes max.

### 2.4 Data Structures

```c
typedef enum {
    LQ_TIER_RAW = -1, LQ_TIER_HAAR = 0,
    LQ_TIER_DB4 = 1,  LQ_TIER_DB8 = 2,
} LQTier;

typedef struct {
    LQTier   tier;
    int      levels;
    int      zeroed;
    int64_t  encode_ns;
    double   mad_thresh;
} LQEncodeResult;

typedef struct {
    uint32_t *packets;
    size_t    num_packets, capacity;
    int       sniper_count, doubletap_count, shotgun_count;
    int       meta_count, zeros_skipped;
    double    norm_factor;
} LQPacketStream;

typedef struct {
    int       is_meta, mode, channel, time_delta;
    double    coefficients[3];
    int       num_coefficients;
    uint16_t  meta_payload;
} LQDecodedPacket;

// Neural atom for semantic dictionary
typedef struct __attribute__((packed, aligned(4))) {
    int16_t waveform[64];       // 64-sample spike template, Q1.11
    int16_t peak_amplitude;     // max |waveform[i]|
    int16_t reserved;           // 4-byte alignment padding
} NeuralAtom;   // 132 bytes each, 256 atoms = 33.75 KB
```

### 2.5 Haar Lifting Scheme (Integer-Only)

Gen 2 replaced Gen 1's convolution-style Haar with an in-place integer lifting
scheme. No floating-point multiplies in the hot path.

```
Q1.16 fixed-point (LIFT_SCALE = 65536.0)

Forward (per level):
  Split: even = buf[2k], odd = buf[2k+1]
  PREDICT: detail[k] = odd - even
  UPDATE:  approx[k] = even + (detail >> 1)   // arithmetic shift = divide by 2

Inverse (per level):
  UNDO UPDATE:  even = approx - (detail >> 1)
  UNDO PREDICT: odd  = detail + even
  Interleave:   buf[2k] = even, buf[2k+1] = odd
```

**Why integer?** RP2350 Hazard3 has no FPU. Soft-float emulation for
multiply costs ~30 cycles vs 1 cycle for add/shift.

### 2.6 MAD-Based Thresholding (Quantizer Module)

```c
double lq_compute_mad_threshold(const double *coeffs,
                                 size_t start, size_t len,
                                 double sigma, LQScratch *sc);
```

Algorithm:
1. Extract absolute values of detail coefficients `[start, len)`.
2. Sort via `qsort`.
3. Median = middle value (or average of two middle values).
4. **Donoho-Johnstone formula:** `threshold = sigma * median / 0.6745`
   - 0.6745 normalizes MAD to Gaussian standard deviation.
   - sigma=1.0: moderate, sigma=2.0: aggressive, sigma=0.5: conservative.

**Why MAD over standard deviation?** One eye blink can be 10x the noise floor.
Standard deviation gets corrupted by outliers; MAD (median) does not.

### 2.7 Semantic Atom Matching (Packer Module)

```c
static int lq_semantic_lookup(const double *coeffs, size_t max_len,
                               double max_abs,
                               uint8_t *out_atom, int8_t *out_gain);
```

Algorithm:
1. Convert 64-sample window to Q1.11 fixed-point.
2. Compute spike peak amplitude and sum.
3. Iterate 256 atoms in dictionary:
   - **Hierarchical rejection:** If `|spike_peak - atom_peak| > 4000`, skip.
   - **L1 distance (Manhattan):** 4x-unrolled loop.
     - On Hazard3: uses `sabs` custom instruction (Zbb extension) for
       branchless absolute difference.
     - Fallback: `diff = (diff < 0) ? -diff : diff`.
   - **Early exit:** If `current_dist >= min_dist`, break inner loop.
4. **Validation gate:** Only accept if `min_dist < 60000`.
5. Compute gain: `sum_amp / (spike_peak * 64)`, quantize to Q1.3 (4-bit).

**Atom dictionary:** 256 entries in `default_dictionary.h`, pre-trained via
K-means clustering on wavelet-domain EEG spikes (generated by
`host_tools/build_dictionary.py`).

### 2.8 Ingestor Module (Hardware Interface)

**Pin assignments:**
```c
ADS_SPI_PORT = spi0, ADS_SPI_BAUD = 4000000 (4 MHz)
PIN_SCK=2, PIN_MOSI=3, PIN_MISO=4, PIN_CS=5
PIN_DRDY=6, PIN_RESET=7, PIN_PWDN=8
```

**ADS1299 configuration:**
- CONFIG1 (0x01) = 0x96: HR mode, 250 SPS
- CONFIG2 (0x02) = 0xC0: Internal clock, test signal off
- CONFIG3 (0x03) = 0xEC: Internal VREF 4.5V, bias enabled
- CHnSET (0x05-0x0C) = 0x60: 24x gain, normal input

**SPI packet:** 3 status bytes + 8 channels x 3 bytes = 27 bytes per DRDY cycle.
Each channel is 24-bit two's complement MSB-first.

**LSB calculation:** `(2 * 4.5V) / (24 * 2^24) = 0.02235 uV`

**Triple-buffer architecture:**
```c
typedef enum { BUF_FREE, BUF_DMA_FILL, BUF_CPU_PROC, BUF_BLE_TX } BufState;
```
- Buffer A: DMA fills from SPI RX.
- Buffer B: CPU runs FWT + threshold + packing.
- Buffer C: BLE stack transmits.
- DMA interrupt handler rotates buffers. If all busy, steals BLE buffer
  (fresh data > old data).

**PWM clock bridge:** Routes DRDY to PWM counter (PWM_DIV_B_RISING, wrap=0xFFFF).
CPU reads atomically -- eliminates metastability from direct GPIO sampling across
clock domains (250 Hz ADC vs 150 MHz CPU).

### 2.9 Main Loop

```c
#define TARGET_CHANNEL     0
#define MAD_SIGMA          1.0
#define ENCODE_DEADLINE_NS 500000000LL   // 500 ms
#define PACKER_ZERO_THRESH 0.001
```

Loop: `__wfi()` until buffer ready -> `lq_encode_tiered()` -> 
`lq_pack_coefficients()` -> assemble BLE PDUs (serialize packets, Fletcher-16
checksum, serialize header) -> `ble_transmit_pdu()` -> `lq_release_cpu_buffer()`.

At 250 SPS with 256-sample buffers: 1.024 seconds per iteration, 500 ms encode
deadline (50% headroom).

### 2.10 Build System

```cmake
# Firmware
target_compile_options(lamquant_fw PRIVATE
    -O3 -march=rv32imac_zba_zbb -ffast-math)
target_link_libraries(lamquant_fw
    pico_stdlib hardware_spi hardware_dma
    hardware_pwm hardware_timer hardware_irq pico_multicore)

# Desktop tests
target_link_libraries(lamquant_test m)
```

`-march=rv32imac_zba_zbb`: Enables Zba (shift-add) and Zbb (bit-manipulation,
including `sabs`) extensions.

### 2.11 Performance

~20-30:1 compression at PRD < 5%. Bare-metal, no malloc in hot path.

### 2.12 File Index

| File | Lines | Purpose |
|------|-------|---------|
| `firmware/inc/lamquant.h` | ~220 | Wire protocol, types, filter externs |
| `firmware/inc/default_dictionary.h` | ~270 | 256 pre-learned neural atoms |
| `firmware/src/main.c` | ~310 | State machine main loop |
| `firmware/src/lifter.c` | ~465 | Haar lifting + db4/db8 convolution + tiered encoder |
| `firmware/src/quantizer.c` | ~140 | MAD threshold + hard thresholding |
| `firmware/src/packer.c` | ~476 | Polymorphic packing, atom matching, Fletcher-16 |
| `firmware/src/ingestor.c` | ~476 | ADS1299 SPI driver, triple-buffer DMA, PWM bridge |
| `tests/test_pipeline.c` | ~200 | Synthetic round-trip test |
| `tests/test_real.c` | ~150 | PhysioNet EEG benchmark |
| `CMakeLists.txt` | ~100 | Dual-target: RP2350 firmware or desktop test |

---

## Gen 3 -- Neural Predictor + Spatial Decorrelation

**Location:** `/mnt/2tb/Gen3 Archive/` (firmware) + `/mnt/2tb/BrainyAI/` (training)
**Date:** ~March 2026
**Language:** C (firmware), Python/PyTorch (training)
**Target:** RP2350 Hazard3

### 3.1 Overview

Gen 3 added neural intelligence: a causal Conv1d predictor that learns to predict
EEG from recent history, so only the (much smaller) residuals need compression.
Two variants: base (lamquant-v3, same as Gen 2) and full (lamquantFull-v3, with
neural pipeline).

### 3.2 Full Pipeline

```
Raw EEG (21ch) -> Prefilter (IIR bandpass 0.5-100Hz + 60Hz notch)
  -> PCA (Jacobi eigendecomposition, 95% energy threshold)
  -> Neural Predictor (3-layer causal Conv1d, INT8)
  -> Residual = Actual - Predicted
  -> Haar/db4/db8 FWT -> MAD Threshold -> Atom Matching
  -> Polymorphic Packing -> BLE 5.3 PDU
```

### 3.3 New Data Structures

```c
// Prefilter
typedef struct {
    double b0, b1, b2, a1, a2, d1, d2;  // biquad Direct Form II
} LQBiquad;

typedef struct {
    LQBiquad highpass[21], lowpass[21], notch[21];
    int num_channels, initialized;
} LQPrefilter;

// PCA
typedef struct {
    int    num_channels, num_keep;
    double mean[21], eigenvalues[21], V[21*21];
    double energy_kept, energy_total;
} LQPca;

// Predictor
#define LQ_PRED_CHANNELS 21
#define LQ_PRED_WINDOW   15
#define LQ_PRED_HIDDEN   16

typedef struct {
    float h1[15*16], h2[15*16], h3[15*16], output[15*21];
} LQPredictorState;

// Atom encoder
typedef struct {
    const NeuralAtom *dictionary;
    int num_atoms, atom_samples;
    double match_threshold, peak_reject_ratio;
} LQAtomEncoder;

typedef struct {
    int matched, atom_id;
    double gain, distance;
} LQAtomMatch;
```

### 3.4 Prefilter Module

Cascaded biquad IIR (Direct Form II transposed):

```c
static inline double bq_process(LQBiquad *bq, double x) {
    double y = bq->b0 * x + bq->d1;
    bq->d1 = bq->b1 * x - bq->a1 * y + bq->d2;
    bq->d2 = bq->b2 * x - bq->a2 * y;
    return y;
}
```

**Hardcoded filter coefficients (fs=250 Hz):**
```
Highpass 0.5 Hz (Butterworth):
  b0=0.993722, b1=-1.987444, b2=0.993722, a1=-1.987389, a2=0.987500

Lowpass 100 Hz (Butterworth):
  b0=0.467108, b1=0.934216, b2=0.467108, a1=0.0, a2=0.131371

Notch 60 Hz (Q=30):
  b0=0.951057, b1=-0.309017, b2=0.951057, a1=-0.309017, a2=0.902113
```

Processing order: Highpass -> Notch -> Lowpass (per sample, per channel).

### 3.5 PCA Module

**`lq_pca_fit(pca, data, num_samples, num_channels, num_keep)`**
1. Compute mean per channel.
2. Build covariance matrix (upper triangle).
3. **Jacobi eigendecomposition:** Iterative Givens rotations to diagonalize,
   max 50 sweeps, convergence threshold 1e-15 (off-diagonal Frobenius norm).
4. Sort eigenvalues descending.
5. If `num_keep <= 0`: auto-select at 95% cumulative energy threshold
   (typically keeps 8-12 of 21 components).

**`lq_pca_forward(pca, in, out)`:** `out[k] = sum((in[c] - mean[c]) * V[c*N+k])`
**`lq_pca_inverse(pca, in, out, nc)`:** `out[c] = mean[c] + sum(in[k] * V[c*N+k])`

### 3.6 Neural Predictor

**Architecture:**
```
Input(21ch, 15 samples)
  -> CausalConv1d(21->16, k=3, dil=1) + ReLU6
  -> CausalConv1d(16->16, k=3, dil=2) + ReLU6
  -> CausalConv1d(16->16, k=3, dil=4) + ReLU6
  -> Conv1d(16->21, k=1)
Output: predicted next sample for all 21 channels
```

Receptive field = `(3-1)*(1+2+4) + 1 = 15` samples = 60 ms at 250 Hz.

**C inference (`predictor.c`):**

```c
static void conv1d_int8(const float *in, const int8_t *w, const float *b,
                        const float *scales, float *out,
                        int in_c, int out_c, int k_s, int dil, int len);
```

Per timestep t, per output channel oc:
- `sum = b[oc]`
- For each input channel ic, kernel position k:
  - Causal index: `t_idx = t - (k_s - 1 - k) * dilation`
  - If `t_idx >= 0`: `sum += in[t_idx*in_c + ic] * (w[oc*in_c*k_s + ic*k_s + k] / scales[oc])`
- `out[t*out_c + oc] = sum`

**Residual computation:**
```c
void lq_compute_residuals(const float *actual, const float *predicted,
                           float *residuals, int num_channels);
// residuals[c] = actual[c] - predicted[c]
```

### 3.7 Model Weights (`model_weights.h`)

INT8 quantized with per-channel float scales:

```
Layer 1: l1_scales[16], l1_w[1008] (21x16x3 int8), l1_b[16] (float)
Layer 2: l2_scales[16], l2_w[768]  (16x16x3 int8), l2_b[16] (float)
Layer 3: l3_scales[16], l3_w[768]  (16x16x3 int8), l3_b[16] (float)
Output:  out_scales[21], out_w[336] (16x21x1 int8), out_b[21] (float)
```

Total: ~2,880 int8 weights + 69 float scales + 69 float biases = ~3 KB.

### 3.8 Atom Module (Full variant)

```c
#define ATOM_MATCH_THRESH 0.15   // normalized distance threshold

LQAtomMatch lq_atom_match(const LQAtomEncoder *enc,
                           const double *segment, int segment_len);
```

Non-overlapping 64-sample window scan. Per window:
1. Compute input peak, energy.
2. Iterate atoms, skip if `atom_peak < 1e-10`.
3. Gain = `input_peak / atom_peak`, reject if `|gain - 1| > 3`.
4. L1 distance normalized by input energy.
5. Accept if distance < 0.15.

### 3.9 Training (PyTorch)

**`host_tools/train.py`:**

```python
class PicoPredictor(nn.Module):
    # 21->16->16->16->21, dilations [1,2,4], ReLU6
    # MSE loss on last timestep, Adam lr=0.001, 50 epochs, batch=4096
    # Output: pico_predictor_gen2.pth
```

**`host_tools/evaluate.py`:**
Sliding 128-sample window, compute `var(original) / var(residual)` = variance
reduction (typically 5-8x -> 50-70x bitrate reduction with FWT).

**Calibration suite (`BrainyAI/`):**
- `LamQuant_Gen3.py`: KL-divergence calibration with 99.99% percentile scaling,
  empirical drift correction by simulating float vs quantized paths.
- `calibrate.py`: Basic 99.99 percentile.
- `calibrate_v3.py`: + bias drift correction.
- `calibrate_parity.py`: Bit-exact final-timestep matching.
- `calibrate_surgical.py`: Full ground-truth corrected biases.
- `calibrate_pro.py`: Per-channel scaling.
- `calibrate_hybrid.py`: INT8 hidden, FP32 output.
- `export_weights.py`: Simple global-scale INT8 export.
- `verify_math.py`: Generate ground-truth test case for C parity.

### 3.10 Differences: v3 vs Full-v3

| Feature | v3 (base) | Full-v3 |
|---------|-----------|---------|
| Prefilter | None | IIR HP + Notch + LP |
| PCA | None | Jacobi, 95% energy |
| Predictor | Sequential only | Multi-channel neural |
| Atoms | Conditional | Full integration |
| Source files | 6 | 9 (+prefilter, pca, atoms) |
| Compression | ~20-30:1 | ~50-70:1 |
| CPU load | ~5% | ~20% |

### 3.11 Performance

- Base: ~20-30:1 compression (same as Gen 2)
- Full: ~50-70:1 compression at PRD < 3%
- Predictor: ~2ms inference per 256-sample chunk on Hazard3

---

## Gen 4 -- Teacher-Student Knowledge Distillation

**Location:** `/mnt/2tb/BrainyAI_Filtered/`
**Date:** ~March 2026
**Language:** Python/PyTorch (training), C (firmware unchanged from Gen 3 Full)
**Target:** RP2350 Hazard3

### 4.1 Overview

Gen 4 introduced a two-stage training pipeline: train a large unconstrained
"teacher" model, then distill its knowledge into a tiny hardware-compatible
"student." The student switched to depthwise separable convolutions.

### 4.2 Teacher: EEGOptimusPrime

```python
class EEGOptimusPrime(nn.Module):
    # 6 layers, 256 hidden, dilations [1,2,4,8,16,32]
    # Input: [batch, 21, 128], Output: [batch, 21] (final timestep)
    layer1 = CausalConv1d(21,  256, k=3, dil=1)   # + ReLU
    layer2 = CausalConv1d(256, 256, k=3, dil=2)   # + ReLU
    layer3 = CausalConv1d(256, 256, k=3, dil=4)   # + ReLU
    layer4 = CausalConv1d(256, 256, k=3, dil=8)   # + ReLU
    layer5 = CausalConv1d(256, 256, k=3, dil=16)  # + ReLU
    layer6 = CausalConv1d(256, 256, k=3, dil=32)  # + ReLU
    out    = Conv1d(256, 21, k=1)
```

Receptive field = `(3-1)*(1+2+4+8+16+32) + 1 = 127` samples = 508 ms.

**Training:** MSE loss, Adam lr=0.001, StepLR(step=20, gamma=0.5), 25 epochs,
batch=2048. Ctrl+C safe (saves weights in `finally` block). Output: `teacher_model.pth`.

### 4.3 Student: Gen4MobileNetStudent

```python
class Gen4MobileNetStudent(nn.Module):
    # Depthwise separable, 32 hidden, dilations [1,2,4]
    # Input: [batch, 21, 15], Output: [batch, 21]
    l1_dw = Conv1d(21, 21, k=3, pad=2, dil=1, groups=21, bias=False)  # depthwise
    l1_pw = Conv1d(21, 32, k=1, bias=True)                             # pointwise
    # -> ReLU6, crop [:-2]

    l2_dw = Conv1d(32, 32, k=3, pad=4, dil=2, groups=32, bias=False)
    l2_pw = Conv1d(32, 32, k=1, bias=True)
    # -> ReLU6, crop [:-4]

    l3_dw = Conv1d(32, 32, k=3, pad=8, dil=4, groups=32, bias=False)
    l3_pw = Conv1d(32, 32, k=1, bias=True)
    # -> ReLU6, crop [:-8]

    out = Conv1d(32, 21, k=1)  # -> return [:, :, -1]
```

Depthwise separable = temporal filtering (per-channel) and channel mixing
(pointwise) decoupled. ~6x fewer MACs than standard Conv1d.

### 4.4 Distillation Dataset

```python
class DistillationDataset(Dataset):
    # teacher_window=128, student_window=15
    # Teacher sees full 128 samples, student only last 15
    # Both predict same next sample
    x_teacher = data[:, idx : idx+128]         # [21, 128]
    x_student = x_teacher[:, -15:]             # [21, 15]
    y_true    = data[:, idx+128]               # [21]
```

### 4.5 Distillation Training

```python
# Dual loss: 50% ground truth, 50% teacher soft targets
loss = 0.5 * MSE(student_pred, y_true) + 0.5 * MSE(student_pred, teacher_pred)

# Hyperparameters
optimizer = AdamW(lr=3e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=30)
epochs = 30, batch = 4096, num_workers = 12
```

Teacher frozen (`eval()`, `requires_grad=False`). Output: `lamquant_student_gen4.pth`.

### 4.6 Dataset Scale-Up

Expanded from 36 subjects (Gen 3) to the full CHB-MIT PhysioNet dataset
(~109 subjects, 64-channel Sharbrough montage).

**Data pipeline:**
```
Raw EDF -> convert_edf.py -> .npy (Volts, all channels)
  -> preprocess_all.py -> processed_npy/*.npy (21ch, 250Hz, bandpass 0.5-50Hz)
  -> prep_dataset.py -> *_filtered.npy (C-engine biquad filtered)
```

Channel selection (standard 10-20): Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2,
F7, F8, T7, T8, P7, P8, Fz, Cz, Pz, A1, A2. Old naming handled:
T3->T7, T4->T8, T5->P7, T6->P8.

### 4.7 C Inference Engine

Gen 4's `predictor.c` uses depthwise separable convolutions:

```c
// Depthwise: one filter per channel, no mixing
void conv1d_depthwise(float *in, const int8_t *w, const float *scales,
                      float *out, int channels, int k_s, int dil, int len);

// Pointwise: 1x1 conv, mixes channels
void conv1d_pointwise(float *in, const int8_t *w, const float *b,
                       const float *scales, float *out,
                       int in_c, int out_c, int len);
```

### 4.8 What Changed from Gen 3

- **Depthwise separable architecture:** ~6x fewer MACs.
- **Knowledge distillation:** Teacher's 128-sample context enables the 15-sample
  student to learn patterns it couldn't discover alone.
- **109-subject dataset** vs 36.
- **7 calibration methods** for different precision/speed tradeoffs.

---

## Gen 5 -- Quantization-Aware Training (QAT)

**Location:** `/mnt/2tb/BrainyAI_Filtered/train_sota.py`
**Date:** ~March 2026
**Language:** Python/PyTorch
**Target:** RP2350 Hazard3

### 5.1 Overview

Gen 5 closed the quantization gap by training with simulated INT8 quantization
in the forward pass. Previous generations trained in float32 and post-hoc
quantized, introducing drift. Gen 5 uses a Straight-Through Estimator (STE) to
backpropagate through the rounding operation.

### 5.2 FakeQuant

```python
class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        max_val = weight.abs().amax(dim=(1, 2), keepdim=True)
        scale = 127.0 / max_val.clamp(min=1e-5)
        q_w = torch.round(weight * scale).clamp(-128, 127)
        return q_w / scale   # dequantize back to float for PyTorch math

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output   # Straight-Through Estimator
```

Per-channel INT8 scaling (matches C engine). `round -> clamp` in forward,
gradient passes through unchanged in backward.

### 5.3 Model: PicoPredictorSOTA

Same depthwise separable architecture as Gen 4's student, but with manual weight
parameters (not `nn.Conv1d`) so `FakeQuant` is applied at every forward pass:

```python
class DepthwiseCausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dil):
        self.weight_dw = nn.Parameter(torch.randn(in_ch, 1, k) * 0.1)
        self.weight_pw = nn.Parameter(torch.randn(out_ch, in_ch, 1) * 0.1)
        self.bias_pw   = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        qw_dw = FakeQuant.apply(self.weight_dw)   # simulate INT8
        qw_pw = FakeQuant.apply(self.weight_pw)
        x = F.conv1d(x, qw_dw, ..., groups=in_ch)  # depthwise
        x = x[:, :, :-padding]                       # causal crop
        x = F.conv1d(x, qw_pw, bias=self.bias_pw)   # pointwise
        return x
```

### 5.4 Compression-Optimized Loss

```python
def compression_loss(pred, target):
    huber = F.huber_loss(pred, target, delta=2.0)   # robust to blink outliers
    var_penalty = torch.var(target - pred)           # minimize residual variance
    return huber + 0.2 * var_penalty
```

Two key changes from MSE:
1. **Huber (delta=2.0):** Quadratic for small errors, linear for large (blinks).
2. **Variance penalty (0.2x):** Directly optimizes for downstream wavelet
   compressibility, not just prediction accuracy.

### 5.5 Training

```
Adam lr=0.002, StepLR(step=20, gamma=0.5), 100 epochs, batch=4096
Data: *_filtered.npy files
Output: pico_sota.pth
```

### 5.6 What Changed from Gen 4

- **No quantization gap:** Model trains under same INT8 constraints as hardware.
- **Compression-aware loss:** Directly optimizes for downstream FWT compression.
- **Huber robustness:** Giant outliers no longer corrupt weight updates.

---

## Gen 6 -- Ternary Neural Codec with Entropy Coding

**Location:** `legacy.7z` -> `LamQuant-Old.7z` / `LamQuant-main.zip`
**Date:** ~March-April 2026
**Language:** Python/PyTorch (training), C (firmware)
**Target:** RP2350 Hazard3 (64KB SRAM, < 4ms latency)
**Repo:** `github.com/Quitetall/lamquant_gen6`
**License:** Firmware GPLv3, Transport AGPLv3, Weights CC BY-NC 4.0

### 6.1 Overview

Gen 6 was a ground-up redesign: from predictor-residual to full autoencoder.
The model encodes EEG into a compact latent space, quantizes via FSQ, entropy-
codes with rANS, and decodes on the receiver. Weights are ternary ({-1, 0, +1}),
enabling lookup-table MAC instead of multiplies.

### 6.2 Teacher Architecture: FP32OracleAutoEncoder

```python
class FocalModulationBlock(nn.Module):
    # Conv1d(in_ch, out_ch, k, padding=k//2) + GroupNorm(4, out_ch)
    # + residual shortcut (Conv1d(k=1) if dims differ, else Identity)
    # Forward: ReLU(GN(conv(x))) + shortcut(x)

class MobileNetV5Focal(nn.Module):     # Encoder
    focal1 = FocalModulationBlock(21 -> 64, k=7)
    focal2 = FocalModulationBlock(64 -> 128, k=5)
    focal3 = FocalModulationBlock(128 -> 256, k=3)
    bottleneck = Conv1d(256 -> 32, k=1)
    # [21, 2500] -> [32, 2500]

class DecoderBlock(nn.Module):         # Decoder
    expand1 = FocalModulationBlock(32 -> 256, k=3)
    expand2 = FocalModulationBlock(256 -> 128, k=5)
    expand3 = FocalModulationBlock(128 -> 64, k=7)
    output  = Conv1d(64 -> 21, k=1)
    # [32, 2500] -> [21, 2500]
```

### 6.3 Teacher Training

**Dataset: Q31Dataset**
- Loads CHB-MIT `.npz` files (21ch x 2500 samples + seizure_mask).
- Q31 normalization: `(int32 / 2147483647.0) * 1000.0`.
- Parallel ingestion (16 threads), persistent binary cache.
- Patient-wise train/val split (20% val).

**Loss: ClinicalHybridLoss**
```python
class EventWeightedMSELoss:
    # base_error * (1 + (penalty-1) * seizure_mask)
    # seizure_penalty_weight = 5.0 (5x on seizure regions)

class SpectralConvergenceLoss:
    # Multi-resolution STFT: fft_sizes=[64, 128, 256]
    # MSE on log10(magnitude), hop=n_fft//4

class ClinicalHybridLoss:
    # (1-alpha)*event_weighted_mse + alpha*spectral_convergence
    # alpha=0.1 (focus on waveform, spectral as regularizer)
```

**Training schedule:**
- 800 epochs total.
- Optimizer: AdamW (fused on CUDA), lr auto-scaled by hardware.
- Scheduler: CosineAnnealingWarmRestarts (T_0=50*batches, eta_min=1e-6).
- SWA starts epoch 600 (SWALR lr=5e-4), BatchNorm update at end.
- Gradient clip: max_norm=20.0.
- AMP: bfloat16 on Ampere+, float16 on older GPUs.
- torch.compile(mode="max-autotune") if available.
- Fidelity audit every 50 epochs (save/load/compare MSE parity).
- Full checkpoint saved every epoch (model + optimizer state).
- Best encoder saved to `teacher_best.ckpt`, decoder to `decoder_best.ckpt`.
- Supports `--resume`, `--logger wandb/mlflow`, `--seed`.

### 6.4 Student Architecture: TernaryMobileNetV5

```
Encoder (on-chip):
  Input [21, 2500]
  -> FocalModBlock(21->96, k=7, stride=1)  + GN(4) + ternary shortcut
  -> FocalModBlock(96->96, k=5, stride=2)  + GN(4) + strided shortcut
  -> FocalModBlock(96->96, k=3, stride=2)  + GN(4) + strided shortcut
  -> FocalModBlock(96->96, k=3, stride=2)  + GN(4) + strided shortcut
  -> Bottleneck Conv1d(96->32, k=1)
  Latent [32, 312]   (8x temporal compression: 2500/1/2/2/2 = 312.5)

Decoder (symmetric expansion)
```

96-wide channels, 4x temporal stride through focal2-4, ~200K parameters.

### 6.5 Ternary Quantization (LSQ)

Learned Step-size Quantization with Straight-Through Estimator:

```python
# Forward:
w_scaled = weight / (lsq_alpha + 1e-8)          # alpha is learnable per-channel
w_ternary = round(clamp(w_scaled, -1, 1))        # {-1, 0, +1}
w_out = w_ternary * lsq_alpha                    # scale back

# Backward: gradient flows through rounding (STE)
```

On hardware: each weight is 2 bits. MAC = lookup `{0, +activation, -activation}`.
No multiplier needed.

### 6.6 Student Training (3-Phase)

**Phase 1: Warm-up (50 epochs)**
- lr=2e-3, quantize=OFF (FP32).
- Loss: MSE only.
- Input: clamp [-50, 50], DC removal.
- Scheduler: CosineAnnealing(eta_min=5e-4).
- Grad clip: 10.0.

**Phase 2: Quantize-Aware (200 epochs)**
- lr=1e-3, quantize=ON (STE ternary weights).
- Loss: MSE only.
- Augmentations: montage permutation + clinical augmentation.
- Scheduler: CosineAnnealing(eta_min=1e-4).
- Grad clip: 10.0.
- Best model tracked by Pearson R.

**Phase 3: Fine-tune (250 epochs)**
- lr=2e-4, quantize=ON, + spectral loss.
- Loss: MSE + 0.1 * SpectralLoss(fft=[64,128,256]).
- Scheduler: CosineAnnealing(eta_min=1e-5).
- Grad clip: 5.0.

Total: 500 epochs, ~8 minutes on RTX 4090.
Output: `student_hardened.ckpt`.

### 6.7 TernaryCodec (Python Codec Interface)

```python
class TernaryCodec:
    def encode(self, x):     # [B, 21, T] -> [B, 32, T/8]
    def decode(self, latent): # [B, 32, T/8] -> [B, 21, T]
    def compress(self, latent):   # latent -> bytes (FSQ + rANS)
    def decompress(self, data):   # bytes -> latent
```

**FSQ (Finite Scalar Quantization):**
1. Normalize latents to [0, 1]: `(latent - vmin) / span`.
2. Quantize to L=16 levels: `symbols = clip(normalized * L, 0, L-1)`.

**rANS (range Asymmetric Numeral Systems):**
1. Build frequency table from symbol distribution.
2. Normalize frequencies to sum = `rans_total_freq` (4096).
3. Encode symbols in reverse: state transitions via `(state//freq)*total + start + (state%freq)`.
4. RANS_L = 1<<15 (32768).

**Wire format:** `'LAMQ'` (4B) + latent_dim (uint16) + latent_T (uint16) +
vmin (float32) + vmax (float32) + freq_table (L x uint16) + rANS bitstream.

### 6.8 Firmware Export

**`export_firmware.py`** generates C headers:

**Ternary weight packing (4 per byte):**
```
00 = 0, 01 = +1, 10 = -1, 11 = reserved
byte |= (bits << (2*j)) for j=0..3
```

**LSQ alphas:** Scaled to Q31 (`int(alpha * 2147483647)`).

**GroupNorm gamma/beta:** Q31 format.

**FSQ lattice:** `fsq_lattice.h` with `FSQ_LEVELS[4] = {3, 3, 3, 3}` (legacy)
or `{8, 6, 5, 5}` (training).

**Toeplitz CS seeds:** `toep_seeds.h`, 21 channels x 4 uint32 LFSR seeds
for compressed sensing reconstruction.

**CRC32:** `firmware_crc.h` with `0xE0EA5AC6`, binary size 3202 bytes.

### 6.9 C Firmware (`focal_modulation.c`)

**Constants:**
```c
#define MAX_T     2500    // input temporal length
#define MAX_CH    96      // max channels after focal1
#define MAX_LAT_T 312     // output latent T = 2500/8
static const int8_t TERNARY_LUT[4] = {0, 1, -1, 0};
```

**Global buffers (SRAM5):**
```c
int16_t act_buf_a[96][2500];      // activation double-buffer A
int16_t act_buf_b[96][2500];      // activation double-buffer B
int16_t shortcut_buf[96][2500];   // residual shortcut accumulator
int32_t latent_output[32][312];   // final latent output
```

**`ternary_conv1d_single(act, t_center, in_ch, kernel_size, groups, group_id, packed_weights, alpha_q31)`**
- Computes one output channel, one time step.
- Weight unpacking: `byte_pos = w_idx/4`, `bit_pos = (w_idx%4)*2`,
  `w = TERNARY_LUT[(byte >> bit_pos) & 0x03]`.
- MAC: `acc += (int16)activation * (int8)weight`.
- Output: `mul_q31(acc, alpha_q31)` -- Q31 multiply `(acc * alpha) >> 31`.

**`groupnorm_relu_inplace(buf, channels, T, gamma_q31, beta_q31)`**
- GroupNorm with groups=4.
- Per-group stats: sum, sum_sq, mean, variance (min=1).
- **Newton's method isqrt** (3 iterations) for inverse std.
- Normalize, scale by gamma, add beta, ReLU, saturate to int16.

**`run_focal_block(act_in, act_out, T_in, in_ch, out_ch, kernel_size, stride, groups, conv_weights, conv_alphas, sc_weights, sc_alphas, norm_gamma, norm_beta)`**
- Main conv path: ternary_conv1d_single per output channel and time step.
- Shortcut: ternary (k=1) if dimensions differ, identity otherwise.
- GroupNorm + ReLU in-place.
- Residual sum with saturation.
- Returns T_out = T_in / stride.

**`run_tnn_encoder_inference(input, T)`**
- Input: `[21][2500]` Q31 activations.
- Convert Q31 -> Q15: `act_buf_a[c][t] = (int16)(input[c][t] >> 16)`.
- focal1: 21->96, k=7, s=1.
- focal2: 96->96, k=5, s=2.
- focal3: 96->96, k=3, s=2.
- focal4: 96->96, k=3, s=2.
- bottleneck: 96->32, k=1, s=1 (no norm).
- Output: `latent_output[32][312]`.

### 6.10 Firmware Export Headers

**`focal_net_weights.h`:** 2850 bytes of packed ternary weights.

**`fsq_lattice.h`:**
```c
#define FSQ_BOUNDS_SIZE 16
static const int32_t FSQ_LEVELS[4] = { 3, 3, 3, 3 };
```

**`toep_seeds.h`:** 21 channels x 4 uint32 seeds (336 bytes):
```c
// ch0:  0x75533621, 0x1E5AEBA7, 0xB9019BC1, 0x179FC550
// ch1:  0x6A1C232D, 0xF6468243, 0x56B1389B, 0x7033279E
// ... (21 channels)
// ch20: 0x219B0BA0, 0xD8B0ABD0, 0x99C9FE26, 0x172EC531
```

**`firmware_crc.h`:**
```c
#define FIRMWARE_CRC32 0xE0EA5AC6u
#define FIRMWARE_BINARY_SIZE 3202
```

**`manifest.json`:**
```json
{
    "timestamp": "2026-03-31T21:46:24.923449",
    "target_mcu": "RP2350 (Hazard3 Dual Core)",
    "memory_pinned": true,
    "exported_headers": ["focal_net_weights.h", "fsq_lattice.h"]
}
```

### 6.11 What Changed from Gen 5

- **Autoencoder, not predictor:** Full encode/decode replaces predict-compress-residual.
- **Ternary weights (W2):** 2-bit weights vs INT8 -- 4x smaller, no multiplier.
- **Focal modulation blocks:** Residual + GroupNorm, much deeper than flat stacks.
- **8x temporal stride:** 2500 samples -> 312 latent timesteps.
- **FSQ + rANS:** Information-theoretic entropy coding replaces polymorphic packing.
- **Q31 firmware:** All integer math, zero floating point.
- **Clinical loss:** Event-weighted MSE (5x seizure) + spectral convergence.
- **SWA:** Stochastic Weight Averaging for jitter-free deployment weights.
- **Production tooling:** Checkpoint resume, W&B/MLflow, CRC, fidelity audits,
  reproducibility manifests, patient-wise splits.

### 6.12 Performance

- Encoder SRAM: ~43KB, activation buffers: 4KB.
- Latency: < 4ms on RP2350.
- Model weights: ~3KB (ternary packed).
- Compression: FSQ+rANS on [32x312] latent (vs [21x2500] raw).

### 6.13 File Index

| File | Purpose |
|------|---------|
| `ai_models/oracle/train_teacher.py` | FP32 teacher training (800 ep, SWA, clinical loss) |
| `ai_models/student/train_student.py` | 3-phase ternary student training (500 ep) |
| `ai_models/student/train_ternary.py` | TernaryMobileNetV5 architecture + augmentations |
| `ai_models/student/harden_artifacts.py` | Latent alignment with strided teacher |
| `lamquant_codec/codec.py` | TernaryCodec: encode/decode/compress/decompress |
| `firmware/export_firmware.py` | C header generation (weights, FSQ, CRC) |
| `firmware/neural/focal_modulation.c` | Ternary MAC, GroupNorm Q31, encoder inference |
| `firmware_export/focal_net_weights.h` | 2850 bytes packed model parameters |
| `firmware_export/fsq_lattice.h` | FSQ levels [3,3,3,3] |
| `firmware_export/toep_seeds.h` | 21x4 Toeplitz CS seeds |
| `firmware_export/firmware_crc.h` | CRC32 + binary size |
| `firmware_export/manifest.json` | Deployment metadata |

---

## Generation Comparison

### Architecture Evolution

| | Gen 1 | Gen 2 | Gen 3 | Gen 4 | Gen 5 | Gen 6 | Gen 7 |
|---|---|---|---|---|---|---|---|
| **Paradigm** | Wavelet tiering | + Atoms + BLE | + Neural predictor | + Distillation | + QAT | Autoencoder | Dual-mode codec |
| **Model** | None | Atom dict | CausalConv1d | DepthwiseSep | DWSep + QAT | TernaryMobileNetV5 | TernaryMNv5 + Vocos |
| **Params** | N/A | N/A | ~16K | ~16K (distilled) | ~16K (QAT) | ~200K | 2.6M enc + 844M dec |
| **Weights** | N/A | N/A | INT8 (post-hoc) | INT8 (post-hoc) | INT8 (QAT) | Ternary W2 (LSQ) | Ternary W2 (LSQ) |
| **Lossless** | No | No | No | No | No | No | LML (2.3x CR) |
| **Compression** | FWT + threshold | FWT + atoms + pack | FWT + prediction | Same | Same | FSQ + rANS | Adaptive SNAC + rANS |
| **Ratio** | 2-5:1 | 20-30:1 | 50-70:1 | ~70:1 | ~70:1 | Latent bottleneck | 50-300:1 (adaptive) |
| **PRD** | < 1% | < 5% | < 3% | < 3% | < 3% | Clinical-grade | TBD (training) |
| **Loss** | N/A | N/A | MSE | MSE + distillation | Huber + variance | Event-weighted + spectral | MSE + R + spectral + GAN |
| **Dataset** | PhysioNet | PhysioNet | CHB-MIT (36) | CHB-MIT (109) | Filtered | CHB-MIT Q31 | TUEG Super (71K) |

### Training Evolution

| | Gen 3 | Gen 4 | Gen 5 | Gen 6 |
|---|---|---|---|---|
| **Model type** | Next-sample predictor | Teacher-student predictor | QAT predictor | Autoencoder |
| **Teacher** | None | EEGOptimusPrime (256h, 6L) | None | FP32OracleAutoEncoder |
| **Activation** | ReLU6 | ReLU6 | ReLU6 | ReLU + GroupNorm |
| **Optimizer** | Adam | AdamW | Adam | AdamW (fused) |
| **LR** | 0.001 | 3e-3 | 0.002 | 2e-3 (auto) |
| **Scheduler** | None | CosineAnnealing | StepLR | SGDR + SWA |
| **Epochs** | 50 | 30 | 100 | 500 (student) / 800 (teacher) |
| **Batch** | 4096 | 4096 | 4096 | 32-64 (auto by VRAM) |
| **Dataset** | 36 subjects | 109 subjects (CHB-MIT) | Filtered subset | Q31 CHB-MIT |
| **Quantization** | Post-hoc KL calib | Post-hoc (7 methods) | FakeQuant STE | Ternary LSQ |
| **Augmentation** | None | None | None | Montage permutation + clinical |

### Hardware Constants

| Parameter | Gen 1 | Gen 2+ |
|-----------|-------|--------|
| MCU | Desktop | RP2350 Hazard3 (150 MHz RISC-V) |
| FPU | Yes | No (soft-float) |
| ADC | Simulated | ADS1299 (24-bit, 250 SPS, 8ch) |
| SPI | N/A | 4 MHz |
| ADC Gain | N/A | 24x |
| LSB | N/A | 0.02235 uV |
| VREF | N/A | 4.5V |
| Buffer | 1024 samples | 256 samples |
| BLE PDU | N/A | 240 bytes |
| Encode deadline | Variable | 500 ms |

### File Location Index

| Generation | Primary Location |
|------------|-----------------|
| Gen 1 | `/mnt/2tb/Gen1 Archive/LamQuant Gen 1/` |
| Gen 2 | `/mnt/2tb/Gen2 Archive/lamquant-v2 backup/` |
| Gen 3 (firmware) | `/mnt/2tb/Gen3 Archive/lamquant-v3/` + `lamquantFull-v3/` |
| Gen 3 (training) | `/mnt/2tb/BrainyAI/` |
| Gen 4 | `/mnt/2tb/BrainyAI_Filtered/` (train_teacher.py, train_student.py) |
| Gen 5 | `/mnt/2tb/BrainyAI_Filtered/train_sota.py` |
| Gen 6 | `legacy.7z` -> `LamQuant-Old.7z` / `LamQuant-main.zip` |
| Gen 6 (firmware export) | `legacy.7z` -> `legacy/firmware_export/` |
| Gen 7 | `github.com/Quitetall/LamQuant` (current repo) |

---

## Gen 7 -- Production Lossless + Neural Codec System

**Location:** `github.com/Quitetall/LamQuant` (main branch)
**Date:** April 2026
**Language:** Rust (lossless codec), Python/PyTorch (neural codec + training), C (firmware)
**Target:** RP2350 Hazard3 (production), ESP32-S3, ESP32-P4 (planned)
**License:** AGPL-3.0

### 7.1 Overview

Gen 7 is a ground-up production system with two independent codec modes:

- **LML (Lossless Mode):** Domain-specific lossless EEG compression achieving
  2-3x CR with bit-exact reconstruction. Replaces EDF as the storage format.
- **LMQ (Neural Mode):** Ternary encoder (W2A16) on MCU + generative Vocos
  iSTFT decoder (up to 844M params) on base station. Targets 50-300x CR.

Both modes are unified under the LMA archive container format, which bundles
compressed EEG signal with annotation files, metadata, and directory structure.

### 7.2 LML Lossless Codec

**Pipeline:**
```
Input [C, T] int16
  -> Le Gall 5/3 integer lifting DWT (in-place, split-buffer for SIMD)
  -> LPC order-2 prediction (Levinson-Durbin)
  -> Bias cancellation (running mean, ctx_len=32, floor division)
  -> Golomb-Rice entropy coding (adaptive k)
  -> CRC-32 per window
Output: LML container (32-byte header + JSON metadata + window payloads)
```

**Key design decisions:**
- Integer lifting (no floating point) -- matches Hazard3 (no FPU)
- BIAS_CTX_LEN=32 validated on 370,823 windows (100% win rate vs ctx=16)
- Floor division matching Python `//` semantics (critical cross-language bug fix)
- Full EDF header preservation (zstd-9 compressed in metadata)
- Non-EEG channel data preserved for bit-exact EDF reconstruction
- 50/50 random file roundtrip SHA-256 verified across entire TUEG corpus

**Implementations:**
- Rust binary (`lml`): ~200 MB/s encode, 32 GB memory throttle, 8-16 threads
- Python (numba JIT): ~15 MB/s encode, shared math with training
- Cross-implementation conformance tested

**Performance:** 2.3x average CR on TUEG (1.7 TB -> 735 GB), 69,671 files.

### 7.3 LMA Archive Container

**Format:**
```
[4B] Magic: 'LMA1'
[4B] Version: uint32
[4B] Entry count: uint32
[4B] Manifest length: uint32
[var] Manifest: zstd-9 compressed JSON
[var] Payloads: concatenated (EDF->LML, text->zstd, compressed->store)
[32B] SHA-256 of everything above
```

**Features:**
- EDF files compressed with LML codec (~2.5x)
- Text/annotations compressed with zstd-9 (~10x)
- Already-compressed files stored as-is
- Pluggable secondary compressor (typed ABC interface: zstd, lzma, none)
- Per-file + archive-level SHA-256 integrity
- Super archive mode: multiple datasets with content deduplication
- Per-dataset extraction, training-split deduplication
- Streaming payloads to disk (constant memory, no OOM on large archives)

### 7.4 Neural Encoder: TernaryMobileNetV5 (Tier L)

**Architecture:**
```
Input [B, 21, 2500]
  -> FocalModBlock(21 -> 256, k=3)  + GroupNorm + ternary shortcut
  -> FocalModBlock(256 -> 256, k=3) + GroupNorm
  -> FocalModBlock(256 -> 256, k=5) + GroupNorm
  -> FocalModBlock(256 -> 256, k=5) + GroupNorm
  -> FocalModBlock(256 -> 256, k=7) + GroupNorm
  -> FocalModBlock(256 -> 256, k=7) + GroupNorm
  -> Bottleneck Conv1d(256 -> 32, k=1)
  -> Adaptive SNAC FSQ (multi-scale: 8+5+5 levels per dim)
  -> rANS entropy coding
Latent [B, 32, 79]
```

**2,609,613 parameters.** Ternary weights (W2) via Learned Step-size
Quantization (LSQ) with Straight-Through Estimator. Configurable width,
block count, and kernel sizes via `TrainingConfig`.

**Changes from Gen 6:**
- Width 256 (was 96) -- 2.7x wider, enabled by flash XIP (weights in 4 MB
  flash, not SRAM)
- 6 focal blocks (was 4) -- deeper receptive field
- Adaptive SNAC FSQ (was flat FSQ) -- variable compression ratio per window,
  SNN-driven (525:1 seizure -> 63:1 normal, avg 321:1)
- Configurable architecture via `encoder_width`, `encoder_blocks`,
  `encoder_kernels` in TrainingConfig

### 7.5 Neural Decoder: Vocos iSTFT (8 Tiers)

| Tier | Params | Dim | Blocks | Output | Use Case |
|------|--------|-----|--------|--------|----------|
| 1 | 14K | 32 | 4 | direct L3 | Testing |
| 2 | 116K | 64 | 6 | direct L3 | Testing |
| 3 | 3.8M | 256 | 8 | iSTFT | Fast iteration |
| 4 | 19.6M | 512 | 12 | iSTFT | Standard |
| 5 | 101M | 896 | 20 | iSTFT | Production candidate |
| 7 | 844M | 1792 | 32 | iSTFT | Full production |
| 8 | 207M | 1280 | 20 | iSTFT | Mobile distillation |

Tiers 5+ use gradient checkpointing to fit in 24 GB VRAM.
All fullband tiers reconstruct [B, 21, 2500] via inverse STFT.
ConvNeXt backbone with optional GRN, Snake activation, SE attention.

### 7.6 Training System

**4-phase schedule:**
1. **Warm-up:** FP32 encoder + decoder, cosine LR, no quantization
2. **QAT:** Ternary STE on encoder, decoder stays FP32, SOAP optimizer
3. **GAN (optional):** Adversarial discriminator for perceptual quality
4. **WSD Infinite LR:** Warmup-Stable-Decay, every checkpoint shippable

**Validated improvements:**
- SOAP optimizer: +0.0135 R over AdamW (3-seed A/B)
- V2-fixed SE decoder: +0.0104 R (SE zero-init, post-MLP)
- SNAC compact preset: +0.0009 R
- BIAS_CTX_LEN=32: +0.2% CR (100% win rate)

**Training infrastructure:**
- ExperimentRunner: A/B sweeps, grid search, recipe system, leaderboard
- TrainingConfig: frozen dataclass, 45+ fields, YAML/JSON round-trip
- PrecomputedL3Dataset with fullband memmap (fp16, memory-mapped I/O)
- Clinical weighted sampling (seizure 5x overweight)
- Per-window SHA-256 provenance in every checkpoint

### 7.7 SNN Seizure Detector

**Architecture:** Bidirectional Mamba SSM (2 layers, d_model=40, d_state=16)
**Parameters:** 57,688
**Training:** 500 epochs on CHB-MIT, DWB class weighting
**Best:** 99.98% sensitivity, 34.8% accuracy (epoch 4, needs TUEG normal data)
**Firmware:** `firmware/snn/snn.c` (C implementation for Core 1)

### 7.8 Firmware: RP2350 Hazard3

**Memory model:**
- TNN weights: 4 MB flash via XIP (not SRAM)
- Activation workspace: 205 KB (ADC buffer temporal reuse)
- Max encoder width: w=222 (int16) or w=445 (int8)
- SCRATCH_X: DSP hot buffers (Core 0)
- SCRATCH_Y: Core 1 stack (SNN)
- 8 FDA safety features: 59 KB SRAM (BLE retry, impedance, pre-ictal buffer,
  seizure diary, EEG quality, session logger, watchdog, power management)
- Binary: 95,268 text + 8,680 data + 505,164 bss = 509.8 KB / 520 KB

**MAC optimization (3 paths):**
1. SW pipelining: 2 cycles/MAC (baseline)
2. XNOR + CPOP bit-serial: 1.5 cycles/MAC (ternary-specific)
3. Zbb min/max saturation: branchless clamp

**Build:** CMake, `-march=rv32imac_zba_zbb_zbs`, no soft-float
(`PICO_PRINTF_SUPPORT_FLOAT=0`)

### 7.9 Interactive CLI

**lamquant.py** — TUI with:
- Splash screen, settings browser (arrow-key nav, search, help per setting)
- Codec Hub (LML lossless / LMQ neural, compress/decompress/verify/inspect)
- Firmware Hub (RP2350, ESP32-S3, ESP32-P4 build/flash/export)
- OpenHuman Portal installer (auto-detect toolchain, modular components)
- Training Cockpit (data pipeline, presets, experiments, live metrics)
- Setup wizard (4-step: system, display, backend, preferences)
- Standardized navigation: b=back, q=menu, x=exit with confirm
- Settings: instant_nav, potato_mode, allow_root, autocomplete
- Shared terminal.py for color/unicode/width detection

### 7.10 Data Pipeline

```
EDF corpus (Temple University Hospital, 71,476 recordings)
  -> LML lossless compression (Rust, 200 MB/s, bit-exact roundtrip)
  -> Annotations merged from 7 TUH subsets (TUSZ, TUAB, TUAR, TUEP, TUEV, TUSL)
  -> LMA super archive (dedup, per-dataset extraction, training split)
  -> L3 windowing (21ch x 2500 samples, Q31 normalization)
  -> Fullband memmap (fp16, memory-mapped I/O)
  -> Training (SOAP + WSD + gradient checkpointing)
```

**TUEG Super Dataset:** 71,476 EDFs + 45,829 annotations from all 7 TUH
subsets, unified with patient-level deduplication.

### 7.11 What Changed from Gen 6

| Feature | Gen 6 | Gen 7 |
|---------|-------|-------|
| **Lossless codec** | None | LML (DWT + LPC + Golomb-Rice, 2.3x CR) |
| **Archive format** | None | LMA (LML + zstd sidecar, dedup, streaming) |
| **Encoder width** | 96 | 256 (configurable) |
| **Encoder blocks** | 4 | 6 (configurable) |
| **Encoder params** | ~200K | 2.6M |
| **Decoder** | Symmetric (same as encoder) | Vocos iSTFT (up to 844M) |
| **Decoder tiers** | 1 | 8 (14K to 844M) |
| **FSQ** | Flat [3,3,3,3] | Adaptive SNAC (SNN-driven CR) |
| **Optimizer** | AdamW | SOAP (+0.0135 R validated) |
| **LR schedule** | SGDR + SWA | WSD Infinite LR (every ckpt shippable) |
| **Dataset** | CHB-MIT (686 files) | TUEG Super (71,476 files + 45,829 annotations) |
| **Training presets** | 1 (500 ep) | 4 (fast/standard/medium/production) |
| **Rust codec** | None | lml binary (200 MB/s, 32 GB memory throttle) |
| **CLI** | Script-based | Interactive TUI (settings, codec hub, firmware hub) |
| **EDF preservation** | None | Bit-exact roundtrip (header + annotation channels) |
| **Firmware targets** | RP2350 only | RP2350 + ESP32-S3 + ESP32-P4 |
| **Safety features** | None | 8 FDA subsystems (59 KB SRAM) |
| **Seizure detector** | None | Mamba SNN (57K params, 99.98% sensitivity) |

### 7.12 File Index

| File | Purpose |
|------|---------|
| `lamquant.py` | Interactive CLI entry point (TUI) |
| `lamquant_codec/edf_to_lml.py` | EDF reader + LML writer + EDF reconstructor |
| `lamquant_codec/lma.py` | LMA archive container (pack/unpack/super/dedup) |
| `lamquant_codec/lossless.py` | Python LML codec (numba JIT) |
| `lamquant_codec/ops/` | DSP primitives (lifting, LPC, bias, WHT, pipeline) |
| `lamquant_codec/cli/` | CLI modules (terminal, box, settings, config, compress) |
| `lamquant-core/src/` | Rust LML codec (container, EDF reader, entropy coder) |
| `lamquant-core/src/bin/lml.rs` | Rust CLI (encode, decode, verify, stats, export) |
| `ai_models/architectures/` | Encoder (TernaryMobileNetV5), Teacher, SNN |
| `ai_models/decoder/` | Vocos iSTFT decoder (8 tiers) |
| `ai_models/student/train_joint.py` | Joint encoder+decoder training |
| `ai_models/student/training_config.py` | Training presets (fast/medium/production) |
| `ai_models/experiment_runner.py` | Experiment orchestrator (A/B, sweep, recipe) |
| `firmware/` | RP2350 firmware (focal MAC, SNN, safety features) |
| `scripts/` | Data sync, annotation index, verification tools |
| `tests/` | 418+ tests (codec, container, integration, training) |
| `decisions/` | 15+ Architecture Decision Records |
