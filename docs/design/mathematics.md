# Mathematical Foundations

Every equation used in LamQuant Gen 7.1 ("Subband"), from training through firmware inference.

---

## 1. Fixed-Point Arithmetic

All firmware math uses integer-only fixed-point representations. No floating point, no `libm`, no soft-float.

### Q31 Format

Represents values in the range [-1.0, +1.0) with 31 fractional bits:

```
value = integer / 2^31
```

Multiplication:

```c
int32_t mul_q31(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * (int64_t)b) >> 31);
}
```

The 64-bit intermediate prevents overflow before the right-shift normalization. Result range: [-1.0, +1.0).

### Q30 Format

Represents values in the range [-2.0, +2.0) with 30 fractional bits:

```
value = integer / 2^30
```

Used for biquad filter coefficients that exceed Q31's range. For example, the highpass biquad's `b1 ≈ -1.98` overflows Q31 but fits in Q30. All three filter stages use Q30 uniformly:

```c
int32_t mul_q30(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * (int64_t)b) >> 30);
}
```

### Saturating Arithmetic

Addition and subtraction clamp to `INT32_MIN` / `INT32_MAX` on overflow using GCC's `__builtin_add_overflow` / `__builtin_sub_overflow`:

```c
int32_t add_sat_q31(int32_t a, int32_t b) {
    int32_t res;
    if (__builtin_add_overflow(a, b, &res))
        res = (a < 0) ? INT32_MIN : INT32_MAX;
    return res;
}
```

These primitives are defined in `firmware/core/math_utils.h` and used throughout the DSP and neural layers.

---

## 2. Biquad IIR Filter (Q30)

Single second-order IIR section per channel (Gen 7.1: HP-only), Direct Form 1:

```
y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
```

All coefficients are Q30 constants precomputed in `export_firmware.py` using `scipy.signal`.

### Stages (Gen 7.1)

| Stage | Type | Cutoff | Order | Purpose |
|-------|------|--------|-------|---------|
| 1 | Highpass (Butterworth) | 0.5 Hz | 2 | Remove DC offset and slow drift |

In Gen 7.1, the lowpass and notch stages are removed. Their anti-aliasing and interference rejection functions are replaced by the 3-level lifting DWT's inherent subband decomposition combined with detail coefficient thresholding. This reduces filter state from 63 delay lines (21 channels x 3 stages) to 21 delay lines (21 channels x 1 stage).

> **Legacy (Gen 7.0)**: 3-stage cascade: HP 0.5 Hz + LP 50 Hz + Notch 60 Hz.

### Coefficient values (Q30, fs=250 Hz)

**Highpass 0.5 Hz:**
```
b0 =  0.99115360 → Q30:  1064243069
b1 = -1.98230719 → Q30: -2128486138
b2 =  0.99115360 → Q30:  1064243069
a1 = -1.98222893 → Q30: -2128402106
a2 =  0.98238545 → Q30:  1054828345
```

To regenerate from Python:
```python
from scipy.signal import butter
Q30 = 1 << 30
b, a = butter(2, 0.5, btype='high', fs=250)
print([int(c * Q30) for c in b])
print([int(c * Q30) for c in a[1:]])
```

---

## 3. LSQ Ternary Quantization

Learned Step Size Quantization with Straight-Through Estimator (STE):

```
W_q = round(clamp(W / alpha, -1, 1))
```

Where:
- `W` = FP32 weight
- `alpha` = per-channel learned scale parameter (always positive: `alpha = |lsq_alpha|`)
- `clamp` restricts to [-1, 1]
- `round` produces ternary values {-1, 0, +1}

The STE passes gradients through the non-differentiable `round` via:

```python
w_grad = (w_q - self.weight).detach() + self.weight
output = conv1d(x, w_grad * alpha, ...)
```

This is equivalent to using the quantized weights in the forward pass while allowing gradients to flow to `self.weight` and `self.lsq_alpha` in the backward pass.

### Weight packing for firmware

Ternary weights are encoded as 2-bit values, packed 4 per byte (little-endian):

| 2-bit code | Weight value |
|------------|-------------|
| 00 | 0 |
| 01 | +1 |
| 10 | -1 |
| 11 | 0 (reserved/padding) |

A byte `0x49` decodes as:
```
bits[1:0] = 01 → +1
bits[3:2] = 10 → -1
bits[5:4] = 00 →  0
bits[7:6] = 01 → +1
```

The firmware uses a lookup table for branchless decoding:

```c
static const int32_t TERNARY_LUT[4] = {0, 1, -1, 0};
acc += (int32_t)act[i] * TERNARY_LUT[(packed_w >> (2*i)) & 0x03];
```

---

## 4. GroupNorm (Integer)

GroupNorm with 4 groups, implemented in pure integer arithmetic:

```
For each group g of channels:
  mean = sum(x) / count
  var  = sum(x^2) / count - mean^2
  std  = isqrt(var)   (Newton's method, 3 iterations)

  For each channel c in group:
    normed = (x[c][t] - mean) * 128 / std
    scaled = mul_q31(normed, gamma[c]) + (beta[c] >> 16)
    output = max(scaled, 0)    // ReLU fused
    output = min(output, 32767) // Saturate to int16
```

The integer square root uses Newton's method with 3 iterations:
```c
std_approx = v32;
std_approx = (std_approx + v32 / std_approx) >> 1;  // iteration 1
std_approx = (std_approx + v32 / std_approx) >> 1;  // iteration 2
std_approx = (std_approx + v32 / std_approx) >> 1;  // iteration 3
```

---

## 5. Ternary Convolution (Firmware)

For each output channel and time step, the ternary 1D convolution computes:

```
acc = sum over (input_channel, kernel_position):
    activation[ic][t + ki] * TERNARY_LUT[packed_weight_bits]

output = mul_q31(acc, alpha_q31)
```

The per-channel Q31 alpha restores the learned scale that was factored out during ternary quantization. This is the "digital gain restoration" step.

---

## 6. Event-Weighted Loss

Training uses an event-weighted MSE that penalizes seizure reconstruction errors 5x more than baseline:

```
L_event = MSE(x, x_hat) * [1 + (lambda - 1) * M]
```

Where `M` is a binary seizure mask from CHB-MIT annotations and `lambda = 5.0`. This forces the bottleneck to preserve ictal morphology — preventing "F1 fragility" where the codec optimizes for baseline at the expense of rare seizure events.

---

## 7. Distillation Loss

Student-teacher alignment uses MSE on reconstructed waveforms plus a Pearson correlation metric:

```
L_distill = alpha * MSE(z_student, z_teacher) + beta * MSE(x_student, x_teacher)
```

The Pearson R between student and teacher outputs is monitored but not directly optimized:

```
R = sum((s - mean(s)) * (t - mean(t))) / (std(s) * std(t))
```

Three-stage curriculum: clean data -> artifact-injected -> seizure-weighted.

---

## 8. Spectral Loss

Multi-resolution STFT loss for waveform fidelity during fine-tuning:

```
L_spectral = (1/K) * sum_{k in fft_sizes} MSE(log10(|STFT_k(pred)|), log10(|STFT_k(target)|))
```

Default FFT sizes: {64, 128, 256}. Hop length = FFT size / 4.

---

## 9. Finite Scalar Quantization (FSQ)

Uniform scalar quantization maps continuous latent values to a discrete grid:

```
bin = clamp(floor((val - vmin) * L / (vmax - vmin)), 0, L-1)
```

Where:
- `L` = number of levels (quality-dependent; see below)
- `vmin`, `vmax` = latent value range (calibrated from training data)

**Firmware optimization**: To avoid runtime division, the inverse range is precomputed at export time as a Q31 multiplier:

```c
fsq_inv_range_q31 = (FSQ_NUM_LEVELS << 31) / (vmax - vmin)
bin = clamp((shifted * fsq_inv_range_q31) >> 31, 0, L-1)
```

### Quality-Dependent FSQ Levels (Gen 7.1)

Gen 7.1 introduces quality-dependent FSQ levels. The FSQ level set `L` varies by quality mode, with MAX_SYMBOLS=32:

| Quality Mode | FSQ Levels (L) | MAX_SYMBOLS | Approx. Compression Ratio |
|-------------|----------------|-------------|---------------------------|
| ALERTING | 2, 3, 5 | 5 | ~150:1 |
| MONITORING | 2, 3, 5, 16 | 16 | ~80:1 |
| CLINICAL | 2, 3, 5, 32 | 32 | ~40:1 |

The L=32 level is used exclusively in CLINICAL mode for maximum fidelity. The latent tensor is [32][79] (32 channels, 79 time steps after 4x stride TNN).

> **Legacy (Gen 7.0)**: Fixed L=2/3/5 (MAX_SYMBOLS=5), latent [32][312].

### 4D FSQ lattice (legacy)

For backward compatibility, the `run_fsq_translation()` function retains the 4D FSQ lattice:

```
FSQ_LEVELS[4] = {8, 6, 5, 5}
```

The 4 quantized values are packed into a single flat index using mixed-radix encoding:

```
index = q[0] + q[1]*8 + q[2]*48 + q[3]*240
```

Total codebook size: 8 x 6 x 5 x 5 = 1200 entries.

### Adaptive gain

`fsq.c` adjusts the quantization scale based on rolling RMS:

```c
if (rms > 50)
    adaptive_fsq_scale = FSQ_QUANT_SCALE_Q31 / ((rms / 50) + 1);
else
    adaptive_fsq_scale = FSQ_QUANT_SCALE_Q31;
```

This prevents dead codes (weak signals not filling bins) and clipping (strong signals exceeding the grid).

---

## 10. rANS Entropy Coding

32-bit asymmetric numeral systems (rANS) encodes FSQ bin indices:

```
Encode symbol s with frequency f[s], cumulative start c[s]:
  while (state >= (L/M)*f[s]*2^16):
      emit state & 0xFF
      state >>= 8
  state = (state/f[s])*M + c[s] + (state % f[s])
```

Where:
- `L` = 2^15 (renormalization bound)
- `M` = total frequency (sum of all symbol frequencies, power of 2, default 4096)
- `f[s]` = frequency count for symbol `s`
- `c[s]` = cumulative frequency for symbol `s`

Encoding proceeds in reverse (rANS is LIFO — last symbol encoded is first decoded). The final state is flushed as 4 trailing bytes.

Frequency tables are calibrated from the trained latent distribution and exported into `focal_net_weights.h`.

---

## 11. Compressed Sensing (Toeplitz)

Binary Toeplitz matrices are generated from an LFSR (Linear Feedback Shift Register):

**LFSR**: 16-bit Fibonacci topology, taps at bits 0, 2, 3, 5. Period: 2^16 - 1 = 65535.

```c
bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1;
state = (state >> 1) | (bit << 15);
```

**Projection**: Each measurement is an inner product of 2500 input samples with a binary ±1 vector:

```
y[i] = sum_{j=0}^{N-1} x[j] * sign(lfsr_bit[j])
```

**Branchless accumulation**: Instead of branching on each bit:

```c
int32_t mask = bit - 1;  // 0x00000000 (bit=1) or 0xFFFFFFFF (bit=0)
acc += (sample ^ mask) - mask;  // bit=1: +sample, bit=0: -sample
```

Each channel uses a different LFSR seed for uncorrelated sensing rows. The batch LFSR generates 32 bits per call to amortize the generation cost.

---

## 12. Le Gall 5/3 Lifting Wavelet

Integer-to-integer wavelet transform, in-place.

**Predict (high-pass details):**
```
d[n] = x[2n+1] - floor((x[2n] + x[2n+2]) / 2)
```

**Update (low-pass approximation):**
```
s[n] = x[2n] + floor((d[n-1] + d[n] + 2) / 4)
```

Boundary handling uses symmetric mirroring for all levels.

### 3-Level Decomposition (Gen 7.1 Golden Path)

In Gen 7.1, a 3-level lifting DWT replaces the former lowpass and notch biquad stages. The transform is applied per-channel on the 2500-sample HP-filtered signal:

```
Level 1: [2500] → approx[1250] + detail[1250]
Level 2: approx[1250] → approx[625] + detail[625]
Level 3: approx[625]  → approx[313] + detail[313]
```

The L3 approximation subband [21,313] becomes the TNN input. Detail subbands are thresholded and selectively transmitted depending on quality mode.

The predict step acts as a highpass filter, extracting detail information. The update step acts as a lowpass filter, producing a smooth approximation. At each level, the signal length is halved (with rounding for odd lengths).

### 2D Lightning Path Transform (Legacy)

The 2D transform for the lightning path applies the temporal pass (32 samples per channel) then the spatial pass (6 channels cross-linked). This remains unchanged from Gen 7.0.

---

## 13. Linear Predictive Coding (LPC)

### Order-8 LPC Analysis (Gen 7.1 Golden Path)

Gen 7.1 uses order-8 LPC analysis per channel with 256-sample autocorrelation windows, applied before the lifting DWT. This captures the spectral envelope of each channel for prediction and delta encoding.

**Autocorrelation** (Q31 samples, int64 accumulator, 256-sample window):
```
R[k] = sum_{n=k}^{255} x[n] * x[n-k],   k = 0..8
```

**Levinson-Durbin recursion** solves for 8 LPC coefficients from R[0..8]:

```
For m = 0 to 7:
  k_m = -(R[m+1] + sum_{j<m} a[j]*R[m-j]) / E
  a_curr[m] = k_m
  For j < m: a_curr[j] = a_prev[j] + k_m * a_prev[m-1-j]
  E = E * (1 - k_m^2)
```

Reflection coefficients `k_m` are in Q31. Coefficients are clamped to `INT32_MIN`/`INT32_MAX`.

**Prediction filter**: The LPC coefficients define a prediction filter applied to each channel. The prediction residual has a flatter spectrum (closer to white noise), which improves the lifting DWT's ability to concentrate energy in the approximation subband.

```
pred[n] = sum_{i=1}^{8} a[i] * x[n-i]
residual[n] = x[n] - pred[n]
```

### LPC Delta Encoding (Gen 7.1)

LPC coefficients are delta-encoded for bandwidth-efficient transmission:

| Frame Type | Encoding | Size per frame (21 channels x 8 coefficients) |
|-----------|----------|------------------------------------------------|
| Keyframe | Full Q31 coefficients | 672 bytes (21 x 8 x 4) |
| Q15 delta | Difference from previous keyframe, Q15 | 336 bytes (21 x 8 x 2) |
| Q8 delta | Difference from previous Q15 frame, Q8 | 168 bytes (21 x 8 x 1) |

Keyframes are sent periodically (e.g., every 10 frames). Between keyframes, Q15 deltas capture coefficient drift. Between Q15 frames, Q8 deltas capture fine changes. The decoder reconstructs full Q31 coefficients by accumulating deltas from the last keyframe.

### Order-4 LPC (Lightning Path, Legacy)

The lightning path retains order-4 LPC on the 6x32 lifting tile:

**Autocorrelation**:
```
R[k] = sum_{n=k}^{N-1} x[n] * x[n-k],   k = 0..4
```

**Forward prediction residuals** (in-place, backwards to preserve needed samples):
```
pred[n] = a[0]*x[n-1] + a[1]*x[n-2] + a[2]*x[n-3] + a[3]*x[n-4]
residual[n] = x[n] - pred[n]
```

---

## 14. Golomb-Rice Coding

Signed residuals are zigzag-encoded to unsigned:

```
mapped = (residual >= 0) ? residual << 1 : (-residual << 1) - 1
```

Then split into quotient and remainder with Rice parameter k=4:

```
quotient = mapped >> k        (unary coded: q ones + 0 terminator)
remainder = mapped & (2^k - 1) (binary coded: k bits)
```

### Golomb-Rice with Run-Length Coding (Gen 7.1 Detail Encoding)

For sparse detail subband coefficients (after thresholding), Gen 7.1 combines Golomb-Rice with run-length coding. After SNN-driven thresholding, most detail coefficients are zero. The encoding proceeds:

1. **Run-length encode zeros**: Count consecutive zero coefficients. Emit a run-length count using Golomb-Rice (k=3) before each non-zero coefficient.
2. **Encode non-zero value**: The non-zero coefficient is zigzag-encoded and Golomb-Rice coded (k=4) as above.
3. **Terminal run**: A final run-length encodes any trailing zeros.

This achieves high compression on the sparse detail subbands, since ALERTING and MONITORING modes threshold most detail coefficients to zero.

---

## 15. CRC32

IEEE 802.3 polynomial 0xEDB88320 (reflected), table-accelerated:

```c
crc ^= 0xFFFFFFFF;
while (len--)
    crc = (crc >> 8) ^ CRC32_TABLE[(crc ^ *data++) & 0xFF];
return crc ^ 0xFFFFFFFF;
```

Compatible with Python's `zlib.crc32()` for cross-platform verification. The expected CRC covers Toeplitz seeds, neural network weights, and FSQ lattice configuration.

---

## 16. Walsh-Hadamard Transform (Gen 7.1)

The 32-point Walsh-Hadamard Transform (WHT) is applied as a pre-rotation on the 32-channel latent dimension before FSQ quantization. It decorrelates latent channels, improving FSQ codebook utilization and entropy coding efficiency.

### Butterfly Decomposition

The WHT is computed in-place using 5 stages of length-2 butterflies (log2(32) = 5 stages):

```
For stage s = 0 to 4:
  block_size = 2^(s+1)
  half = block_size / 2
  For each block starting at index j:
    For i = 0 to half-1:
      a = data[j + i]
      b = data[j + i + half]
      data[j + i]        = a + b
      data[j + i + half] = a - b
```

After all 5 stages, the output is normalized by dividing by sqrt(32) (or equivalently, right-shifting by 5 for the integer version with a final rounding correction).

### Application to Latent Tensor

At each of the T=79 time steps, the WHT is applied along the 32-channel dimension:

```
For t = 0 to 78:
  wht32_forward(latent[:, t])
```

The inverse WHT (used at the decoder) is identical to the forward WHT up to the normalization factor, since the Hadamard matrix is symmetric and orthogonal.

---

## 17. Detail Coefficient Thresholding (Gen 7.1)

After the 3-level lifting DWT, detail coefficients are thresholded to produce a sparse representation. The threshold is estimated from the data using the Median Absolute Deviation (MAD), a robust noise estimator.

### MAD Noise Estimation

```
sigma_hat = MAD(detail_coeffs) / 0.6745
```

Where:
- `MAD(x) = median(|x - median(x)|)`
- `0.6745` is the MAD-to-sigma conversion factor for Gaussian noise

### Quality-Dependent Multipliers

The threshold is `T = lambda * sigma_hat`, where `lambda` depends on the quality mode:

| Quality Mode | Lambda | Effect |
|-------------|--------|--------|
| ALERTING | 4.0 | Aggressive: only large transients survive, most detail zeroed |
| MONITORING | 2.0 | Moderate: preserves clinically relevant detail |
| CLINICAL | 0.5 | Conservative: preserves nearly all detail for diagnostic fidelity |

### Hard Thresholding

```
detail_out[n] = (|detail[n]| >= T) ? detail[n] : 0
```

The resulting sparse array is then encoded using Golomb-Rice with run-length coding (Section 14).

---

## 18. SNN Classification on L3 Subband (Gen 7.1)

The Spiking Neural Network (SNN) classifies the L3 approximation subband for quality mode selection and path decision. Topology: 21 -> 64 -> 8.

- **Input**: L3 approximation [21,313], stride 1
- **Hidden layer**: 64 leaky integrate-and-fire (LIF) neurons
- **Output layer**: 8 class neurons (softmax over spike rates)
- **Weights**: ~8 KB (up from ~5 KB in Gen 7.0 due to wider input: 21 channels vs. 8)

> **Legacy (Gen 7.0)**: Input [21,2500] stride 8, topology 8->64->8, weights ~5 KB.
