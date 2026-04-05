# Mathematical Foundations

Every equation used in LamQuant, from training through firmware inference.

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

Three cascaded second-order IIR sections per channel, Direct Form 1:

```
y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
```

All coefficients are Q30 constants precomputed in `export_firmware.py` using `scipy.signal`.

### Stages

| Stage | Type | Cutoff | Order | Purpose |
|-------|------|--------|-------|---------|
| 1 | Highpass (Butterworth) | 0.5 Hz | 2 | Remove DC offset and slow drift |
| 2 | Lowpass (Butterworth) | 50 Hz | 2 | Anti-alias, remove high-frequency noise |
| 3 | Notch | 60 Hz (Q=30) | 2 | Suppress power line interference |

### Coefficient values (Q30, fs=250 Hz)

**Highpass 0.5 Hz:**
```
b0 =  0.99115360 → Q30:  1064243069
b1 = -1.98230719 → Q30: -2128486138
b2 =  0.99115360 → Q30:  1064243069
a1 = -1.98222893 → Q30: -2128402106
a2 =  0.98238545 → Q30:  1054828345
```

**Lowpass 50 Hz:**
```
b0 =  0.20657208 → Q30:   221805086
b1 =  0.41314417 → Q30:   443610172
b2 =  0.20657208 → Q30:   221805086
a1 = -0.36952738 → Q30:  -396777000
a2 =  0.19581571 → Q30:   210255520
```

**Notch 60 Hz (Q=30):**
```
b0 =  0.97547839 → Q30:  1047411946
b1 = -0.12250159 → Q30:  -131535080
b2 =  0.97547839 → Q30:  1047411946
a1 = -0.12250159 → Q30:  -131535080
a2 =  0.95095678 → Q30:  1021082069
```

To regenerate these from Python:
```python
from scipy.signal import butter, iirnotch
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
- `L` = number of levels (default 16)
- `vmin`, `vmax` = latent value range (calibrated from training data)

**Firmware optimization**: To avoid runtime division, the inverse range is precomputed at export time as a Q31 multiplier:

```c
fsq_inv_range_q31 = (FSQ_NUM_LEVELS << 31) / (vmax - vmin)
bin = clamp((shifted * fsq_inv_range_q31) >> 31, 0, L-1)
```

### 4D FSQ lattice

For the `fsq.c` adaptive quantizer, 4 latent dimensions are quantized with independent level counts:

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

Integer-to-integer wavelet transform, in-place:

**Predict (high-pass details):**
```
d[n] = x[2n+1] - floor((x[2n] + x[2n+2]) / 2)
```

**Update (low-pass approximation):**
```
s[n] = x[2n] + floor((d[n-1] + d[n] + 2) / 4)
```

The 2D transform applies the temporal pass (32 samples per channel) then the spatial pass (6 channels cross-linked). Boundary handling uses symmetric mirroring.

---

## 13. Linear Predictive Coding (LPC)

Order-4 LPC removes temporal redundancy after the lifting wavelet.

**Autocorrelation** (Q31 samples, int64 accumulator):
```
R[k] = sum_{n=k}^{N-1} x[n] * x[n-k],   k = 0..4
```

**Levinson-Durbin recursion** solves for 4 LPC coefficients from R[0..4]:

```
For m = 0 to 3:
  k_m = -(R[m+1] + sum_{j<m} a[j]*R[m-j]) / E
  a_curr[m] = k_m
  For j < m: a_curr[j] = a_prev[j] + k_m * a_prev[m-1-j]
  E = E * (1 - k_m^2)
```

Reflection coefficients `k_m` are in Q31. Coefficients are clamped to `INT32_MIN`/`INT32_MAX`.

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
