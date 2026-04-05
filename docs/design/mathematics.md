# LamQuant Gen 7 Mathematics

## 1. Event-Weighted Loss

$$L_{event} = MSE(x, \hat{x}) \cdot [1 + (\lambda - 1) \cdot M]$$

Seizure mask $M=1$ scales error by $\lambda=5.0$, prioritizing ictal morphology preservation.

## 2. LSQ Ternary Quantization (STE)

$$W_q = \text{round}\left(\text{clamp}\left(\frac{W}{\alpha}, -1, 1\right)\right)$$

Straight-through estimator passes gradients through the non-differentiable `round`. Per-channel $\alpha$ is learned via LSQ. Weights are ternary: $\{-1, 0, +1\}$.

## 3. Distillation Loss

$$L_{distill} = \alpha \cdot MSE(z_s, z_t) + \beta \cdot D_{KL}(P_s \| P_t)$$

Aligns student latent $z_s$ and output distribution $P_s$ with teacher. Three-stage curriculum: clean → artifact-injected → seizure-weighted.

## 4. Q31 Fixed-Point Inference

$$y = \left\lfloor \frac{A \cdot W_q \cdot \alpha_{q31}}{2^{31}} \right\rfloor$$

All firmware arithmetic uses `int32 × int32 → int64 >> 31`. No float, no libm. Q30 variant (shift by 30) used for biquad coefficients exceeding $|1.0|$.

## 5. FSQ + rANS Compression

Uniform scalar quantization maps latent values to $L=16$ bins:

$$\text{bin} = \text{clamp}\left(\left\lfloor\frac{(v - v_{min}) \cdot L}{v_{max} - v_{min}}\right\rfloor, 0, L-1\right)$$

32-bit rANS entropy codes bin indices using frequency tables calibrated from the trained latent distribution. Typical compression: 5–8x at R ≥ 0.85.
