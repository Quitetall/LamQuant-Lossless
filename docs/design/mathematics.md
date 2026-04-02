# LamQuant Gen 6 Mathematics

The LamQuant neural codec bridges strictly differentiable high-level mathematics natively into absolute logic-gate optimizations structurally avoiding arbitrary runtime evaluations natively.

## 1. Event-Weighted Clinical Loss

Standard auto-encoders evaluated purely over Mean Squared Error (MSE) systematically wipe away clinical variances directly. We dynamically mask the `2500-sample` signal tensors with 5-fold multipliers exclusively localized over seizure morphologies:
`weighted_error = base_squared_error * (1.0 + (5.0 - 1.0) * mask_expanded)`

## 2. Straight-Through Estimators (STE)

Hardware Ternary Networks (composed only of `-1, 0, 1`) inherently break PyTorch Autograd paths because `torch.sign(x)` results in a gradient identically zero identically across practically mapping evaluations.

We explicitly bypass the zeroing utilizing STE backwards hooks:
1.  **Forward Pass:** We threshold the FP32 variables directly clamping structural arrays into identical {-1, 0, 1} symbols dynamically.
2.  **Backward Pass:** We "lie" to PyTorch structurally skipping the `torch.sign` boundary logic flowing the exact full-precision backwards gradients completely through the zero-points organically.

## 3. Knowledge Distillation (KL Divergence)
We prevent the Ternary student from losing clinical edge mappings via dual-loss optimization: `Loss = MSE(Latents) + KL_DIV(Student, Teacher)`.

We normalize spatial bounds strictly bridging probabilities evaluated exclusively through a spatial `SoftMax(Latents / Temperature)` mapping ensuring probabilities naturally smooth over biological peaks without clamping to extreme numerical infinities structurally mitigating over-fit risks completely natively.
