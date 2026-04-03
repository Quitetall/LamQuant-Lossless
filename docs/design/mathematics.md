# LamQuant Gen 6 Mathematics

The LamQuant neural codec uses specialized mathematical primitives to enable high-fidelity signal compression on resource-constrained hardware.

### 1. Event-Weighted Clinical Loss ($L_{event}$)
$$L_{event} = MSE(x, \hat{x}) \cdot [1 + (\lambda - 1) \cdot M]$$
The loss function scales the error by a multiplier $\lambda$ (default = 5.0) where the binary seizure mask $M=1$, forcing the manifold to preserve critical ictal morphology.

### 2. Straight-Through Estimator (STE)
$$W_q = \text{round}\left(\text{clamp}\left(\frac{W_{fp}}{\alpha}, -1, 1\right)\right)$$
The STE allows backpropagation gradients to bypass non-differentiable quantization functions, enabling the optimization of discrete ternary weights ({-1, 0, 1}).

### 3. Knowledge Distillation ($L_{distill}$)
$$L_{distill} = \alpha \cdot MSE(z_s, z_t) + \beta \cdot D_{KL}(P_s || P_t)$$
The objective function aligns the student's latent manifold $z_s$ and output probabilities $P_s$ with the teacher, ensuring the quantized model retains the teacher's feature extraction capability.

### 4. Q31 Fixed-Point Scaling
$$y = \text{truncate}\left(\frac{A \cdot W_q \cdot \alpha_{q31}}{2^{31}}\right)$$
The inference engine uses 32-bit integer multiplication and bit-shifting to scale accumulations by the LSQ step size, achieving 100% deterministic math without an FPU.
