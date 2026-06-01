# LamQuant v7.7 Neural Architecture

## Overview

Three neural models form the codec's neural compression path:

1. **Encoder** (`TernaryMobileNetV5_Subband`) — 435K params, ternary, ships to MCU
2. **Decoder** (`VocosDecoder` Tier 3) — ~15M params, FP32, runs on base station
3. **Activity Detector** (`MambaSNN`) — 57K params, INT4, ships to MCU

```
Raw EEG [21, 2500] @ 250 Hz
    |
    +-- Lossless path (Mode 3): LPC -> Lifting DWT -> Golomb-Rice
    |   CR ~ 4.8:1, R = 1.000
    |
    +-- Neural path (Modes 0-2):
            |
            +-- Lifting DWT -> L3 approximation [21, 313]
            |
            +-- MambaSNN: L3 -> activity level per timestep
            |   -> FSQ level schedule [79] (2/3/5 levels per step)
            |
            +-- Encoder: L3 -> latent [32, 79]
            |   -> FSQ quantize per schedule -> rANS entropy code
            |   CR ~ 40-525:1 depending on activity
            |
            +-- Decoder (base station): latent -> reconstructed [21, 2500]
                R ~ 0.87-0.93 depending on quality mode
```

---

## 1. Encoder: TernaryMobileNetV5_Subband

**Source:** `lamquant_neural/models/encoder.py`

### Specifications

| Property | Value |
|----------|-------|
| Input | `[B, 21, 313]` L3 approximation subband |
| Output | `[B, 32, 79]` latent (uniform [-1, 1]) |
| Parameters | 435K |
| Weight precision | Ternary ({-1, 0, +1}), bottleneck INT8 |
| Firmware footprint | 42 KB SRAM |
| Total stride | 4x (313 -> 79) |

### Architecture

```
[21, 313] L3 subband input
    | premix: TernaryConv1d(21->21, k=1, s=1)
    |   Spatial decorrelation (channel mixing)
    |
    | focal1: TernaryConv1d(21->128, k=3, s=2) + GroupNorm(8) + ReLU
    |   ZeroPadShortcut(21->128, s=2) residual (free, no learned params)
    |   Output: [128, 157]
    |
    | focal2: TernaryFocalBlock(128->128, k=5, s=1)
    |   Feature extraction at constant resolution
    |   Output: [128, 157]
    |
    | focal3: TernaryFocalBlock(128->128, k=7, s=2)
    |   Wide kernel late = full spike-wave temporal context (225ms)
    |   Output: [128, 79]
    |
    | dw_gate: TernaryConv1d(128->128, k=3, depthwise)
    |   3-step causal temporal context before gating decision
    |
    | GLU bottleneck:
    |   value = INT8Conv1d(128->32, k=1)    -- 256-level channel mixing
    |   gate  = sigmoid(TernaryConv1d(128->32, k=1, input=dw_gate output))
    |   latent = value * gate
    |
    | Cayley rotation: Q = (I-A)(I+A)^{-1}, A skew-symmetric [32x32]
    |   Learned orthogonal rotation into FSQ-optimal coordinates
    |   0% dead codes vs 6.2% with fixed WHT
    |
    | CDF-LUT: per-channel empirical quantile lookup table
    |   32 breakpoints per channel, binary search + linear interpolation
    |   Maps ANY distribution to uniform [-1, 1] regardless of kurtosis
    |   STE: gradient flows through piecewise-linear segments
    |
    | DitheredFSQ (training only):
    |   Random L in {2, 3, 5, 8, 16, 32} per batch
    |   Bernoulli mask: 50% passthrough, 50% quantize
    |   Builds robustness to all FSQ levels simultaneously
    v
[32, 79] latent (uniform [-1, 1])
```

### Design Rationale

- **Kernel progression narrow->wide (3->5->7):** L3 is already lowpass-filtered 3x by the lifting DWT. Wide kernels at the start waste capacity on already-smooth data. At 79 timesteps, k=7 covers 9% of the sequence (~225ms = one full spike-wave complex).

- **GLU gating with temporal context:** Every top Mamba/SSM gating mechanism includes local temporal context before the gate decision. The depthwise k=3 conv gives 3 timesteps of context for 336 ternary weights (84 bytes).

- **INT8 bottleneck value path:** The 128->32 channel projection is the information bottleneck. Ternary ({-1,0,+1}) is too coarse for fine-grained channel mixing. INT8 gives 256-level precision at 4 KB (vs 0.5 KB ternary). This single layer carries most of the reconstruction quality.

- **Cayley rotation:** Parameterized as skew-symmetric matrix A, with Q = (I-A)(I+A)^{-1} guaranteed orthogonal. Learns to align latent axes with directions of maximum reconstruction-relevant variance. Eliminates dead FSQ codes that plague fixed transforms.

- **CDF-LUT:** Per-channel kurtosis varies from 0 to 700+ across EEG channels. A global uniform quantizer loses quality on high-kurtosis channels. The CDF-LUT adaptively maps each channel's empirical distribution to uniform, computed once from training data, frozen forever. 2 KB firmware footprint (32 channels x 32 entries x 2 bytes INT16).

### Mini-decoder (training only)

The encoder includes a symmetric ternary decoder (expand1/expand2/expand3/output) used only during the warm training phase. It is NOT deployed to firmware and is discarded after training. The production decoder is VocosDecoder.

---

## 2. Decoder: VocosDecoder (Tier 3)

**Source:** `ai_models/decoder/vocos_decoder.py`

### Specifications

| Property | Value |
|----------|-------|
| Input | `[B, 32, 79]` FSQ-dequantized latent |
| Output | `[B, 21, 2500]` reconstructed raw EEG |
| Parameters | ~15M (Tier 3) |
| Weight precision | FP32 (base station, no constraint) |
| Architecture | ConvNeXt backbone + iSTFT head |

### Architecture

```
[32, 79] latent
    | Conv1d(32 -> 256, k=1): channel embedding
    |
    | SpatialCoherenceLayer:
    |   Electrode-topology graph attention (10-20 montage adjacency)
    |   Learned spatial mixing enforces biophysical relationships
    |   Zero-init gate: starts as identity, learns spatial structure
    |
    | 8x ConvNeXt blocks (constant 79 timesteps):
    |   +-- dwconv(256, k=7, depthwise)     -- token mixer
    |   +-- LayerNorm(256)
    |   +-- Linear(256 -> 768)              -- 3x expansion
    |   +-- GELU                            -- activation
    |   +-- GRN(768)                        -- global response normalization
    |   +-- Linear(768 -> 256)              -- contraction
    |   +-- SEBlock(256, r=4)               -- squeeze-excitation attention
    |   +-- LayerScale                      -- learnable per-channel scaling
    |   +-- residual connection
    |
    | SubPixelShuffle1d(r=4): [256, 79] -> [64, 316]
    |   Depth-to-space temporal upsampling (cleaner than transposed conv)
    |
    | ISTFTHead:
    |   Conv1d(64 -> 378, k=1)
    |     -> 189 = 21ch * 9bins for log-magnitude
    |     -> 189 = 21ch * 9bins for phase angle
    |   magnitude = exp(clamp(log_mag, -8, 8))
    |   phase = cos(phi) + j*sin(phi) (unit circle projection)
    |   complex_stft = magnitude * (cos + j*sin)
    |   signal = istft(complex_stft, n_fft=16, hop=8, length=2500)
    v
[21, 2500] reconstructed EEG
```

### Design Rationale

- **Vocos-style constant resolution:** Process at the same temporal resolution throughout (79 frames), then upsample at the very end. Avoids progressive upsampling artifacts that create spectral aliasing.

- **iSTFT output head:** Predicting magnitude + phase in frequency domain, then inverse-STFT to time domain. More stable than direct waveform prediction for quasi-periodic EEG oscillations. Magnitude via exp ensures non-negative. Phase via unit-circle projection avoids wrapping discontinuities.

- **ConvNeXt blocks:** Modern architecture (Liu et al. 2022) adapted for 1D. Depthwise conv for local patterns, pointwise for channel mixing. Layer-scale + residual for stable deep stacking.

- **SE attention:** Global channel context via squeeze-and-excite. Zero-init ensures identity at start, learns useful channel weighting during training. Validated +0.0104 R in A/B test.

- **SpatialCoherenceLayer:** EEG channels have known spatial relationships (volume conduction). Graph attention using electrode adjacency captures cross-channel correlations that temporal-only processing misses.

### Tier System

| Tier | dim | blocks | output | params | use case |
|------|-----|--------|--------|--------|----------|
| 1 | 32 | 4 | direct (L3) | ~50K | embedded/test |
| 2 | 64 | 6 | direct (L3) | ~200K | lightweight |
| 3 | 256 | 8 | iSTFT (2500) | ~15M | production |
| 4 | 512 | 12 | iSTFT (2500) | ~80M | research |
| 5 | 896 | 20 | iSTFT (2500) | ~300M | high-quality |
| 8 | 1280 | 20 | iSTFT (2500) | ~200M | mobile (INT8) |

---

## 3. Activity Detector: MambaSNN

**Source:** `ai_models/snn/mamba_ssm_minimal.py`

### Specifications

| Property | Value |
|----------|-------|
| Input | `[B, 21, T]` raw EEG or L3 subband |
| Output | `[B, 8, T_out]` activity logits + spike rate |
| Parameters | 57K |
| Weight precision | INT4 (firmware), FP32 (training) |
| Firmware footprint | 14.1 KB at W2A8 |
| Temporal complexity | O(T) linear (vs O(T^2) for attention) |

### Architecture

```
[21, T] EEG input
    | spatial_mix: Linear(21 -> 40)
    |   880 params. Channel mixing without temporal assumption.
    |
    | BidirectionalSSM block 1 (d_model=40, d_state=16, expand=2):
    |   +-- LayerNorm(40)
    |   +-- Forward SelectiveSSM:
    |   |     in_proj: Linear(40 -> 160)  [x_proj, z gate]
    |   |     causal conv1d(80, k=4, depthwise)
    |   |     SiLU activation
    |   |     x_proj: Linear(80 -> 33)  [B:16, C:16, dt:1]
    |   |     dt = softplus(dt_raw + bias)  -- data-dependent step size
    |   |     A = -exp(A_log)  -- HiPPO-initialized diagonal decay
    |   |     h[t] = dt*A*h[t-1] + dt*B*x[t]  -- state update
    |   |     y[t] = C * h[t]  -- output read
    |   |     y = y * SiLU(z)  -- output gating
    |   |     out_proj: Linear(80 -> 40)
    |   +-- Backward SelectiveSSM (same, reversed time)
    |   +-- average(forward, backward) + residual
    |   ~28K params
    |
    | BidirectionalSSM block 2 (same architecture)
    |   ~28K params
    |
    | readout: Linear(40 -> 8)
    |   328 params. Project to 8 spatial electrode groups.
    |
    | stride-8 mean pooling (raw input only)
    |   [8, T] -> [8, T/8]
    |
    | Per-timestep classification:
    |   max across 8 groups -> activity scalar
    |   threshold: <0.0 -> L=2 (quiescent)
    |              >0.0 -> L=3 (active)
    |              >0.5 -> L=5 (seizure/event)
    v
[T_out] FSQ level schedule per timestep
```

### Design Rationale

- **Selective State Space Model (Mamba):** Linear O(T) complexity. The "selective" mechanism makes B, C, and dt data-dependent — the model learns WHAT to remember at each timestep rather than applying fixed decay. Critical for seizure detection where onset timing is unpredictable.

- **HiPPO-LegS initialization for A:** Decay rates initialized as -(n + 1/2) for n=0..15. This spaces memory timescales from ~2s (n=0) to ~0.03s (n=15) at the effective 31.25 Hz sample rate. Seizure patterns (5-30s) accumulate across recurrent steps, while fast spikes (<100ms) are captured by high-n components.

- **Bidirectional:** Forward + backward SSM outputs averaged. Seizure spikes have distinctive morphology in both time directions. FEMBA (2024) showed bidirectional Mamba improves EEG classification.

- **INT4 weight QAT:** Symmetric 4-bit quantization via STE during training. 16 discrete weight levels. 4x memory reduction (57K params -> 14.1 KB). INT3 quantization on decay parameters (A_log) preserves temporal dynamics with just 8 discrete values.

- **Chunked parallel scan:** Training uses chunks of 32 timesteps — vectorized within chunks, sequential between. Avoids 313 Python loop iterations while keeping exp() within float32 range.

- **HomeostaticThresholdAdapter:** On-device per-patient threshold adaptation using EMA of spike rates. No backprop needed. 200 bytes firmware state. Adapts to individual patient baselines within 5 minutes. Inspired by corticohippocampal metaplasticity.

### Output: Adaptive Compression Schedule

The SNN's per-timestep output drives variable-rate compression:

| Activity Level | FSQ Levels | Bits/symbol | Typical CR |
|---------------|-----------|-------------|-----------|
| Quiescent (< 0.0) | 2 | 1 | 525:1 |
| Active (> 0.0) | 3 | 1.6 | 210:1 |
| Seizure/Event (> 0.5) | 5 | 2.3 | 63:1 |

Average across typical recordings: ~321:1 (vs fixed 83:1 without SNN).

---

## Firmware Deployment Summary

| Component | Size | Precision | Location |
|-----------|------|-----------|----------|
| Encoder weights | 42 KB | Ternary + INT8 bottleneck | SRAM4 |
| MambaSNN weights | 14.1 KB | W2A8 | Flash (XIP) |
| CDF-LUT | 2 KB | INT16 | Flash (XIP) |
| Cayley rotation Q | 4 KB | Q31 | Flash (XIP) |
| FSQ lattice | 0.5 KB | INT16 | Flash (XIP) |
| **Total weights** | **~63 KB** | | |
| Activation buffers | ~8 KB | INT16 | SCRATCH_X/Y |
| **Total SRAM** | **~50 KB** | | of 520 KB available |

---

## Training Pipeline

```
Phase 1 — Warm (50 epochs):
    Encoder (FP32) + mini-decoder, no quantization
    Optimizer: AdamW, lr=2e-3
    Loss: MSE + 0.5*(1-R) + 0.1*PRD + 0.1*spectral

Phase 2 — QAT (200+ epochs, WSD-infinity schedule):
    Encoder (ternary STE) + VocosDecoder (FP32)
    Optimizer: SOAP, lr=1e-3, warmup 10 epochs then stable
    Loss: same multi-scale loss against fullband target [21, 2500]
    EMA checkpoint selection: save EMA weights when EMA R > live R
    Warm-phase best seeds CheckpointManager (no regression from QAT start)

SNN trained separately:
    MambaSNN on activity labels from CHB-MIT + TUSZ seizure annotations
    DWB (Distance-Weighted Binary) cross-entropy loss
    INT4 QAT throughout
```
