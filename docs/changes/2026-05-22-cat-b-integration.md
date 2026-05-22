# Cat B crate integration — added + smoke-tested (2026-05-22)

User direction: *"Add all the libraries and test them as a separate
thing, compare to the original, then see which is better."*

All Cat B candidates with **feasible smoke tests** are wired today.
Items with no current use site or no in-tree equivalent are
documented under "Deferred — no comparison surface."

## Active comparators (opt-in via Cargo feature)

| Crate | Where | Feature flag | Smoke result | A/B vs original |
|---|---|---|---|---|
| `constriction 0.4` | `lamquant-core` | `experimental_arithmetic` | 6/6 pass | **Original wins.** rANS overhead > Golomb-Rice per user direction; constriction-backed coder kept opt-in for differential testing only. Don't promote to default. |
| `pulp 0.21` | `lamquant-core` dev-dep | always (dev) | 1/1 pass (`pulp_dispatch_sum_smoke`) | Untested at codec hot path. Lifting kernel is already 800 MiB/s scalar; pulp dispatch is the right SIMD primitive when we rewrite a real kernel. Bookmark. |
| `realfft 3` | `lamquant-core` dev-dep | always (dev) | 1/1 pass (`realfft_roundtrip_8_point`) | No in-tree real-valued FFT today. Use when fullband spectrogram lands. |
| `fixed 1` | `lamquant-firmware` | `cat-b-fixed` | 1/1 pass (`fixed_q15_add_mul_smoke`) | Hand-rolled raw `i32` Q-format is faster + simpler. Use `fixed` only for new code where the type-safety wins outweigh the wrapper cost. |
| `microfft 0.6` (size-1024) | `lamquant-firmware` | `cat-b-microfft` | 1/1 pass (`microfft_8_point_dc_smoke`) | Hazard3 has no on-MCU FFT today. When spectrogram-on-device lands, microfft is the right choice. |
| `idsp 0.21` | `lamquant-firmware` | `cat-b-idsp` | 1/1 pass (`idsp_cossin_smoke`, ~2.3e-5 err) | Hand-rolled biquad already beats C on RP2350 (memory entry "Rust DSP perf — RESOLVED"). idsp is the alternative if we ever need `cossin` / `unwrap` / `Lowpass<>` plumbing. |

## Skipped — no FPU / no trigger met

| Crate | Reason |
|---|---|
| `micromath` | Hazard3 has no FPU. Firmware is **pure Q-format** as of 2026-05-21 — `f32` is gone entirely. micromath would re-introduce float we already removed. |
| `faer` | No dense linear-algebra in scope. Watch-list only. |
| `burn-flex`, `burn 0.18` | Rust-side QAT not in plan. PyTorch stays canonical. |
| `candle 0.9.2` | Reference inference, not production runtime. |
| `tch-rs` | Pulls full libtorch; would bloat the basestation binary. |
| `muriscv-nn` | Hazard3 has no V extension. Re-evaluate if RVV port lands. |
| `cmsis_dsp` (FFI) | Cortex-M-only. Hazard3 is RV32IMAC. |
| `microflow` | TFLite-Micro compiler; no ternary / Conv1D / ISTFT — doesn't fit our model shape. |
| `bitvec` | Use `bitstream-io` if we ever want bit-level streams (single source of truth). |
| `crossbeam`, `flume` | `tokio::mpsc` + `parking_lot` already cover BLUT. Adding a third channel lib is dep bloat. |
| `rkyv` | Conformance assets < 200 MB threshold. Re-evaluate at scale. |
| `trouble` (BLE host) | No MCU↔phone link in scope yet. |
| `embassy-rp` + `rp235x-hal` async | Single-loop scheduler suffices. |
| `loom` | No non-trivial dual-core shared-state code today. |
| `ort` / `tract` | Basestation Vocos deployment is Phase W12 (post real production weights). |
| `cmsis_nn` | Cortex-M-only; same rationale as `cmsis_dsp`. |

## Smoke commands

```
# Cat A (always-on; included in default test suite)
cargo test -p lamquant-core --features host        # snapshot + property + cat_b host smokes
cargo test -p lamquant-ipc-types                   # PostcardEnvelope + EnvelopeError
cargo test -p lamquant-firmware --test conformance # 9 firmware integration tests

# Cat B firmware smokes (opt-in)
cargo test --release -p lamquant-firmware --features cat-b-all --test cat_b_smoke

# constriction A/B (already-shipped opt-in feature)
cargo test --features "host experimental_arithmetic" -p lamquant-core --lib arithmetic
```

## Verification snapshot

| Suite | Count | Status |
|---|---|---|
| firmware conformance | 9/9 | PASS |
| firmware comms (Cat A7) | 4/4 | PASS |
| firmware cat-b-all | 3/3 | PASS |
| lamquant-core unit + snapshot + proptest | 446 | PASS |
| lamquant-core cat_b_smoke (pulp + realfft) | 2/2 | PASS |
| lamquant-core arithmetic (constriction) | 6/6 | PASS |
| lamquant-ipc-types | 3/3 | PASS |

## What stays gated

- `constriction` stays behind `experimental_arithmetic` — known
  worse than Golomb-Rice for our window sizes.
- `fixed`, `microfft`, `idsp` stay behind `cat-b-*` features — opt-in
  comparators, not yet replacing any in-tree kernel.
- `pulp`, `realfft` are dev-deps only — never enter a release binary.

Next perf experiments (deferred — separate sessions):
- pulp on lifting kernel (current 800 MiB/s; target +30%)
- idsp biquad parity bench vs hand-rolled
- constriction differential test at small block sizes (verify FP error
  vs the in-tree Golomb path under degenerate inputs)
