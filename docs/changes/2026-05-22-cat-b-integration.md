# Cat B crate integration — added + smoke-tested (2026-05-22)

User direction: *"Add all the libraries and test them as a separate
thing, compare to the original, then see which is better."*

All Cat B candidates with **feasible smoke tests** are wired today.
Items with no current use site or no in-tree equivalent are
documented under "Deferred — no comparison surface."

## Active comparators (opt-in via Cargo feature)

| Crate | Where | Feature flag | Smoke result | A/B vs original |
|---|---|---|---|---|
| `constriction 0.4` | `lamquant-core` | `experimental_arithmetic` | 6/6 pass | **Original wins on REAL data (empirical, 2026-05-22).** Real CHB-MIT `chb01_01` ch0 post-LPC residual: golomb-rice **1251 bytes / 2.35 GiB/s**, constriction rANS **1260 bytes / 403 MiB/s** — constriction is **+0.7% bigger AND 5.8× slower**. Synthetic ±15 bench (favourable to rANS) showed −1.9% CR / 6.5× slower; real data exposes that the static Laplace prior misfits actual EEG residual heavy tails. Keep opt-in only. |
| `pulp 0.21` | `lamquant-core` dev-dep | always (dev) | 1/1 pass (`pulp_dispatch_sum_smoke`) | Untested at codec hot path. Lifting kernel is already 800 MiB/s scalar; pulp dispatch is the right SIMD primitive when we rewrite a real kernel. Bookmark. |
| `realfft 3` | `lamquant-core` dev-dep | always (dev) | 1/1 pass (`realfft_roundtrip_8_point`) | No in-tree real-valued FFT today. Use when fullband spectrogram lands. |
| `fixed 1` | `lamquant-firmware` | `cat-b-fixed` | 1/1 pass (`fixed_q15_add_mul_smoke`) | Hand-rolled raw `i32` Q-format is faster + simpler. Use `fixed` only for new code where the type-safety wins outweigh the wrapper cost. |
| `microfft 0.6` (size-1024) | `lamquant-firmware` | `cat-b-microfft` | 1/1 pass (`microfft_8_point_dc_smoke`) | Hazard3 has no on-MCU FFT today. When spectrogram-on-device lands, microfft is the right choice. |
| `idsp 0.21` | `lamquant-firmware` (smoke) + `lamquant-core` (bench) | `cat-b-idsp` + dev-dep | 1/1 pass + biquad A/B | **idsp wins Q-format by ~30%** (empirical 2026-05-22, see "Empirical bench" section below). f32 tied (2.11 vs 2.09 GiB/s); Q30 i32 idsp 1.74 GiB/s vs hand-rolled 1.34 GiB/s. firmware swap is a real perf opportunity post-ship. Wire-format-locked biquad coeffs gate the actual replacement. |

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

## Empirical bench commands + results (2026-05-22)

```
cargo bench --features "host experimental_arithmetic" \
  -p lamquant-core --bench cat_b_compare
```

Two harnesses live in `lamquant-core/benches/cat_b_compare.rs`:

1. **Synthetic** (`cat_b_entropy_encode_1250_residuals`) — xorshift
   ±15 i64 stream, mimics a perfectly-decorrelated LPC residual.
2. **Real CHB-MIT** (`cat_b_entropy_encode_chbmit_real`) — loads
   `chb01/chb01_01.edf` from `/mnt/4tb/data/Archive/edf/physionet/
   chbmit/` (with fallback to `/mnt/4tb/data/Archive/lma/physionet/
   chbmit.lma`), runs `lpc::analyze(channel0[..1250], order=8)`,
   feeds the resulting residual stream to both encoders.

Output on `onyx-maurader-BrianBigPC`:

### Synthetic ±15 (favourable to rANS — matches its Laplace prior)

```
golomb_rice       bytes=864  (5.53 bits/sym)   thrpt=2.43 GiB/s
constriction_rans bytes=848  (5.43 bits/sym)   thrpt=381 MiB/s
                  (-1.9%)                       (6.5x slower)
```

### Real CHB-MIT chb01_01 channel 0 (post-LPC residual)

```
golomb_rice       bytes=1251 (8.01 bits/sym)   thrpt=2.35 GiB/s
constriction_rans bytes=1260 (8.06 bits/sym)   thrpt=403 MiB/s
                  (+0.7% BIGGER)                (5.8x slower)
```

**Verdict (real data):** Constriction loses on **both** axes. The
prior synthetic A/B was an unrealistic best-case for rANS — the
xorshift ±15 distribution matches constriction's static Laplace
model. Real CHB-MIT EEG residuals are heavier-tailed; golomb-rice's
per-block parameter adaptation handles them with smaller output AND
much higher throughput. Keep constriction behind
`experimental_arithmetic` for differential testing only — DO NOT
promote to default.

### idsp biquad A/B (2026-05-22 — bench `cat_b_biquad_df1_*`)

```
cat_b_biquad_df1_f32_1024_samples
  handrolled_df1_f32        2.11 GiB/s
  idsp_directform1_f32      2.09 GiB/s     (tie, ±1% noise)

cat_b_biquad_df1_q30_i32_1024_samples
  handrolled_df1_q30_i32    1.34 GiB/s
  idsp_directform1_q30_i32  1.74 GiB/s     (+30% — idsp wins)
```

**Verdict (firmware-realistic Q-format):** idsp's `Biquad<Q<i32, i64,
30>>` + `DirectForm1Wide` is ~30% faster than a clean hand-rolled
DF1 i32 path with the same `i64` accumulator + Q30 round-then-shift.
The win likely comes from idsp's tighter MULH/MULL instruction
scheduling (verified on x86_64; effect on RP2350 Hazard3 RV32IMAC
unknown until benched on-MCU).

NOT swapped into firmware yet — `HpFilterBank::run` is on the
wire-format-locked path (biquad coefficients are part of the
encoder/decoder contract). Any swap needs (a) a Hazard3-target
bench to confirm the +30% transfers, (b) coefficient-format
audit so the swap stays bit-equivalent, and (c) a full
conformance roundtrip pass. Treat as a tracked post-ship perf
opportunity, not a blocker.

Next perf experiments (deferred — separate sessions):
- pulp on lifting kernel (current 800 MiB/s; target +30%)
- idsp Q30 i32 bench on Hazard3 RV32IMAC target to confirm the
  +30% generalises off x86_64
- constriction differential test at much larger block sizes (16k+
  residuals) to find the actual rate/throughput crossover
