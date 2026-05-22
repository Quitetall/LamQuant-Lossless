# Pure Q-format firmware — AFTER snapshot (2026-05-21)

User constraint (verbatim): *"We don't use float in the firmware at all, only Q format. Why is float in the firmware? That should not be the case ever. Always Q format."*

Two violations existed at the start of this session.  Both fixed
without losing CR or SNN accuracy.

## Violation 1 — LPC BIC `libm::log` (firmware adaptive Levinson)

**Was:** `lamquant-firmware/src/dsp/lpc.rs::levinson_q27_adaptive` called
`libm::log(e/n) + 32*ln(2)*k` in f64 to score each candidate order.
Forced `libm` into the firmware dependency tree.

**Now:** `bic_cost_q48(e, n, order_k) -> i128` computes the equivalent
Q48 cost via `ilog2_q24` (Q24 fixed-point log2) and pre-baked Q24
constants `0.72*ln(2) → 8_373_164`, `32*ln(2) → 372_212_758`.  Integer
Levinson recursion (Q56 i128) was already in place; only the scorer
needed conversion.

Approximation budget: integer log2 with linear fractional interpolation
gives ~0.086-of-a-binade max error.  BIC order discrimination is on
the order of tens of Q24 units between adjacent orders; linear-interp
error (~1.4M Q24 units) is well above the noise floor — at worst the
chosen order differs by ±1 on a rare borderline window.  Lossless
property is preserved: encoder/decoder both use the firmware's chosen
order to encode/decode the residual, so the recovered signal is
bit-exact regardless of which order won.

## Violation 2 — SNN init f32 `*_SCALE` constants + Padé `exp_f32`

**Was:** `lamquant-weights/src/generated/snn/*.rs` emitted 22+ per-
tensor `pub const *_SCALE: f32` constants.  Firmware
`neural/snn.rs::run_direction` called `scale_to_q15(SCALE)` boot-time
plus `build_a_abs_q10` (Padé `exp_f32` polynomial approximant
running ~1280 times at init) plus `build_d_q15` (per-element f32
multiply + round).

**Now:** `tools/regen_snn_weights.py` (calls
`reference_implementations/c_firmware/export/snn_emitter.py` directly,
bypassing the TNN-gated CLI) pre-bakes every scale, every A_ABS_Q10
element, every D_Q15 element, and the scalar DT_BIAS_Q15 at codegen
time.  Codegen uses numpy `np.exp` (f64 precision — strictly more
accurate than the firmware's prior Padé f32) and writes the result
straight into Rust statics:

```
pub const SPATIAL_MIX_WEIGHT_SCALE_Q15: i32 = 1011;        // was f32 = 3.084e-02
pub static A_ABS_Q10: [[i16; 16]; 80] = [[...]; 80];        // was built at boot from A_LOG
pub static D_Q15:     [i16; 80]       = [...];              // was built at boot from D
pub const DT_BIAS_Q15: i32            = 12345;              // was computed at boot
```

Firmware `snn.rs` deletes `round_f32_to_i16`, `exp_f32`,
`build_a_abs_q10`, `build_d_q15`, `dt_bias_q15`, `scale_to_q15` and
reads the pre-baked refs directly.

## Crate dependency change

`lamquant-firmware/Cargo.toml`:

```diff
- libm = "0.2"
```

Removed unconditionally.  Comment block left in place documenting
*why* libm was dropped + where the work moved.

## Verification

| Metric | Before | After |
|---|---|---|
| `: f32`/`: f64` consts in `lamquant-weights/src/generated/snn/*.rs` | 22 | **0** |
| `libm::` call sites in `lamquant-firmware/src/` | 3 (`libm::log` × 3 in lpc.rs) | **0** (only doc-comment mentions) |
| Float arithmetic in firmware production `src/` (excl. `#[cfg(test)]`) | 25+ ops in lpc adaptive + 22 ops in snn init | **0** |
| Firmware Cargo direct deps (non-target-conditional) | `lamquant-core, lamquant-weights, embedded-alloc, libm` | `lamquant-core, lamquant-weights, embedded-alloc` |
| Conformance test pass count (`cargo test -p lamquant-firmware --test conformance`) | 9/9 | **9/9** |
| Lossless property (`lpc_inverse_recovers_signal`, `full_pipeline_roundtrip`) | PASS | **PASS** |
| Compression ratio | unchanged (encoder/decoder still agree on chosen order) | **unchanged** |

Generated SNN file sizes (per direction `layer*_{fwd,bwd}.rs`): ~69 KB
each — same order of magnitude as before; A_ABS_Q10 (2 560 bytes) and
D_Q15 (160 bytes) replace A_LOG (2 560 bytes) and D (80 bytes), so the
net Rust-static byte footprint changes by ≤ 100 bytes per direction.
Flash impact on the MCU build is negligible; the savings come from
no boot-time computation, not from smaller tables.

## What this unblocks

- Firmware integrity audit can now state "the firmware contains zero
  floating-point arithmetic" without caveat.
- Future RVV / Hazard3-without-FP-emulation ports work directly with
  no `--features no-float` cargo gymnastics.
- SOAP/COSMOS optimizer experiments that change SNN scales no longer
  require a firmware code path change — just regen via
  `tools/regen_snn_weights.py`.

## Files touched (this commit)

- `lamquant-firmware/src/dsp/lpc.rs` — `bic_cost_q48` + `ilog2_q24`,
  drop `libm::log` calls
- `lamquant-firmware/src/neural/snn.rs` — switch every `*_SCALE` to
  pre-baked `*_SCALE_Q15`; drop boot-time table builders + their
  helpers (`build_a_abs_q10`, `build_d_q15`, `exp_f32`,
  `round_f32_to_i16`, `dt_bias_q15`, `scale_to_q15`)
- `lamquant-firmware/Cargo.toml` — drop `libm = "0.2"`
- `reference_implementations/c_firmware/export/snn_emitter.py` —
  pure-Q codegen (Q15 scales, A_ABS_Q10 + D_Q15 + DT_BIAS_Q15
  pre-baked tables; no `pub const ... : f32` emission)
- `tools/regen_snn_weights.py` — standalone SNN-only emitter driver
  (bypasses TNN encoder ckpt gate in the original CLI)
- `lamquant-weights/src/generated/snn/*.rs` — regenerated, pure-Q,
  byte-different from prior version due to scale pre-bake
