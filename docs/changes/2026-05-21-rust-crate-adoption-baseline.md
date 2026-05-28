# Rust Crate Adoption — Baseline (2026-05-21)

Snapshot captured before Phase-Plan Category A adoption (insta, cargo-fuzz
expansion, heapless migration, defmt, postcard, crc).  Use this file to
diff against post-adoption state.

## Workspace at HEAD `5b791ad`

- 17 workspace crates direct
- 478 unique transitive Cargo dependencies (from `cargo tree --workspace --prefix none --no-dedupe | sort -u | wc -l`)

## Per-crate dependency manifest (load-bearing only)

### `lamquant-core` (host + wasm + python bindings)

Already in:
- `sha2 = "0.10", default-features = false` — load-bearing for SHA chain
- `serde`, `serde_json`, `tempfile`, `libm`, `base64`
- optional features: `constriction 0.4` (behind `experimental_arithmetic`),
  `pyo3 0.25` + `numpy 0.25` (python), `wasm-bindgen`, `aes-gcm`, `hmac`,
  `zeroize`, `argon2`, `tokio`, `reqwest`, `parquet`, `arrow-array/schema`,
  `hdf5-metno`, `ndarray`, `dicom-object/core`, `aws-config/sdk-s3`,
  `keyring`, `notify`, `ratatui 0.30` + tui-* (legacy lml TUI)
- dev: `criterion 0.5`

Not present (Category A targets):
- **`crc`** — CRC-32 should be explicit; verify whether implemented native or absent entirely
- **`insta`** — no snapshot harness
- **`proptest`** — only ad-hoc roundtrip tests today

### `lamquant-firmware` (RP2350 Hazard3 RISC-V no_std)

Already in:
- `lamquant-core` (default-features=false)
- `lamquant-weights` (features=`subband_v1`)
- `embedded-alloc 0.6`
- `libm 0.2`
- `riscv 0.12` (target-conditional)

Not present (Category A targets):
- **`defmt`, `defmt-rtt`, `panic-probe`**
- **`heapless 0.8`**
- **`postcard`**
- **`fixed`** (Q-format arithmetic on no-FPU Hazard3)

Source: 7,412 LOC across `src/{main,lib,scheduler,power,stack_guard,
integrity}.rs + neural/{ssm_block,layer_norm,fsq,focal,ternary_mac,snn,mod}.rs +
afe/{ads1299,mod}.rs + safety/{state,mod}.rs + codec/{hybrid_entropy,
fsq_adaptive,lpc_delta,mailbox,detail_threshold,quality,rans_context,mod}.rs +
transport/{usb,ble,mod}.rs + dsp/{lpc,lifting,biquad,wht,mod}.rs`

Hot-path heap users flagged in audit (2026-05-21):
- `src/neural/focal.rs:150` — `vec![vec![0i16; t]; channels]` (per-call, MIGRATE)
- `src/transport/usb.rs:108` — `VecSink(Vec::new())` (init-time, keep)

### `crates/lamquant-ops`

Already in: `serde`, `serde_json`, `sha2 = "0.10"`; dev: `jsonschema 0.30`.

### `blut`

Already in: `serde`, `serde_json`, `serde_yaml`, `thiserror`, `anyhow`,
`tracing`, `tokio`, `async-trait`, `nix`, `dirs`, `parking_lot`, `clap`,
`humantime`, `rusqlite`, `tempfile`, `sha2`, `memmap2`, `faster-hex`,
`uuid`, `toml`, `schemars`, `tokio-util`, `rayon`, `which`, `bincode`,
`rand`, `ratatui`, `crossterm`, `fuzzy-matcher`.

### `crates/lmafs`

Already in: `lamquant-core` (features=host), `fuser 0.14`, `libc`, `clap`,
`tracing`.

### `crates/lamquant-history`

Already in: `serde`, `serde_json`.

## Fuzz coverage (existing)

Targets in `lamquant-core/fuzz/fuzz_targets/`:

1. `container_header.rs` — LML container header parser
2. `decompress.rs` — LML decompress path
3. `lma_manifest.rs` — LMA manifest JSON
4. `offset_table.rs` — LMLFOOT1 offset table
5. `roundtrip.rs` — encode→decode roundtrip property

Per the audit, the lmafs FUSE attack surface goes through `lma_read_entry`
+ container parser + LML decoder, all of which are nominally covered by the
five existing targets.  Gaps to verify:
- FSQ rANS decoder path (per `lamquant-firmware/src/codec/rans_context.rs`
  mirror on host side) — likely under `decompress.rs` but confirm.
- `LmlReader::next_window` random-access seek with random offset-table inputs.

## Test counts (pre-commit baseline)

Per recent commit logs: 446 tests pass in ~12.4 s across codec (386),
container (16), edf_reader (44).  Captured by the `pre-commit` hook on
every commit; verify post-adoption matches or grows.

## Bench baselines (criterion, encode/decode throughput)

Recent commits `bce08c6` + `18f1a78` + `f281a70`:
- `stage_lifting_forward_3level/2500`    ~786 MiB/s
- `stage_lpc_analyze_with_mode/anytime`  ~362 MiB/s  ← bottleneck
- `stage_lpc_analyze_with_mode/fixed`    ~486 MiB/s  (~30 % faster)
- `stage_golomb_encode_dense/1250`       ~607 MiB/s
- end-to-end ~144 MiB/s on 4-channel × 2500-sample windows

## Firmware build status

Not built this session.  Pre-W work shipped per-stage MAC kernels and
linker overlays (memory entry #50, #81, #90); flash budget ~449.9 / 520 KB
(memory entry "Firmware memory model").

## Diff plan post-adoption

After landing Cat A (A1 sha2/crc, A2 insta, A3 cargo-fuzz expansion,
A4 proptest, A5 defmt, A6 heapless, A7 postcard, A8 probe-rs, A9 size
audits, A10 nextest):

| Metric | Before | After (target) |
|---|---|---|
| Workspace transitive deps | 478 | +5–10 (defmt+heapless+postcard+crc+insta+proptest are small) |
| Firmware Cargo deps | 4 (core, weights, embedded-alloc, libm) | ~8 (+defmt, defmt-rtt, panic-probe, heapless, postcard, fixed?) |
| Firmware flash size | unknown today | re-measure via `cargo-bloat` |
| Fuzz targets | 5 | ≥6 (verify FSQ rANS explicit, add lmafs end-to-end if useful) |
| Test count | 446 | ≥446 + new proptest harnesses + insta snapshots |
| Bench numbers | as above | re-run after each commit; expect no regression |
