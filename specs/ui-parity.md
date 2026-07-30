# UI Parity Spec

Single source of truth for the LamQuant front-end inventory. Three implementations must match this document exactly:

1. **Python TUI** — `lamquant.py` + `lamquant_codec/cli/*`
2. **Rust+Ratatui TUI** — `lamquant-core/src/tui/*` + `lamquant-core/src/bin/lml.rs`
3. **Tauri+Svelte 5 GUI** — `gui/`

When any of the three drifts from this spec, the spec wins. Drift is a CI failure via `crates/lamquant-ops/tests/schema_parity.rs`.

## Workflow inventory (10 top-level)

The main menu is hub-and-spoke. Workflow order, key, and screen ID are all canonical. Front-ends MAY render differently (TUI as a list, GUI as tiles) but the order, keys, and IDs do not vary.

| Key | ID | Title | Subtitle |
|---|---|---|---|
| `1` | `lossless` | LML Lossless | Encode / Decode / Verify / Info / Stats / Recover |
| `2` | `neural` | LMQ Neural | Same shape as LML, neural codec |
| `3` | `codec_hub` | Codec Hub | Browse, copy-path, diff |
| `4` | `eagle` | Eagle Validation | quick + LQS-{L,C,M,A} + perf + RD + H2H |
| `5` | `firmware` | Firmware Hub | Toolchain status + export + build + size + flash |
| `6` | `train` | Training Cockpit | Pipeline + GPU + tmux + presets + runs + leaderboard + queue + metrics |
| `7` | `vision` | Vision (live device) | Impedance + signals + recording/export + compare |
| `s` | `settings` | Settings | 9 sections, `LamQuantConfig` editor with search/help |
| `i` | `setup` | Install / Syscheck | Wizard + benchmarks + recommended config |
| `t` | `test` | Diagnostics | pytest runner + KAT + CRC + crash log viewer |

Auxiliary keys (consistent across all three UIs): `h` help, `?` what's new, `q` quit, `x` exit, `b` back, `Esc` back, `Enter` confirm, `Ctrl+C` cancel running op (on second press, route through exit-confirm panel).

## Screen IDs (per workflow)

Workflow → list of well-known screen IDs. New screens MUST land in this table before they're implemented in any UI. IDs are `snake_case`, ASCII, ≤32 bytes.

### `lossless` / `neural` (same shape)
- `lossless` / `neural` — landing page (5 ops)
- `lossless.encode` — input + output picker → run
- `lossless.decode`
- `lossless.verify` — input only
- `lossless.info` — input only, sync result
- `lossless.stats` — input only, sync result
- `lossless.recover` — input + output, recovery from corrupted file

### `codec_hub`
- `codec_hub` — landing
- `codec_hub.browse` — modal: recursive list of `.lml`/`.lmq`
- `codec_hub.copy_path` — modal: search + numbered selection → clipboard
- `codec_hub.diff` — input1 + input2 picker

### `eagle`
- `eagle` — landing (9 sub-modules)
- `eagle.lqs` — sub-landing (level picker) → runner
- `eagle.lqs.{l,c,m,a}` — runner per level
- `eagle.quick` — 10-window sanity check
- `eagle.perf` — throughput + p50/p95/p99
- `eagle.rd` — rate-distortion sweep
- `eagle.h2h` — head-to-head vs gzip/zstd
- `eagle.downstream` — stub (Phase 3 in TUI)
- `eagle.hallucination` — stub
- `eagle.badge` — stub
- `eagle.leaderboard` — stub
- `eagle.metrics_explorer` — stub
- `eagle.preflight.hardware` — device hardware self-test (existing GUI)
- `eagle.preflight.software` — pytest + C host tests
- `eagle.preflight.benchmarks` — host benchmarks

### `firmware`
- `firmware` — landing
- `firmware.toolchain` — env probes
- `firmware.export` — export weights from checkpoint
- `firmware.configure` — cmake configure for target
- `firmware.build` — make -j
- `firmware.size` — riscv64-elf-size on artifact
- `firmware.flash` — UF2 + bootloader detect + flash

### `train` (cockpit)
- `train` — landing dashboard (16 panels visible)
- `train.pipeline` — data prep status
- `train.gpu` — GPU/CPU/RAM probes
- `train.tmux` — active sessions, attach/view
- `train.presets` — fast/medium/production
- `train.hyperparams` — show preset config
- `train.runs` — historical CSV logs
- `train.leaderboard` — best-R per run
- `train.compare` — side-by-side metrics
- `train.queue` — stage list with status pills
- `train.metrics` — live tail-f or matplotlib
- `train.experiments` — sweep launcher
- `train.reset` — destructive reset options

### `vision`
- `vision` — landing
- `vision.impedance` — head diagram + per-channel readings
- `vision.signals` — live waveform + recording controls
- `vision.export` — post-recording export
- `vision.compare` — codec mode comparison
- `vision.firmware` — alias for `firmware.flash` to preserve clinical bookmarks

### `settings`
- `settings` — list + search
- `settings.help` — full-screen help for current field

### `setup` (install)
- `setup` — landing
- `setup.wizard` — 4-step quick setup
- `setup.syscheck` — system probe + recommended config
- `setup.bench` — compress benchmark + corpus estimate

### `test` (diagnostics)
- `test` — landing
- `test.conformance`, `test.codec`, `test.full`, `test.paranoid` — runners
- `test.crash_logs` — view ~/.lamquant_crashes/

### Global
- `main` — top-level workflow hub (10 tiles / 10 menu lines)
- `exit_confirm`, `help`, `tutorial`, `root_warn`, `resume`

## Op contract

Every long-running operation in any UI MUST flow through `lamquant-ops`. The wire
format is the JSON Schema at `specs/plan-projections.schema.json`. Front-ends do
not invent new protocol payloads — extending the schema requires the parity test
to pass first.

### Op IDs (canonical strings)

These map 1:1 to the `operation` field of the initial `planned` projection and
to the Rust launcher registry. This marker block is machine-checked against
every codec, external, BLUT, and parameterized-install operation:

<!-- canonical-operation-ids:start -->
- `encode_lma`
- `encode_lml_siblings`
- `decode`
- `verify`
- `info`
- `stats`
- `bench`
- `archive`
- `extract`
- `list_archive`
- `verify_archive`
- `verify_manifest`
- `export_csv`
- `export_npy`
- `export_raw`
- `recover`
- `diff`
- `train_encoder`
- `train_snn`
- `train_tnn`
- `train_resume`
- `eagle_quick`
- `eagle_full`
- `eagle_bench`
- `eagle_lqs_l`
- `eagle_lqs_c`
- `eagle_lqs_m`
- `eagle_lqs_a`
- `eagle_perf`
- `eagle_rd`
- `eagle_h2h`
- `test_conformance`
- `test_full`
- `test_paranoid`
- `test_codec`
- `setup_pip`
- `setup_extras`
- `setup_cargo`
- `setup_musl`
- `setup_windows`
- `gui`
- `viz_lamquant-gui`
- `viz_eeglab`
- `viz_mne`
- `viz_legacy_OpenBCIGUI`
- `viz_legacy_BVAnalyzer`
- `viz_legacy_besa`
- `viz_OpenBCIGUI`
- `viz_BVAnalyzer`
- `viz_besa`
- `viz_install_lamquant_gui`
- `viz_install_mne`
- `viz_install_scope_tui`
- `viz_install_bottom`
- `viz_install_television`
- `viz_install_csvlens`
- `viz_install_gitui`
- `viz_uninstall_lamquant_gui`
- `viz_uninstall_mne`
- `viz_uninstall_scope_tui`
- `viz_uninstall_bottom`
- `viz_uninstall_television`
- `viz_uninstall_csvlens`
- `viz_uninstall_gitui`
- `cockpit_reset`
- `cockpit_checkpoints`
- `cockpit_metrics`
- `fw_list_devices`
- `fw_build_rp2350`
- `fw_build_nrf54l15`
- `fw_build_esp32p4`
- `fw_build_stm32n6`
- `fw_flash_rp2350`
- `fw_flash_nrf54l15`
- `fw_flash_esp32p4`
- `fw_flash_stm32n6`
- `fw_size_rp2350`
- `fw_size_stm32n6`
- `fw_size_esp32p4`
- `fw_size_nrf54l15`
- `fw_check_rp2350`
- `fw_check_stm32n6`
- `fw_check_esp32p4`
- `fw_check_nrf54l15`
- `fw_export`
- `fw_legacy_esp32s3`
- `cockpit_jobs`
- `cockpit_export`
- `syscheck_py`
- `cockpit_data_prep`
- `cockpit_train_encoder`
- `cockpit_train_snn`
- `cockpit_train_oracle`
- `setup_install_lml`
- `setup_install_eagle`
- `setup_install_lqt`
<!-- canonical-operation-ids:end -->

Adding a new op ID requires:
1. Entry here in `ui-parity.md`
2. Entry in the applicable Rust codec/launcher registry
3. Exact registry and front-end parity tests passing

## Lexicon (frozen)

Drift-prone synonym pairs. Spec freezes the canonical token; the other is a lint failure.

| Canonical | Banned synonyms |
|---|---|
| `encode` | `compress` |
| `decode` | `decompress` |
| `verify` (file integrity) | (preflight is NOT a synonym) |
| `preflight` (device hardware self-test) | (verify is NOT a synonym) |
| `info` | `inspect`, `show` |
| `stats` | `statistics` |
| `archive` | `pack` |
| `extract` | `unpack` |
| `recover` | `repair` |
| `export {csv|npy|raw}` | `dump`, `convert` |
| `bench` | `benchmark` (the noun is fine, the op id is `bench`) |
| `diff` | `compare` (compare is a Vision route name, not an op) |

UI labels MAY use a longer human-readable form (e.g. "Encode (LML lossless)") but the underlying op ID and route token use the canonical.

## State files

| Path | Schema | Writer | Readers |
|---|---|---|---|
| `lamquant.toml` (search: `./` → `$XDG_CONFIG_HOME/lamquant/` → `/etc/lamquant/`) | `specs/config-schema.yaml` | `lml settings --apply` only | all three UIs |
| `~/.config/lamquant/history.json` (Linux) ; `~/Library/Application Support/lamquant/history.json` (macOS) ; `%APPDATA%\lamquant\history.json` (Windows) | `specs/history-schema.json` | any UI via fcntl-locked atomic write | all three UIs |
| `<output>/.lamquant_state.json` | `specs/resume-state-schema.json` | `lml encode/decode/verify` | all three UIs (resume detection at boot) |
| `~/.lamquant_crashes/<ts>.txt` | (free-form text, not JSON) | each UI's panic hook | user only |

Round-trip rule: writes go through ONE binary per file (`lml` for state mutation, `lml settings --apply` for TOML). No UI directly serializes TOML or `history.json` from its own code path. Readers MAY parse directly.

## Bindings

Schemas in `specs/` are language-neutral. Each UI/runner must keep plan-projection
bindings in sync with the protocol:

- **Rust:** `crates/lamquant-ops` exports binding types for `PlanProjection`.
- **Python:** `lamquant_codec/cli` keeps Python bindings consistent with `plan-projections`.
- **TS:** `gui/` keeps TypeScript bindings consistent with `plan-projections`.

Producers may emit `planned`, `progress`, `artifact`, and `diagnostic`
observations while executing. They never invent terminal `receipt` or `failure`
projections; the supervising graph `PlanExecutor` owns terminal
`Receipt`/`Failure`.

Product process plans MUST be constructed from the sealed canonical operation
registry. They declare the `source-bound-first-party-v1` implementation class
and one compatible versioned observation protocol: `exit-status-v1`,
`plan-projection-v1`, or `blut-status-jsonl-v1`. Third-party Python, CUDA, C++,
vendor, and legacy Node implementations MUST NOT use this runner. They require
an Ed25519-verified capability manifest and the BPC2 v2 control protocol through
the dedicated supervised-process transport. Raw command construction is
test-feature-only and is not part of the product API.

Registered operations whose semantics depend on checkout-relative paths MUST
provide an explicit canonical working directory. That directory is hashed into
the compiled plan and revalidated before spawn. Every interpreted script or
source-tree root declared by that registered operation is recursively hashed
into the same plan and revalidated immediately before spawn. Generated caches
are excluded by fixed runtime policy; callers cannot supply their own source
closure.

Vision software preflight uses `test_full`; benchmark preflight uses
`eagle_bench`. Both run through one app-lifetime `useOp` handle per operation
kind, render executor-issued terminal receipts plus bounded diagnostic history,
and expose invocation-scoped cancellation. One operation kind cannot have two
active GUI invocations. Required benchmark skips and zero-audit runs fail
closed. Direct Tauri test, benchmark, or Python-decoder subprocess commands are
forbidden.

The schema parity test in `crates/lamquant-ops/tests/schema_parity.rs` round-trips
a YAML/JSON fixture through all three bindings. If any of them rejects the
fixture, the build fails.

## Versioning

This spec is `parity-version: 2`. Version 2 replaces the retired `OpEvent`
protocol with executor-owned plan projections. Bumping the version is a
deliberate breaking change.

Every front-end operation MUST load `history.json` through the checked shared
reader before compiling a process plan. A missing file creates a version-2
history. A file missing `parity_version` is version 1. Malformed history and
any version mismatch fail closed before execution. Normal history writes never
change parity. Completing the setup wizard is the explicit migration action:
it preserves compatible history fields, writes `schema_version: "2.0"` and
`parity_version: 2`, then permits operations. Downgrade remains an alternative.
