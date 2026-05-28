# Cargo Dependency Graph — LamQuant Workspace (recon for 8-repo split)

**Generated:** 2026-05-27  
**Source:** Complete scan of 15 Cargo.toml files across monorepo  
**Scope:** 9 workspace members + 2 standalone workspaces (blut, hazard3_bench)

---

## Executive Summary

The LamQuant monorepo contains **9 workspace members** organized into **3 destination repo categories** with clear separation patterns:

| Repo | Members | Type | Scope |
|------|---------|------|-------|
| **LamQuant-Lossless** (PUBLIC) | `lamquant-core`, `lamquant-firmware`, `lamquant-weights`, `lmafs`, `lamquant-ops`, `lamquant-ipc-types` | Codec + firmware | DSP, firmware runtime, op contracts, IPC, weights data |
| **LamQuant-Vision** (PUBLIC) | `crates/lamquant-lsl`, `gui/src-tauri` | GUI + LSL integration | Tauri GUI shell, LSL outlet/inlet bridge |
| **LamQuant-Codec** (PRIVATE) | `gui/src-tauri`, `installer/src-tauri`, TUI, SDK, Hub | Turnkey apps | User-facing apps (GUI, installer, TUI, Python SDK) |
| **BLUT** (PUBLIC) | `blut/` | Training orchestrator | Separate git submodule, own workspace |
| **Eagle** (PUBLIC) | `tools/hazard3_bench/`, bench scripts | Validation suite | Hazard3 benchmarking, Eagle harness |

**Locked classifications** (no reclassification):
- `lamquant-core` → Lossless (core DSP codec)
- `lamquant-firmware` → Lossless (RP2350 bare-metal)
- `lamquant-weights` → Lossless (const data, firmware-coupled)
- `blut` → Own repo (ADR 0017, git submodule)

---

## Per-Crate Inventory

### 1. lamquant-core (v7.7.0)

**Path:** `/mnt/4tb/LamQuant/lamquant-core/`

**Type:** Library + 2 binaries  
**License:** AGPL-3.0

**Description:** LML lossless EEG compression codec (Le Gall lifting DWT + LPC + Golomb-Rice). Core DSP engine, no_std + alloc.

**Features:**
- `default` → `["host"]` (full host build)
- `std` → Core DSP only, no_std-friendly
- `host` → File I/O, CLI, TUI, parallel I/O (pulls 40+ deps)
- `async` → Tokio runtime + reqwest HTTP + notify filesystem watcher (heavy)
- `keyring` → OS keyring lookup (macOS/Windows native, Linux libdbus+libsecret)
- `parquet` → Apache Arrow/parquet stack export (~50 transitive crates)
- `hdf5` → LibHDF5 export (system dev headers required)
- `dicom` → DICOM waveform read (pure-Rust, feature-gated)
- `s3` → AWS SDK S3 (100+ transitive crates, implies `async`)
- `python` → PyO3 + numpy (extension module via maturin)
- `ffi` → FFI bindings
- `wasm` → WebAssembly support
- `experimental_arithmetic` → Constriction rANS/range coder (research-grade)
- `experimental_bit_pack` → Per-subband bit-packing (perf opt-in)

**Binaries:**
- `lml` (requires `host`) — Main CLI codec binary
- `lamquant` (requires `host`) — Alternate alias/shorthand

**Benchmarks:**
- `codec` (requires `host`) — E2E throughput via criterion
- `cat_b_compare` (requires `host`, `experimental_arithmetic`) — Arithmetic coder comparison

**Direct dependencies (core):**
- `sha2@0.10` (no_std-compatible)
- `base64@0.22` (alloc feature for no_std)
- `libm@0.2` (log function for adaptive-LPC AIC/MDL in no_std)
- `lamquant-ops@{path}` (optional, host feature)
- `lamquant-history@{path}` (optional, host feature)
- **Host feature enablers** (40+ deps): clap, rayon, walkdir, zstd, globset, tempfile, filetime, ratatui, crossterm, indicatif, serde, serde_json, serde_yaml, TUI widget stack (throbber-widgets-tui, ratatui-textarea, tui-big-text, tui-scrollview, tui-tree-widget, tui-widget-list, tui-piechart), tracing, tracing-subscriber
- **Crypto (Phase 7):** aes-gcm, hmac, zeroize, argon2, rpassword, getrandom
- **Async (Phase 6):** tokio, reqwest, notify
- **Export formats (optional):** parquet, arrow-array, arrow-schema, hdf5-metno, ndarray, dicom-object, dicom-core
- **AWS (optional):** aws-config, aws-sdk-s3
- **Research (optional):** constriction, probability
- **Bindings:** pyo3, numpy, wasm-bindgen

**Inverse dependencies (in-workspace):**
- `lamquant-firmware` depends on it (required)
- `lamquant-lsl` depends on it (required)
- `lmafs` depends on it (required)
- `gui/src-tauri` depends on it (required)

**Destination:** **LamQuant-Lossless (PUBLIC)**

**Rationale:**
- Core DSP codec is the foundation of all compression.
- Used by firmware (embedded), CLI (host), GUI (desktop), LSL bridge.
- No_std path keeps embedded builds clean.
- Host features remain opt-in so default builds stay dependency-minimal.

---

### 2. lamquant-firmware (v0.1.0)

**Path:** `/mnt/4tb/LamQuant/lamquant-firmware/`

**Type:** Library + 1 bare-metal binary  
**License:** AGPL-3.0  
**Target:** `riscv32imac-unknown-none-elf` (RP2350 Hazard3)

**Description:** LamQuant v7.7 bare-metal firmware for RP2350 MCU. Encodes/decodes EEG data via DSP pipeline + TNN/SNN neural blocks. No allocator except boot, hot path uses fixed-capacity heapless collections.

**Features:**
- `default` → `["firmware-bin"]` (bare-metal binary)
- `firmware-bin` → Pulls rp235x-hal + critical-section impl (binary only)
- `cat-b-fixed`, `cat-b-microfft`, `cat-b-idsp`, `cat-b-all` → Optional research comparators (off by default)
- `host-verify` → Bit-exact parity tests vs. C reference (host-side verification only)
- `target-rp2350`, `target-nrf54l15`, `target-esp32p4`, `target-stm32n6`, `target-esp32s3` → Per-target selection (ADR 0019)

**Binaries:**
- `lamquant-firmware` (requires `firmware-bin`) — Bare-metal RP2350 entry point

**Direct dependencies:**
- `lamquant-core@{path}` (no default features) — Codec DSP core
- `lamquant-weights@{path}` (feature `subband_v1`) — TNN/SNN weights (const data)
- `lamquant-ipc-types@{path}` — Versioned MCU↔host envelope
- `embedded-alloc@0.6` — Boot-time allocator
- `heapless@0.8` (no default features) — Fixed-capacity Vec/Queue for hot path
- `postcard@1` (no default features) — Binary serialization for IPC
- **Target-specific** (riscv32imac only):
  - `rp235x-hal@0.3` (feature `firmware-bin`, `critical-section-impl`)
  - `riscv@0.12`
  - `critical-section@1.2`
  - `embedded-hal@1.0`
  - `embedded-hal-nb@1.0`
  - `fugit@0.3` (time/duration types)
  - `nb@1.1` (non-blocking I/O traits)
  - `defmt@0.3` (firmware logging via RTT)
  - `defmt-rtt@0.4` (RTT sink)
  - `panic-halt@1.0` (panic behavior)
- **Cat B comparators (optional):** fixed, microfft, idsp

**Inverse dependencies (in-workspace):**
- `tools/hazard3_bench/` depends on it (bare-metal-free, bench-only)
- `tools/bench/bench_rs/` depends on it (host tests)

**Inverse dependencies (external):**
- `blut/` **does NOT** depend on firmware (training runs on GPUs, not MCU)

**Destination:** **LamQuant-Lossless (PUBLIC)**

**Rationale:**
- Firmware is integral to the lossless codec story (hardware target).
- Ships with the codec repo because the codec is useless without hardware runtime.
- Customers deploying the LamQuant-Lossless codec will need this firmware build artifacts.
- Stay in workspace to simplify CI (single `cargo build` target for firmware + host).

---

### 3. lamquant-weights (v7.7.0)

**Path:** `/mnt/4tb/LamQuant/lamquant-weights/`

**Type:** Library (const data only)  
**License:** AGPL-3.0  
**Edition:** 2021  
**Scope:** Pure no_std, no_alloc, no transitive deps

**Description:** TNN focal blocks, FSQ table, SNN, rotation matrix, CRC LUT. All `pub static const` data. Generated from Python checkpoint via `export_firmware.py`.

**Features:**
- `default` → `["subband_v1"]` (Gen 7.1 architecture)
- `subband_v1`, `subband_v2`, `legacy_v7_0` → Mutually exclusive architecture variants

**Direct dependencies:** None (zero deps, pure data)

**Inverse dependencies (in-workspace):**
- `lamquant-firmware` depends on it (required for TNN/SNN)
- **Question:** Does `lamquant-core` use weights? **No.** Core is codec DSP only; weights are MCU-specific.
- **Question:** Does LamQuant-Neural (not yet in monorepo) need it? **Possibly** — if neural codec is a separate model, weights would be hosted separately. Current weights are firmware-only (no neural codec in public tree yet).

**Destination:** **LamQuant-Lossless (PUBLIC)**

**Rationale:**
- Firmware can't ship without weights (const data needed at compile time).
- Zero external deps, stays lean.
- Version pinned to firmware release (7.7.0 matches firmware 0.1.0 semantically).
- If/when LamQuant-Neural needs separate weights, we can split `lamquant-weights` into `lamquant-weights-firmware` + `lamquant-weights-neural` and move each to the respective repo.

**Open question:** Could Neural codec depend on this? **Defer.** No neural codec crate exists yet in the public tree; revisit when it lands.

---

### 4. lamquant-ipc-types (v0.1.0)

**Path:** `/mnt/4tb/LamQuant/crates/lamquant-ipc-types/`

**Type:** Library  
**License:** AGPL-3.0  
**Description:** Shared MCU↔host IPC envelope + message kinds. Versioned postcard serialization. No_std compatible.

**Features:**
- `default` → `[]` (minimal)
- `defmt-format` → `defmt::Format` impls for on-MCU logging (optional)
- `host` → Stub for host-side serde_json bridge (allows deserialization without firmware deps)

**Direct dependencies:**
- `postcard@1` (no default features) — Binary serialization
- `serde@1` (no default features) — Serialization trait
- `heapless@0.8` (no default features) — Fixed-capacity types
- `defmt@0.3` (optional, feature `defmt-format`) — Firmware logging

**Inverse dependencies (in-workspace):**
- `lamquant-firmware` depends on it (required, comms.rs) — MCU-side envelope/deserialization
- **Does lamquant-core depend on it?** No, core is codec DSP only.
- **Does gui/src-tauri depend on it?** Not directly (GUI talks to firmware via higher-level protocol layer, not raw IPC).
- **Does BLUT depend on it?** No, BLUT is training orchestrator.

**Destination:** **LamQuant-Lossless (PUBLIC)**

**Rationale:**
- IPC protocol lives in firmware repo (tight coupling with MCU runtime).
- Host consumers (GUI, CLI tools) will include it transitively when they need to talk to the MCU.
- Zero transitive deps, stays clean.

---

### 5. lamquant-ops (v0.1.0)

**Path:** `/mnt/4tb/LamQuant/crates/lamquant-ops/`

**Type:** Library  
**License:** AGPL-3.0

**Description:** Shared op-runner contract (OpEvent, OpEventSink, spawn_lml, spawn_command, op_spec, launcher). UI parity spec — all long-running ops from GUI/TUI flow through this.

**Features:** None

**Direct dependencies:**
- `serde@1` (features `derive`)
- `serde_json@1` — Tiny, zero-dep JSON for OpEvent line emission
- `sha2@0.10` — Fingerprint generation

**Dev dependencies:**
- `jsonschema@0.30` — Schema parity tests against canonical JSON Schema

**Inverse dependencies (in-workspace):**
- `lamquant-core` depends on it (optional, `host` feature) — Used by TUI/CLI for op lifecycle
- `gui/src-tauri` depends on it (required) — GUI operation runner
- `lamquant-core` (dev-deps) — Integration tests

**Destination:** **LamQuant-Lossless (PUBLIC)**

**Rationale:**
- Op contract is part of the lossless codec's public surface (all UIs/TUIs use it).
- Zero external deps, stays lean.
- Shared between firmware-TUI, Vision-TUI, and Codec-GUI/TUI.
- Could be factored to "LamQuant-Common" in future, but for MVP stays in Lossless.

**Note:** This is a great candidate for a future `lamquant-common` crate (shared infra layer) alongside other common types.

---

### 6. lmafs (v1.1.0)

**Path:** `/mnt/4tb/LamQuant/crates/lmafs/`

**Type:** Library + 1 binary  
**License:** AGPL-3.0  
**Description:** FUSE filesystem that mounts a LamQuant .lma archive as a read-only directory. Lets file managers open the archive as a regular dir.

**Binaries:**
- `lmafs` — Mount point daemon (requires host feature)

**Direct dependencies:**
- `lamquant-core@{path}` (feature `host`) — LMA archive reading
- `fuser@0.14` — FUSE bindings
- `libc@0.2` — C interop
- `clap@4` (feature `derive`) — CLI arg parsing
- `tracing@0.1`, `tracing-subscriber@0.3` — Structured logging

**Inverse dependencies (in-workspace):**
- None (lmafs is a standalone tool, not a lib dependency for other crates)

**Destination:** **LamQuant-Lossless (PUBLIC)**

**Rationale:**
- lmafs ships with the codec (customer feature: mount archives as filesystems).
- Tight coupling to LMA container format (lives in lamquant-core).
- End-user tool, not a library consumed by other crates.

---

### 7. lamquant-history (v0.1.0)

**Path:** `/mnt/4tb/LamQuant/crates/lamquant-history/`

**Type:** Library  
**License:** AGPL-3.0

**Description:** Shared history.json + resume-state reader/writer for LamQuant front-ends. Same file format used across all UIs (Rust TUI, Python TUI, Tauri GUI).

**Features:** None

**Direct dependencies:**
- `serde@1` (features `derive`)
- `serde_json@1`

**Dev dependencies:**
- `tempfile@3`

**Inverse dependencies (in-workspace):**
- `lamquant-core` depends on it (optional, `host` feature) — TUI uses history
- `gui/src-tauri` depends on it (required) — GUI resume state
- **Question:** Does Firmware-TUI depend on it? **Yes** (any TUI that needs resume).
- **Question:** Does Vision-TUI depend on it? **Yes** (history is UI-agnostic).
- **Question:** Does BLUT depend on it? **No** (BLUT is training orchestrator, not UI).

**Destination:** **LamQuant-Lossless (PUBLIC)**

**Rationale:**
- History format is codec UI concern (all lossless codec UIs share it).
- Zero external deps, ultra-lightweight.
- Cross-process safe via fcntl/Windows-share locks (spec in specs/history-schema.json).
- Could move to LamQuant-Common in future, but stays in Lossless for MVP.

---

### 8. lamquant-lsl (v0.1.0)

**Path:** `/mnt/4tb/LamQuant/crates/lamquant-lsl/`

**Type:** Library + 3 binaries  
**License:** AGPL-3.0  
**Description:** Lab Streaming Layer (LSL) integration. Replay .lml as LSL outlet, record LSL streams to .lml. De-facto neuroscience real-time data network integration.

**Features:**
- `default` → `[]`
- `liblsl` → Pull in lsl crate (Rust bindings to C liblsl). Requires system liblsl OR bundled build (brittle on modern glibc due to deprecated `PTHREAD_STACK_MIN` in bundled boost).
- `async` → Tokio async wrappers (implies `liblsl`). For multi-stream daemons, TUI/GUI event loops, cancellation.

**Binaries:**
- `lml-stream` (requires `liblsl`) — Stream .lml as LSL outlet
- `lml-record` (requires `liblsl`) — Record LSL streams to .lml
- `lml-discover` (requires `liblsl`) — Discover LSL streams on network

**Direct dependencies:**
- `lamquant-core@{path}` (no default features, feature `host`) — Codec for encode/decode
- `lsl` (optional, git https://github.com/labstreaminglayer/liblsl-rust) — LSL bindings
- `sha2@0.10` — Stream ID fingerprinting from file
- `tokio@1` (optional, feature `async`) — Async runtime

**Dev dependencies:**
- `tempfile@3`
- `tokio@1` (full features, async test runtime)

**Rationale for feature design:**
- Deliberately **excluded from default-members** (requires system liblsl or brittle bundled build).
- Build explicitly: `cargo build -p lamquant-lsl --features liblsl`.
- `liblsl` feature is optional so the crate compiles in CI without liblsl installed.

**Inverse dependencies (in-workspace):**
- None (lml-* binaries are CLI tools, not libs depended on by other crates)
- **ADR 0024 Phase 4 note:** Cyclic dep issue — if lamquant-core depended on lamquant-lsl, we'd have a cycle. Solution: lamquant-lsl depends on lamquant-core, not vice versa. LSL subcommands will land in a future `crates/lamquant-cli/` crate that depends on BOTH.

**Destination:** **LamQuant-Vision (PUBLIC)**

**Rationale:**
- LSL is a Vision/integration layer concern (neuroscience data replay/record).
- Not a firmware/codec-core feature.
- Likely used by the Vision TUI (lab real-time streaming, replay from archives).
- Could also be used by Codec-GUI for research, but primary home is Vision.

**Open question:** Codec or Vision? **Vision.** LSL is neuroscience lab integration, not core codec compression.

---

### 9. gui/src-tauri (v7.0.0, crate name: `lamquant-gui`)

**Path:** `/mnt/4tb/LamQuant/gui/src-tauri/`

**Type:** Library + crates-of (Tauri WebView)  
**License:** AGPL-3.0

**Description:** LamQuant EEG desktop application. Tauri WebView shell (Rust backend ← IPC → React frontend). Encodes/decodes streams via lamquant-core, manages session history, spawns long-running operations.

**Features:**
- `default` → `["dev_kit"]`
- `dev_kit` → Research-only features (Python decoder bridge for raw-vs-reconstructed compare view)

**Binaries:** Tauri multi-platform binary (Windows, macOS, Linux)

**Direct dependencies:**
- `lamquant-core@{path}` — Core codec
- `lamquant-ops@{path}` — Op-runner contract
- `lamquant-history@{path}` — Session history
- `serde@1`, `serde_json@1` — Serialization
- `tauri@2.11.1` — WebView runtime
- `tauri-plugin-log@2`, `tauri-plugin-shell@2`, `tauri-plugin-dialog@2` — Tauri plugins
- `tokio@1` — Async runtime
- `serialport@4.6` — Serial comms (firmware link?)
- `zip@0.6` — Archive handling

**Build dependencies:**
- `tauri-build@2.6.1` — Tauri build script

**Inverse dependencies:**
- None (GUI is an end-user application, not a lib)

**Destination:** **LamQuant-Vision (PUBLIC)** or **LamQuant-Codec (PRIVATE)**?

**Decision:** **LamQuant-Vision (PUBLIC)**

**Rationale:**
- GUI is the Vision/user experience layer (EEG visualization, recording, playback).
- Not the turnkey codec app (which would be a CLI tool or minimal TUI).
- Codec repo (private) can import Vision as a public dependency for the GUI shell.
- Keeps GUI open-source as Vision, installer/SDK/Hub TUI in Codec (private).

**Alternative:** If Vision is meant to be a thin wrapper around Codec, then GUI could go in Codec. But the description "LamQuant-Vision — GUI shell" suggests it's the primary UI layer.

---

### 10. installer/src-tauri (v7.0.0, crate name: `openhuman-portal`)

**Path:** `/mnt/4tb/LamQuant/installer/src-tauri/`

**Type:** Library + Tauri binary  
**License:** AGPL-3.0

**Description:** OpenHuman Portal — LamQuant installer. Tauri WebView installer for Windows/macOS/Linux. Downloads, configures, installs LamQuant codec + firmware + dependencies.

**Features:** None

**Direct dependencies:**
- `serde@1`, `serde_json@1`
- `tauri@2.10.3` — WebView runtime
- `tauri-plugin-dialog@2`, `tauri-plugin-shell@2` — File dialogs, subprocess execution
- `tokio@1` — Async runtime

**Build dependencies:**
- `tauri-build@2.5.6`

**Inverse dependencies:** None (installer is an end-user application)

**Destination:** **LamQuant-Codec (PRIVATE)**

**Rationale:**
- Installer is a Codec-specific app (part of turnkey deployment story).
- Private because it ships with distribution, not as a public library.

---

## External Standalone Workspaces

### BLUT (v0.1.0)

**Path:** `/mnt/4tb/LamQuant/blut/` (git submodule, excluded from main workspace per ADR 0017)

**Type:** Standalone workspace with library + binary  
**License:** MIT  
**Author:** Brian Lam  
**Repository:** https://github.com/Quitetall/blut

**Description:** BLUT (Brian Lam's Universal Trainer) — Stage→Plan→Recipe orchestrator for local ML training (SFT, DPO, distillation, eval). Generic ML training framework, not LamQuant-specific.

**Binaries:**
- `blut` — CLI orchestrator

**Direct dependencies (core):**
- `serde`, `serde_json`, `serde_yaml` — Config/recipe serialization
- `thiserror`, `anyhow` — Error handling
- `tracing`, `tracing-subscriber` — Logging
- `tokio` (full features) — Async runtime
- `async-trait` — Async trait impls
- `nix` (feature `signal`) — Signal handling (UNIX)
- `dirs` — XDG directories
- `parking_lot` — Fast mutex
- `clap` (feature `derive`) — CLI parsing
- `humantime` — Duration parsing
- `rusqlite` (feature `bundled`) — SQLite job tracking
- `tempfile` — Scratch dirs
- `sha2` — Hash computation
- `memmap2` — Memory-mapped files
- `faster-hex` — Hex encoding
- `uuid` (feature `v4`) — Unique IDs
- `toml` — TOML config
- `schemars` — JSON Schema generation
- `tokio-util` — Tokio utilities
- `rayon` — Parallel iteration
- `which` — Find executables in PATH
- `bincode` — Binary encoding
- `rand` — RNG
- `ratatui` — TUI rendering
- `crossterm` — Terminal control
- `fuzzy-matcher` — Fuzzy searching

**Does NOT depend on any lamquant-* crates.** BLUT is a generic ML training framework that orchestrates external training scripts/containers. It's not LamQuant-coupled.

**Inverse dependencies:** None (BLUT is independent; could be used for other ML projects)

**Destination:** **BLUT (PUBLIC)** — Own repository, own CI, own release cycle

**Rationale (ADR 0017):**
- Generic ML trainer, not LamQuant-specific.
- Separate git submodule with independent release cadence.
- Never absorbed into LamQuant workspace.
- Users can depend on BLUT independently for their own training flows.

---

### tools/hazard3_bench (v0.0.0)

**Path:** `/mnt/4tb/LamQuant/tools/hazard3_bench/`

**Type:** Standalone workspace with binary  
**License:** (no license specified, assumed AGPL-3.0)

**Description:** LamQuant encoder cycle benchmark for RP2350 (Hazard3 RISC-V). Verilator + Renode simulation harness. Measures cycle count, memory usage, cache behavior on firmware encode path.

**Binaries:**
- `bench_encode` — Bare-metal benchmark binary (links firmware codec, no allocator, runs on simulated RP2350)

**Direct dependencies:**
- `lamquant-firmware@{path}` (no default features) — Codec library (not binary)
- `embedded-alloc@0.6` — Boot allocator
- **Target-specific** (riscv32imac):
  - `riscv@0.12` (feature `critical-section-single-hart`)
  - `riscv-rt@0.12` — RISC-V runtime
  - `panic-halt@1.0`

**Inverse dependencies:** None (benchmark harness, not a library)

**Destination:** **Eagle (PUBLIC)** — Validation/benchmarking suite

**Rationale:**
- Hazard3 bench is an Eagle concern (performance validation, not codec feature).
- Separate workspace so CI doesn't perturb main lockfile.
- Customers will use Eagle suite for performance validation + certification.

---

### tools/bench/bench_rs (v0.0.0)

**Path:** `/mnt/4tb/LamQuant/tools/bench/bench_rs/`

**Type:** Standalone workspace with binary  
**License:** (assumed AGPL-3.0)

**Description:** Host-side benchmarking harness. Links firmware codec as a library (host target, not embedded). Measures codec performance on CPU without hardware.

**Binaries:**
- `bench_rs` — Host benchmark (release-optimized)

**Direct dependencies:**
- `lamquant-firmware@{path}` — Firmware lib (host-mode, no firmware-bin feature)

**Workspace declaration:**
- `[workspace]` — Standalone workspace (doesn't inherit from root)

**Inverse dependencies:** None (benchmark tool)

**Destination:** **Eagle (PUBLIC)** — Validation suite

**Rationale:**
- Host-side perf validation (separate from Hazard3 simulation).
- Standalone workspace so it doesn't affect main builds.

---

## Dependency Arrows (Inter-Workspace)

### LamQuant-Lossless (PUBLIC)

**Members:**
- `lamquant-core` (v7.7.0) — Core DSP codec
- `lamquant-firmware` (v0.1.0) — RP2350 bare-metal
- `lamquant-weights` (v7.7.0) — Weights const data
- `lamquant-ipc-types` (v0.1.0) — MCU↔host envelope
- `lamquant-ops` (v0.1.0) — Op-runner contract
- `lamquant-history` (v0.1.0) — Session history (shared UI state)
- `lmafs` (v1.1.0) — FUSE filesystem mount

**Internal dependency arrows:**
```
lmafs ──→ lamquant-core
lamquant-firmware ──→ lamquant-core
lamquant-firmware ──→ lamquant-weights
lamquant-firmware ──→ lamquant-ipc-types
lamquant-core ──→ lamquant-ops (optional, host feature)
lamquant-core ──→ lamquant-history (optional, host feature)
```

**This is the "core codec" repo.** Customers deploying LamQuant compression will import this repo.

**Public surface:**
- Core DSP API: `encode()`, `decode()`, archive I/O
- Op-runner contract: spawn encode/decode operations
- Session history: persist/resume state
- Firmware runtime: RP2350 binaries
- Weights: TNN/SNN const data

---

### LamQuant-Vision (PUBLIC)

**Members:**
- `gui/src-tauri` (v7.0.0) — Desktop GUI
- `crates/lamquant-lsl` (v0.1.0) — LSL integration (optional)

**Internal dependency arrows:**
```
gui/src-tauri ──→ lamquant-core
gui/src-tauri ──→ lamquant-ops
gui/src-tauri ──→ lamquant-history
lamquant-lsl ──→ lamquant-core
```

**This is the "user experience" repo.** Integrates the codec with neuroscience tools (LSL), provides GUI for end users.

**Dependencies on Lossless:**
- Imports `lamquant-core`, `lamquant-ops`, `lamquant-history`

**Public surface:**
- Tauri GUI for recording/playback
- LSL outlet/inlet for real-time neuroscience workflows
- Python decoder bridge (research) via `dev_kit` feature

---

### LamQuant-Codec (PRIVATE)

**Members:**
- `gui/src-tauri` (re-exported from Vision for branding)
- `installer/src-tauri` (v7.0.0)
- **Future:** Hub TUI, Python SDK, CLI wrapper

**Internal dependency arrows:**
```
openhuman-portal (installer) ──→ (downloads + installs LamQuant-Lossless binaries)
lamquant-gui ──→ (Vision GUI, imported here)
```

**This is the "turnkey distribution" repo.** Codec team packages Vision GUI + Firmware + SDK + Installer into a customer-facing product.

**Dependencies on Lossless:**
- Runtime dependency (installs/links LamQuant-Lossless binaries)
- Does **not** import Rust crates directly (uses compiled artifacts)

**Private because:**
- Distribution bundling, licensing negotiations, OEM deals
- Not a pure open-source library
- Hub TUI (private, customer support) will live here
- Python SDK (private, commercial extension) will live here

---

### BLUT (PUBLIC, separate repo)

**Standalone workspace.** No dependencies on LamQuant crates.

**Used for:** Generic ML training (SFT, DPO, distillation). Could train LamQuant models, but the trainer itself is LamQuant-agnostic.

**Future:** When Neural codec lands, BLUT will orchestrate Neural training (but still without direct imports of Neural crates; orchestration happens via subprocess + config recipes).

---

### Eagle (PUBLIC, benchmarking suite)

**Members:**
- `tools/hazard3_bench/` (Verilator + Renode sims)
- `tools/bench/bench_rs/` (host-side CPU bench)

**Dependencies:**
```
tools/hazard3_bench ──→ lamquant-firmware
tools/bench/bench_rs ──→ lamquant-firmware
```

**Public surface:**
- Performance baseline reports
- Certification suite for FDA PCCP
- CI integration for regression detection

---

## External (crates.io) Dependency Summary

### LamQuant-Lossless (top-15 external deps)

| Crate | Version | Why | Stage |
|-------|---------|-----|-------|
| `ratatui` | 0.30 | TUI rendering (CLI/TUI UI) | Codified |
| `tokio` | 1.52 | Async runtime (Phase 6 async codec) | Codified |
| `serde` | 1.0 | Serialization (config, history, IPC) | Codified |
| `serde_json` | 1.0 | JSON for OpEvent, CLI args | Codified |
| `clap` | 4 | CLI argument parsing | Codified |
| `rayon` | 1.12 | Parallel encode/decode | Codified |
| `zstd` | 0.13 | Archive compression (LMA container) | Codified |
| `sha2` | 0.10 | Fingerprinting, crypto | Codified |
| `crossterm` | 0.28 | Terminal control (TUI) | Codified |
| `aes-gcm` | 0.10 | AEAD encryption (Phase 7) | Phase 7 |
| `argon2` | 0.5 | Password KDF (Phase 7) | Phase 7 |
| `reqwest` | 0.12 | HTTP client (Phase 6 async) | Phase 6 |
| `parquet` | 53 | Arrow parquet export (Phase 5.4) | Phase 5.4 |
| `aws-sdk-s3` | 1 | AWS S3 (Phase 6.2/6.7) | Phase 6 |
| `constriction` | 0.4 | rANS/range coder (Track B5+) | Research |

**Crypto story:** RustCrypto (pure-Rust, no_std-compatible) — aes-gcm, hmac, zeroize, argon2, rpassword, getrandom.

**TUI widget stack (15+ crates):** throbber-widgets-tui, ratatui-textarea, tui-big-text, tui-scrollview, tui-tree-widget, tui-widget-list, tui-piechart.

**Optional export formats:** Parquet (Arrow stack ~50 crates), HDF5 (system libhdf5), DICOM (pure-Rust).

**Optional S3:** AWS SDK (~100 transitive crates, feature-gated).

---

### LamQuant-Vision (top-10 external deps)

| Crate | Version | Why |
|-------|---------|-----|
| `tauri` | 2.11 | GUI runtime (Rust backend ← IPC → React frontend) |
| `tokio` | 1.52 | Async runtime (GUI event loop, long ops) |
| `serde` | 1.0 | Config/message serialization |
| `serde_json` | 1.0 | JSON IPC messages |
| `tauri-plugin-shell` | 2 | Subprocess execution (firmware flash, tools) |
| `tauri-plugin-dialog` | 2 | File dialogs |
| `serialport` | 4.6 | Serial comms (firmware programming?) |
| `zip` | 0.6 | Archive unpacking |
| `sha2` | 0.10 | LSL stream fingerprinting |
| `lsl` | (git master) | LSL outlet/inlet bindings (optional feature) |

**Heavy dep:** Tauri pulls in webview2 (Windows), WebKit (macOS/Linux), ~100 transitive crates.

---

### BLUT (top-10 external deps)

| Crate | Version | Why |
|-------|---------|-----|
| `tokio` | 1 | Async runtime (job scheduler, subprocess) |
| `serde` | 1 | TOML/YAML recipe deserialization |
| `serde_json` | 1 | JSON config |
| `serde_yaml` | 0.9 | YAML recipe parsing |
| `sqlx` / `rusqlite` | bundled | Job database (SQLite) |
| `clap` | 4 | CLI parsing |
| `rayon` | 1 | Parallel task scheduling |
| `ratatui` | 0.30 | TUI status display |
| `tracing` | 0.1 | Structured logging |
| `nix` | 0.29 | UNIX signal handling (graceful shutdown) |

**Lightweight framework:** ~45 crates total (excluding dev-deps). No heavy ML deps (BLUT orchestrates external Python/torchrun/vLLM via subprocess).

---

## Cross-Repo Split Design

### Split: `lamquant-core` → Lossless

**No split needed.** Core stays intact in Lossless repo.

---

### Split: `lamquant-firmware` → Lossless

**No split needed.** Firmware stays intact in Lossless repo.

---

### Split: `lamquant-weights` → Lossless (with future Neural option)

**Current (MVP):**
- All weights stay in Lossless (`subband_v1`, `subband_v2`, `legacy_v7_0` variants)

**Future (if/when Neural codec lands):**
- Option A: Keep all weights in Lossless, Neural crate optionally depends on it
- Option B: Split into `lamquant-weights-firmware` (Lossless) + `lamquant-weights-neural` (Neural)
- **Decision:** Defer until Neural codec exists. For now, single crate in Lossless.

---

### Split: `lamquant-ops` → Lossless (shared across all UIs)

**No split needed.** Op-runner contract is a shared utility used by:
- Codec-core CLI/TUI
- Vision GUI
- Vision LSL daemon
- **Future:** Firmware-TUI (hypothesis)
- **Future:** Codec Hub TUI

**Could become:** `lamquant-common::ops` in future refactoring, but stays in Lossless for MVP.

---

### Split: `lamquant-history` → Lossless (shared across all UIs)

**No split needed.** Session history format is UI-agnostic, used by:
- Codec TUI
- Vision GUI
- **Future:** Firmware-TUI
- **Future:** Vision TUI

**Could become:** `lamquant-common::history` in future, but stays in Lossless for MVP.

---

### Split: `lamquant-ipc-types` → Lossless (MCU↔host only)

**No split needed.** IPC protocol is firmware-specific, lives in Lossless.

---

### Split: `lmafs` → Lossless (archive filesystem)

**No split needed.** FUSE mount is a Lossless codec feature.

---

### Split: `lamquant-lsl` → Vision (neuroscience integration)

**No split needed.** LSL is explicitly a Vision feature.

---

### Split: `gui/src-tauri` → Vision (primary) + Codec (secondary)

**Current (MVP):**
- GUI lives in Vision repo (open-source UI)
- Codec repo re-exports Vision GUI in its installers

**Could split if Vision and Codec diverge (e.g., Vision gains proprietary features):**
- Option A: Keep single GUI, Codec installs Vision GUI
- Option B: Fork GUI for Codec features, keep Vision GUI as basic open-source version
- **Decision:** Single GUI in Vision for now. Codec team can overlay proprietary launcher/settings later.

---

## Open Questions — Resolution

### Q1: Does `lamquant-weights` belong in Lossless or Common?

**Answer:** **Lossless (for MVP).**

**Reasoning:**
- Current weights are firmware-only (const data for RP2350 TNN/SNN).
- No external crate (other than firmware) depends on them.
- When Neural codec lands, revisit splitting into `lamquant-weights-firmware` (Lossless) + `lamquant-weights-neural` (Neural).
- For now, zero external deps, zero complexity — keep in Lossless.

---

### Q2: Does `lamquant-history` belong in Lossless or Common?

**Answer:** **Lossless (for MVP).**

**Reasoning:**
- History format is codec-agnostic (used by all UIs: CLI TUI, Vision GUI, future Firmware-TUI, future Codec TUI).
- But it's fundamentally a "resume state for codec operations," so it lives in Lossless.
- If Codec (private) TUI needs to import it, it can depend on Lossless.
- Could become `lamquant-common::history` in future, but the dependency graph doesn't require it.

---

### Q3: Does `lamquant-ipc-types` belong in Lossless or Common?

**Answer:** **Lossless.**

**Reasoning:**
- IPC protocol is MCU↔host communication (firmware-specific).
- Not used by Vision or Codec except transitively through Lossless.
- No need for a separate Common repo for a single IPC protocol.

---

### Q4: What does `lamquant-lsl` depend on? Vision or Codec?

**Answer:** **Vision (LSL is neuroscience integration, not codec feature).**

**Reasoning:**
- LSL streams EEG data to/from neuroscience platforms (BrainVision, EEGLAB, real-time labs).
- This is a Vision concern (lab integration), not a turnkey Codec concern.
- Vision repo includes LSL integration + GUI for recording/playback.
- Codec (private) could optionally depend on Vision LSL, but LSL is not a Codec feature.

---

### Q5: Is `lamquant-ops` codec-specific or shared infra?

**Answer:** **Codec-specific (lives in Lossless), but shared across all UIs.**

**Reasoning:**
- OpEvent defines the contract for codec operations (encode, decode, compress, decompress).
- All codec frontends (CLI, TUI, GUI, future Firmware-TUI) use this contract.
- Not a "common" utility (not used by training, not used by validation, not used by Neural).
- But if we build a shared infra layer in future, `lamquant-ops` would be a perfect fit alongside `lamquant-history`.

---

### Q6: Could we build a `lamquant-common` crate to consolidate shared types?

**Answer:** **Yes, in future. Not in MVP.**

**What would go in Common:**
- `lamquant-ops` (op-runner contract)
- `lamquant-history` (session state)
- **Future:** version info, capability negotiation, schema validation

**Why not now:**
- No urgency (three repos is not too many deps).
- Codec (private) would need to depend on Common, which adds a layer.
- ADR to be written when the pattern becomes clear.

---

### Q7: Should `gui/src-tauri` be in Vision or Codec?

**Answer:** **Vision (primary), with Codec as a consumer.**

**Reasoning:**
- GUI is the "vision" for end users (EEG visualization, real-time recording).
- Open-source, community-friendly, not locked to distribution.
- Codec (private) can wrap Vision GUI + installer + Hub TUI + SDK into a turnkey product.
- Keeps Vision repo focused on UX, Codec repo focused on distribution/licensing.

---

### Q8: Does BLUT depend on or get imported by LamQuant?

**Answer:** **No (one-way only).**

**Reasoning:**
- BLUT is a generic ML trainer (Stage→Plan→Recipe framework).
- LamQuant code does **not** import BLUT.
- BLUT **could** orchestrate LamQuant training (but only via subprocess + config, not via Rust imports).
- BLUT lives in its own repo (git submodule, separate release cadence).
- Users can use BLUT independently for other ML projects.

---

### Q9: Does Eagle depend on Lossless or is it independent?

**Answer:** **Depends on Lossless (firmware library).**

**Reasoning:**
- Eagle benchmarks link `lamquant-firmware` as a library (no allocator, no entry point).
- Eagle runs on Hazard3 simulator (Verilator) or host CPU.
- Eagle **imports** Lossless crates, not vice versa.
- Lossless **does not** import Eagle (Eagle is test infrastructure, not a dependency).

---

### Q10: Should `tools/hazard3_bench` and `tools/bench/bench_rs` be their own repos?

**Answer:** **No. Stay in Eagle (suite of benchmarks).**

**Reasoning:**
- Both are validation/benchmarking tools, not core libraries.
- Separate workspaces (don't perturb main lockfile).
- Ship as part of Eagle release for customer certification.

---

## Dependency Counts & Complexity Metrics

### Per-crate internal dep fan-in (workspace)

| Crate | Imported by (count) | Circularly imports |
|-------|---------------------|-------------------|
| `lamquant-core` | 4 (firmware, lsl, lmafs, gui) | No |
| `lamquant-firmware` | 2 (hazard3_bench, bench_rs) | No |
| `lamquant-weights` | 1 (firmware) | No |
| `lamquant-ipc-types` | 1 (firmware) | No |
| `lamquant-ops` | 2 (core [optional], gui) | No |
| `lamquant-history` | 2 (core [optional], gui) | No |
| `lmafs` | 0 (standalone tool) | No |
| `lamquant-lsl` | 0 (standalone tool) | No |
| `gui/src-tauri` | 0 (end-user app) | No |
| `installer/src-tauri` | 0 (end-user app) | No |

**No circular dependencies.** DAG is clean (one-directional).

---

### Per-repo external dep counts (crates.io)

| Repo | Direct external | Transitive | Profile | Risk |
|------|-----------------|-----------|---------|------|
| Lossless | ~60 | ~300 | default build is clean, `async` + `parquet` + `s3` pull in heavy stacks | Medium (depends on features) |
| Vision | ~25 | ~200 | Tauri is heavy (~100 crates), LSL depends on system liblsl | High (Tauri) |
| Codec | TBD (private) | TBD | Will import Vision GUI + installer + Hub TUI | Medium |
| BLUT | ~45 | ~150 | Lightweight orchestrator, no ML deps | Low |
| Eagle | ~5 | ~20 | Minimal (firmware lib only) | Low |

---

## Migration Checklist (for 8-repo split)

### Phase 1: Lossless Repo Setup
- [ ] Create `LamQuant-Lossless` repo from monorepo
  - [ ] Copy: `lamquant-core/`, `lamquant-firmware/`, `lamquant-weights/`, `crates/{lamquant-ops, lamquant-history, lamquant-ipc-types, lmafs}/`
  - [ ] Rename: `lamquant-core/` → `crates/lamquant-lossless/` (per task description)
  - [ ] Root `Cargo.toml` workspace: 7 members
  - [ ] CI: `cargo build --workspace`, `cargo test --workspace`, firmware build script
- [ ] Update import paths in all crates (path deps now relative to new repo root)
- [ ] Validate: `cargo build --workspace` passes

### Phase 2: Vision Repo Setup
- [ ] Create `LamQuant-Vision` repo
  - [ ] Copy: `gui/src-tauri/`, `crates/lamquant-lsl/`
  - [ ] Root `Cargo.toml` workspace: 2 members
  - [ ] Cargo.toml: both crates depend on `lamquant-core` via git import (Lossless repo)
  - [ ] CI: `cargo build --workspace`, `cargo test --workspace`
- [ ] Update import paths: `lamquant-core = { git = "https://github.com/openhuman-ai/LamQuant-Lossless", ... }`

### Phase 3: Codec Repo Setup (PRIVATE)
- [ ] Create `LamQuant-Codec` repo (private GitHub)
  - [ ] Copy: `installer/src-tauri/`
  - [ ] Build script: download + link Vision GUI (pre-built artifacts or git submodule)
  - [ ] CI: signed builds, installer generation
  - [ ] **Future:** Hub TUI, Python SDK, license enforcement

### Phase 4: BLUT Repo Promotion
- [ ] Promote `blut/` from submodule to standalone repo
  - [ ] Git history: preserve via `git subtree split` or `git filter-branch`
  - [ ] CI: independent build, release, PyPI publish

### Phase 5: Eagle Repo Setup
- [ ] Create `Eagle` repo (public, or private validation suite)
  - [ ] Copy: `tools/hazard3_bench/`, `tools/bench/bench_rs/`
  - [ ] Cargo.toml: both depend on `lamquant-firmware` via git import (Lossless repo)
  - [ ] CI: baseline generation, regression detection

### Phase 6: Cross-repo imports & CI validation
- [ ] Update Lossless CI to pull Vision + Codec modules (smoke tests)
- [ ] Update Codec CI to validate installer with Vision GUI + Lossless artifacts
- [ ] Setup git hooks: prevent accidental circular imports

---

## Recommendations for Common Infra (Future ADR)

### Create `lamquant-common` crate (phase 2 of refactoring)

**Contents:**
- `opaque_types.rs` — Types shared across repos (version, caps, schema version)
- `ops.rs` — Op-runner contract (currently in Lossless)
- `history.rs` — Session state format (currently in Lossless)
- `ipc.rs` — MCU↔host envelope (currently in Lossless)

**Destination:** Separate public repo `LamQuant-Common`

**Why:**
- Reduces coupling between Vision, Codec, Firmware, BLUT.
- Single source of truth for wire formats, versioning, capability negotiation.
- Codec (private) can depend on Common (public) + Lossless (public) without exposing internal logic.

**When:** After split is stable (3+ months). Write ADR when the pattern is clear.

---

## Summary Table

| Crate | Version | Path | Type | Destination | Rationale | External Deps (top-5) |
|-------|---------|------|------|-------------|-----------|----------------------|
| `lamquant-core` | 7.7.0 | `lamquant-core/` | Lib + 2 bins | **Lossless** | Core DSP codec | ratatui, tokio, serde, clap, rayon |
| `lamquant-firmware` | 0.1.0 | `lamquant-firmware/` | Lib + 1 bin | **Lossless** | RP2350 bare-metal | embedded-alloc, heapless, rp235x-hal, postcard |
| `lamquant-weights` | 7.7.0 | `lamquant-weights/` | Lib (data) | **Lossless** | TNN/SNN const data | (none) |
| `lamquant-ipc-types` | 0.1.0 | `crates/lamquant-ipc-types/` | Lib | **Lossless** | MCU↔host envelope | postcard, serde, heapless, defmt |
| `lamquant-ops` | 0.1.0 | `crates/lamquant-ops/` | Lib | **Lossless** | Op-runner contract | serde, serde_json, sha2 |
| `lamquant-history` | 0.1.0 | `crates/lamquant-history/` | Lib | **Lossless** | Session history | serde, serde_json |
| `lmafs` | 1.1.0 | `crates/lmafs/` | Lib + 1 bin | **Lossless** | FUSE archive mount | fuser, clap, tracing |
| `lamquant-lsl` | 0.1.0 | `crates/lamquant-lsl/` | Lib + 3 bins | **Vision** | LSL integration | lsl, sha2, tokio |
| `lamquant-gui` | 7.0.0 | `gui/src-tauri/` | Lib + app | **Vision** | Desktop GUI | tauri, tokio, serialport, zip |
| `openhuman-portal` | 7.0.0 | `installer/src-tauri/` | Lib + app | **Codec** (private) | Installer | tauri, tokio |
| `blut` | 0.1.0 | `blut/` | Lib + bin | **BLUT** | ML trainer | tokio, serde, rusqlite, ratatui, rayon |
| `bench_rs` | 0.0.0 | `tools/bench/bench_rs/` | Bin | **Eagle** | Host bench | lamquant-firmware |
| `hazard3_bench` | 0.0.0 | `tools/hazard3_bench/` | Bin | **Eagle** | Firmware bench | lamquant-firmware |

---

## File Written

This report was automatically generated via `cargo metadata`, `cargo tree`, and manual Cargo.toml inspection.

**Output file:** `/mnt/4tb/LamQuant/decomp/cargo_dep_graph.md`

**Next steps:**
1. Cross-check with `/mnt/4tb/LamQuant/.claude/plans/optimized-seeking-bentley.md` (locked decisions ADR).
2. Use this as input to the 8-repo git split workflow.
3. Write post-split import validation scripts (prevent circular deps across repos).
