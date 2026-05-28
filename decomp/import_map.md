# Import Map — lamquant_core consumers

**Generated:** 2026-05-27  
**Total hits:** 375 references across 84 files  
**Scope:** All `.rs`, `.toml`, `.py`, `.sh` files (excluded: `target/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.numba_cache/`, `node_modules/`, `dist/`, `build/`, `graphify-out/`)

---

## Summary

| Category | Count |
|----------|-------|
| **Rust files with `use` imports** | 41 |
| **Python files (`import lamquant_core`)** | 18 |
| **TOML files (dependency declarations)** | 10 |
| **Shell scripts & config files (path references)** | 15 |
| **Comment/doc references only** | Many |

---

## Categorization by Import Type

### Common-bound imports
**Files:** 13 directly importing common modules  
**Key modules:** `crc32`, `edf`, `lma`, `source`, `error`, `paths`, `lma`

### Lossless-bound imports
**Files:** 34 directly importing lossless modules  
**Key modules:** `lml`, `container`, `lpc`, `backend`, `stream`, `tui`, `security`, `lifting`, `golomb`, `rans`, `bit_pack`, `offset_table`, `range`, `codec_stages`, `async_io`, `ffi`, `wasm`

### Mixed imports
**Files:** 18 importing the full `lamquant_core` PyO3 wheel (uses both common + lossless)

### Ambiguous imports (in the codebase itself)
**Files:** 5 with compound `use lamquant_core::{...}` statements

---

## Detailed Import Map by File

### Rust Library Crates (Canonical consumers)

#### `crates/lamquant-lsl/` — LSL streaming integration
**Dependency declaration:** `Cargo.toml:64`
```toml
lamquant-core = { path = "../../lamquant-core", default-features = false, features = ["host"] }
```

**Files requiring rewrites:**

| File | Line | Import | Type | Rewrite Target |
|------|------|--------|------|-----------------|
| `crates/lamquant-lsl/src/error.rs` | 6 | `use lamquant_core::error::LmlError;` | Common | `lamquant-common::error::LmlError` |
| `crates/lamquant-lsl/src/inlet.rs` | 33 | `use lamquant_core::lpc::LpcMode;` | Lossless | `lamquant-lossless::lpc::LpcMode` |
| `crates/lamquant-lsl/src/inlet.rs` | 172 | `lamquant_core::lml::compress_with_mode(...)` | Lossless | `lamquant_lossless::lml::compress_with_mode(...)` |
| `crates/lamquant-lsl/src/inlet.rs` | 486 | `lamquant_core::lml::decompress(&encoded)` | Lossless | `lamquant_lossless::lml::decompress(&encoded)` |
| `crates/lamquant-lsl/src/metadata_lite.rs` | 51 | `lamquant_core::container::parse_header(&bytes)` | Lossless | `lamquant_lossless::container::parse_header(&bytes)` |
| `crates/lamquant-lsl/src/outlet.rs` | 87 | `lamquant_core::container::read_file(lml_path)` | Lossless | `lamquant_lossless::container::read_file(lml_path)` |
| `crates/lamquant-lsl/src/stream_id.rs` | 47 | `lamquant_core::container::parse_header(&bytes)` | Lossless | `lamquant_lossless::container::parse_header(&bytes)` |
| `crates/lamquant-lsl/src/xdf.rs` | 256 | `lamquant_core::container::read_file(lml_path)` | Lossless | `lamquant_lossless::container::read_file(lml_path)` |
| `crates/lamquant-lsl/src/xdf.rs` | 365 | `lamquant_core::container::read_file(lml_path)` | Lossless | `lamquant_lossless::container::read_file(lml_path)` |

**Tests (same rewrites apply):**
- `xdf_clockoffset.rs`: 2 imports (`container`, `lpc::LpcMode`)
- `xdf_export.rs`: 2 imports (`container`, `lpc::LpcMode`)
- `xdf_multistream.rs`: 2 imports (`container`, `lpc::LpcMode`)
- `xdf_timestamps.rs`: 2 imports (`container`, `lpc::LpcMode`)
- `outlet_actor.rs`: 2 imports (`container`, `lpc::LpcMode`)
- `multi_stream_async.rs`: 2 imports (`container`, `lpc::LpcMode`)

**Cargo.toml rewrite:**
```toml
# BEFORE
lamquant-core = { path = "../../lamquant-core", default-features = false, features = ["host"] }

# AFTER (Phase 2 decision: depends on split strategy)
lamquant-common = { path = "../../lamquant-common" }
lamquant-lossless = { path = "../../lamquant-lossless", features = ["host"] }
```

---

#### `crates/lmafs/` — FUSE filesystem for LMA archives
**Dependency declaration:** `Cargo.toml:15`
```toml
lamquant-core = { path = "../../lamquant-core", features = ["host"] }
```

**Imports (all Common):**

| File | Line | Import | Rewrite Target |
|------|------|--------|-----------------|
| `src/main.rs` | 161 | `lamquant_core::lma::list_archive(&archive_path)` | `lamquant_common::lma::list_archive(&archive_path)` |
| `src/main.rs` | 451 | `lamquant_core::lma::read_entry_decoded(&self.archive_path, ...)` | `lamquant_common::lma::read_entry_decoded(...)` |
| `src/main.rs` | 460 | `lamquant_core::lma::read_entry(&self.archive_path, ...)` | `lamquant_common::lma::read_entry(...)` |

**Cargo.toml rewrite:**
```toml
# BEFORE
lamquant-core = { path = "../../lamquant-core", features = ["host"] }

# AFTER
lamquant-common = { path = "../../lamquant-common" }
```

---

#### `lamquant-firmware/` — RP2350 embedded firmware
**Dependency declaration:** `Cargo.toml:70`
```toml
lamquant-core = { path = "../lamquant-core", default-features = false }
```

**Imports:**

| File | Line | Import | Type | Rewrite Target |
|------|------|--------|------|-----------------|
| `src/integrity.rs` | 11 | `use lamquant_core::crc32::{crc32_update, CRC32_INIT};` | Common | `lamquant_common::crc32::{crc32_update, CRC32_INIT}` |
| `src/dsp/lpc.rs` | 864 | `use lamquant_core::lpc as core_lpc;` | Lossless | `lamquant_lossless::lpc as core_lpc` |
| `src/dsp/lifting.rs` | 337 | `use lamquant_core::lifting as core;` | Lossless | `lamquant_lossless::lifting as core` |

**Cargo.toml rewrite:**
```toml
# BEFORE
lamquant-core = { path = "../lamquant-core", default-features = false }

# AFTER
lamquant-common = { path = "../lamquant-common" }
lamquant-lossless = { path = "../lamquant-lossless", default-features = false }
```

---

#### `gui/src-tauri/` — Electron/Tauri desktop GUI
**Dependency declaration:** `Cargo.toml:47`
```toml
lamquant-core = { path = "../../lamquant-core" }
```

**Imports (all Lossless/TUI):**

| File | Line | Import | Rewrite Target |
|------|------|--------|-----------------|
| `src/config.rs` | 34 | `lamquant_core::tui::config::config_path()` | `lamquant_lossless::tui::config::config_path()` |
| `src/config.rs` | 39 | `lamquant_core::tui::config::LamQuantConfig::load()` | `lamquant_lossless::tui::config::LamQuantConfig::load()` |
| `src/config.rs` | 113 | `lamquant_core::tui::config::LamQuantConfig::load()` | `lamquant_lossless::tui::config::LamQuantConfig::load()` |
| `src/config.rs` | 121 | `lamquant_core::tui::config::LamQuantConfig` (type) | `lamquant_lossless::tui::config::LamQuantConfig` |
| `src/state_bridge.rs` | 30 | `use lamquant_core::tui::panel::Action;` | `lamquant_lossless::tui::panel::Action` |
| `src/state_bridge.rs` | 31 | `use lamquant_core::tui::router::{Router, SCREEN_MAIN};` | `lamquant_lossless::tui::router::{Router, SCREEN_MAIN}` |
| `src/state_bridge.rs` | 32 | `use lamquant_core::tui::snapshot::StateSnapshot;` | `lamquant_lossless::tui::snapshot::StateSnapshot` |
| `src/state_bridge.rs` | 33 | `use lamquant_core::tui::state::AppState as TuiAppState;` | `lamquant_lossless::tui::state::AppState as TuiAppState` |

**Cargo.toml rewrite:**
```toml
# BEFORE
lamquant-core = { path = "../../lamquant-core" }

# AFTER
lamquant-lossless = { path = "../../lamquant-lossless" }
```

---

### Core Crate Internals — `lamquant-core/`

#### Self-imports within lamquant-core (no external action needed)

**Main binary: `src/bin/lml.rs`**  
- 20 Common imports (edf, source, lma, paths, error)
- 52 Lossless imports (container, lpc, lml, stream, backend, security, async_io, range)
- **Decision:** This file will be split into two binaries post-phase-2
  - Lossless-specific logic stays in `lamquant-lossless/src/bin/lml.rs`
  - Common-only operations (e.g., EDF→LMA conversion) may move to a separate tool

**Main binary: `src/bin/lamquant.rs`**  
- Lossless only (TUI cockpit)
- Stays in `lamquant-lossless/src/bin/lamquant.rs`

**Tests (canonical users of public API):**

| File | Type | Imports | Rewrite |
|------|------|---------|----------|
| `tests/byte_equal_backends.rs` | Lossless | `backend`, `lpc::LpcMode` | `lamquant_lossless::` |
| `tests/config_save.rs` | Lossless | `tui::config::*` | `lamquant_lossless::tui::` |
| `tests/conformance.rs` | Mixed | `lifting`, `lml`, `lpc` (ambiguous due to compound use) | Split |
| `tests/cross_platform_bytes.rs` | Lossless | `container`, `lpc::LpcMode` | `lamquant_lossless::` |
| `tests/dicom_parity.rs` | Common | `source::DicomWaveformReader` | `lamquant_common::source::` |
| `tests/edf_safety.rs` | Common | `edf::read_edf`, `error::LmlError` | `lamquant_common::` |
| `tests/lma_conformance.rs` | Common | `lma::*` | `lamquant_common::lma::` |
| `tests/op_e2e.rs` | Lossless | `tui::state::AppState` | `lamquant_lossless::tui::` |
| `tests/output_panel_cancel.rs` | Lossless | `tui::operations::*`, `tui::panel::*`, `tui::panels::output::*` | `lamquant_lossless::tui::` |
| `tests/property_roundtrip.rs` | Lossless | `lml` | `lamquant_lossless::lml` |
| `tests/reducer_unit.rs` | Lossless | `tui::app::*`, `tui::panel::Action`, `tui::router` | `lamquant_lossless::tui::` |
| `tests/snapshot_wire_format.rs` | Common | `lma` | `lamquant_common::lma` |
| `tests/tui_smoke.rs` | Lossless | `tui::app::App` | `lamquant_lossless::tui::` |
| `tests/window_size_transition.rs` | Common | `lma::{list_archive, pack_archive, unpack_archive}` | `lamquant_common::lma::` |
| `tests/wizard_buffer.rs` | Lossless | `tui::*` | `lamquant_lossless::tui::` |

**Benchmarks:**

| File | Type | Imports | Rewrite |
|------|------|---------|----------|
| `benches/cat_b_compare.rs` | AMBIGUOUS | `use lamquant_core::{edf, golomb, lma, lpc};` | **DECISION NEEDED:** Split into two `use` statements or move to separate bench crates |
| `benches/codec.rs` | Lossless | `backend`, `container`, `golomb`, `lifting`, `lml`, `lpc` | `lamquant_lossless::` |

**Fuzz targets:**

| File | Type | Imports |
|------|------|---------|
| `fuzz/fuzz_targets/container_header.rs` | Lossless | `container::read_from` |
| `fuzz/fuzz_targets/decompress.rs` | Lossless | `lml::decompress` |
| `fuzz/fuzz_targets/lma_manifest.rs` | Common | `lma::list_archive` |
| `fuzz/fuzz_targets/lml_random_seek.rs` | Lossless | `stream::LmlReader` |
| `fuzz/fuzz_targets/offset_table.rs` | Lossless | `offset_table::OffsetTable` |
| `fuzz/fuzz_targets/roundtrip.rs` | Lossless | `lml::{compress, decompress}` |

---

### Python Integration Points

**All Python files import the full `lamquant_core` PyO3 wheel** (compiled by maturin).  
Post-split, these will need to adapt depending on the PyO3 binding strategy:

#### Option A: Dual PyO3 wheels
- `lamquant-common` publishes `lamquant_common` wheel (CRC-32, EDF, LMA)
- `lamquant-lossless` publishes `lamquant_lossless` wheel (compression + container I/O)
- Python code imports both

#### Option B: Single wheel with re-exports
- `lamquant-lossless` re-exports `lamquant-common` symbols at its crate root
- Python code continues to import single `lamquant_lossless` wheel
- Simpler for Python consumers but tighter coupling

#### Option C: Separate CLI-only wheel
- Keep current `lamquant_core` wheel for backward compatibility
- Deprecate it in favor of `lamquant-lossless` + `lamquant-common`

**Files requiring rewrite (depend on option chosen):**

| File | Imported Functions | Decision Point |
|------|-------------------|-----------------|
| `ai_models/snn/lma_dataset.py` | `lma_read_entry`, `container_read_bytes` | Both common + lossless |
| `scripts/m1_stage_labels.py` | `lma_read_entry` | Common |
| `scripts/m2_pack_per_dataset.py` | `lma_read_entry` | Common |
| `scripts/audit_lma_upstream_drift.py` | `lma_read_entry` | Common |
| `scripts/bulk_lml_to_lma.py` | `lma_read_entry` | Common |
| `scripts/w1_stage_labels_v2.py` | `lma_read_entry` | Common |
| `scripts/build_snn_train_val_split.py` | `lma_read_entry` | Common |
| `scripts/benchmark_dataloader_throughput.py` | `lma_read_entry` | Common |
| `tests/conftest.py` | Full module | Test infrastructure |
| `tests/helpers/rust_bindings.py` | Full module | Test helper |
| `tests/integration/test_lma_dataset.py` | `lma_read_entry`, `container_read_bytes` | Both |
| `tests/snn/test_lma_dataset_coverage.py` | Full module | Test |
| `tests/codec_python_smoke/test_lma_dataset_deep.py` | Full module | Test |
| `tests/codec/test_entropy_coders.py` | `golomb_encode_dense` | Lossless |
| `reference_implementations/python_codec/lamquant_codec/ops/golomb.py` | `golomb_encode_dense`, `golomb_decode_dense` | Lossless |
| `reference_implementations/python_codec/lamquant_codec/ops/rans.py` | `rans_encode`, `rans_decode` | Lossless |
| `reference_implementations/python_codec/lamquant_codec/training/lma_dataset.py` | `lma_read_entry` | Common |
| `legacy/one_shots/audit_lml_footer_coverage.py` | Full module | Legacy (deprecated) |

---

### TOML Dependency Declarations

All of these will need to be split post-Phase-2:

| File | Current | Rewrite Target |
|------|---------|-----------------|
| `crates/lamquant-lsl/Cargo.toml:64` | `lamquant-core = { path = "../../lamquant-core", default-features = false, features = ["host"] }` | Split: `lamquant-common` + `lamquant-lossless` with `host` feature |
| `crates/lmafs/Cargo.toml:15` | `lamquant-core = { path = "../../lamquant-core", features = ["host"] }` | `lamquant-common = { path = "../../lamquant-common" }` |
| `gui/src-tauri/Cargo.toml:47` | `lamquant-core = { path = "../../lamquant-core" }` | `lamquant-lossless = { path = "../../lamquant-lossless" }` |
| `lamquant-firmware/Cargo.toml:70` | `lamquant-core = { path = "../lamquant-core", default-features = false }` | Split: `lamquant-common` + `lamquant-lossless` without `host` |
| `lamquant-core/fuzz/Cargo.toml:20` | `[dependencies.lamquant-core]` | N/A (internal) |
| `Cargo.toml:10, :28` | Workspace members | Update to include `lamquant-common`, `lamquant-lossless` |
| `pyproject.toml:56` | `rust = ["lamquant-core>=0.2.0"]` | Update to reference published wheels (decision pending on wheel strategy) |

---

### Shell Scripts & Build Tooling

**Path references (not code imports, but build commands):**

| File | Reference | Action |
|------|-----------|--------|
| `scripts/verify_reproducible_build.sh:42` | `--manifest-path lamquant-core/Cargo.toml` | Update to split path or conditional logic |
| `scripts/regen_cli_reference.sh:23, :39` | References to `lamquant-core/Cargo.toml` and `lamquant-core/src/bin/lml.rs` | Update paths post-split |
| `scripts/run_benchmarks.sh:20, :75, :85, :93, :102` | Builds against `lamquant-core` | Conditional: update for new split crates |
| `tools/build_conformance_vectors.py:41` | Path reference to `lamquant-core/Cargo.toml` | Update path reference |
| `crates/lamquant-ops/src/launcher.rs:48-50` | Cargo build commands with `lamquant-core/Cargo.toml` | Update manifest paths in launcher strings |
| Installation scripts (`installer/install*.sh`) | References to `lamquant-core` binary | Update binary references if split |

---

### Documentation & Comments Only

These files reference `lamquant-core` in comments/docs but don't import code:

**Markdown/docs/config files:**
- `.tarpaulin.toml:12` — Coverage config comment
- `lamquant-core/cbindgen.toml:1-2, :10` — C FFI generation config (specific to `lamquant-core` crate)
- `pyproject.toml:31-32, :55, :86` — Project metadata
- Many test/script docstrings explaining internal structure

**No rewrites needed** for documentation-only references; these naturally update with file moves.

---

## Mixed Files (Require Split-Rewrite)

Files importing **both** Common and Lossless symbols in the same Rust compilation unit:

### Rust Files (2)

**`lamquant-core/src/bin/lml.rs`** — Main CLI binary
- Common: `edf`, `source`, `lma`, `paths`, `error` (20 references)
- Lossless: `container`, `lpc`, `lml`, `stream`, `backend`, `security`, `async_io`, `range` (52 references)
- **Action:** Post-Phase-2, this binary may be split into separate tool or dual-import within lamquant-lossless (with lamquant-common as public dep)

**`lamquant-core/benches/cat_b_compare.rs`** — Benchmark comparing backend performance
- Line 26: `use lamquant_core::{edf, golomb, lma, lpc};`
  - `edf` (Common), `lma` (Common), `golomb` (Lossless), `lpc` (Lossless)
- **Action:** Split into two use statements:
  ```rust
  use lamquant_common::{edf, lma};
  use lamquant_lossless::{golomb, lpc};
  ```

### Python Files (18)

All Python files import the **full PyO3 wheel** dynamically:
```python
import lamquant_core  # or: import lamquant_core as _lc, _lml, _rs, etc.
```

This is **intentional mixing** (Python calls both LMA + compression functions from same module).

**Files:**
1. `ai_models/snn/lma_dataset.py:75, :545` — Calls `lma_read_entry` + `container_read_bytes`
2. `legacy/one_shots/audit_lml_footer_coverage.py:32` — Full module
3. `reference_implementations/python_codec/lamquant_codec/ops/golomb.py:22` — Just Lossless (golomb_*)
4. `reference_implementations/python_codec/lamquant_codec/ops/rans.py:26` — Just Lossless (rans_*)
5. `reference_implementations/python_codec/lamquant_codec/training/lma_dataset.py:83` — Just Common (lma_read_entry)
6. `scripts/audit_lma_upstream_drift.py:67` — Just Common (lma_read_entry)
7. `scripts/benchmark_dataloader_throughput.py:54` — Just Common (lma_read_entry)
8. `scripts/build_snn_train_val_split.py:76` — Just Common (lma_read_entry)
9. `scripts/bulk_lml_to_lma.py:227` — Just Common (lma_read_entry)
10. `scripts/m1_stage_labels.py:42, :57` — Just Common (lma_read_entry)
11. `scripts/m2_pack_per_dataset.py:125` — Just Common (lma_read_entry)
12. `scripts/w1_stage_labels_v2.py:33` — Just Common (lma_read_entry)
13. `tests/codec/test_entropy_coders.py:39` — Just Lossless (golomb_encode_dense)
14. `tests/codec_python_smoke/test_lma_dataset_deep.py:153` — Just Common (lma_read_entry)
15. `tests/conftest.py:142` — Full module (test fixture)
16. `tests/helpers/rust_bindings.py:18` — Full module (test helper)
17. `tests/integration/test_lma_dataset.py:147` — Both (lma + container)
18. `tests/snn/test_lma_dataset_coverage.py:45` — Full module (test fixture)

**Action (Phase 3 — Post PyO3 split decision):**
- **If dual wheels:** Update imports to select the right wheel per file
  ```python
  # Before (generic)
  import lamquant_core as _lc
  _lc.lma_read_entry(...)
  
  # After (split)
  import lamquant_common as _lc
  _lc.lma_read_entry(...)
  ```
- **If single wheel:** No change needed (backward compat).

---

## Ambiguous Cases (Require Human Decision)

### Files with compound `use` statements mixing categories:

**`lamquant-core/benches/cat_b_compare.rs:26`**
```rust
use lamquant_core::{edf, golomb, lma, lpc};
                     └─Common─┘ └Lossless┘
```
**Decision:** Split into two statements (already noted in Mixed section above).

**`lamquant-core/tests/conformance.rs:6`**
```rust
use lamquant_core::{lifting, lml, lpc};
                    └─ Lossless ─┘  
```
**Decision:** All three are Lossless. Rewrite as:
```rust
use lamquant_lossless::{lifting, lml, lpc};
```

**`lamquant-core/src/bin/lml.rs:11`** (CRITICAL — main binary)
```rust
use lamquant_core::{container, edf, lma, lml, tui};
                    └─Lossless─┘ └Common┘ └Lossless─┘
```
**Decision:** Split into two:
```rust
use lamquant_common::{edf, lma};
use lamquant_lossless::{container, lml, tui};
```

**`lamquant-firmware/src/dsp/lpc.rs:864`**
```rust
use lamquant_core::lpc as core_lpc;
```
**Classification:** Lossless (cross-check function). Clear rewrite.

**`lamquant-firmware/src/dsp/lifting.rs:337`**
```rust
use lamquant_core::lifting as core;
```
**Classification:** Lossless (cross-check function). Clear rewrite.

---

## Implementation Roadmap

### Phase 1: Analysis (CURRENT) ✓
- Identify all `lamquant_core` consumers
- Classify imports (done above)
- Document rewrites per consumer

### Phase 2: File Migration
1. Create two new crate directories:
   - `lamquant-common/` (copy shared modules from `lamquant-core/src/`)
   - Rename `lamquant-core/` → `lamquant-lossless/` (keep lossless-specific modules + TUI)

2. Update `Cargo.toml` manifests:
   - Workspace: add `lamquant-common`, remove `lamquant-core`, keep `lamquant-lossless`
   - Each consumer: rewrite dependency statements (see table above)

3. Rewrite all Rust imports per the table above

4. Python wheels (conditional on strategy choice — see below)

### Phase 3: PyO3 Wheel Strategy (Decision Required)

**Option A: Dual wheels (recommended for modularity)**
- `lamquant-common` publishes PyO3 wheel `lamquant_common` with LMA/EDF functions
- `lamquant-lossless` publishes PyO3 wheel `lamquant_lossless` with compression/container functions
- Python code imports one or both as needed
- **Pros:** Clean boundaries, Neural code can depend on `lamquant-common` only via crates.io
- **Cons:** More wheel maintenance, Python code must know about two imports

**Option B: Single re-export wheel (backward-compatible)**
- `lamquant-lossless` Cargo.toml: `lamquant-common = { version = "X", path = "../lamquant-common" }`
- `lamquant-lossless/src/lib.rs`:
  ```rust
  pub use lamquant_common::{crc32, edf, lma, source, error, paths, io, ingest, pipeline};
  ```
- `lamquant-lossless` PyO3 wheel re-exports all common functions
- Python continues to `import lamquant_core` (or rename to `lamquant_lossless`)
- **Pros:** Minimal Python changes, single wheel, simpler for CLI users
- **Cons:** Tighter coupling, violates Unix philosophy

**Option C: Keep current wheel, deprecate**
- Maintain `lamquant-core` PyO3 wheel as-is for backward compat
- Warn users to migrate to split wheels over 2–3 releases
- **Pros:** No immediate Python breakage
- **Cons:** Maintains technical debt, confusing naming

### Phase 4: Testing & Verification
- Run full test suite with new import paths
- Verify cross-crate compilation
- Benchmark to ensure no performance regression
- Test firmware build with dual imports
- Test LSL inlet/outlet with split crates
- Update CI/CD workflows

### Phase 5: Publishing
- Publish `lamquant-common` to crates.io
- Rename local build to `lamquant-lossless` or publish as-is (depends on Phase 3 decision)
- Update documentation and installation instructions

---

## Statistics Summary

| Metric | Count |
|--------|-------|
| **Total reference lines** | 375 |
| **Unique files** | 84 |
| **Rust files with code imports** | 41 |
| **Python files (PyO3 wheel)** | 18 |
| **TOML dependency files** | 10 |
| **Shell/config references** | 15 |
| **Common-bound modules** | 7 (`crc32`, `edf`, `lma`, `source`, `error`, `paths`, `io`, `ingest`, `pipeline`) |
| **Lossless-bound modules** | 16 (`lml`, `container`, `lpc`, `backend`, `stream`, `tui`, `security`, `lifting`, `golomb`, `rans`, `bit_pack`, `offset_table`, `range`, `codec_stages`, `async_io`, `ffi`, `wasm`) |
| **External crates depending on lamquant-core** | 4 (`lamquant-lsl`, `lmafs`, `lamquant-firmware`, `gui/src-tauri`) |
| **Internal consumers (tests/benches/bins)** | Many (see lamquant-core/ above) |
| **Mixed files (code + test + script)** | 2 Rust + 18 Python |
| **Ambiguous (compound use)** | 4 Rust |

---

## Notes for Phase 2 Engineers

1. **Preserve test structure:** Tests in `lamquant-lossless/tests/` should compile against both `lamquant-common` + `lamquant-lossless` (both available post-Phase-2).

2. **Firmware is no_std:** Ensure `lamquant-common` compiles with `default-features = false` (CRC-32, lifting, LPC, errors are all core/no_std; EDF/LMA are host-only).

3. **LSL dependency on TUI:** `lamquant-lsl` uses TUI types indirectly (does not—see imports above; it only uses `error::LmlError`, `container::`, `lpc::LpcMode`, `lml::`, which are all available without TUI).

4. **CLI binary split:** `lamquant-core/src/bin/lml.rs` is large and touches both common + lossless. Post-Phase-2, either:
   - Keep in `lamquant-lossless/src/bin/lml.rs` with dual imports
   - Move common-only commands to a separate `lamquant-archive` or similar binary in `lamquant-common/src/bin/`
   - Decision defers to Phase 2 architect

5. **Backward compat for Python:** Decide on PyO3 wheel strategy **before** migration. Delaying this decision will create mid-Phase breakage.

6. **Test fixtures:** `lamquant-core/tests/fixtures/dicom/` — determine ownership post-split. Likely stays with lossless (used by codec tests) but document in Phase 2 PR.

---

## Appendix: Full Module Ownership

### ✓ Common (to `lamquant-common`)
- `crc32` — CRC-32 ISO 3309 (no_std)
- `edf` — EDF file reader (host)
- `error` / `codec_errors` — Error types (no_std core + host)
- `lma` — LMA archive format (host)
- `source` — Signal source readers (DICOM, EEG, CSV, BrainVision, etc.) (host)
- `paths` — Path utilities (host)
- `io` — I/O abstractions (host)
- `ingest` — Data ingestion pipeline (host)
- `pipeline` — Pipeline traits (host)

### ✓ Lossless (to `lamquant-lossless`)
- `backend` — Compute backend dispatch (host)
- `bit_pack` — Bit packing utilities (no_std)
- `lifting` — Le Gall 5/3 integer DWT (no_std)
- `lml` — LML packet compress/decompress (no_std)
- `lpc` — LPC analysis/synthesis (no_std)
- `golomb` — Golomb-Rice entropy coding (no_std)
- `rans` — rANS entropy coding (no_std)
- `arithmetic` — Arithmetic coding (experimental, host)
- `container` — LML v1 container file I/O (host)
- `stream` — Parallel streaming I/O (host)
- `codec_stages` — Codec pipeline stages (host)
- `offset_table` — LML footer index (no_std core + host)
- `range` — Range query support (host)
- `security` — Encryption/HMAC (host)
- `async_io` — Async file and network I/O (host, tokio)
- `tui` — Terminal UI cockpit (host)
- `tui_experimental` — Experimental TUI features (host)
- `ffi` — C FFI bindings (host)
- `wasm` — WebAssembly bindings (wasm)

