# lamquant-core Comprehensive Audit & Module Decomposition Map

**Date:** 2026-05-27  
**Scope:** Complete `Cargo.toml` + module tree analysis for 8-repo decomposition  
**Target:** Data for `lamquant-common` + `lamquant-lossless` Cargo.toml creation

---

## Task 1: Cargo.toml Metadata & Configuration

### Package Metadata
```toml
[package]
name = "lamquant-core"
version = "7.7.0"
edition = "2021"
rust-version = "1.81"
description = "LML — lossless EEG compression codec (Le Gall lifting DWT + LPC + Golomb-Rice)"
license = "AGPL-3.0"
repository = "https://github.com/openhuman-ai/LamQuant"
categories = ["compression", "science"]
keywords = ["eeg", "lossless", "codec", "medical", "neuroscience"]
build = "build.rs"
```

### Library Configuration
```
name = "lamquant_core"
crate-type = ["rlib"]
```

### Binary Declarations
- **lml** (9,653 lines) — Main CLI. SPLIT REQUIRED (common + lossless imports)
- **lamquant** (236 lines) — TUI cockpit. LOSSLESS

### Features & Flags (14 total)
| Feature | Destination |
|---------|-------------|
| default, std, host, async, python, ffi, wasm | lamquant-lossless |
| parquet, hdf5, dicom, s3, keyring | lamquant-lossless |
| experimental_arithmetic, experimental_bit_pack | lamquant-lossless |

### Dependencies Summary

**Always Required (Core, no_std + alloc):**
- sha2, base64, libm → lamquant-common

**Host-only Large:**
- CLI: clap, clap_complete, clap_mangen
- Files: walkdir, zstd, globset, tempfile, filetime
- UI: ratatui, crossterm, indicatif, tui-* widgets
- Serialization: serde, serde_json, serde_yaml
- Logging: tracing, tracing-subscriber
→ All to lamquant-lossless (host feature)

**Phase 7 Security (RustCrypto, bundled in host):**
- aes-gcm, hmac, zeroize, argon2, rpassword, getrandom, keyring
→ lamquant-lossless (host feature)

**Phase 6 Async (gated by async feature):**
- tokio, reqwest, notify
→ lamquant-lossless

**Export Formats & Experimental:**
- parquet, arrow-*, hdf5, dicom, aws-*, constriction, probability
→ lamquant-lossless

---

## Task 2: Module Tree from src/lib.rs

### Core (no_std + alloc)
- backend, bit_pack, codec_errors, crc32, error, golomb, lifting, lml, lpc, rans
- arithmetic (experimental_arithmetic feature)

### Host-only
- async_io (async), codec_stages, container, edf, ingest, io, lma, offset_table, paths, pipeline, range, security, source, stream, tui, tui_experimental

### Bindings
- ffi (ffi feature), wasm (wasm feature)

### PyO3 Module (feature = "python")
13 exposed functions:
- golomb_encode_dense, golomb_decode_dense
- rans_encode, rans_decode
- lml_compress, lml_decompress
- container_write, container_read, container_read_bytes, container_read_window_np, container_read_phys_f32, container_metadata
- lma_read_entry

---

## Task 3: Top-Level .rs Files

| File | Lines | Destination |
|------|-------|-------------|
| arithmetic | 250 | LOSSLESS |
| async_io | 592 | LOSSLESS |
| backend | 262 | LOSSLESS |
| bit_pack | 305 | LOSSLESS |
| codec_errors | 379 | LOSSLESS |
| codec_stages | 300 | LOSSLESS |
| container | 1,242 | LOSSLESS |
| crc32 | 106 | COMMON |
| edf | 544 | COMMON |
| error | 96 | COMMON |
| ffi | 141 | LOSSLESS |
| golomb | 490 | LOSSLESS |
| io | 410 | COMMON |
| lifting | 177 | LOSSLESS |
| lma | 4,241 | COMMON |
| lml | 1,868 | LOSSLESS |
| lpc | 895 | LOSSLESS |
| offset_table | 447 | LOSSLESS |
| paths | 211 | COMMON |
| pipeline | 214 | COMMON |
| range | 409 | LOSSLESS |
| rans | 392 | LOSSLESS |
| security | 663 | LOSSLESS |
| stream | 558 | LOSSLESS |
| wasm | 42 | LOSSLESS |

**Summary:** COMMON = 8 files (~2,142 lines); LOSSLESS = 18 files (~13,508 lines)

---

## Task 4: Subdirectories

### bin/ (2 files, 9,889 lines)
- lml.rs (9,653) — Main CLI. SPLIT REQUIRED
- lamquant.rs (236) — TUI cockpit. LOSSLESS

### source/ (11 files, 3,180 lines)
- mod.rs, ascii.rs, bitstream.rs, bundle.rs, reader.rs
- edf_reader.rs, brainvision.rs, cnt.rs, eeglab.rs, raw.rs, dicom.rs
- All COMMON (data source readers)

### ingest/ (3 files, 717 lines)
- mod.rs, ascii_lines.rs, edf_synth.rs
- All COMMON (non-EDF format ingestion)

### tui/ (14 files, 5,976 lines)
- All LOSSLESS (host-only, application layer)
- Key files: app.rs, router.rs, panel.rs, state.rs, operations.rs, panels/, config.rs, history.rs, theme.rs

### tui_experimental/ (4 files, 679 lines)
- All LOSSLESS (host-only)
- Files: codec.rs, menu.rs, style.rs

---

## Task 5: Intra-File Split Analysis

### src/bin/lml.rs (SPLIT REQUIRED, 9,653 lines)

Recommendation: Keep in lamquant-lossless/src/bin/lml.rs with dual imports:
```rust
use lamquant_common::{edf, source, lma, paths, error};
use lamquant_lossless::{container, lpc, lml, stream, backend, security, async_io, range, tui};
```

---

## Task 6: lamquant-common/Cargo.toml

```toml
[package]
name = "lamquant-common"
version = "0.1.0"
edition = "2021"
rust-version = "1.81"
description = "LamQuant shared primitives"
license = "GPL-3.0-or-later"
repository = "https://github.com/Quitetall/LamQuant-Lossless"
categories = ["compression", "science"]
keywords = ["eeg", "edf", "lma", "archive"]

[lib]
name = "lamquant_common"
crate-type = ["rlib"]

[dependencies]
sha2 = { version = "0.10", default-features = false }
base64 = { version = "0.22", default-features = false, features = ["alloc"] }
libm = { version = "0.2" }
walkdir = { version = "2", optional = true }
zstd = { version = "0.13", optional = true }
globset = { version = "0.4", optional = true }
tempfile = { version = "3", optional = true }
filetime = { version = "0.2", optional = true }
serde = { version = "1", features = ["derive"], optional = true }
serde_json = { version = "1", default-features = false, features = ["alloc"], optional = true }
serde_yaml = { version = "0.9", optional = true }
dicom-object = { version = "0.8", optional = true }
dicom-core = { version = "0.8", optional = true }
tracing = { version = "0.1.40", optional = true }
tracing-subscriber = { version = "0.3.18", features = ["env-filter"], optional = true }
pyo3 = { version = "0.25", features = ["extension-module"], optional = true }
numpy = { version = "0.25", optional = true }

[features]
default = ["host"]
std = []
host = ["base64/std", "sha2/std", "dep:walkdir", "dep:zstd", "dep:globset", "dep:tempfile", "dep:filetime", "dep:serde", "dep:serde_json", "dep:serde_yaml", "dep:tracing", "dep:tracing-subscriber"]
dicom = ["host", "dep:dicom-object", "dep:dicom-core"]
python = ["host", "dep:pyo3", "dep:numpy"]

[dev-dependencies]
serde_json = "1"
insta = { version = "1", features = ["yaml"] }
proptest = "1"
```

---

## Task 7: lamquant-lossless/Cargo.toml

```toml
[package]
name = "lamquant-lossless"
version = "7.7.0"
edition = "2021"
rust-version = "1.81"
description = "LML — lossless EEG compression codec"
license = "GPL-3.0-or-later"
repository = "https://github.com/Quitetall/LamQuant-Lossless"
categories = ["compression", "science"]
keywords = ["eeg", "lossless", "codec", "medical", "neuroscience"]
build = "build.rs"

[lib]
name = "lamquant_lossless"
crate-type = ["rlib"]

[[bin]]
name = "lml"
path = "src/bin/lml.rs"
required-features = ["host"]

[[bin]]
name = "lamquant"
path = "src/bin/lamquant.rs"
required-features = ["host"]

[[bench]]
name = "codec"
harness = false
required-features = ["host"]

[[bench]]
name = "cat_b_compare"
harness = false
required-features = ["host"]

[dependencies]
lamquant-common = { path = "../crates/lamquant-common", features = ["host"] }
sha2 = { version = "0.10", default-features = false }
base64 = { version = "0.22", default-features = false, features = ["alloc"] }
libm = { version = "0.2" }
lamquant-ops = { path = "../crates/lamquant-ops", optional = true }
lamquant-history = { path = "../crates/lamquant-history", optional = true }
clap = { version = "4", features = ["derive"], optional = true }
clap_complete = { version = "4", optional = true }
clap_mangen = { version = "0.2", optional = true }
rayon = { version = "1", optional = true }
walkdir = { version = "2", optional = true }
zstd = { version = "0.13", optional = true }
globset = { version = "0.4", optional = true }
tempfile = { version = "3", optional = true }
filetime = { version = "0.2", optional = true }
ratatui = { version = "0.30", optional = true }
crossterm = { version = "0.28", optional = true }
indicatif = { version = "0.17", optional = true }
serde_json = { version = "1", default-features = false, features = ["alloc"], optional = true }
serde = { version = "1", features = ["derive"], optional = true }
serde_yaml = { version = "0.9", optional = true }
throbber-widgets-tui = { version = "0.11", optional = true }
ratatui-textarea = { version = "0.8", features = ["crossterm"], optional = true }
tui-big-text = { version = "0.8", optional = true }
tui-scrollview = { version = "0.6", optional = true }
tui-tree-widget = { version = "0.24", optional = true }
tui-widget-list = { version = "0.15", optional = true }
tui-piechart = { version = "0.3", optional = true }
tracing = { version = "0.1.40", optional = true }
tracing-subscriber = { version = "0.3.18", features = ["env-filter"], optional = true }
pyo3 = { version = "0.25", features = ["extension-module"], optional = true }
numpy = { version = "0.25", optional = true }
wasm-bindgen = { version = "0.2", optional = true }
aes-gcm = { version = "0.10", default-features = false, features = ["aes", "alloc"], optional = true }
hmac = { version = "0.12", default-features = false, optional = true }
zeroize = { version = "1.7", default-features = false, features = ["zeroize_derive"], optional = true }
argon2 = { version = "0.5", default-features = false, features = ["alloc", "password-hash"], optional = true }
rpassword = { version = "7", optional = true }
getrandom = { version = "0.2", optional = true }
keyring = { version = "3", optional = true }
tokio = { version = "1", default-features = false, features = ["rt-multi-thread", "macros", "fs", "io-util", "sync", "time", "signal"], optional = true }
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls", "json"], optional = true }
notify = { version = "6", default-features = false, features = ["macos_fsevent"], optional = true }
parquet = { version = "53", default-features = false, features = ["arrow", "snap"], optional = true }
arrow-array = { version = "53", default-features = false, optional = true }
arrow-schema = { version = "53", default-features = false, optional = true }
hdf5-metno = { version = "0.12", optional = true }
ndarray = { version = "0.16", optional = true }
constriction = { version = "0.4", default-features = false, features = ["std"], optional = true }
probability = { version = "0.20", optional = true }
aws-config = { version = "1", default-features = false, features = ["rustls", "behavior-version-latest"], optional = true }
aws-sdk-s3 = { version = "1", default-features = false, features = ["rustls", "behavior-version-latest"], optional = true }

[features]
default = ["host"]
std = []
host = ["lamquant-common/host", "base64/std", "sha2/std", "dep:clap", "dep:clap_complete", "dep:clap_mangen", "dep:rayon", "dep:walkdir", "dep:zstd", "dep:globset", "dep:tempfile", "dep:filetime", "dep:ratatui", "dep:crossterm", "dep:indicatif", "dep:serde_json", "dep:serde", "dep:serde_yaml", "dep:lamquant-ops", "dep:lamquant-history", "dep:throbber-widgets-tui", "dep:ratatui-textarea", "dep:tui-big-text", "dep:tui-scrollview", "dep:tui-tree-widget", "dep:tui-widget-list", "dep:tui-piechart", "dep:tracing", "dep:tracing-subscriber", "dep:aes-gcm", "dep:hmac", "dep:zeroize", "dep:argon2", "dep:rpassword", "dep:getrandom"]
async = ["host", "dep:tokio", "dep:reqwest", "dep:notify"]
keyring = ["host", "dep:keyring"]
parquet = ["host", "dep:parquet", "dep:arrow-array", "dep:arrow-schema"]
hdf5 = ["host", "dep:hdf5-metno", "dep:ndarray"]
dicom = ["host", "lamquant-common/dicom"]
s3 = ["async", "dep:aws-config", "dep:aws-sdk-s3"]
python = ["host", "dep:pyo3", "dep:numpy"]
ffi = ["host"]
wasm = ["std", "dep:wasm-bindgen"]
experimental_arithmetic = ["host", "dep:constriction", "dep:probability"]
experimental_bit_pack = ["host"]

[dev-dependencies]
tempfile = "3"
lamquant-ops = { path = "../crates/lamquant-ops" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
criterion = { version = "0.5", features = ["html_reports", "cargo_bench_support"] }
insta = { version = "1", features = ["yaml"] }
proptest = "1"
pulp = "0.21"
realfft = "3"
idsp = "0.21"
dsp-process = "0.2"
dsp-fixedpoint = "0.1"
loom = "0.7"
rkyv = { version = "0.7", default-features = false, features = ["alloc", "validation", "size_32"] }
bitstream-io = "4"
faer = "0.22"
```

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total lines in lamquant-core/src | 15,650 |
| Destination: lamquant-common | ~2,142 lines |
| Destination: lamquant-lossless | ~13,508 lines |
| Top-level .rs files | 26 |
| Subdirectories | 5 |
| Core modules (no_std) | 11 |
| Host-only modules | 15 |
| Files requiring split-rewrite | 2 |
| External consumers | 4 |
| Feature flags | 14 |
| Dependencies (runtime) | 55+ |

**Audit completed.** Ready for Phase 2 surgical extraction.
