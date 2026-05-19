# OpenHuman LamQuant Changelog

## lossless-codec v1.4 (2026-05-19)

### KDE Ark integration (Kerfuffle CliInterface plugin)

- New `installer/ark-plugin/` — C++/Qt6 plugin (`kerfuffle_clilma`) subclassing `Kerfuffle::CliInterface`. Lets Ark open `.lma` archives natively — entries listed with original size, compressed size, method, and SHA-256; extract on demand. Shells out to the existing `lml` binary. Read-only by design (add/delete/move not exposed to protect the lossless archive contract). License: BSD-2-Clause.
- `installer/install-ark-plugin.sh --user|--system` — distro-agnostic build + install script. Detects missing build deps via pacman/dpkg/rpm probes against `/etc/os-release`. Checks `lml ls --long` compat before installing. Writes a Plasma session env snippet (`~/.config/plasma-workspace/env/lamquant-ark-plugin.sh`) for per-user installs so `QT_PLUGIN_PATH` is set on next login. Records a build sentinel for Ark ABI drift detection across upgrades.

### `lml ls --long` versioned wire format

- New `--long` flag on `lml ls` emits a TAB-separated, machine-readable listing for OS plugin shell-outs. First stdout line is `#lml-ls schema=1` (versioned marker so consumers fail closed on unknown versions). Per-entry columns: `<original_size>\t<compressed_size>\t<method>\t<sha256>\t<path>`. The Kerfuffle plugin validates the schema line before accepting any entry lines.

### Plugin parse hardening

- Strict 5-column TSV check in `clilma.cpp` — malformed lines are rejected, not silently skipped.
- Path-traversal defense at both consumer (plugin rejects `..`, leading `/`, NUL, CR, LF) and producer (`lml ls --long` strips control bytes before output).

### Installer hardening

- `install-mime.sh`: rewrote `Exec=` line rewrite in pure bash parameter expansion (dropped `sed`); no longer clobbers an existing `xdg-mime default` on re-run.
- `lma-open`: stale FUSE-mount recovery via `/proc/self/mountinfo` with timeout; `rmdir` failure now bails loudly; dynamic mount-stability poll (replaced fixed `sleep 2`); spawns `lmafs` under `systemd-run --user --scope` so it survives launcher scope teardown; synchronous `xdg-open`.

### lmafs strict-decode default

- `lmafs` now surfaces codec failure as `EIO` instead of silently falling back to raw stored bytes. New `--allow-raw-fallback` flag opts into the legacy behavior for forensic triage of pre-v1.1 archives.
- Directory tree built from flat manifest paths — nested entries like `chb06/chb06_01.edf` now create proper subdirectory nodes in the mount.

### lma core hardening

- Added `MAX_ENTRY_ORIGINAL_SIZE = 16 GB` cap at all four read sites (`extract_entry`, `verify_archive`, `read_entry`, `read_entry_decoded`). Previously only compressed size was bounded, allowing a zstd-bomb pattern (small compressed → multi-GB decompressed → OOM).

### Two coexisting open paths for .lma

- FUSE path: `xdg-mime default lamquant.desktop application/x-lma` → Dolphin double-click mounts via `lmafs`.
- Ark path: `xdg-mime default org.kde.ark.desktop application/x-lma` → opens directly in Ark without a FUSE mount.
- `install-mime.sh` no longer clobbers a user's existing choice on re-run.

### Test totals

- 1271 / 1271 integration tests green
- 277 / 277 lamquant-core lib tests
- 7 / 7 lmafs unit tests
- 20 / 20 lma module tests

---

## LamQuant v7.7.1 (2026-04-21)

### Firmware Optimization Session

**Safety Subsystems (+59 KB, 8 subsystems)**
- Added `firmware/core/safety_features.c/.h`: 8 persistent safety monitoring subsystems
- BLE retry buffer (0.5 KB): retransmit lost neural packets over BLE
- Impedance monitor (0.8 KB): per-channel contact quality with threshold alerts
- Pre-ictal ring buffer (32.0 KB): ~8 s circular buffer of L3 windows for seizure onset detection
- Channel quality scores (0.2 KB): per-channel SNR and artifact flags
- Seizure diary log (8.0 KB): persistent on-device seizure event log
- Event timestamp log (2.0 KB): general clinical event timestamps
- Battery/power monitor (0.1 KB): supply voltage and charge state tracking
- Firmware fault log (1.0 KB): watchdog resets, PMP faults, BLE errors
- Total SRAM: 509.8 KB / 520 KB (98.0%), up from 449.9 KB (86.5%)

**Pedantic Compiler Warnings**
- Enabled max pedantic warning flags across 18 firmware headers
- Zero warnings: all pre-existing issues resolved
- Catches implicit truncations, shadowed variables, and missing prototypes at compile time

**MAC Throughput Optimization (3-path)**
- Achieved ~1.2 cyc/MAC (down from ~3 cyc/MAC)
- Path 1: Software pipelining of weight decode + accumulate loops
- Path 2: XNOR + CPOP (popcount) via Zbb `cpop` instruction for packed ternary
- Path 3: Zbb saturating arithmetic for activation clamp (no branch on overflow)
- Combined throughput measured across all focal block configurations

**Inference Optimizations**
- W2A8 support: INT8 activation path alongside existing INT16 (W2A16)
- DMA prefetch infrastructure: weight prefetch during activation compute
- Fused conv+shortcut: reduces memory bandwidth for residual connections
- XIP cache tiling: aligned weight layout for cache-line-optimal flash reads

**Scratchpad Optimization (+4.4 KB headroom)**
- Moved Core 0 DSP hot buffers to SCRATCH_X (4 KB dedicated scratchpad):
  `hp_filters` (HP biquad state, 21ch), LPC coefficient working set, entropy coding buffers
- Moved Core 1 stack to SCRATCH_Y (4 KB dedicated scratchpad)
- Freed equivalent main RAM; net +4.4 KB headroom over naive placement
- SCRATCH_X: 2.6 KB used, 1.5 KB free; SCRATCH_Y: 2.0 KB used, 2.0 KB free

**Soft-Float Elimination (-8.5 KB flash)**
- Removed all soft-float operations from firmware
- `.text` section: 95.3 KB (down from ~103.8 KB)
- Firmware is now 100% pure integer (Q31/Q30 fixed-point)

**Floor Division Fix (cross-language correctness)**
- `BIAS_CTX_LEN=32` context bias cancellation now uses floor division consistently
- C firmware and Python/Rust reference implementations now produce bit-identical output
- Affects `ops/bias.py` and `lamquant-core/src/lpc.rs` (already correct); firmware was the divergent path

---

## OH! v1.0.0 / LamQuant v7.7.0 (2026-04-19)

### Architecture Refactor
- DSP code consolidated into `lamquant_codec/ops/` (lifting, lpc, bias, wht, pipeline, constants)
- `ai_models/student/subband_preprocess.py` is now a backward-compat re-export layer; all imports unchanged
- Wire-format constants centralized in `lamquant_codec/ops/constants.py` (single source of truth)
- Neural per-window packet magic: `LMQ1` (historical names LMQ2/LMQ3/LMQ5 are retired)
- Lossless per-window packet magic: `LML1` (historical names LMQ4/LMQ5 retired; legacy files still readable)
- Deleted modules: `lamquant_codec/ops/fse.py`, `fsq.py`, `laplacian_rans.py`, `lamquant_codec/packet.py`
- Fused pipeline (`ops/pipeline.py`): 3.57× faster encode (5.89 ms → 1.65 ms)

### LML v1 Lossless Codec
- SOTA lossless EEG compression: 2.26:1 on 16-bit clinical EEG
- 94% of Shannon entropy limit (0.54 bits/sample gap)
- CRC-32 per window + SHA-256 per file integrity
- Human-readable file header (`head -1 file.lml`)
- Noise-aware mode: configurable LSB stripping (0-32 bits)
- Double-strip prevention (refuses to strip already-stripped data)
- Golomb-Rice entropy coder (JIT'd, 2.5x speedup)
- Backward compatible: reads legacy `LMQ5` and `LMQ4` per-window packets

### OH! CLI
- Interactive guided menu with prefix matching
- Live compression dashboard (10 Hz, braille spinner)
- Crash-safe state file with automatic resume
- Line-buffered audit log for regulatory reproducibility
- Tab completion via prompt_toolkit (optional)
- `oh compress`, `oh train`, `oh info`, `oh syscheck`
- Smart defaults from config — no flags needed

### Training Cockpit
- 14 screens: pipeline status, GPU monitoring, tmux integration
- Experiment runner with systematic A/B sweeps
- Model leaderboard ranked by validation R
- Run comparison with side-by-side metrics
- Hyperparameter editor reading TrainingConfig
- Preset management: fast / medium / production

### OpenHuman Eagle (LQS Benchmark)
- LQS v1.0: four compliance levels (L/C/M/A)
- Per-band fidelity requirements (delta through gamma)
- Downstream task preservation (seizure, sleep, pathology)
- Compliance badge with printable certificate
- Open standard: any codec can run the same suite

### Infrastructure
- Typed dataclasses: LQFileInfo, LQPacketHeader, ConversionResult, SystemProfile
- Box/SplitBox guaranteed ASCII alignment
- Version derived from pyproject.toml (single source of truth)
- 23 conformance tests, 100-file adversarial verification
- Full EDF→LML→NPZ pipeline verified byte-identical

### Bug Fixes
- Dead R gradient: pearson_r_batch().item() → pearson_r_torch()
- SE decoder: inplace ReLU, wrong placement, Kaiming init
- Clinical split: shuffle before capping (was all CHB-MIT)
- TUH seizure files: reclassified as epilepsy_patient
- Gradient checkpointing for Tier 5+ decoders
- Worker cap: now cpu_count-1 (was arbitrarily 8)
