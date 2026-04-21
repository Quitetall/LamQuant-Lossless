# OpenHuman LamQuant Changelog

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
