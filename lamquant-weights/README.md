# lamquant-weights

Generated TNN/SNN weights crate for LamQuant firmware. **Do not hand-edit
`src/generated/`** — it's rebuilt from a model checkpoint via:

```bash
python firmware/export_firmware.py \
    --target rust \
    --schema firmware/export_schema.toml \
    --checkpoint weights/student_subband_gold.ckpt \
    --arch subband_v1
```

## Architecture variants

Selected at build time via Cargo features (mutually exclusive):

| Feature | Encoder | Width | Description |
|---------|---------|-------|-------------|
| `subband_v1` (default) | `TernaryMobileNetV5_Subband` | 128 | Gen 7.1 production |
| `subband_v2` | `TernaryMobileNetV5_Subband_V2` | 216 | Gen 7.6.1 (depthwise-separable) |
| `legacy_v7_0` | `TernaryMobileNetV5` | 96 | Gen 7.0 (full-band, deprecated) |

## Crate layout

```
src/
├── lib.rs              # Re-exports + feature gating
├── types.rs            # Hand-written typed wrappers (TernaryConvWeights, ...)
├── metadata.rs         # GENERATED: model version, CRC, ckpt SHA-256, timestamp
└── generated/
    ├── mod.rs          # GENERATED: re-export tree
    ├── focal/          # Per-layer ternary conv weights
    ├── rotation.rs     # Cayley rotation Q15
    ├── fsq.rs          # FSQ lattice + rANS tables
    ├── toeplitz.rs     # LFSR seeds
    └── snn/            # Mamba state-space model weights
```

## Boot integrity

`metadata::FIRMWARE_CRC32` is the CRC-32 over all weight byte arrays in
deterministic enumeration order. Firmware verifies this at boot via
`lamquant_firmware::integrity::check_firmware_crc()`. Mismatch → safe mode.

## Reproducibility

`.exportlock.json` records the source checkpoint SHA-256, git commit, and
exporter version. Re-running the codegen tool against the same checkpoint
must produce byte-identical output. Tested by `tools/verify_export_parity.py`.
