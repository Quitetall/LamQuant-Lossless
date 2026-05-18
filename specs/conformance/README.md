# LML wire-format conformance test suite

Phase 8 / Item E. Publishable test vectors that any LML reader
implementation can run against itself for spec conformance.

## What this suite covers

The suite focuses on the **wire format**: header parsing, window-CRC
verification, footer (LMLFOOT1) parsing, metadata fidelity, and
typed-error behaviour on corrupted vectors. It does **not** run the
full entropy / LPC inverse / lifting inverse stack — that's exercised
by `cargo test` in the LamQuant Rust crate.

If you are writing a third-party LML reader (Python, C, Go, …),
run `verify.py` against your reader's parse function. Every positive
vector should `PASS`, every negative vector should produce a typed
error matching the documented kind.

## Vector catalogue

| # | Vector | Category | Exercises |
|---|---|---|---|
| 1 | `basic_4ch_1024s_250hz.lml` | round-trip | baseline; 4ch × 1024 samples @ 250 Hz, default window |
| 2 | `single_sample_1ch.lml` | edge | n_channels=1, total_samples=1 (minimum non-trivial) |
| 3 | `single_channel_long.lml` | edge | n_channels=1, total_samples=1 M (lots of windows) |
| 4 | `max_channels_512.lml` | edge | n_channels=512 (within `u16::MAX`) |
| 5 | `sample_rate_50hz.lml` | rate/depth | low rate, default 16-bit |
| 6 | `sample_rate_4000hz.lml` | rate/depth | high rate (ECoG-grade) |
| 7 | `bit_depth_8.lml` | rate/depth | 8 noise_bits stripped |
| 8 | `metadata_unicode.lml` | metadata | UTF-8 channel labels (中文, ñ, emoji) |
| 9 | `metadata_large_json.lml` | metadata | 64 KB metadata payload |
| 10 | `corrupt_crc.lml` | corruption | window-level CRC flipped → `CrcMismatch` |
| 11 | `truncated_window.lml` | corruption | last window payload chopped → `Truncated` |
| 12 | `bitflipped_signal_byte.lml` | corruption | one byte in a payload flipped → `CrcMismatch` |
| 13 | `legacy_no_footer.lml` | version compat | pre-Phase-0.6 wire (flag bit 0 cleared, footer absent) |

Vectors 1-9 + 13 are **positive**: the parse function should succeed.
Vectors 10-12 are **negative**: the parse function should error with
the documented kind.

## Run against the LamQuant Rust crate

```
cargo build --release --bin lml --features host
python3 specs/conformance/verify.py specs/conformance/vectors/*.lml
```

Exit 0 means every vector matched its expected behaviour.

## Run against your own LML reader

`verify.py` reads each vector's sibling `.expected.json`. The
expected JSON pins the per-vector category + the documented error
kind for negative vectors. Wire your reader in place of the `--lml-
binary` argument:

```
python3 specs/conformance/verify.py --reader my-lml-reader \
    specs/conformance/vectors/*.lml
```

The reader binary must:
- Print a one-line JSON object `{"status":"OK","n_channels":...,...}`
  on stdout for positive vectors.
- Exit with a non-zero code AND a `{"status":"ERR","kind":"<KIND>"}`
  line on stderr for negative vectors, where `<KIND>` ∈
  `{InvalidMagic, InvalidHeader, CrcMismatch, Truncated, UnsupportedVersion}`.

## Regenerating the vectors

```
python3 tools/build_conformance_vectors.py
```

This runs the LamQuant `lml encode` binary and post-processes its
output (for the legacy + corruption vectors) to produce the
deterministic byte content. Pin SEED in the generator if you need
reproducibility across machines.

## Honest scope note (Item E, decision recorded in plan)

`verify.py` is **validation-only**, not a reference decoder. It
inspects the header + footer + window CRC structure + metadata, but
it does NOT invert the entropy / LPC / lifting passes — those live
in `lamquant-core` (Rust) and are the only authoritative full-decode
implementation today. If you build a clean-room Python decoder we
will happily link it here.
