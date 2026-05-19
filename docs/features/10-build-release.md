# Build / Release

> Supply chain: build matrix, reproducible builds, SBOM, fuzz harness,
> conformance suite, release signing. The bucket for contributors and
> packagers, not end users.

LamQuant ships reproducible static-musl binaries with SBOM, daily
security audits, fuzz coverage on 5 targets, and a 13-vector
conformance suite. Build artifacts are byte-identical across host
machines when fed `SOURCE_DATE_EPOCH`.

## At a glance

| Feature | Path / lane | Status | First shipped | Notes |
|---|---|---|---|---|
| Static musl Linux binary | CI release lane | shipped | v1.0 (2.3) | `x86_64-unknown-linux-musl` for `lml` + `lamquant` |
| macOS CI lane (x86_64 + aarch64) | `ci.yml` matrix | shipped | v1.0 (2.1) | macos-13 + macos-14 |
| Windows CI lane | `ci.yml` matrix | shipped | v1.0 (2.2) | windows-latest MSVC, long-paths |
| Reproducible builds | `SOURCE_DATE_EPOCH` | shipped | v1.0 (2.4) | Tag-commit `%ct`, `--remap-path-prefix` |
| SBOM (SPDX 2.3) | cargo-sbom | shipped | v1.0 (2.5) | Generated per release |
| SBOM (CycloneDX 1.5) | cargo-cyclonedx | shipped | v1.0 (2.5) | Parallel format |
| cargo-audit | `audit.yml` | shipped | v1.0 (8.8) | Daily cron + per-PR; RUSTSEC advisories |
| cargo-deny | `deny.toml` | shipped | v1.0 (8.8) | License + bans + sources policy |
| All-features build lane | `ci.yml` | shipped | v1.0 (D) | parquet + hdf5 + s3 + dicom + keyring + async |
| Fuzz harness | `fuzz/` | shipped | v1.0 (F) | 5 targets (decompress, roundtrip, container_header, offset_table, lma_manifest) |
| Conformance test suite | `specs/conformance/` | shipped | v1.0 (E) | 13 vectors, CI `conformance` job |
| Release notarize (macOS) | `release.yml` | shipped | v1.0 (7.7) | Opt-in via repo secrets |
| Release Authenticode (Windows) | `release.yml` | shipped | v1.0 (7.7) | Opt-in via repo secrets |
| cargo-tarpaulin Rust coverage | `.tarpaulin.toml` | shipped | v1.2 (T.1) | llvm engine, fail-under 70% |
| Python coverage on `lamquant_codec` | `pyproject.toml` | shipped | v1.2 (T.2) | Coverage scope extended |
| 1M-file stress harness | `scripts/stress_1m_files.sh` | shipped | v1.0 (8.5) | synth N EDFs → encode → archive → verify |
| >100 GB file fixture recipe | `docs/STRESS_TESTING.md` | shipped | v1.0 (8.6) | Pipeline scaling test |

## Build matrix

CI runs on three OS lanes plus an all-features lane:

| Lane | Runner | What it builds |
|---|---|---|
| Linux glibc | `ubuntu-latest` | Default features; runs full test suite |
| Linux musl | `ubuntu-latest` cross | Static `x86_64-unknown-linux-musl` release binaries |
| macOS x86_64 | `macos-13` | Default features; runs full test suite |
| macOS aarch64 | `macos-14` | Default features; runs full test suite |
| Windows MSVC | `windows-latest` | Default features; long-paths config enabled |
| All-features | `ubuntu-latest` | `--features async,s3,parquet,hdf5,dicom,keyring` |

Static musl is the recommended download for production: no glibc
version dependency, runs on every modern Linux.

## Reproducible builds

Two-step:

1. **`SOURCE_DATE_EPOCH`** — pinned to the commit's `%ct` (committer
   date in Unix seconds) at the release tag. Eliminates timestamp
   embedding in the binary.
2. **`--remap-path-prefix`** — strips the absolute build path from
   debug symbols and panic messages. Two machines with different
   checkout paths produce the same binary.

Verify with `scripts/verify_reproducible_build.sh` — builds twice on
different paths, diff'd byte-for-byte.

```sh
SOURCE_DATE_EPOCH=$(git -C /path/to/LamQuant log -1 --format=%ct v1.4) \
  cargo build --release --target x86_64-unknown-linux-musl
```

CI's release lane runs this verifier as a job step; mismatches fail
the release.

## SBOM (SPDX + CycloneDX)

Two formats generated in CI:

- **SPDX 2.3** via `cargo-sbom` — `lamquant.spdx.json`
- **CycloneDX 1.5** via `cargo-cyclonedx` — `lamquant.cyclonedx.json`

Both ship as release artifacts. Compliance teams use them for license
audits and known-CVE matching. Tooling: `syft`, `grype`, `cyclonedx-cli`.

## cargo-audit + cargo-deny

| Tool | Purpose | Trigger |
|---|---|---|
| `cargo-audit` | RUSTSEC advisory match against `Cargo.lock` | Daily cron + every PR |
| `cargo-deny` | License allowlist + crate bans + source restrictions | Every PR |

`deny.toml` policy explicitly:

- Allowed licenses: MIT, Apache-2.0, BSD-3-Clause, etc.
- Banned: GPL family (incompatible with the project's MIT licensing)
- Sources: crates.io + known-good mirrors only

A failure in either lane blocks merge. Audit-2026-05-11 hardened this
chain after the cargo-deny lane briefly drifted out of CI.

## Fuzz harness

`fuzz/fuzz_targets/` contains 5 libfuzzer-sys targets:

| Target | Coverage |
|---|---|
| `decompress` | LML container decode under adversarial input |
| `roundtrip` | Encode → decode → compare end-to-end |
| `container_header` | LML1 header parse |
| `offset_table` | LMLFOOT1 footer parse |
| `lma_manifest` | LMA1 manifest decompress + JSON parse |

CI runs 60 seconds per target on every PR. Crashes are saved as
corpus entries; the corpus tarball is uploaded as an artifact.

For continuous fuzzing (longer runs, OSS-Fuzz integration), upstream
the harness via `cargo fuzz run <target>`.

## Conformance test suite (Group E)

`specs/conformance/` ships 13 versioned test vectors that any
third-party LML reader can run against to claim conformance:

```
specs/conformance/
├── README.md          (vector list + protocol)
├── verify.py          (structural-only Python validator)
└── vectors/
    ├── 01-empty.lml
    ├── 02-single-channel-single-window.lml
    ├── ...
    └── 13-large-multichannel.lml
```

Each vector has:
- Source bytes (synthetic, deterministic)
- Expected `.lml` output (byte-exact)
- Per-window CRC-32 expectations
- Decoded SHA-256

CI's `conformance` job runs the suite on every PR; a regression in
the codec that breaks any vector fails the merge.

A pure-Python clean-room decoder is on the deferred-but-planned list.
The Rust crate is the authoritative reference until then.

## cargo-tarpaulin (v1.2 T.1)

Rust line coverage via `llvm` engine, configured in `.tarpaulin.toml`:

```toml
fail-under = 70
engine = "llvm"
```

CI's coverage job uploads:
- HTML report as an artifact
- Optional Codecov upload (`CODECOV_TOKEN` repo secret)

Run locally:
```sh
cargo install cargo-tarpaulin
cargo tarpaulin --config .tarpaulin.toml
```

## Python coverage extended (v1.2 T.2)

`pyproject.toml`'s coverage scope expanded from `[ai_models,
firmware]` to also include `lamquant_codec`. Same `pytest-cov` run
now covers the Python codec layer plus the AI / firmware layers.

```sh
pytest --cov=lamquant_codec --cov=ai_models --cov=firmware --cov-report=term
```

## Release pipeline notarize + Authenticode (Phase 7.7)

`.github/workflows/release.yml` has opt-in slots:

| Platform | Tool | Triggered by repo secrets |
|---|---|---|
| macOS | `notarytool` | `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID` |
| Windows | `signtool.exe` | `WINDOWS_CERT_PFX_B64`, `WINDOWS_CERT_PASSWORD` |

Without secrets configured, the lane builds unsigned binaries. With
secrets, every release ships notarized macOS `.dmg` and Authenticode-signed
Windows `.exe`. See [Cryptography](./06-cryptography.md) for the
broader signing surface.

## 1M-file stress harness

`scripts/stress_1m_files.sh`: synthesize N short EDFs, encode each,
bundle into archives, verify integrity, time the whole pipeline.
Default N = 1 000 000; tune with `STRESS_N` env. Run on a beefy
machine with plenty of inodes available.

```sh
STRESS_N=10000 ./scripts/stress_1m_files.sh   # smaller smoke run
```

The harness catches scaling regressions (e.g., O(N^2) directory
walks) that small-corpus benches don't.

## >100 GB file fixture

`docs/STRESS_TESTING.md` ships a recipe to generate a >100 GB synthetic
EDF for stress-testing the codec's streaming path. Required for
ECoG / long-term-monitoring workflows where a single recording may run
days at high sample rate.

## Feature-gated builds

| Cargo feature | Adds | Default? |
|---|---|---|
| `async` | tokio runtime, `lml watch` / `fetch` / `notify` / `metrics` | no |
| `s3` | aws-sdk-s3 (rustls), S3 read / write / ETag | no |
| `parquet` | parquet + arrow, `lml export -f parquet` | no |
| `hdf5` | hdf5-metno, `lml export -f hdf5` | no |
| `dicom` | dicom-rs, `.dcm` source reader | no |
| `keyring` | OS keyring crate, library-only API | no |
| `host` | Default desktop build (rayon + AVX2) | yes |

Release artifacts use the all-features build. Custom downstream
packagers can pick a smaller subset.

## Source layout

| Path | What's there |
|---|---|
| `lamquant-core/` | Library + `lml` / `lamquant` binaries |
| `crates/lmafs/` | FUSE filesystem ([OS Integration](./07-os-integration.md)) |
| `crates/lamquant-history/` | Append-only event log helpers |
| `crates/lamquant-ops/` | `OpEvent` wire format (for `--emit-json-events`) |
| `specs/conformance/` | Codec conformance vectors |
| `fuzz/` | libfuzzer-sys harnesses |
| `dist/systemd/` | systemd unit templates |
| `installer/` | MIME, FUSE, Ark plugin install scripts |
| `tests/integration/` | Python integration test suite |
| `scripts/` | Bench, stress, repro-verify scripts |
| `docs/` | This doc tree |

## Error cases

| Trigger | Behavior |
|---|---|
| Reproducible-build verifier sees byte diff | CI release lane fails |
| cargo-audit hits a fresh RUSTSEC advisory | CI daily cron fails; PRs flag the same advisory |
| cargo-deny sees a new GPL-family dep | refuse merge |
| Fuzz target crashes in CI | corpus entry saved, artifact uploaded; test job fails |
| Conformance vector mismatch | CI `conformance` job fails |
| tarpaulin coverage drops below `fail-under = 70` | CI coverage job fails |
| macOS notarize secrets missing | release lane logs "unsigned build" and continues |

## Related

- **Other buckets**:
  - [Verification](./03-verification.md) — conformance suite enforces the wire format
  - [Cryptography](./06-cryptography.md) — release-pipeline notarize / Authenticode lanes
  - [Operational](./09-operational.md) — `--features async` / `--features s3` build flags
- **Source files**:
  - `.github/workflows/ci.yml` — main CI matrix
  - `.github/workflows/release.yml` — release lane + signing
  - `.github/workflows/audit.yml` — daily cargo-audit cron
  - `.tarpaulin.toml` — coverage config
  - `deny.toml` — cargo-deny policy
  - `fuzz/fuzz_targets/` — fuzz harness
  - `specs/conformance/` — codec conformance vectors
  - `scripts/verify_reproducible_build.sh` — repro-build verifier
  - `scripts/stress_1m_files.sh` — stress harness
- **Tests**:
  - `lamquant-core/tests/cross_platform_bytes.rs` — x86 / ARM / RISC-V byte equality
- **Commits**:
  - Phase 2.3 — static musl Linux binary
  - Phase 2.4 — reproducible builds
  - Phase 2.5 — SBOM (SPDX + CycloneDX)
  - Phase 8.8 — cargo-audit + cargo-deny lanes
  - Phase 8.9 — publishable conformance suite
  - v1.2 T.1 — cargo-tarpaulin + CI coverage job
  - v1.2 T.2 — Python coverage extended to `lamquant_codec`
- **Cross-cutting docs**:
  - [`../BUILDING.md`](../BUILDING.md) — full build walkthrough
  - [`../FEATURES.md`](../FEATURES.md) §9 (CI hardening + supply-chain)
  - [`../STRESS_TESTING.md`](../STRESS_TESTING.md) — large-file harness recipe
