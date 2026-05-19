# Lossless Codec Feature Catalogue

LamQuant's lossless codec surface, sorted by what you want to do.
Pick a bucket and the doc covers every command, flag, wire-format
nuance, and known error in that area.

## Buckets

1. [Compression](./01-compression.md) — get bytes INTO an archive
2. [Decompression](./02-decompression.md) — get bytes OUT of an archive
3. [Verification](./03-verification.md) — integrity + provenance
4. [Archive Ops](./04-archive-ops.md) — split / concat / append / strip-pii / volume lifecycle
5. [Browse / Inspect](./05-browse-inspect.md) — `ls` / `cat` / `info` read-only paths
6. [Cryptography](./06-cryptography.md) — encrypt / decrypt / sign / verify
7. [OS Integration](./07-os-integration.md) — FUSE, KDE Ark, MIME, double-click-to-open
8. [Export](./08-export.md) — MATLAB / Parquet / HDF5 / BIDS sinks
9. [Operational](./09-operational.md) — watch, systemd, webhooks, metrics, S3
10. [Build / Release](./10-build-release.md) — musl, SBOM, fuzz, conformance
11. [CLI UX](./11-cli-ux.md) — colors, verbosity, completions, man, `--force`, `--emit-json-events`

## Cross-cutting docs

- [`../INSTALL_FILE_MANAGER_INTEGRATION.md`](../INSTALL_FILE_MANAGER_INTEGRATION.md) — Ark / FUSE / MIME install walkthrough
- [`../CLI_REFERENCE.md`](../CLI_REFERENCE.md) — auto-generated full flag listing
- [`../CLI_FEATURE_MATRIX.md`](../CLI_FEATURE_MATRIX.md) — chronological audit trail with commit refs
- [`../FEATURES.md`](../FEATURES.md) — value-ordered inventory of every shipped capability
- [`../lml-format-v1.md`](../lml-format-v1.md) — frozen LML wire-format spec
- [`../FAQ.md`](../FAQ.md) — install / encode / decode / encrypt / watch troubleshooting
- [`../PROBLEM.md`](../PROBLEM.md) — what LamQuant is for and how it compares to gzip / zstd / FLAC
- [`../BUILDING.md`](../BUILDING.md) — build from source

## Status legend

- shipped — works in the current release tag (`lossless-codec-v1.4`)
- partial — implemented behind a feature flag or with known gaps
- planned — on the roadmap, not yet started
- not feasible — investigated and ruled out (see linked rationale)
