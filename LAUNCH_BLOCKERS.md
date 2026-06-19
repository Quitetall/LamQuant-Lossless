# lml Crate Launch Blockers

Scope: publish LamQuant Lossless as the `lml` Rust crate and keep `lml` as the CLI/TUI entrypoint.

## Blocking

- `lml` name ownership on crates.io is unresolved. Local packaging reports that crates.io already has `lml` versions `0.1.0` and `0.0.0`; publishing `7.7.1` requires owner access or a renamed crate.
- Publish order is not complete. `lml` depends on local `lmq-common`, so `lmq-common` must be published first or replaced by an in-crate module before `lml` can be packaged for crates.io.
- Workspace-local deps leak into the publish graph through default features. `lamquant-ops` and `lamquant-history` are local path crates with `publish = false`; either publish/split them, gate them behind non-default internal features, or remove them from the crates.io build surface.
- `lmafs` cannot publish until `lml 7.7.1` exists on crates.io because it depends on `package = "lml", version = "7.7.1"`.

## Release Hygiene

- Repo-wide `cargo fmt --check` currently reports pre-existing formatting drift. Do not run broad formatting during launch cleanup unless the diff is intentionally scoped.
- `cargo check -p lml --no-default-features` currently fails because the no-std profile lacks allocator, panic, and unwinding setup. Decide whether no-default-features is a supported published configuration or document it as firmware-integration-only.
- The root worktree is dirty from the mode split and launch cleanup. Cut the publish commit only after the dependency graph and crate name issue are resolved.

## Done In This Cleanup

- Crate, Python, citation, README, TUI, WASM, and firmware-facing license metadata now use `AGPL-3.0-or-later`.
- Default `lml encode` behavior remains MCU mode; basestation mode is opt-in behind `experimental_basestation`.
- NWB/HDF5 support is available through the `nwb` feature while preserving MCU integer-only defaults.
