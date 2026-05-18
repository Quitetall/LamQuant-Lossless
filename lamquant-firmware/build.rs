//! Place memory.x in OUT_DIR so the linker (-Tmemory.x) finds our
//! memory layout, dispatching to the per-target file (ADR 0019).
//!
//! Per-target Cargo features select the right linker script:
//!
//!   --features target-rp2350   → memory/rp2350.x
//!   --features target-nrf54l15 → memory/nrf54l15.x
//!   --features target-esp32p4  → memory/esp32p4.x
//!   --features target-stm32n6  → memory/stm32n6.x
//!
//! Mutually exclusive — selecting multiple is a hard error. Selecting
//! none falls back to the repo-root `memory.x` (currently RP2350), so
//! existing `cargo build --target riscv32imac-unknown-none-elf` flows
//! continue to work unchanged while the per-target HAL bringup lands
//! incrementally.
//!
//! memory.x defines MEMORY + REGION_ALIAS, then INCLUDEs riscv-rt's
//! link.x (or cortex-m-rt's link.x for ARM targets). Named `memory.x`
//! (not `device.x`) to avoid collision with PAC-emitted device.x.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"));
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo"));

    // Snapshot active per-target features. cargo exports
    // CARGO_FEATURE_TARGET_<UPPER_ID>=1 for each `--features
    // target-<id>` invocation. The list must stay synchronized with
    // the `[features]` table in Cargo.toml — adding a new target =
    // one feature + one memory/<id>.x + one row here.
    let candidates = [
        ("CARGO_FEATURE_TARGET_RP2350", "rp2350"),
        ("CARGO_FEATURE_TARGET_NRF54L15", "nrf54l15"),
        ("CARGO_FEATURE_TARGET_ESP32P4", "esp32p4"),
        ("CARGO_FEATURE_TARGET_STM32N6", "stm32n6"),
    ];
    let selected: Vec<&str> = candidates
        .iter()
        .filter(|(env_var, _)| env::var_os(env_var).is_some())
        .map(|(_, id)| *id)
        .collect();

    // Mutual exclusion — a build with two target-* features active is
    // a configuration error (it would produce a wrong-layout binary).
    if selected.len() > 1 {
        panic!(
            "lamquant-firmware: multiple target-* features active ({:?}); \
             pick exactly one via --features target-<id>",
            selected
        );
    }

    // Resolve memory.x source. None → repo-root memory.x (RP2350
    // legacy default, kept for backward compat while HAL bringup
    // progresses). Some(id) → memory/<id>.x.
    let src_path: PathBuf = match selected.first() {
        None => manifest_dir.join("memory.x"),
        Some(id) => manifest_dir.join("memory").join(format!("{id}.x")),
    };
    let bytes = fs::read(&src_path).unwrap_or_else(|e| {
        panic!(
            "lamquant-firmware: failed to read linker script {}: {}",
            src_path.display(),
            e
        );
    });
    fs::write(out.join("memory.x"), bytes).unwrap();
    println!("cargo:rustc-link-search={}", out.display());

    // Re-run when any candidate source changes. cargo's
    // rerun-if-changed is per-file, not per-directory, so list each.
    rerun_if_changed(&manifest_dir.join("memory.x"));
    for (_, id) in &candidates {
        rerun_if_changed(&manifest_dir.join("memory").join(format!("{id}.x")));
    }
    println!("cargo:rerun-if-changed=build.rs");
}

fn rerun_if_changed(p: &Path) {
    println!("cargo:rerun-if-changed={}", p.display());
}
