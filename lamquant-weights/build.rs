//! Build-time invariants for the generated crate.
//!
//! Currently a no-op. Future: verify `.exportlock.json` matches the SHA-256
//! of `src/generated/`, refuse to build if codegen output has been
//! hand-edited or corrupted.

fn main() {
    println!("cargo:rerun-if-changed=src/generated/");
    println!("cargo:rerun-if-changed=.exportlock.json");
}
