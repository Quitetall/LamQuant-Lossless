//! Sync LSL inlet — Phase 3 scaffolding.
//!
//! Phase 3 will land the LSL → `.lml` encoding bridge here:
//!   * `Inlet::subscribe(name)` finds an LSL stream by name.
//!   * `Inlet::record_to_lml(path)` pulls samples, buffers into
//!     codec windows, encodes to disk via `lml::compress_with_mode`.
//!   * Timestamp metadata (LSL-clock + host wall-clock) is stored
//!     in the LML metadata JSON for reproducible re-replay.
//!
//! Empty for Phase 1; the module exists so `lib.rs` doesn't have
//! to grow gates around its `pub mod` declaration when Phase 3
//! lands.

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        // Phase 3 lands real tests. This one keeps the module's
        // test target alive so `cargo test --lib` doesn't report
        // an empty module warning when other phases add tests.
    }
}
