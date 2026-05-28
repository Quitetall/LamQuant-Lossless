//! Subprocess runner — re-export from `lamquant-ops` for backwards compat.
//!
//! The actual implementation moved to `crates/lamquant-ops/src/runner.rs`
//! during the UI parity refactor (Phase 1). New code should `use
//! lamquant_ops::runner::*` directly.

pub use lamquant_ops::runner::{spawn_blut, spawn_command, spawn_lml, OpHandle};
