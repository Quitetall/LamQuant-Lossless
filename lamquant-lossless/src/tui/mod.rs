//! LML interactive TUI — the shared framework, driven by the codec's own manifest.
//!
//! This module used to be a bare `pub use lamquant_tui::*;` whose own comment
//! said "the framework IS the LML TUI". That was the ADR 0053 noncompliance
//! stated out loud: the shared crate held this product's build label, tile
//! table and home screen, so `lamquant-tui` could not be reused by a front-end
//! that wanted different ones without editing the framework.
//!
//! The framework still supplies the shell — event loop, router, layout,
//! widgets, terminal lifecycle, panic recovery. What moved is the product: see
//! `manifest.rs`.
//!
//! THE GLOB RE-EXPORT IS GONE, and what it was actually holding up is worth
//! recording. `pub use lamquant_tui::*;` published the framework's entire
//! surface — `app`, `panel`, `router`, `config`, everything — through the
//! codec, and the only consumers of that breadth were two of the codec's own
//! test files: `reducer_unit.rs` (10 tests over the App reducer and router) and
//! `config_save.rs` (3 over config persistence). Neither contained a single
//! codec reference. They tested the FRAMEWORK, and sat here purely because the
//! glob made the framework reachable from this crate.
//!
//! They now live in `crates/lamquant-tui/tests/`, beside the code they
//! constrain — the same relocation ADR 0185's ceiling note records for the four
//! files moved on 2026-08-22, and for the same reason: a test that asserts
//! nothing about the codec should not be a codec test.
//!
//! What remains is one function. `bin/lml.rs` calls `tui::run_interactive()`;
//! that is the entire surface this module needs from the framework now.
//!
//! This does NOT by itself close the lossless(L1) -> meta(L3) layer inversion,
//! and claiming otherwise would be the easy misreading: `lamquant-tui` still
//! lives in the private meta, so the [dependencies] edge survives. What it does
//! is shrink that edge from "the whole framework, transitively" to a single
//! entry point, which is what makes the real fix — publishing the framework, or
//! moving it to a public module — a mechanical change rather than an audit.

pub mod manifest;

/// Run the codec's TUI.
///
/// Was `lamquant_tui::run_interactive()`, which resolved to
/// `ShellProfile::Spoke` — the framework choosing this binary's shell from a
/// closed enum of the front-ends it happened to know about. Now the codec
/// states its own manifest and the framework composes it. Raw mode, the boot
/// splash, panic recovery and terminal restoration are unchanged: they live in
/// `run_interactive_with_manifest`, which is the part every front-end wants
/// identical.
pub fn run_interactive() -> i32 {
    lamquant_tui::run_interactive_with_manifest(manifest::lml())
}
