//! LML interactive TUI — the shared framework, driven by the codec's own manifest.
//!
//! This module used to be a bare `pub use lamquant_tui::*;` whose own comment
//! said "the framework IS the LML TUI". That was the ADR 0053 noncompliance
//! stated out loud: the shared crate held this product's build label, tile
//! table and home screen, so `lamquant-tui` could not be reused by a front-end
//! that wanted different ones without editing the framework.
//!
//! The framework still supplies the shell — event loop, router, layout,
//! widgets, terminal lifecycle, panic recovery — and that is what the glob
//! re-export below is for. What moved is the product: see `manifest.rs`.

pub mod manifest;

// Re-export the framework's shell types. `run_interactive` arrives through this
// glob too, but the explicit definition below shadows it — an inherent item
// always wins over a glob import, which is what lets this module keep the
// entry-point name `bin/lml.rs` already calls while changing what it does.
pub use lamquant_tui::*;

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
