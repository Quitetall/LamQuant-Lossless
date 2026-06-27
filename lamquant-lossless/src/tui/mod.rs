//! LML interactive TUI — thin re-export from the shared `lamquant-tui` framework.
//!
//! The framework's default `App` starts at `SCREEN_CODEC_HUB` with a single
//! Codec Hub tile and the "codec lml · AGPL-3.0-or-later" splash branding.
//! No LML-specific extensions needed — the framework IS the LML TUI.

// Re-export everything from the shared framework.
pub use lamquant_tui::*;

// Provide the entry point the binary expects (`tui::run_interactive()`).
// The framework's `run_interactive()` handles: panic hook → raw mode →
// alternate screen → boot splash → App::new().run() → restore terminal.
