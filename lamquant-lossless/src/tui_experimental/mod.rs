//! Interactive TUI for the `lml` binary.
//!
//! When `lml` is invoked with no arguments, this module provides a
//! full-screen interactive menu matching the Python `lamquant.py` TUI.
//! With arguments, the existing CLI path runs unchanged.

pub mod codec;
pub mod menu;
pub mod style;

pub use menu::run_interactive;
