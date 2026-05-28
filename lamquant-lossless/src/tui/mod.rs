//! LamQuant TUI — modular full-screen terminal interface.
//!
//! Architecture:
//!   App         — owns terminal, runs event loop, manages layout
//!   Router      — screen stack, navigation history
//!   Panel trait — renderable, focusable, event-handling widget
//!   Operation   — background task that streams events to UI
//!
//! Adding a new feature:
//!   1. Implement `Panel` trait (render + handle_event)
//!   2. Optionally implement `Operation` trait for background work
//!   3. Register in Router with a screen ID
//!   4. Done — framework handles layout, focus, key dispatch

pub mod app;
pub mod art;
pub mod clipboard;
pub mod config;
pub mod crash;
pub mod history;
pub mod layout;
pub mod operations;
pub mod panel;
pub mod panels;
pub mod peers_config;
pub mod router;
pub mod snapshot;
pub mod state;
pub mod theme;

use app::App;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::io;

/// Entry point — called when `lml` is run with no arguments.
pub fn run_interactive() -> i32 {
    match run_app() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("TUI error: {}", e);
            1
        }
    }
}

fn run_app() -> io::Result<()> {
    // Restore the TTY before any panic touches stderr, otherwise the user is
    // left with a corrupt terminal that no longer echoes input.
    crash::install_panic_hook();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Paint the boot splash IMMEDIATELY after entering the alternate
    // screen — before App::new() does its config/history/peers I/O.
    // Without this the user sees a blank terminal for the full
    // App::new() construction window. The full SplashPanel takes
    // over once App is built and runs its 700ms tick budget on top
    // of whichever screen the boot logic landed on.
    terminal.draw(|f| panels::splash::render_boot(f, f.area()))?;

    let result = App::new().run(&mut terminal);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}
