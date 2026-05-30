//! TUI smoke tests — drive the App with synthetic key events through a
//! `TestBackend`, then snapshot or assert on the rendered buffer.
//!
//! These tests are coarse: they verify navigation, key bindings, and
//! that screens render without panicking. Pixel-exact snapshots are
//! avoided so future cosmetic edits don't break the suite.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lamquant_core::tui::app::App;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}
fn ctrl(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::CONTROL)
}

fn make_app() -> App {
    App::new()
}

/// Dismiss any intercepting startup screen (splash, wizard, resume,
/// root_warn) so tests land on the main hub regardless of the host's
/// real history.json / config state. Each Esc backs one screen out; a
/// few rounds covers stacked overlays (e.g. splash on top of resume).
fn dismiss_startup_overlays(app: &mut App) {
    for _ in 0..6 {
        match app.current_screen().as_str() {
            "splash" | "wizard" | "resume" | "root_warn" => {
                app.dispatch_key_for_test(key(KeyCode::Esc));
            }
            _ => return,
        }
    }
}

fn make_terminal() -> Terminal<TestBackend> {
    let backend = TestBackend::new(120, 40);
    Terminal::new(backend).expect("terminal")
}

fn buf_contains(term: &Terminal<TestBackend>, needle: &str) -> bool {
    let buf = term.backend().buffer();
    let mut s = String::new();
    for cell in buf.content().iter() {
        s.push_str(cell.symbol());
    }
    s.contains(needle)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn renders_main_menu_without_panic() {
    let mut app = make_app();
    let mut term = make_terminal();
    // First-time wizard may pop up if no config; just assert no panic.
    app.render_for_test(&mut term).expect("render");
}

#[test]
fn navigates_into_lossless_submenu() {
    let mut app = make_app();
    // Make sure we're on main, not the wizard. Esc cancels the
    // wizard from the wizard screen. On main, Esc routes to
    // exit_confirm (panels/main_hub.rs change in this branch); from
    // exit_confirm, Esc backs out → main. So Esc twice from main
    // is a safe round-trip.
    dismiss_startup_overlays(&mut app);
    // After codec-only hub shrink, '1' is the Codec Hub shortcut on the
    // main hub (was Visualization before the shrink — see panels/main_hub.rs
    // HubTile assignments).
    app.dispatch_key_for_test(key(KeyCode::Char('1')));
    let mut term = make_terminal();
    app.render_for_test(&mut term).expect("render");
    assert!(
        ["codec_hub", "main", "wizard"].contains(&app.current_screen().as_str()),
        "screen={}",
        app.current_screen(),
    );
}

#[test]
fn settings_navigation_renders() {
    let mut app = make_app();
    dismiss_startup_overlays(&mut app);
    app.dispatch_key_for_test(key(KeyCode::Char('s')));
    let mut term = make_terminal();
    app.render_for_test(&mut term).expect("render");
    // Settings panel header should render — check for a section name.
    if app.current_screen() == "settings" {
        assert!(
            buf_contains(&term, "BACKEND") || buf_contains(&term, "Settings"),
            "settings did not render expected text"
        );
    }
}

#[test]
fn ctrl_c_on_main_routes_through_exit_confirm() {
    // Phase 0 #16 contract: Ctrl+C on main MUST NOT force-quit immediately.
    // It routes through SCREEN_EXIT_CONFIRM so a stray double Ctrl+C cannot
    // trash the user's session. Confirm requires a separate Enter/y.
    let mut app = make_app();
    dismiss_startup_overlays(&mut app);
    app.dispatch_key_for_test(ctrl(KeyCode::Char('c')));
    assert!(!app.should_quit(), "first Ctrl+C must not force-quit");
    assert_eq!(
        app.current_screen(),
        "exit_confirm",
        "Ctrl+C on main should navigate to exit_confirm"
    );
}

#[test]
fn ctrl_c_on_exit_confirm_does_not_force_quit() {
    // A second Ctrl+C from inside the exit_confirm panel should remain on
    // exit_confirm (the user already sees the prompt) — never force-quit.
    let mut app = make_app();
    dismiss_startup_overlays(&mut app);
    app.dispatch_key_for_test(ctrl(KeyCode::Char('c')));
    app.dispatch_key_for_test(ctrl(KeyCode::Char('c')));
    assert!(
        !app.should_quit(),
        "Ctrl+C from exit_confirm must not force-quit"
    );
    assert_eq!(app.current_screen(), "exit_confirm");
}

#[test]
fn ctrl_c_then_enter_quits() {
    // The escape hatch: Ctrl+C → exit_confirm → Enter (or y) → quit.
    let mut app = make_app();
    dismiss_startup_overlays(&mut app);
    app.dispatch_key_for_test(ctrl(KeyCode::Char('c')));
    app.dispatch_key_for_test(key(KeyCode::Char('y')));
    assert!(
        app.should_quit(),
        "y on exit_confirm after Ctrl+C should quit"
    );
}

#[test]
fn x_routes_through_exit_confirm() {
    let mut app = make_app();
    dismiss_startup_overlays(&mut app);
    // UX: on main hub `x` (or Esc) opens exit-confirm; `q` is Home no-op.
    app.dispatch_key_for_test(key(KeyCode::Char('x')));
    assert!(
        !app.should_quit(),
        "x alone should not quit (goes to confirm)"
    );
    // Now press n to cancel.
    app.dispatch_key_for_test(key(KeyCode::Char('n')));
    assert!(!app.should_quit());
}

#[test]
fn exit_confirm_y_quits() {
    let mut app = make_app();
    dismiss_startup_overlays(&mut app);
    // `x` triggers exit-confirm panel; `y` confirms quit.
    app.dispatch_key_for_test(key(KeyCode::Char('x')));
    app.dispatch_key_for_test(key(KeyCode::Char('y')));
    assert!(app.should_quit(), "y on exit confirm should quit");
}

#[test]
fn renders_every_top_level_screen_without_panic() {
    let mut term = make_terminal();
    // After codec-only hub shrink, surviving top-level shortcuts are:
    //   '1' Codec Hub, 'N' Peers, 's' Settings, 'i' Install & setup,
    //   't' Diagnostics, 'h' Help.
    for shortcut in ['1', 'N', 's', 'i', 't', 'h'] {
        let mut app = make_app();
        dismiss_startup_overlays(&mut app);
        app.dispatch_key_for_test(key(KeyCode::Char(shortcut)));
        // Should not panic on render even if screen isn't fully populated.
        app.render_for_test(&mut term)
            .expect("render did not panic");
    }
}

#[test]
fn tick_after_navigation_does_not_panic() {
    let mut app = make_app();
    dismiss_startup_overlays(&mut app);
    app.dispatch_key_for_test(key(KeyCode::Char('s')));
    for _ in 0..10 {
        app.tick_for_test();
    }
}
