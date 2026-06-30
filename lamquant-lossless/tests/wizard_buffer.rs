// DISABLED (cfg(any()) = never compiled): stale post-W2-extract remnant.
// Drives WizardPanel via crossterm events that `src/tui` no longer accepts
// (headless model has its own event type; crossterm isn't a dep). Preserved as
// migration reference; re-home to crates/lamquant-tui or delete. See task:
// "Relocate stale TUI render tests".
#![cfg(any())]
//! Phase 0 #18 contract: the worker-count input on the wizard's step 3
//! caps at 8 characters. The previous 4-char cap silently truncated
//! 99999 → "9999" with no UI feedback. 8 chars accommodates large
//! Threadripper / Xeon Phi node counts plus a margin.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lamquant_core::tui::config::LamQuantConfig;
use lamquant_core::tui::panel::Panel;
use lamquant_core::tui::panels::wizard::WizardPanel;
use lamquant_core::tui::state::AppState;

fn key(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
}

#[test]
fn wizard_worker_buffer_caps_at_eight_chars() {
    let mut w = WizardPanel::new(LamQuantConfig::default());
    w.set_step_for_test(2);
    // The fresh buffer carries the default workers value (0). Clear it
    // by sending Backspace until empty.
    while !w.workers_buffer_for_test().is_empty() {
        w.handle_event(
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
            &AppState::new(),
        );
    }
    assert_eq!(w.workers_buffer_for_test(), "");

    // Fire 12 digit keypresses — buffer must accept the first 8 only.
    for c in "123456789012".chars() {
        w.handle_event(key(c), &AppState::new());
    }

    let buf = w.workers_buffer_for_test();
    assert_eq!(buf.len(), 8, "buffer length should cap at 8, got {:?}", buf);
    assert_eq!(buf, "12345678", "first 8 digits should be retained");
}

#[test]
fn wizard_worker_buffer_accepts_full_eight_digit_value() {
    let mut w = WizardPanel::new(LamQuantConfig::default());
    w.set_step_for_test(2);
    while !w.workers_buffer_for_test().is_empty() {
        w.handle_event(
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
            &AppState::new(),
        );
    }
    for c in "12345678".chars() {
        w.handle_event(key(c), &AppState::new());
    }
    assert_eq!(w.workers_buffer_for_test(), "12345678");
}
