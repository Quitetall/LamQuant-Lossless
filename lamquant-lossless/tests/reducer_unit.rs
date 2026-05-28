//! Reducer-level unit tests — assert that `App::dispatch(Action)` mutates
//! `AppState` and `Router` in the expected ways without going through the
//! key-event translation layer.
//!
//! Added in the reactive-store post-refactor cleanup pass. Establishes
//! the harness for type-checking the dispatch chokepoint independently
//! of TUI rendering and key parsing.

use lamquant_core::tui::app::App;
use lamquant_core::tui::panel::Action;
use lamquant_core::tui::router;

fn fresh_app() -> App {
    // App::new() may navigate to wizard or root_warn screen on first run;
    // tests don't care which screen is active at construction.
    App::new()
}

#[test]
fn navigate_pushes_screen_onto_router() {
    let mut app = fresh_app();
    app.dispatch_for_test(Action::Navigate(router::SCREEN_SETTINGS.to_string()));
    assert_eq!(app.current_screen(), router::SCREEN_SETTINGS);
}

#[test]
fn back_pops_to_previous_screen() {
    let mut app = fresh_app();
    let initial = app.current_screen().to_string();
    app.dispatch_for_test(Action::Navigate(router::SCREEN_SETTINGS.to_string()));
    assert_eq!(app.current_screen(), router::SCREEN_SETTINGS);
    app.dispatch_for_test(Action::Back);
    assert_eq!(app.current_screen(), initial);
}

#[test]
fn status_message_lands_in_state() {
    let mut app = fresh_app();
    app.dispatch_for_test(Action::StatusMessage("hello world".to_string()));
    assert_eq!(
        app.state_for_test().status_message.as_deref(),
        Some("hello world"),
    );
}

#[test]
fn set_status_aliases_status_message() {
    let mut app = fresh_app();
    app.dispatch_for_test(Action::SetStatus("from launcher".to_string()));
    assert_eq!(
        app.state_for_test().status_message.as_deref(),
        Some("from launcher"),
    );
}

#[test]
fn quit_from_exit_confirm_sets_should_quit() {
    let mut app = fresh_app();
    // Get to exit_confirm: dispatch Quit when not on exit_confirm navigates there.
    app.dispatch_for_test(Action::Quit);
    assert_eq!(app.current_screen(), "exit_confirm");
    // Second Quit on exit_confirm sets should_quit.
    app.dispatch_for_test(Action::Quit);
    assert!(
        app.should_quit(),
        "should_quit should flip on Quit-from-exit_confirm"
    );
}

#[test]
fn home_from_main_routes_to_exit_confirm() {
    let mut app = fresh_app();
    // Navigate to main first (might start on wizard).
    app.dispatch_for_test(Action::Navigate(router::SCREEN_MAIN.to_string()));
    assert_eq!(app.current_screen(), router::SCREEN_MAIN);
    // Home from main escalates to exit_confirm.
    app.dispatch_for_test(Action::Home);
    assert_eq!(app.current_screen(), "exit_confirm");
}

#[test]
fn home_from_other_screen_routes_to_main() {
    let mut app = fresh_app();
    app.dispatch_for_test(Action::Navigate(router::SCREEN_SETTINGS.to_string()));
    app.dispatch_for_test(Action::Home);
    assert_eq!(app.current_screen(), router::SCREEN_MAIN);
}

#[test]
fn back_one_pops_router_unconditionally() {
    let mut app = fresh_app();
    app.dispatch_for_test(Action::Navigate(router::SCREEN_SETTINGS.to_string()));
    let pre = app.current_screen().to_string();
    app.dispatch_for_test(Action::BackOne);
    assert_ne!(
        app.current_screen(),
        pre,
        "BackOne should pop the router stack"
    );
}

#[test]
fn ignored_and_consumed_are_noops() {
    let mut app = fresh_app();
    let before = app.current_screen().to_string();
    app.dispatch_for_test(Action::Ignored);
    app.dispatch_for_test(Action::Consumed);
    assert_eq!(
        app.current_screen(),
        before,
        "Ignored/Consumed must not navigate"
    );
}

#[test]
fn run_operation_status_renders_op_id() {
    let mut app = fresh_app();
    app.dispatch_for_test(Action::RunOperation {
        op_id: "encode".to_string(),
        args: vec!["foo.edf".into()],
    });
    let msg = app.state_for_test().status_message.as_deref().unwrap_or("");
    assert!(
        msg.contains("encode"),
        "status should mention op_id, got {:?}",
        msg
    );
}
