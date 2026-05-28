//! Phase 0 #17 contract: when the runner emits an Error event whose message
//! contains "cancelled", the OutputPanel sets cancelled=true (NOT
//! failed=true) so the border + header render in theme::warning() yellow
//! rather than theme::error() red. A genuine error keeps failed=true.

use lamquant_core::tui::operations::{channel, OpEvent, OpEventSink};
use lamquant_core::tui::panel::Panel;
use lamquant_core::tui::panels::output::OutputPanel;

fn ts() -> i64 {
    1_730_000_000_000
}

#[test]
fn cancelled_message_marks_panel_cancelled_not_failed() {
    let mut panel = OutputPanel::new();
    let (sink, rx) = channel();
    panel.start("encode".into(), rx);

    sink.emit(OpEvent::Started {
        ts_ms: ts(),
        op_id: "encode".into(),
        total: None,
    });
    sink.emit(OpEvent::Error {
        ts_ms: ts(),
        message: "encode cancelled by user".into(),
    });
    // P3 of refactor: tick() no longer drains; pull each event via
    // try_recv_event() and apply via consume() — same flow App.tick_panels uses.
    while let Some(ev) = panel.try_recv_event() {
        panel.consume(ev);
    }
    panel.tick();

    assert!(panel.is_done(), "panel should reach done state");
    assert!(
        panel.is_cancelled(),
        "expected cancelled=true on cancel message"
    );
    assert!(!panel.is_failed(), "failed must stay false on cancel");
}

#[test]
fn real_error_message_marks_panel_failed_not_cancelled() {
    let mut panel = OutputPanel::new();
    let (sink, rx) = channel();
    panel.start("encode".into(), rx);

    sink.emit(OpEvent::Started {
        ts_ms: ts(),
        op_id: "encode".into(),
        total: None,
    });
    sink.emit(OpEvent::Error {
        ts_ms: ts(),
        message: "out of disk space".into(),
    });
    // P3 of refactor: tick() no longer drains; pull each event via
    // try_recv_event() and apply via consume() — same flow App.tick_panels uses.
    while let Some(ev) = panel.try_recv_event() {
        panel.consume(ev);
    }
    panel.tick();

    assert!(panel.is_done());
    assert!(panel.is_failed(), "expected failed=true on real error");
    assert!(
        !panel.is_cancelled(),
        "cancelled must stay false on real error"
    );
}

#[test]
fn done_message_marks_neither_failed_nor_cancelled() {
    let mut panel = OutputPanel::new();
    let (sink, rx) = channel();
    panel.start("encode".into(), rx);

    sink.emit(OpEvent::Started {
        ts_ms: ts(),
        op_id: "encode".into(),
        total: None,
    });
    sink.emit(OpEvent::Done {
        ts_ms: ts(),
        message: "ok".into(),
    });
    // P3 of refactor: tick() no longer drains; pull each event via
    // try_recv_event() and apply via consume() — same flow App.tick_panels uses.
    while let Some(ev) = panel.try_recv_event() {
        panel.consume(ev);
    }
    panel.tick();

    assert!(panel.is_done());
    assert!(!panel.is_failed());
    assert!(!panel.is_cancelled());
}
