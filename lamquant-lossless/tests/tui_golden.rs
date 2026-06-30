// DISABLED (cfg(any()) = never compiled): stale post-W2-extract remnant.
// Tests ratatui rendering (TestBackend) that moved to the `tui/` crate; `src/tui`
// here is now headless (no ratatui/crossterm deps), so this can't compile.
// Preserved as the migration reference; re-home to crates/lamquant-tui or delete
// once covered there. See task: "Relocate stale TUI render tests".
#![cfg(any())]
//! TUI golden-render baselines.
//!
//! These tests snapshot the *current* rendered output of TUI panels
//! into golden files under `tests/golden/`. The goal is NOT correctness
//! per se — it is to lock today's visible surface so that a future
//! migration (e.g. the W2 lamquant-core extract) can prove byte-for-byte
//! equality with the LamQuant-Lossless reference implementation.
//!
//! Conventions:
//!   - Buffer is fixed at 80×24 — small enough that a human can eyeball
//!     a diff, and large enough to fit the main hub plus its tip line
//!     under `ui_preference = "ask"` (the default).
//!   - The buffer is serialised to a plain UTF-8 string where each cell
//!     becomes one symbol; rows are joined with '\n' and each row ends
//!     with '\n'. Styling/colour is dropped on purpose: this baseline
//!     covers character layout only. A future companion baseline can
//!     capture style attributes if migration ever needs to prove that
//!     too.
//!   - Set `LAMQUANT_REGEN_GOLDENS=1` (or delete the golden file) to
//!     refresh the baseline. Otherwise the test asserts byte-equality
//!     against the on-disk file.
//!
//! Why a separate `tui_golden.rs` rather than appending to
//! `tui_smoke.rs` or `panels/main_hub.rs`: smoke tests deliberately
//! avoid pixel-exact assertions (so cosmetic edits don't break them);
//! the panel module currently has no `#[cfg(test)]` block at all.
//! Keeping the byte-exact baselines in their own file makes the W2
//! migration's diff target obvious.

use std::path::PathBuf;

use lamquant_core::tui::config::LamQuantConfig;
use lamquant_core::tui::panel::Panel;
use lamquant_core::tui::panels::main_hub::{HubTile, MainHubPanel};
use lamquant_core::tui::router;
use lamquant_core::tui::state::AppState;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

/// Mirrors the production main-hub tile inventory in
/// `src/tui/app.rs::register_panels` — keep in sync with that constructor.
/// If app.rs changes the tile order/text and this list is not updated,
/// the regen step will refresh the golden file and a human reviewer
/// must accept the diff.
fn make_main_hub_panel() -> MainHubPanel {
    MainHubPanel::new(
        router::SCREEN_MAIN,
        vec![
            HubTile::new(
                "1",
                "Codec Hub",
                "Compress · decompress · browse · verify",
                router::SCREEN_CODEC_HUB,
            ),
        ],
        vec![
            HubTile::new(
                "N",
                "Peers",
                "Remote LamQuant devices · SSH targets",
                router::SCREEN_PEERS,
            ),
            HubTile::new(
                "s",
                "Settings",
                "Workers · paths · device profiles",
                router::SCREEN_SETTINGS,
            ),
            HubTile::new(
                "i",
                "Install & setup",
                "Wizard · dependencies · syscheck · GPU probe",
                router::SCREEN_SETUP,
            ),
            HubTile::new(
                "t",
                "Diagnostics",
                "Internal Testing Suite · Crashlog Viewer · Health Check",
                router::SCREEN_TEST,
            ),
        ],
    )
}

/// AppState constructed for golden rendering. We *intentionally* do not
/// call `AppState::new()` because that reads cwd / on-disk config /
/// history — non-deterministic. Instead we start from `AppState::new()`
/// and then overwrite the fields that influence the main-hub render to
/// pinned defaults.
fn make_deterministic_state() -> AppState {
    let mut s = AppState::new();
    s.cfg = LamQuantConfig::default(); // ui_preference="ask" → tip line shows
    s
}

/// Serialise a buffer into a deterministic newline-joined string where
/// each cell becomes one symbol. Styling is dropped.
fn buffer_to_string(term: &Terminal<TestBackend>) -> String {
    let buf = term.backend().buffer();
    let w = buf.area.width as usize;
    let h = buf.area.height as usize;
    let mut out = String::with_capacity((w + 1) * h);
    for y in 0..h {
        for x in 0..w {
            let cell = &buf.content()[y * w + x];
            out.push_str(cell.symbol());
        }
        out.push('\n');
    }
    out
}

fn golden_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("golden");
    p.push(name);
    p
}

fn assert_or_regen(actual: &str, golden_name: &str) {
    let path = golden_path(golden_name);
    let regen = std::env::var("LAMQUANT_REGEN_GOLDENS")
        .ok()
        .as_deref()
        == Some("1")
        || !path.exists();
    if regen {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create golden dir");
        }
        std::fs::write(&path, actual.as_bytes()).expect("write golden");
        // When regenerating, still succeed so CI in regen mode is green.
        return;
    }
    let expected = std::fs::read_to_string(&path).expect("read golden");
    assert_eq!(
        actual, expected,
        "main_hub render diverged from golden at {}",
        path.display()
    );
}

#[test]
fn main_hub_golden_v0() {
    let panel = make_main_hub_panel();
    let state = make_deterministic_state();

    let backend = TestBackend::new(80, 24);
    let mut term = Terminal::new(backend).expect("terminal");
    term.draw(|f| {
        let area = f.area();
        panel.render(&state, f, area);
    })
    .expect("draw");

    let actual = buffer_to_string(&term);
    assert_or_regen(&actual, "main_hub_v0.txt");
}
