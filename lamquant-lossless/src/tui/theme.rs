//! Theme — LamQuant brand colors and reusable styles.
//!
//! Honours `NO_COLOR`, `TERM=dumb`, and `cfg.output.color = "never"`.
//! When color is disabled, every style getter returns `Style::default()`
//! so the renderer emits no ANSI escapes.

use ratatui::style::{Color, Modifier, Style};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

pub const CYAN: Color = Color::Cyan;
pub const GREEN: Color = Color::Green;
pub const RED: Color = Color::Red;
pub const YELLOW: Color = Color::Yellow;
pub const DIM: Color = Color::DarkGray;
pub const WHITE: Color = Color::White;

// ── Runtime detection ────────────────────────────────────────────────────
// Detected once at App::new() (or first style call). Atomics so panels can
// read without a Mutex.

static COLOR_INITIALIZED: AtomicBool = AtomicBool::new(false);
/// 0 = enabled, 1 = disabled.
static COLOR_DISABLED: AtomicBool = AtomicBool::new(false);
/// 0 = unicode, 1 = ascii.
static CHARSET: AtomicU8 = AtomicU8::new(0);

/// Re-detect from environment (and optionally an explicit cfg pref).
/// Should be called once at startup. cfg_color: "auto" | "always" | "never".
/// cfg_charset: "auto" | "unicode" | "ascii".
pub fn detect(cfg_color: &str, cfg_charset: &str) {
    let no_color = std::env::var_os("NO_COLOR").is_some();
    let term_dumb = std::env::var("TERM").map(|t| t == "dumb").unwrap_or(false);
    let color_off = match cfg_color {
        "never" => true,
        "always" => false,
        _ => no_color || term_dumb,
    };
    COLOR_DISABLED.store(color_off, Ordering::Relaxed);

    let lang = std::env::var("LANG").unwrap_or_default();
    let lc_ctype = std::env::var("LC_CTYPE").unwrap_or_default();
    let utf8_locale = lang.to_uppercase().contains("UTF-8")
        || lang.to_uppercase().contains("UTF8")
        || lc_ctype.to_uppercase().contains("UTF-8");
    let charset_ascii = match cfg_charset {
        "ascii" => true,
        "unicode" => false,
        _ => term_dumb || !utf8_locale,
    };
    CHARSET.store(if charset_ascii { 1 } else { 0 }, Ordering::Relaxed);

    COLOR_INITIALIZED.store(true, Ordering::Relaxed);
}

#[inline]
fn color_enabled() -> bool {
    !COLOR_DISABLED.load(Ordering::Relaxed)
}

/// True when the terminal cannot reliably render Unicode (e.g. TERM=dumb or non-UTF-8 locale).
pub fn ascii_only() -> bool {
    CHARSET.load(Ordering::Relaxed) == 1
}

#[inline]
fn maybe(s: Style) -> Style {
    if color_enabled() {
        s
    } else {
        Style::default()
    }
}

pub fn title() -> Style {
    maybe(Style::default().fg(CYAN).add_modifier(Modifier::BOLD))
}
pub fn heading() -> Style {
    maybe(Style::default().fg(WHITE).add_modifier(Modifier::BOLD))
}
pub fn normal() -> Style {
    maybe(Style::default().fg(WHITE))
}
pub fn dim() -> Style {
    maybe(Style::default().fg(DIM))
}
pub fn highlight() -> Style {
    maybe(Style::default().fg(CYAN).add_modifier(Modifier::BOLD))
}
pub fn selected() -> Style {
    // In monochrome mode, selection is signalled with REVERSED.
    if color_enabled() {
        Style::default()
            .bg(Color::DarkGray)
            .fg(WHITE)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().add_modifier(Modifier::REVERSED)
    }
}
pub fn success() -> Style {
    maybe(Style::default().fg(GREEN))
}
pub fn error() -> Style {
    maybe(Style::default().fg(RED))
}
pub fn warning() -> Style {
    maybe(Style::default().fg(YELLOW))
}
pub fn key_hint() -> Style {
    maybe(Style::default().fg(CYAN))
}
pub fn key_label() -> Style {
    maybe(Style::default().fg(DIM))
}
pub fn status_bar() -> Style {
    maybe(Style::default().bg(Color::DarkGray).fg(WHITE))
}
pub fn status_msg() -> Style {
    maybe(Style::default().bg(Color::DarkGray).fg(GREEN))
}

/// Repeat the unicode horizontal-rule glyph `width` times, clamped to
/// MAX_DASH_WIDTH so an adversarial `COLUMNS=999999` env can't blow
/// up render-loop memory. Real terminals top out around 10k cols; the
/// 4096 cap is way past any realistic display.
pub fn dash(width: usize) -> String {
    const MAX_DASH_WIDTH: usize = 4096;
    let g = if ascii_only() { "-" } else { "─" };
    // Precondition: width is finite by type (usize). No assertion
    // needed on input — clamping is the contract.
    let clamped = width.min(MAX_DASH_WIDTH);
    let out = g.repeat(clamped);
    // Postcondition: output character count never exceeds the cap.
    // chars() walks the grapheme stream rather than bytes so the
    // assertion is portable across ascii / unicode rules.
    debug_assert!(
        out.chars().count() == clamped,
        "dash glyph count must equal clamped width"
    );
    debug_assert!(
        out.chars().count() <= MAX_DASH_WIDTH,
        "dash width exceeded cap"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    /// Process-wide lock for theme tests. cargo test runs cases in
    /// parallel by default; without this lock two cases that mutate the
    /// same env vars + global atomics race and observe each other's
    /// state. The mutex pairs them up so they execute sequentially while
    /// the rest of the suite still parallelises around them.
    fn env_lock() -> &'static Mutex<()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn detect_no_color_disables() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("LANG");
        std::env::set_var("NO_COLOR", "1");
        detect("auto", "auto");
        assert!(!color_enabled());
        std::env::remove_var("NO_COLOR");
    }

    #[test]
    fn explicit_always_overrides_no_color() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("NO_COLOR", "1");
        detect("always", "auto");
        assert!(color_enabled());
        std::env::remove_var("NO_COLOR");
    }

    #[test]
    fn explicit_never_disables_even_without_no_color() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("NO_COLOR");
        detect("never", "auto");
        assert!(!color_enabled());
    }

    #[test]
    fn term_dumb_forces_ascii() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("TERM", "dumb");
        detect("auto", "auto");
        assert!(ascii_only());
        std::env::remove_var("TERM");
    }

    #[test]
    fn explicit_ascii_charset() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        detect("auto", "ascii");
        assert!(ascii_only());
    }

    #[test]
    fn explicit_unicode_charset_even_on_dumb_term() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("TERM", "dumb");
        detect("auto", "unicode");
        assert!(!ascii_only());
        std::env::remove_var("TERM");
    }
}
