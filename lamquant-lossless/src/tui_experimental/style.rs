//! ANSI styling helpers — matches Python's lamquant_codec/cli/menu.py colors.

use std::io::{self, IsTerminal};

/// Whether stdout supports ANSI colors.
pub fn supports_color() -> bool {
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    if std::env::var("TERM").as_deref() == Ok("dumb") {
        return false;
    }
    io::stdout().is_terminal()
}

/// Whether stdout supports Unicode box-drawing characters.
pub fn supports_unicode() -> bool {
    // Conservative: assume yes on modern terminals
    supports_color()
}

// ANSI escape codes — conditional on TTY
pub struct Colors {
    pub dim: &'static str,
    pub cyn: &'static str,
    pub grn: &'static str,
    pub red: &'static str,
    pub yel: &'static str,
    pub bld: &'static str,
    pub rst: &'static str,
}

pub const COLORS_ON: Colors = Colors {
    dim: "\x1b[90m",
    cyn: "\x1b[36m",
    grn: "\x1b[32m",
    red: "\x1b[31m",
    yel: "\x1b[33m",
    bld: "\x1b[1m",
    rst: "\x1b[0m",
};

pub const COLORS_OFF: Colors = Colors {
    dim: "",
    cyn: "",
    grn: "",
    red: "",
    yel: "",
    bld: "",
    rst: "",
};

pub fn colors() -> &'static Colors {
    if supports_color() {
        &COLORS_ON
    } else {
        &COLORS_OFF
    }
}

/// Unicode/ASCII symbols
pub fn check() -> &'static str {
    if supports_unicode() {
        "✓"
    } else {
        "OK"
    }
}
pub fn cross() -> &'static str {
    if supports_unicode() {
        "✗"
    } else {
        "X"
    }
}
pub fn dot() -> &'static str {
    if supports_unicode() {
        "·"
    } else {
        "-"
    }
}
pub fn hrule() -> &'static str {
    if supports_unicode() {
        "─"
    } else {
        "-"
    }
}

/// Clear screen
pub fn clear() {
    if io::stdout().is_terminal() {
        print!("\x1b[2J\x1b[H");
    }
}

/// Clear screen + scrollback
pub fn clear_full() {
    if io::stdout().is_terminal() {
        print!("\x1b[3J\x1b[H\x1b[2J\x1b[H");
    }
}

/// Repeat a string N times
pub fn repeat(s: &str, n: usize) -> String {
    s.repeat(n)
}
