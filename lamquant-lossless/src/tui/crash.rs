//! Crash report capture — `std::panic::set_hook` that restores the TTY
//! before writing a structured report to `~/.lamquant_crashes/<ts>.json`,
//! so a panic mid-render leaves a usable terminal and a forensic trail.

use std::fs;
use std::io;
use std::panic::{self, PanicHookInfo};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, LeaveAlternateScreen};

/// Install a custom panic hook. Idempotent — second call replaces the first.
/// Wraps the default hook so panic messages still print to stderr.
pub fn install_panic_hook() {
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info: &PanicHookInfo<'_>| {
        // 1. Restore terminal so the user can read the panic message.
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);

        // 2. Write structured crash report.
        let _ = write_report(info);

        // 3. Defer to default hook for stderr backtrace.
        default_hook(info);
    }));
}

fn crash_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".lamquant_crashes"))
}

fn write_report(info: &PanicHookInfo<'_>) -> io::Result<()> {
    let dir = crash_dir().ok_or_else(|| io::Error::other("no $HOME"))?;
    fs::create_dir_all(&dir)?;

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("crash_{}.json", ts));

    let location = info
        .location()
        .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
        .unwrap_or_else(|| "<unknown>".to_string());

    let payload = info.payload();
    let message = if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    };

    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let pkg_version = env!("CARGO_PKG_VERSION");

    // Hand-rolled JSON to avoid pulling serde at this layer.
    let json = format!(
        "{{\n  \"timestamp_unix\": {},\n  \"location\": \"{}\",\n  \"message\": \"{}\",\n  \"cwd\": \"{}\",\n  \"exe\": \"{}\",\n  \"version\": \"{}\"\n}}\n",
        ts,
        escape(&location),
        escape(&message),
        escape(&cwd),
        escape(&exe),
        escape(pkg_version),
    );
    fs::write(&path, json)?;

    eprintln!();
    eprintln!("--- LamQuant crashed ---");
    eprintln!("Crash report written to: {}", path.display());
    Ok(())
}

/// JSON string escape — covers backslash, quote, common whitespace,
/// AND all C0 control characters (0x00-0x1F) as `\uXXXX`. A raw
/// 0x01 in a panic message previously produced invalid JSON and
/// broke any forensic parser of the crash log.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0C' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_handles_special_chars() {
        assert_eq!(escape("hello"), "hello");
        assert_eq!(escape(r#"path\with"quotes"#), r#"path\\with\"quotes"#);
        assert_eq!(escape("multi\nline"), "multi\\nline");
    }

    #[test]
    fn crash_dir_is_under_home() {
        if let Ok(home) = std::env::var("HOME") {
            let dir = crash_dir().expect("HOME set in test env");
            assert!(dir.starts_with(home));
            assert_eq!(dir.file_name().unwrap(), ".lamquant_crashes");
        }
    }
}
