//! Clipboard helper — best-effort copy to system clipboard.
//!
//! Tries, in order: wl-copy (Wayland), xclip, xsel, pbcopy (macOS).
//! Returns the name of the backend used, or an error message if all failed.
//!
//! No external deps — purely subprocess-based with stdin pipe.
//!
//! The actual subprocess work runs in a worker thread with a short timeout.
//! Without that hedge, a stale X server with a hung xclip backend would
//! freeze the entire TUI: the panel handler calls this synchronously, and
//! `child.wait()` will block forever waiting on a process that never reads
//! from its stdin. Bounding the foreground wait keeps the UI responsive
//! even when the system clipboard is wedged.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// How long the foreground call will wait for the worker thread before
/// giving up. The actual clipboard write may still succeed in the background;
/// the user just won't see a confirmation if the backend is exceptionally slow.
const COPY_TIMEOUT: Duration = Duration::from_millis(750);

pub fn copy_to_clipboard(text: &str) -> Result<&'static str, String> {
    let owned = text.to_string();
    let (tx, rx) = mpsc::channel();
    // Detach the actual blocking work to a worker thread so an unresponsive
    // clipboard backend (e.g. xclip against a stale X server) cannot freeze
    // the TUI render loop. We don't join the handle — if it outlives the
    // timeout the OS will reap it when its work completes.
    thread::spawn(move || {
        let result = copy_blocking(&owned);
        let _ = tx.send(result);
    });
    match rx.recv_timeout(COPY_TIMEOUT) {
        Ok(r) => r,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            Err("clipboard backend slow (stale X server?) — copy may complete in background".into())
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => Err("clipboard worker thread died".into()),
    }
}

fn copy_blocking(text: &str) -> Result<&'static str, String> {
    let attempts: &[(&str, &[&str])] = &[
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("xsel", &["--clipboard", "--input"]),
        ("pbcopy", &[]),
    ];
    let mut errors = Vec::new();
    for (prog, args) in attempts {
        match try_pipe(prog, args, text) {
            Ok(()) => return Ok(prog),
            Err(e) => errors.push(format!("{}: {}", prog, e)),
        }
    }
    Err(format!(
        "no clipboard backend available ({})",
        errors.join("; ")
    ))
}

fn try_pipe(prog: &str, args: &[&str], text: &str) -> Result<(), String> {
    let mut child = Command::new(prog)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;
    {
        let stdin = child.stdin.as_mut().ok_or("no stdin")?;
        stdin
            .write_all(text.as_bytes())
            .map_err(|e| e.to_string())?;
    } // drop stdin so child sees EOF
    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("exit {}", status.code().unwrap_or(-1)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_missing_backends_gracefully() {
        // We can't easily mock spawn() success, but copying empty str should
        // either succeed (if a backend is installed) or return a clean error.
        let r = copy_to_clipboard("test");
        match r {
            Ok(_backend) => {} // some backend worked
            Err(msg) => assert!(msg.contains("no clipboard backend")),
        }
    }
}
