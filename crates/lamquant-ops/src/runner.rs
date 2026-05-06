//! Subprocess runner — spawns either the `lml` self-binary or an arbitrary
//! external command, streaming stdout/stderr line-by-line as `OpEvent`s
//! through a sink.
//!
//! The runner is generic over `OpEventSink`. In-process consumers (Rust TUI)
//! provide `MpscSink`; cross-process consumers (Tauri GUI) provide their own
//! sink that updates a shared snapshot AND emits state-transition events.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::sink::OpEventSink;
use crate::OpEvent;

/// Returned to the caller when an op is launched. Holds a kill channel.
#[derive(Debug)]
pub struct OpHandle {
    kill_tx: mpsc::Sender<()>,
    /// Tracks whether kill() has been called so callers can avoid double-send.
    killed: Arc<Mutex<bool>>,
}

impl OpHandle {
    /// Request graceful cancellation. Idempotent — second call is a no-op.
    pub fn kill(&mut self) {
        let mut guard = self.killed.lock().expect("kill flag poisoned");
        if !*guard {
            *guard = true;
            // Best-effort send; supervisor may have already exited.
            let _ = self.kill_tx.send(());
        }
    }

    /// Whether kill has been requested.
    pub fn was_killed(&self) -> bool {
        *self.killed.lock().expect("kill flag poisoned")
    }
}

/// Spawn a subprocess of the current `lml` binary (resolved via current_exe).
pub fn spawn_lml<S: OpEventSink>(args: Vec<String>, sink: S) -> OpHandle {
    let op_id = args.first().cloned().unwrap_or_default();
    let exe_label = "lml".to_string();
    spawn_internal(SpawnTarget::Lml, op_id, exe_label, args, sink)
}

/// Spawn an arbitrary external command (training, eagle, pytest, etc.).
pub fn spawn_command<S: OpEventSink>(program: String, args: Vec<String>, sink: S) -> OpHandle {
    let op_id = program.clone();
    let label = program.clone();
    spawn_internal(SpawnTarget::Cmd(program), op_id, label, args, sink)
}

enum SpawnTarget {
    Lml,
    Cmd(String),
}

fn spawn_internal<S: OpEventSink>(
    target: SpawnTarget,
    op_id: String,
    exe_label: String,
    args: Vec<String>,
    sink: S,
) -> OpHandle {
    let (kill_tx, kill_rx) = mpsc::channel::<()>();
    let killed = Arc::new(Mutex::new(false));
    let killed_supervisor = Arc::clone(&killed);

    // Sink lives on the supervisor thread; clones go to stdout/stderr readers.
    let sink = Arc::new(sink);

    thread::spawn(move || {
        sink.emit(OpEvent::Started {
            ts_ms: OpEvent::now_ms(),
            op_id: op_id.clone(),
            total: None,
        });
        sink.emit(OpEvent::Log {
            ts_ms: OpEvent::now_ms(),
            message: format!("$ {} {}", exe_label, args.join(" ")),
        });

        let spawn_result: Result<Child, String> = match target {
            SpawnTarget::Lml => match std::env::current_exe() {
                Ok(exe) => Command::new(&exe)
                    .args(&args)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("spawn lml: {}", e)),
                Err(e) => Err(format!("current_exe: {}", e)),
            },
            SpawnTarget::Cmd(prog) => Command::new(&prog)
                .args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("spawn {}: {}", prog, e)),
        };

        let mut child = match spawn_result {
            Ok(c) => c,
            Err(e) => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: e,
                });
                return;
            }
        };

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let sink_out = Arc::clone(&sink);
        let sink_err = Arc::clone(&sink);

        let h_out = stdout.map(|s| {
            thread::spawn(move || {
                for line in BufReader::new(s).lines().map_while(|r| r.ok()) {
                    // If the line is itself a JSON-encoded OpEvent (because
                    // the child was invoked with --emit-json-events), forward
                    // it as the structured event. Otherwise emit as Log.
                    let ev = OpEvent::from_json_line(&line).unwrap_or_else(|_| OpEvent::Log {
                        ts_ms: OpEvent::now_ms(),
                        message: line,
                    });
                    sink_out.emit(ev);
                }
            })
        });
        let h_err = stderr.map(|s| {
            thread::spawn(move || {
                for line in BufReader::new(s).lines().map_while(|r| r.ok()) {
                    sink_err.emit(OpEvent::Log {
                        ts_ms: OpEvent::now_ms(),
                        message: format!("[stderr] {}", line),
                    });
                }
            })
        });

        // Supervisor poll loop: watch for kill signal OR child exit.
        let mut was_killed = false;
        let exit_status = loop {
            match kill_rx.try_recv() {
                Ok(_) => {
                    was_killed = true;
                    sink.emit(OpEvent::Log {
                        ts_ms: OpEvent::now_ms(),
                        message: format!(">> Cancelling (PID {})...", child.id()),
                    });
                    let _ = child.kill();
                    break child.wait();
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    // Caller dropped the handle — they no longer care about
                    // cancel. Just wait for the child to finish.
                }
            }

            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => thread::sleep(Duration::from_millis(50)),
                Err(e) => {
                    sink.emit(OpEvent::Error {
                        ts_ms: OpEvent::now_ms(),
                        message: format!("wait failed: {}", e),
                    });
                    return;
                }
            }
        };

        if was_killed {
            *killed_supervisor.lock().expect("kill flag poisoned") = true;
        }

        if let Some(h) = h_out {
            let _ = h.join();
        }
        if let Some(h) = h_err {
            let _ = h.join();
        }

        match exit_status {
            Ok(_) if was_killed => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("{} cancelled by user", op_id),
                });
            }
            Ok(s) if s.success() => {
                sink.emit(OpEvent::Done {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("{} completed (exit 0)", op_id),
                });
            }
            Ok(s) => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("{} exited with code {}", op_id, s.code().unwrap_or(-1)),
                });
            }
            Err(e) => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("wait failed: {}", e),
                });
            }
        }
    });

    OpHandle { kill_tx, killed }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::channel;
    use std::time::Instant;

    /// Cross-platform "sleep 30" fixture. Unix uses `sleep` (coreutils);
    /// Windows uses `powershell -Command Start-Sleep`.
    fn long_running_cmd() -> (String, Vec<String>) {
        #[cfg(windows)]
        {
            (
                "powershell".to_string(),
                vec!["-NoProfile".to_string(), "-Command".to_string(), "Start-Sleep -Seconds 30".to_string()],
            )
        }
        #[cfg(not(windows))]
        {
            ("sleep".to_string(), vec!["30".to_string()])
        }
    }

    fn fast_exit_cmd() -> (String, Vec<String>) {
        #[cfg(windows)]
        {
            ("cmd".to_string(), vec!["/c".to_string(), "exit".to_string(), "0".to_string()])
        }
        #[cfg(not(windows))]
        {
            ("true".to_string(), vec![])
        }
    }

    #[test]
    fn cancels_long_running_command() {
        let (sink, rx) = channel();
        let (prog, args) = long_running_cmd();
        let mut handle = spawn_command(prog, args, sink);

        thread::sleep(Duration::from_millis(200));
        let start = Instant::now();
        handle.kill();

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut saw_cancel = false;
        while Instant::now() < deadline {
            if let Ok(ev) = rx.recv_timeout(Duration::from_millis(100)) {
                if let OpEvent::Error { message, .. } = &ev {
                    if message.contains("cancelled") {
                        saw_cancel = true;
                        break;
                    }
                }
            }
        }
        let elapsed = start.elapsed();
        assert!(saw_cancel, "expected cancel event within 2s");
        assert!(elapsed < Duration::from_secs(2), "kill took too long: {:?}", elapsed);
        assert!(handle.was_killed());
    }

    #[test]
    fn fast_command_completes_normally() {
        let (sink, rx) = channel();
        let (prog, args) = fast_exit_cmd();
        let _handle = spawn_command(prog, args, sink);
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut saw_done = false;
        while Instant::now() < deadline {
            if let Ok(ev) = rx.recv_timeout(Duration::from_millis(100)) {
                if matches!(ev, OpEvent::Done { .. }) {
                    saw_done = true;
                    break;
                }
            }
        }
        assert!(saw_done, "expected Done event from fast-exit fixture");
    }
}
