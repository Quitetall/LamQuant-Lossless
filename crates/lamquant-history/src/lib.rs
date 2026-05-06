//! Shared `history.json` reader/writer for all three LamQuant front-ends.
//!
//! Wire format is canonical at `specs/history-schema.json`. The Rust TUI,
//! Tauri GUI, and Python TUI all read and write the same file via this
//! crate (Rust callers) and `lamquant_codec.cli.menu` (Python callers).
//!
//! Every mutation goes through the [`History::save`] path which:
//!   1. Acquires an OS-level advisory lock on a sibling `*.lock` file so
//!      simultaneous writers from multiple front-ends cooperate.
//!   2. Re-reads the on-disk file under the lock and merges in our own
//!      additions (recent_paths union, recent_operations append) so two
//!      writers don't clobber each other's history.
//!   3. Writes a temp file, fsync, rename — the rename is atomic on every
//!      supported OS.
//!
//! On rename failure (cross-fs, antivirus lock on Windows) we explicitly
//! clean up the temp file so repeated saves don't litter the user's config
//! directory.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Per-OS resolution. Honoured precedence:
///   1. `LAMQUANT_HISTORY` env (test override).
///   2. `$XDG_CONFIG_HOME/lamquant/history.json` (Linux + explicit XDG).
///   3. `~/Library/Application Support/lamquant/history.json` (macOS).
///   4. `%APPDATA%\lamquant\history.json` (Windows).
///   5. `~/.config/lamquant/history.json` (Linux fallback).
pub fn history_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LAMQUANT_HISTORY") {
        return Some(PathBuf::from(p));
    }
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join("lamquant").join("history.json"));
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return Some(
                PathBuf::from(home)
                    .join("Library")
                    .join("Application Support")
                    .join("lamquant")
                    .join("history.json"),
            );
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            return Some(PathBuf::from(appdata).join("lamquant").join("history.json"));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return Some(
            PathBuf::from(home)
                .join(".config")
                .join("lamquant")
                .join("history.json"),
        );
    }
    None
}

/// One recorded operation in the rolling 50-entry log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryOp {
    /// Canonical op id from `specs/ui-parity.md::Op IDs`.
    pub action: String,
    /// Human-readable target (typically the input filename basename).
    pub target: String,
    /// ISO 8601 UTC timestamp.
    pub when: String,
    /// One of: ok, error, cancelled, partial.
    pub result: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RecentPaths {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// On-disk shape per `specs/history-schema.json`. The "interrupted" / "last_*"
/// fields are additive resume markers used by the Rust TUI's resume panel.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct History {
    #[serde(rename = "schema_version")]
    pub schema_version: String,
    pub recent_operations: Vec<HistoryOp>,
    pub recent_paths: RecentPaths,

    // Resume markers — the schema's additionalProperties: false is overridden
    // for these so a future schema bump can promote them.
    pub interrupted: bool,
    pub last_op: Option<String>,
    pub last_input: Option<String>,
    pub last_output: Option<String>,
}

impl History {
    pub fn load() -> Self {
        let Some(path) = history_path() else { return Self::default(); };
        Self::load_from(&path)
    }

    pub fn load_from(path: &Path) -> Self {
        let Ok(text) = fs::read_to_string(path) else { return Self::default(); };
        Self::parse(&text)
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = history_path() else { return Ok(()); };
        self.save_to(&path)
    }

    pub fn save_to(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let lock_path = path.with_extension("json.lock");
        let _lock = LockGuard::acquire(&lock_path);

        // Re-read under lock and merge so concurrent writers don't lose
        // each other's additions. Last-writer-wins for duplicates.
        let mut merged = self.clone();
        if let Ok(text) = fs::read_to_string(path) {
            let on_disk = Self::parse(&text);
            for s in on_disk.recent_paths.inputs.into_iter().rev() {
                merged.recent_paths.inputs.retain(|x| x != &s);
                merged.recent_paths.inputs.insert(0, s);
            }
            for s in on_disk.recent_paths.outputs.into_iter().rev() {
                merged.recent_paths.outputs.retain(|x| x != &s);
                merged.recent_paths.outputs.insert(0, s);
            }
            // recent_operations: ours come first (more recent), then any
            // on-disk entries we don't already have. Dedupe by the
            // (action, target, when) triple — `when` is generated by
            // record_op so a true concurrent insert from another front-end
            // gets a unique timestamp; our own previously-saved ops match
            // exactly and get skipped here.
            for op in on_disk.recent_operations {
                let dup = merged.recent_operations.iter().any(|x| {
                    x.action == op.action && x.target == op.target && x.when == op.when
                });
                if !dup {
                    merged.recent_operations.push(op);
                }
            }
        }
        merged.recent_paths.inputs.truncate(20);
        merged.recent_paths.outputs.truncate(20);
        merged.recent_operations.truncate(50);
        if merged.schema_version.is_empty() {
            merged.schema_version = "1.0".into();
        }

        let body = serde_json::to_string_pretty(&merged)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let tmp = path.with_extension("json.tmp");
        {
            let mut f = File::create(&tmp)?;
            f.write_all(body.as_bytes())?;
            let _ = f.sync_all();
        }
        match fs::rename(&tmp, path) {
            Ok(()) => Ok(()),
            Err(rename_err) => {
                let _ = fs::remove_file(&tmp);
                Err(rename_err)
            }
        }
    }

    /// Mutators (chainable) ─ small enough to inline at the call site too.
    pub fn add_input(&mut self, p: &str) {
        self.recent_paths.inputs.retain(|x| x != p);
        self.recent_paths.inputs.insert(0, p.to_string());
        self.recent_paths.inputs.truncate(20);
    }

    pub fn add_output(&mut self, p: &str) {
        self.recent_paths.outputs.retain(|x| x != p);
        self.recent_paths.outputs.insert(0, p.to_string());
        self.recent_paths.outputs.truncate(20);
    }

    pub fn record_op(&mut self, action: &str, target: &str, result: &str) {
        let when = utc_now_iso();
        self.recent_operations.insert(
            0,
            HistoryOp {
                action: action.to_string(),
                target: target.to_string(),
                when,
                result: result.to_string(),
            },
        );
        self.recent_operations.truncate(50);
    }

    pub fn mark_running(&mut self, op: &str, input: &str, output: Option<&str>) {
        self.last_op = Some(op.to_string());
        self.last_input = Some(input.to_string());
        self.last_output = output.map(|s| s.to_string());
        self.interrupted = true;
    }

    pub fn mark_complete(&mut self, result: &str) {
        if let (Some(op), Some(input)) = (self.last_op.clone(), self.last_input.clone()) {
            self.record_op(&op, &input, result);
        }
        self.interrupted = false;
    }

    fn parse(text: &str) -> Self {
        // Spec format first.
        if let Ok(h) = serde_json::from_str::<History>(text) {
            // If schema_version is empty AND we got nothing, it might be the
            // legacy flat format — fall back below.
            if !h.recent_paths.inputs.is_empty()
                || !h.recent_paths.outputs.is_empty()
                || !h.recent_operations.is_empty()
                || h.schema_version != ""
            {
                return h;
            }
        }
        // Legacy flat shape: pre-Phase-2 had `recent_inputs` / `recent_outputs`
        // at the top level. Migrate quietly so users don't lose history.
        #[derive(Deserialize)]
        #[serde(default)]
        struct LegacyHistory {
            recent_inputs: Vec<String>,
            recent_outputs: Vec<String>,
            last_op: Option<String>,
            last_input: Option<String>,
            last_output: Option<String>,
            interrupted: bool,
        }
        impl Default for LegacyHistory {
            fn default() -> Self {
                Self {
                    recent_inputs: Vec::new(),
                    recent_outputs: Vec::new(),
                    last_op: None,
                    last_input: None,
                    last_output: None,
                    interrupted: false,
                }
            }
        }
        if let Ok(l) = serde_json::from_str::<LegacyHistory>(text) {
            return History {
                schema_version: "1.0".into(),
                recent_operations: Vec::new(),
                recent_paths: RecentPaths {
                    inputs: l.recent_inputs,
                    outputs: l.recent_outputs,
                },
                interrupted: l.interrupted,
                last_op: l.last_op,
                last_input: l.last_input,
                last_output: l.last_output,
            };
        }
        Self::default()
    }
}

// ── ISO 8601 timestamp without chrono ─────────────────────────────────

fn utc_now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs / 86_400;
    let rem = secs.rem_euclid(86_400);
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let s = rem % 60;
    let (y, mo, d) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, m, s)
}

fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z / 146097 } else { (z - 146096) / 146097 };
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y_out = y + i64::from(m <= 2);
    (y_out as i32, m as u32, d as u32)
}

// ── Locking ────────────────────────────────────────────────────────────

struct LockGuard {
    _file: Option<File>,
    #[cfg(unix)]
    fd: Option<i32>,
}

impl LockGuard {
    fn acquire(path: &Path) -> Self {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        match OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)
        {
            Ok(file) => {
                #[cfg(unix)]
                {
                    use std::os::unix::io::AsRawFd;
                    let fd = file.as_raw_fd();
                    unsafe {
                        let _ = libc_flock(fd, 2); // LOCK_EX
                    }
                    return LockGuard { _file: Some(file), fd: Some(fd) };
                }
                #[cfg(windows)]
                {
                    return LockGuard { _file: Some(file) };
                }
                #[cfg(not(any(unix, windows)))]
                {
                    return LockGuard { _file: Some(file) };
                }
            }
            Err(_) => LockGuard {
                _file: None,
                #[cfg(unix)]
                fd: None,
            },
        }
    }
}

#[cfg(unix)]
unsafe fn libc_flock(fd: i32, op: i32) -> i32 {
    extern "C" {
        fn flock(fd: i32, op: i32) -> i32;
    }
    unsafe { flock(fd, op) }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            if let Some(fd) = self.fd {
                unsafe {
                    let _ = libc_flock(fd, 8); // LOCK_UN
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
    }

    fn with_history_path<F: FnOnce(&Path)>(f: F) {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.json");
        let prev = std::env::var_os("LAMQUANT_HISTORY");
        unsafe { std::env::set_var("LAMQUANT_HISTORY", &path); }
        f(&path);
        match prev {
            Some(v) => unsafe { std::env::set_var("LAMQUANT_HISTORY", v) },
            None => unsafe { std::env::remove_var("LAMQUANT_HISTORY") },
        }
    }

    #[test]
    fn round_trip_recent_paths_preserves_order() {
        with_history_path(|_| {
            let mut h = History::default();
            h.add_input("/data/a.edf");
            h.add_input("/data/b.edf");
            h.save().expect("save");
            let loaded = History::load();
            assert_eq!(loaded.recent_paths.inputs, vec!["/data/b.edf", "/data/a.edf"]);
        });
    }

    #[test]
    fn record_op_appends_newest_first() {
        with_history_path(|_| {
            let mut h = History::default();
            h.record_op("encode", "a.edf", "ok");
            h.record_op("decode", "a.lml", "error");
            h.save().expect("save");
            let loaded = History::load();
            assert_eq!(loaded.recent_operations.len(), 2);
            assert_eq!(loaded.recent_operations[0].action, "decode");
        });
    }

    #[test]
    fn save_caps_at_documented_limits() {
        with_history_path(|_| {
            let mut h = History::default();
            for i in 0..30 {
                h.add_input(&format!("/data/{}.edf", i));
            }
            for i in 0..60 {
                h.record_op("encode", &format!("{}.edf", i), "ok");
            }
            h.save().expect("save");
            let loaded = History::load();
            assert_eq!(loaded.recent_paths.inputs.len(), 20);
            assert_eq!(loaded.recent_operations.len(), 50);
        });
    }

    #[test]
    fn load_handles_legacy_flat_shape() {
        with_history_path(|path| {
            std::fs::write(
                path,
                "{\"recent_inputs\":[\"x.edf\"],\"recent_outputs\":[\"y.lml\"],\"interrupted\":false}\n",
            )
            .unwrap();
            let loaded = History::load();
            assert_eq!(loaded.recent_paths.inputs, vec!["x.edf"]);
            assert_eq!(loaded.recent_paths.outputs, vec!["y.lml"]);
        });
    }

    #[test]
    fn save_then_concurrent_save_does_not_lose_data() {
        with_history_path(|path| {
            let mut a = History::default();
            a.add_input("/data/a.edf");
            a.record_op("encode", "a.edf", "ok");
            a.save_to(path).expect("save a");

            let mut b = History::load_from(path);
            b.add_input("/data/b.edf");
            b.record_op("decode", "a.lml", "ok");
            b.save_to(path).expect("save b");

            let loaded = History::load_from(path);
            assert!(loaded.recent_paths.inputs.contains(&"/data/a.edf".into()));
            assert!(loaded.recent_paths.inputs.contains(&"/data/b.edf".into()));
            assert_eq!(loaded.recent_operations.len(), 2);
        });
    }
}
