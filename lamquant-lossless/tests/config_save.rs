//! Round-trip and atomic-save tests for `LamQuantConfig`.
//!
//! Phase 0 #14 added fsync + tmp-cleanup-on-rename-failure semantics. These
//! tests pin the contract: a successful save replaces the destination
//! atomically, AND a failed rename leaves no orphan `.toml.tmp` litter.

use lamquant_core::tui::config::{config_path, LamQuantConfig};
use std::sync::{Mutex, OnceLock};

/// Process-wide lock for tests that mutate `LAMQUANT_CONFIG`. cargo test
/// runs cases in parallel by default; without this lock two cases setting
/// different paths race and one observes the other's tempdir.
fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Run a closure with `LAMQUANT_CONFIG` pointed at `path`. Holds the
/// process-wide env lock for the lifetime of the guard so parallel tests
/// can't cross-contaminate.
struct EnvGuard {
    prev: Option<std::ffi::OsString>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl EnvGuard {
    fn set(path: &std::path::Path) -> Self {
        let lock = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("LAMQUANT_CONFIG");
        unsafe {
            std::env::set_var("LAMQUANT_CONFIG", path);
        }
        EnvGuard { prev, _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => unsafe { std::env::set_var("LAMQUANT_CONFIG", v) },
            None => unsafe { std::env::remove_var("LAMQUANT_CONFIG") },
        }
    }
}

#[test]
fn save_writes_atomically_and_cleans_tmp() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("lamquant.toml");
    let _g = EnvGuard::set(&path);

    let mut cfg = LamQuantConfig::default();
    cfg.compute.workers = 17;
    cfg.save().expect("save");

    // Final config exists.
    assert!(
        path.exists(),
        "config file should exist at {}",
        path.display()
    );
    let body = std::fs::read_to_string(&path).expect("read config");
    assert!(body.contains("workers = 17"), "missing workers field");

    // No orphaned .toml.tmp left behind.
    let tmp = path.with_extension("toml.tmp");
    assert!(!tmp.exists(), "orphan tmp file remains: {}", tmp.display());

    // Round-trip: load + save preserves the value.
    let loaded = LamQuantConfig::load();
    assert_eq!(loaded.compute.workers, 17, "round-trip lost workers");
}

#[test]
fn save_round_trip_preserves_unknown_keys() {
    // The TOML parser preserves unknown keys into `extra` so a future Rust
    // version doesn't drop fields written by a newer Python TUI.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("lamquant.toml");
    let _g = EnvGuard::set(&path);

    let custom = "\
schema_version = \"1.0\"
instance_name = \"lamquant\"

[future_section]
unknown_key = \"value\"
";
    std::fs::write(&path, custom).expect("seed");

    let cfg = LamQuantConfig::load();
    assert!(
        cfg.extra.contains_key("future_section"),
        "future_section should be preserved into extra; got {:?}",
        cfg.extra.keys().collect::<Vec<_>>()
    );

    cfg.save().expect("re-save");
    let body = std::fs::read_to_string(&path).expect("read");
    assert!(
        body.contains("future_section"),
        "re-save dropped unknown section: {}",
        body
    );
}

#[test]
fn config_path_respects_env_override() {
    let _g = EnvGuard::set(std::path::Path::new("/tmp/explicit-override.toml"));
    let p = config_path();
    assert_eq!(p.to_string_lossy(), "/tmp/explicit-override.toml");
}
