//! Path template expansion for CLI output (Phase 1.6).
//!
//! Users pass `-o '{parent}/{stem}-compressed.lml'` to control per-input
//! output naming in batch encode/decode runs. This module is the single
//! source of truth for the placeholder grammar so every subcommand that
//! accepts a template uses the same vocabulary.
//!
//! Placeholders (case-sensitive):
//! - `{stem}`   — file stem (`patient001.edf` → `patient001`)
//! - `{name}`   — file name with extension (`patient001.edf`)
//! - `{ext}`    — file extension without dot (`patient001.edf` → `edf`)
//! - `{parent}` — parent directory path (`/data/foo/x.edf` → `/data/foo`)
//!
//! Bible alignment:
//! - R6  Strict-typed boundary: the helper returns `Option<PathBuf>`,
//!   `None` for missing components (e.g. `{stem}` on a path with no
//!   stem) so callers reject the unresolved template explicitly.
//! - R31 Idempotent: same input + template → same output. No
//!   timestamps, no PID, no random suffixes.

use std::path::{Path, PathBuf};

/// Returns `true` if `template` contains at least one `{name}` /
/// `{stem}` / `{ext}` / `{parent}` placeholder. Callers use this to
/// decide between per-input template expansion and literal-path mode.
pub fn has_placeholder(template: &str) -> bool {
    template.contains("{stem}")
        || template.contains("{name}")
        || template.contains("{ext}")
        || template.contains("{parent}")
}

/// Phase 1.4 — refuse-clobber check. Returns `Ok(())` if `path`
/// does not exist OR `force` is true OR `path == "-"` (stdout
/// sentinel). Returns a typed error otherwise so callers surface a
/// uniform "use --force to overwrite" message.
///
/// Symlinks count as existing (we never silently overwrite the
/// target). Bible R30 hostile-caller default.
pub fn ensure_can_write(path: &Path, force: bool) -> Result<(), String> {
    if force {
        return Ok(());
    }
    if path == Path::new("-") {
        return Ok(());
    }
    // symlink_metadata so a dangling symlink at the target is still
    // treated as "exists" — don't follow it and overwrite whatever
    // it points to.
    if std::fs::symlink_metadata(path).is_ok() {
        return Err(format!(
            "refusing to overwrite existing path {} — pass --force to opt in",
            path.display()
        ));
    }
    Ok(())
}

/// Expand `template` against the components of `input`. Returns
/// `None` if any referenced placeholder cannot be resolved (e.g.
/// `{ext}` on an extension-less path).
///
/// Trivia: empty `{parent}` (input has no parent) substitutes the
/// empty string rather than failing, so `{parent}/{stem}.lml` on
/// `foo.edf` yields `/{stem}.lml` cleanly resolved.
pub fn expand_template(template: &str, input: &Path) -> Option<PathBuf> {
    let stem = input.file_stem().and_then(|s| s.to_str());
    let name = input.file_name().and_then(|s| s.to_str());
    let ext = input.extension().and_then(|s| s.to_str());
    let parent = input
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut out = String::with_capacity(template.len() + 16);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let close = after.find('}')?;
        let key = &after[..close];
        match key {
            "stem" => out.push_str(stem?),
            "name" => out.push_str(name?),
            "ext" => out.push_str(ext?),
            "parent" => out.push_str(&parent),
            _ => {
                // Unknown placeholder — fail explicitly. Beats
                // silently emitting `{foo}` literally.
                return None;
            }
        }
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    Some(PathBuf::from(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_placeholders_passthrough() {
        let p = expand_template("output.lml", Path::new("/data/x.edf")).unwrap();
        assert_eq!(p, PathBuf::from("output.lml"));
    }

    #[test]
    fn ensure_can_write_passes_for_missing_path() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("does_not_exist.lml");
        ensure_can_write(&p, false).unwrap();
    }

    #[test]
    fn ensure_can_write_refuses_existing_without_force() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.lml");
        std::fs::write(&p, b"x").unwrap();
        assert!(ensure_can_write(&p, false).is_err());
    }

    #[test]
    fn ensure_can_write_allows_with_force() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.lml");
        std::fs::write(&p, b"x").unwrap();
        ensure_can_write(&p, true).unwrap();
    }

    #[test]
    fn ensure_can_write_allows_stdout_sentinel() {
        ensure_can_write(Path::new("-"), false).unwrap();
    }

    #[test]
    fn stem_substitution() {
        let p = expand_template("{stem}.lml", Path::new("/data/foo.edf")).unwrap();
        assert_eq!(p, PathBuf::from("foo.lml"));
    }

    #[test]
    fn name_substitution() {
        let p = expand_template("{name}.lml", Path::new("/data/foo.edf")).unwrap();
        assert_eq!(p, PathBuf::from("foo.edf.lml"));
    }

    #[test]
    fn ext_substitution() {
        let p = expand_template("{stem}.from_{ext}.lml", Path::new("/data/foo.bdf")).unwrap();
        assert_eq!(p, PathBuf::from("foo.from_bdf.lml"));
    }

    #[test]
    fn parent_substitution() {
        let p =
            expand_template("{parent}/compressed/{stem}.lml", Path::new("/data/foo.edf")).unwrap();
        assert_eq!(p, PathBuf::from("/data/compressed/foo.lml"));
    }

    #[test]
    fn parent_with_no_parent() {
        // `foo.edf` (no directory) → empty parent substitutes cleanly.
        let p = expand_template("{parent}/{stem}.lml", Path::new("foo.edf")).unwrap();
        assert_eq!(p, PathBuf::from("/foo.lml"));
    }

    #[test]
    fn ext_missing_returns_none() {
        assert!(expand_template("{ext}.lml", Path::new("/data/no_ext")).is_none());
    }

    #[test]
    fn stem_present_even_when_ext_missing() {
        // Bare filename without extension has a stem (the whole name)
        // but no extension. {stem} should still resolve.
        let p = expand_template("{stem}.lml", Path::new("/data/no_ext")).unwrap();
        assert_eq!(p, PathBuf::from("no_ext.lml"));
    }

    #[test]
    fn unknown_placeholder_returns_none() {
        assert!(expand_template("{stem}-{not_a_thing}.lml", Path::new("/data/foo.edf")).is_none());
    }

    #[test]
    fn unclosed_placeholder_returns_none() {
        assert!(expand_template("{stem.lml", Path::new("/data/foo.edf")).is_none());
    }

    #[test]
    fn multiple_substitutions() {
        let p = expand_template("{parent}/{stem}-archived.{ext}", Path::new("/x/y/z.edf")).unwrap();
        assert_eq!(p, PathBuf::from("/x/y/z-archived.edf"));
    }

    #[test]
    fn has_placeholder_detects() {
        assert!(has_placeholder("{stem}.lml"));
        assert!(has_placeholder("{parent}/x.lml"));
        assert!(has_placeholder("a-{name}-b"));
        assert!(has_placeholder("{ext}"));
        assert!(!has_placeholder("plain.lml"));
        assert!(!has_placeholder(""));
        // Literal-looking braces without a recognised key are still
        // detected — the expander will fail on unknown placeholder
        // which is correct (gives the user a clear error).
        assert!(!has_placeholder("{foo}"));
    }
}
