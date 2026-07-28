//! LMA directory archives on the BCS2 forensic-capsule wire.
//!
//! ADR 0139 contract 5 wants one native wire family. The LMA container
//! (`LMA1`/`LMA2`) predates it and is the last large store-side holdout: a
//! bespoke header, a bespoke JSON manifest, a bespoke EOCD footer, all
//! describing something BCS2 already models — a directory tree with per-entry
//! integrity and metadata, which is exactly [`ProfileId::FORENSIC_TREE_V1`].
//!
//! # Why the mapping needed a wire change first
//!
//! An LMA archive does not store its entries verbatim. EEG files are LML-coded,
//! most other files are zstd-compressed, already-compressed files are stored
//! raw. A forensic capsule's per-entry `content_id` is its chain-of-custody
//! anchor — it answers *"is this the file that was archived?"* — so the naive
//! mapping, hashing whatever bytes happen to be stored, would have kept every
//! check passing while silently redefining the one claim the capsule exists to
//! make.
//!
//! The stored-form declaration added to `abir-bcs` is what makes the mapping
//! honest: `content_id` stays the hash of the *file*, and the frame separately
//! records what it holds and which capability recovers it. So:
//!
//! | LMA `Method` | frame holds | declares |
//! |---|---|---|
//! | `Store` | the file, verbatim | nothing |
//! | `Zstd` | a zstd stream | `CAP_ZSTD` |
//! | `Lml` | an LML bitstream | `CAP_LML_LOSSLESS_V1` |
//! | `Lml` + synthetic | an LML bitstream + template | `CAP_LML_LOSSLESS_V1 | CAP_LMA_SYNTHETIC_REEMIT` |
//!
//! The synthetic case is the one worth stating plainly. Those entries came from
//! files that are not EEG containers (ASCII sample-per-line, say): they were
//! converted to an intermediate EDF, coded, and the original's byte-level shape
//! kept as a template. A reader that can decode LML but cannot re-emit would
//! produce a *plausible wrong answer* — a valid EDF that is not the archived
//! file. The second capability bit is what turns that into a refusal.
//!
//! # What this module does not do
//!
//! It does not retire anything. `LMA1`/`LMA2` remain fully readable through
//! [`crate::lma`], no existing archive is rewritten, and conversion is opt-in
//! and non-destructive — it reads an archive and writes a new capsule beside it.
//! Retirement is a later phase and is gated on this path first being measured
//! superior.

// `Method` is non-exhaustive for downstream users. Internal matches retain
// wildcard arms so future variants fail closed or remain explicitly classified.
// Current compiler sees those defensive arms as unreachable.
#![allow(unreachable_patterns)]

use crate::lma::{encode_archive_entry, ArchiveSummary, EncodedEntry, Method, SyntheticFromInfo};
use semantic_abir_bcs::{
    encode_forensic_tree, raw_content_id, ForensicContentTransform, ForensicEntry,
    ForensicFileType, ForensicTimestamp, ForensicTree, ForensicTreeView, ForensicXattr,
    ResourceBounds, CAP_LMA_SYNTHETIC_REEMIT, CAP_LML_LOSSLESS_V1, CAP_ZSTD,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

type Error = Box<dyn std::error::Error + Send + Sync>;

/// xattr carrying the archive-time SHA-256 of the original file.
///
/// Redundant with the capsule's own blake3 `content_id` as an integrity check,
/// and kept anyway: it is the exact provenance string every existing LMA
/// manifest records, so a converted capsule can still be reconciled against the
/// archive it came from. Dropping it would make conversion lossy in the one
/// dimension an archive is judged on.
const XATTR_SHA256: &[u8] = b"user.lamquant.sha256";

/// xattr carrying the synthetic-source descriptor as JSON.
///
/// Opaque here on purpose — this module does not know any template's shape, the
/// same way the LMA manifest did not. It is round-tripped verbatim.
const XATTR_SYNTHETIC: &[u8] = b"user.lamquant.synthetic_from";

/// xattr recording which LMA method produced the frame.
///
/// The capability mask already says what is needed to read the frame, so this is
/// not load-bearing for correctness. It exists so a converted capsule can report
/// the same per-method counts the LMA tooling reports, without inferring them
/// from capability bits — an inference that would silently break the first time
/// two methods share a capability.
const XATTR_METHOD: &[u8] = b"user.lamquant.method";

/// Capabilities a fully equipped LamQuant reader offers.
pub const READER_CAPABILITIES: u64 = CAP_ZSTD | CAP_LML_LOSSLESS_V1 | CAP_LMA_SYNTHETIC_REEMIT;

/// Every ancestor directory of `path`, shortest first.
///
/// The forensic profile requires each entry's parent to be present as a
/// directory entry. LMA never guaranteed that — it stores a flat file list plus
/// a separate directory table that only covers directories that existed at pack
/// time — so parents are synthesised here rather than assumed.
fn ancestors_of(path: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let components: Vec<&str> = path.split('/').collect();
    for component in &components[..components.len().saturating_sub(1)] {
        if !current.is_empty() {
            current.push('/');
        }
        current.push_str(component);
        out.push(current.clone());
    }
    out
}

fn timestamp_of(seconds: Option<u64>, nanos: Option<u32>) -> Option<ForensicTimestamp> {
    let seconds = seconds?;
    Some(ForensicTimestamp {
        // A pre-1970 mtime is representable in the capsule (i64) but not in the
        // u64 an LMA manifest stores, so this cast cannot lose a real value.
        seconds: i64::try_from(seconds).unwrap_or(i64::MAX),
        nanoseconds: nanos.unwrap_or(0).min(999_999_999),
    })
}

fn blank_entry(path: &str, file_type: ForensicFileType, mode: u32) -> ForensicEntry {
    ForensicEntry {
        path: path.as_bytes().to_vec(),
        file_type,
        mode,
        owner: None,
        timestamps: [None; 4],
        acl: None,
        xattrs: Vec::new(),
        hardlink_target: None,
        symlink_target: None,
        sparse_extents: Vec::new(),
        flags: 0,
        device: None,
        special_type: None,
        content: None,
        content_transform: None,
    }
}

/// Capability mask required to recover a file stored under `method`.
fn capabilities_for(method: Method, synthetic: bool) -> u64 {
    let mut mask = match method {
        Method::Store => 0,
        Method::Zstd => CAP_ZSTD,
        Method::Lml => CAP_LML_LOSSLESS_V1,
        // `Method` is `#[non_exhaustive]`. A method this build does not know
        // cannot be described honestly, so callers must not reach here; the
        // builder below rejects it rather than guessing a mask.
        _ => u64::MAX,
    };
    if synthetic {
        mask |= CAP_LMA_SYNTHETIC_REEMIT;
    }
    mask
}

/// Build one regular-file capsule entry from an encoded LMA entry.
///
/// `stored` is what the archive holds; `original_size`/`sha256` describe the
/// file itself. Keeping those two apart is the whole point of the stored form.
#[allow(clippy::too_many_arguments)]
fn file_entry(
    rel_path: &str,
    stored: Vec<u8>,
    method: Method,
    original_size: u64,
    sha256: &str,
    mtime: Option<u64>,
    mtime_nanos: Option<u32>,
    mode: Option<u32>,
    synthetic_from: Option<&SyntheticFromInfo>,
    logical_content_id: Option<semantic_abir::ContentId>,
) -> Result<ForensicEntry, Error> {
    let capabilities = capabilities_for(method, synthetic_from.is_some());
    if capabilities == u64::MAX {
        return Err(format!(
            "lma capsule: entry `{rel_path}` uses a compression method this build does not \
             recognise; refusing to write a capsule that misdescribes how to read it"
        )
        .into());
    }

    let mut entry = blank_entry(rel_path, ForensicFileType::Regular, mode.unwrap_or(0o644));
    entry.timestamps[1] = timestamp_of(mtime, mtime_nanos);

    // xattr names must be unique and ascending; build sorted.
    let mut xattrs: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    xattrs.insert(XATTR_SHA256.to_vec(), sha256.as_bytes().to_vec());
    xattrs.insert(
        XATTR_METHOD.to_vec(),
        method_name(method).as_bytes().to_vec(),
    );
    if let Some(info) = synthetic_from {
        let json = serde_json::json!({
            "format": info.format,
            "sample_rate": if info.sample_rate.is_finite() { info.sample_rate } else { 0.0 },
            "template": info.template_json,
        });
        xattrs.insert(
            XATTR_SYNTHETIC.to_vec(),
            serde_json::to_vec(&json).map_err(|e| format!("lma capsule: {e}"))?,
        );
    }
    entry.xattrs = xattrs
        .into_iter()
        .map(|(name, value)| ForensicXattr { name, value })
        .collect();

    if capabilities == 0 {
        // Stored verbatim: the frame IS the file, so no transform to declare.
        // Guard the invariant rather than trusting the caller, because a
        // mismatch here would make `original_size` a lie about the frame.
        if stored.len() as u64 != original_size {
            return Err(format!(
                "lma capsule: entry `{rel_path}` is stored verbatim but its {} stored bytes do \
                 not match its {original_size}-byte original",
                stored.len()
            )
            .into());
        }
        entry.content = Some(stored);
    } else {
        let logical = match logical_content_id {
            Some(id) => id,
            // Only reachable when packing from disk, where the original bytes
            // are in hand. The conversion path always supplies the id it
            // recomputed from the decoded original.
            None => {
                return Err(format!(
                    "lma capsule: entry `{rel_path}` is transformed but its original content id \
                     is unknown"
                )
                .into())
            }
        };
        entry.content = Some(stored);
        entry.content_transform = Some(ForensicContentTransform {
            capabilities,
            logical_content_id: logical,
            logical_len: original_size,
        });
    }
    Ok(entry)
}

fn method_name(method: Method) -> &'static str {
    match method {
        Method::Lml => "lml",
        Method::Zstd => "secondary",
        Method::Store => "store",
        _ => "unknown",
    }
}

fn method_from_name(name: &str) -> Option<Method> {
    match name {
        "lml" => Some(Method::Lml),
        "secondary" | "zstd" => Some(Method::Zstd),
        "store" => Some(Method::Store),
        _ => None,
    }
}

/// Assemble a capsule from file entries, synthesising the directory entries the
/// profile requires and sorting everything into the canonical order.
fn build_tree(
    mut files: Vec<ForensicEntry>,
    dir_mtimes: &BTreeMap<String, (Option<u64>, Option<u32>)>,
) -> ForensicTree {
    let mut directories: BTreeSet<String> = BTreeSet::new();
    for entry in &files {
        let path = String::from_utf8_lossy(&entry.path).into_owned();
        for ancestor in ancestors_of(&path) {
            directories.insert(ancestor);
        }
    }
    for path in dir_mtimes.keys() {
        directories.insert(path.clone());
        for ancestor in ancestors_of(path) {
            directories.insert(ancestor);
        }
    }

    let mut entries: Vec<ForensicEntry> = directories
        .into_iter()
        .map(|path| {
            let mut entry = blank_entry(&path, ForensicFileType::Directory, 0o755);
            if let Some((seconds, nanos)) = dir_mtimes.get(&path) {
                entry.timestamps[1] = timestamp_of(*seconds, *nanos);
            }
            entry
        })
        .collect();
    entries.append(&mut files);
    // The profile requires strictly ascending paths.
    entries.sort_by(|a, b| a.path.cmp(&b.path));

    ForensicTree {
        platform: platform_tag(),
        entries,
    }
}

fn platform_tag() -> String {
    // Constrained to the profile's charset: alphanumerics, dot, underscore, dash.
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Resource ceilings sized for one directory archive.
///
/// A directory archive is routinely larger than the conservative
/// single-artifact defaults, in both entry count and per-frame size — a single
/// multi-gigabyte EDF exceeds the 64 MiB default frame cap on its own. The
/// ceilings are raised to fit *this* tree rather than disabled, so a malformed
/// capsule is still refused rather than being allowed to allocate without
/// limit.
///
/// `largest_frame` of 0 means "unknown" (the read paths, which have not parsed
/// the catalog yet); the file's own length is then the only honest bound
/// available, and it is a true upper bound on any frame inside it.
fn bounds_for(entry_count: usize, largest_frame: u64) -> ResourceBounds {
    let headroom = |value: u64| -> u32 {
        u32::try_from(value.saturating_mul(2).saturating_add(1 << 20)).unwrap_or(u32::MAX)
    };
    let mut bounds = ResourceBounds::default();
    bounds.max_index_entries = bounds
        .max_index_entries
        .max(u32::try_from(entry_count.saturating_add(16)).unwrap_or(u32::MAX));
    bounds.max_frame_bytes = bounds.max_frame_bytes.max(headroom(largest_frame));
    // The metadata document grows with the entry count and is itself a frame,
    // so the catalog bound has to track the tree size too.
    bounds.max_catalog_bytes = bounds
        .max_catalog_bytes
        .max(headroom(entry_count as u64 * 1024));
    bounds
}

/// Pack a directory into a BCS2 forensic capsule.
///
/// Same compression cascade and the same per-file encoder as
/// [`crate::lma::pack_archive`], so an entry's stored bytes are identical either
/// way — only the container differs.
pub fn pack_directory(
    input_dir: &Path,
    output_path: &Path,
    zstd_level: i32,
    verbose: bool,
    progress_fn: Option<&dyn Fn(usize, usize, &str)>,
) -> Result<ArchiveSummary, Error> {
    let meta = std::fs::symlink_metadata(input_dir)
        .map_err(|e| format!("Cannot stat {}: {}", input_dir.display(), e))?;
    if !meta.file_type().is_dir() {
        return Err(format!(
            "Not a directory (or is a symlink to one): {}",
            input_dir.display()
        )
        .into());
    }

    let files = crate::lma::walk_files(input_dir);
    let symlinks = crate::lma::walk_symlinks(input_dir);
    if !symlinks.is_empty() {
        // Matches `pack_archive`: refuse loudly rather than silently dropping or
        // silently following. The capsule format could represent them, but
        // changing the safety posture is not this migration's business.
        return Err(format!(
            "Refusing to archive {} symlink(s) under {}: resolve them first",
            symlinks.len(),
            input_dir.display()
        )
        .into());
    }

    let tmp_dir = output_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let total = files.len();
    let mut entries: Vec<ForensicEntry> = Vec::with_capacity(total);
    let mut summary = ArchiveSummary {
        n_files: 0,
        original_bytes: 0,
        archive_bytes: 0,
        cr: 0.0,
        counts_lml: 0,
        counts_zstd: 0,
        counts_store: 0,
        errors: Vec::new(),
    };
    let mut stored_bytes: u64 = 0;

    for (index, (full_path, rel_path)) in files.iter().enumerate() {
        if let Some(report) = progress_fn {
            report(index, total, rel_path);
        }
        match encode_archive_entry(full_path, rel_path, zstd_level, &tmp_dir, verbose) {
            EncodedEntry::Skipped { rel_path, msg } => summary.errors.push((rel_path, msg)),
            EncodedEntry::Ready {
                rel_path,
                compressed,
                method,
                original_size,
                file_hash,
                mtime,
                mtime_nanos,
                mode,
                synthetic_from,
                warnings,
            } => {
                summary.errors.extend(warnings);
                let original = std::fs::read(full_path)?;
                let logical = raw_content_id(&original);
                stored_bytes += compressed.len() as u64;
                summary.original_bytes += original_size;
                match method {
                    Method::Lml => summary.counts_lml += 1,
                    Method::Zstd => summary.counts_zstd += 1,
                    Method::Store => summary.counts_store += 1,
                    _ => {}
                }
                summary.n_files += 1;
                entries.push(file_entry(
                    &rel_path,
                    compressed,
                    method,
                    original_size,
                    &file_hash,
                    mtime,
                    mtime_nanos,
                    mode,
                    synthetic_from.as_ref(),
                    Some(logical),
                )?);
            }
        }
    }

    let dir_mtimes = collect_dir_mtimes(input_dir);
    let tree = build_tree(entries, &dir_mtimes);
    let bounds = bounds_for(tree.entries.len(), stored_bytes);
    let capsule = encode_forensic_tree(&tree, bounds)
        .map_err(|e| format!("lma capsule: encode failed: {e:?}"))?;
    std::fs::write(output_path, &capsule)?;

    summary.archive_bytes = capsule.len() as u64;
    summary.cr = if summary.archive_bytes > 0 {
        summary.original_bytes as f64 / summary.archive_bytes as f64
    } else {
        0.0
    };
    Ok(summary)
}

fn collect_dir_mtimes(root: &Path) -> BTreeMap<String, (Option<u64>, Option<u32>)> {
    let mut out = BTreeMap::new();
    for (rel, seconds) in crate::lma::walk_dirs(root) {
        out.insert(rel, (Some(seconds), None));
    }
    out
}

/// One entry as recovered from a capsule.
pub struct CapsuleEntry {
    pub path: String,
    pub original_size: u64,
    pub stored_size: u64,
    pub method: Method,
    pub sha256: Option<String>,
    pub required_capabilities: u64,
    pub synthetic_from: Option<SyntheticFromInfo>,
    pub mtime: Option<u64>,
    pub mtime_nanos: Option<u32>,
    pub mode: Option<u32>,
}

fn xattr_of<'a>(
    entry: &'a semantic_abir_bcs::ForensicEntryMetadata,
    name: &[u8],
) -> Option<&'a [u8]> {
    entry
        .xattrs
        .iter()
        .find(|xattr| xattr.name == name)
        .map(|xattr| xattr.value.as_slice())
}

fn synthetic_of(
    entry: &semantic_abir_bcs::ForensicEntryMetadata,
) -> Result<Option<SyntheticFromInfo>, Error> {
    let Some(raw) = xattr_of(entry, XATTR_SYNTHETIC) else {
        return Ok(None);
    };
    let value: serde_json::Value =
        serde_json::from_slice(raw).map_err(|e| format!("lma capsule: synthetic_from: {e}"))?;
    let format = value
        .get("format")
        .and_then(|v| v.as_str())
        .ok_or("lma capsule: synthetic_from missing `format`")?
        .to_string();
    let sample_rate = value
        .get("sample_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let template_json = value
        .get("template")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    Ok(Some(SyntheticFromInfo {
        format,
        sample_rate,
        template_json,
    }))
}

/// List a capsule's regular-file entries.
pub fn list_capsule(path: &Path) -> Result<Vec<CapsuleEntry>, Error> {
    let bytes = std::fs::read(path)?;
    let bounds = bounds_for(0, bytes.len() as u64);
    let view = ForensicTreeView::parse(&bytes, READER_CAPABILITIES, bounds)
        .map_err(|e| format!("lma capsule: parse failed: {e:?}"))?;
    let mut out = Vec::new();
    for entry in view.entries() {
        if entry.file_type != ForensicFileType::Regular {
            continue;
        }
        let method = xattr_of(entry, XATTR_METHOD)
            .and_then(|raw| std::str::from_utf8(raw).ok())
            .and_then(method_from_name)
            // A capsule written by another producer has no LamQuant method
            // xattr. Its capability mask still says how to read it, so report
            // the method the mask implies rather than refusing to list it.
            .unwrap_or(match entry.required_capabilities() {
                0 => Method::Store,
                mask if mask & CAP_LML_LOSSLESS_V1 != 0 => Method::Lml,
                _ => Method::Zstd,
            });
        out.push(CapsuleEntry {
            path: String::from_utf8_lossy(&entry.path).into_owned(),
            original_size: entry.content_len.unwrap_or(0),
            stored_size: entry.frame_len().unwrap_or(0),
            method,
            sha256: xattr_of(entry, XATTR_SHA256)
                .and_then(|raw| std::str::from_utf8(raw).ok())
                .map(str::to_string),
            required_capabilities: entry.required_capabilities(),
            synthetic_from: synthetic_of(entry)?,
            mtime: entry.timestamps[1].map(|t| t.seconds.max(0) as u64),
            mtime_nanos: entry.timestamps[1].map(|t| t.nanoseconds),
            mode: Some(entry.mode),
        });
    }
    Ok(out)
}

/// Recover one entry's original bytes from its stored frame.
fn recover(
    stored: &[u8],
    method: Method,
    original_size: u64,
    synthetic_from: Option<&SyntheticFromInfo>,
    tmp_dir: Option<&Path>,
) -> Result<Vec<u8>, Error> {
    let bytes = match method {
        Method::Store => stored.to_vec(),
        Method::Zstd => crate::lma::decode_zstd_bounded(
            stored,
            usize::try_from(original_size)
                .map_err(|_| "lma capsule: entry too large for this platform")?,
            "lma capsule entry",
        )?,
        Method::Lml => {
            let edf = crate::lma::decode_lml_to_edf(stored, Some(original_size), tmp_dir)?;
            match synthetic_from {
                Some(info) => crate::lma::re_emit_synthetic(&edf, info)?,
                None => edf,
            }
        }
        _ => return Err("lma capsule: unknown compression method".into()),
    };
    Ok(bytes)
}

/// Unpack a capsule into `out_dir`, restoring bytes, mtimes and modes.
pub fn unpack_capsule(
    capsule_path: &Path,
    out_dir: &Path,
    verbose: bool,
) -> Result<ArchiveSummary, Error> {
    let bytes = std::fs::read(capsule_path)?;
    let bounds = bounds_for(0, bytes.len() as u64);
    let view = ForensicTreeView::parse(&bytes, READER_CAPABILITIES, bounds)
        .map_err(|e| format!("lma capsule: parse failed: {e:?}"))?;

    let mut summary = ArchiveSummary {
        n_files: 0,
        original_bytes: 0,
        archive_bytes: bytes.len() as u64,
        cr: 0.0,
        counts_lml: 0,
        counts_zstd: 0,
        counts_store: 0,
        errors: Vec::new(),
    };

    for entry in view.entries() {
        let rel = String::from_utf8_lossy(&entry.path).into_owned();
        // The profile already rejects absolute paths and `..` components at
        // parse time, so this join cannot escape `out_dir`.
        let target = out_dir.join(&rel);
        match entry.file_type {
            ForensicFileType::Directory => {
                std::fs::create_dir_all(&target)?;
                continue;
            }
            ForensicFileType::Regular => {}
            other => {
                summary
                    .errors
                    .push((rel, format!("unsupported entry type {other:?}")));
                continue;
            }
        }

        let method = xattr_of(entry, XATTR_METHOD)
            .and_then(|raw| std::str::from_utf8(raw).ok())
            .and_then(method_from_name)
            .unwrap_or(match entry.required_capabilities() {
                0 => Method::Store,
                mask if mask & CAP_LML_LOSSLESS_V1 != 0 => Method::Lml,
                _ => Method::Zstd,
            });
        let synthetic = synthetic_of(entry)?;
        let stored = view
            .stored_bytes(entry)
            .ok_or_else(|| format!("lma capsule: entry `{rel}` has no frame"))?;
        let original_size = entry.content_len.unwrap_or(0);

        let recovered = match recover(
            stored,
            method,
            original_size,
            synthetic.as_ref(),
            target.parent(),
        ) {
            Ok(bytes) => bytes,
            Err(e) => {
                if verbose {
                    eprintln!("  WARN: {rel}: {e}");
                }
                summary.errors.push((rel, e.to_string()));
                continue;
            }
        };

        // The capsule's content_id is the archived file's identity. Checking the
        // recovered bytes against it is what makes extraction verified rather
        // than merely attempted -- and it is the check the LMA path could only
        // do via the manifest's separate sha256 field.
        if let Some(expected) = entry.content_id {
            let actual = raw_content_id(&recovered);
            if actual != expected {
                summary.errors.push((
                    rel.clone(),
                    "recovered bytes do not match the archived content id".into(),
                ));
                continue;
            }
        }

        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, &recovered)?;
        restore_metadata(&target, entry)?;

        summary.n_files += 1;
        summary.original_bytes += recovered.len() as u64;
        match method {
            Method::Lml => summary.counts_lml += 1,
            Method::Zstd => summary.counts_zstd += 1,
            Method::Store => summary.counts_store += 1,
            _ => {}
        }
    }

    summary.cr = if summary.archive_bytes > 0 {
        summary.original_bytes as f64 / summary.archive_bytes as f64
    } else {
        0.0
    };
    Ok(summary)
}

fn restore_metadata(
    target: &Path,
    entry: &semantic_abir_bcs::ForensicEntryMetadata,
) -> Result<(), Error> {
    if let Some(stamp) = entry.timestamps[1] {
        let mtime = filetime::FileTime::from_unix_time(stamp.seconds, stamp.nanoseconds);
        // Best-effort, matching the LMA extract path: a filesystem that cannot
        // hold the timestamp must not fail the extraction.
        let _ = filetime::set_file_mtime(target, mtime);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(target, std::fs::Permissions::from_mode(entry.mode));
    }
    Ok(())
}

/// Recover an entry's original bytes and the bytes to store for it, from an LMA
/// archive.
///
/// [`crate::lma::read_entry`] is deliberately *not* uniform: for `Store` and
/// `Zstd` it returns the decoded original (and SHA-verifies it against the
/// manifest on the way out), while for `Lml` it returns the raw bitstream
/// because the manifest hash describes the pre-encode EDF, not the payload.
/// Treating that as one contract silently feeds already-decompressed bytes to a
/// decompressor — which is exactly the failure this function exists to avoid.
///
/// The LML bitstream is carried across verbatim rather than re-encoded: it is
/// the expensive artifact, and re-encoding would risk a different payload for
/// the same input. Zstd frames are re-encoded, which is cheap and produces a
/// valid encoding of the same original — all the capsule claims about a frame
/// is what recovers it, not which encoder produced it.
fn recover_for_conversion(
    lma_path: &Path,
    entry: &crate::lma::ArchiveEntry,
    zstd_level: i32,
    tmp_dir: Option<&Path>,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let bytes = crate::lma::read_entry(lma_path, &entry.path)?;
    match entry.method {
        Method::Store => Ok((bytes.clone(), bytes)),
        Method::Zstd => {
            let stored = zstd::encode_all(bytes.as_slice(), zstd_level)
                .map_err(|e| format!("lma capsule: re-encode `{}`: {e}", entry.path))?;
            Ok((bytes, stored))
        }
        Method::Lml => {
            let edf = crate::lma::decode_lml_to_edf(&bytes, Some(entry.original_size), tmp_dir)?;
            let original = match entry.synthetic_from.as_ref() {
                Some(info) => crate::lma::re_emit_synthetic(&edf, info)?,
                None => edf,
            };
            Ok((original, bytes))
        }
        _ => Err(format!(
            "lma capsule: entry `{}` uses a compression method this build does not recognise",
            entry.path
        )
        .into()),
    }
}

/// Convert an existing `LMA1`/`LMA2` archive into a forensic capsule.
///
/// Non-destructive: the source archive is read and left exactly as it is.
///
/// Every entry is recovered to its original bytes and checked against the
/// manifest's own SHA-256 before it is written. A straight payload copy would be
/// faster and would carry a corrupt entry into the new capsule, stamping it with
/// a fresh and entirely confident identity — the one outcome a chain-of-custody
/// format must not produce.
pub fn convert_archive(
    lma_path: &Path,
    output_path: &Path,
    zstd_level: i32,
) -> Result<ArchiveSummary, Error> {
    let manifest = crate::lma::list_archive(lma_path)?;
    let mut entries: Vec<ForensicEntry> = Vec::with_capacity(manifest.len());
    let mut summary = ArchiveSummary {
        n_files: 0,
        original_bytes: 0,
        archive_bytes: 0,
        cr: 0.0,
        counts_lml: 0,
        counts_zstd: 0,
        counts_store: 0,
        errors: Vec::new(),
    };
    let mut stored_bytes: u64 = 0;

    for entry in &manifest {
        let (original, stored) =
            match recover_for_conversion(lma_path, entry, zstd_level, output_path.parent()) {
                Ok(pair) => pair,
                Err(e) => {
                    summary.errors.push((entry.path.clone(), e.to_string()));
                    continue;
                }
            };
        let actual_sha = crate::lma::sha256_hex(&original);
        if actual_sha != entry.sha256 {
            summary.errors.push((
                entry.path.clone(),
                format!(
                    "source archive integrity: manifest records {} but the entry recovers to {}",
                    entry.sha256, actual_sha
                ),
            ));
            continue;
        }

        stored_bytes += stored.len() as u64;
        summary.original_bytes += entry.original_size;
        summary.n_files += 1;
        match entry.method {
            Method::Lml => summary.counts_lml += 1,
            Method::Zstd => summary.counts_zstd += 1,
            Method::Store => summary.counts_store += 1,
            _ => {}
        }
        entries.push(file_entry(
            &entry.path,
            stored,
            entry.method,
            entry.original_size,
            &entry.sha256,
            entry.mtime,
            entry.mtime_nanos,
            entry.mode,
            entry.synthetic_from.as_ref(),
            Some(raw_content_id(&original)),
        )?);
    }

    let tree = build_tree(entries, &BTreeMap::new());
    let bounds = bounds_for(tree.entries.len(), stored_bytes);
    let capsule = encode_forensic_tree(&tree, bounds)
        .map_err(|e| format!("lma capsule: encode failed: {e:?}"))?;
    std::fs::write(output_path, &capsule)?;

    summary.archive_bytes = capsule.len() as u64;
    summary.cr = if summary.archive_bytes > 0 {
        summary.original_bytes as f64 / summary.archive_bytes as f64
    } else {
        0.0
    };
    Ok(summary)
}

/// True when `path` holds a BCS2 artifact rather than an LMA container.
///
/// Lets one CLI verb accept either, so the wire change is invisible to callers.
pub fn is_capsule(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic).is_ok() && magic == semantic_abir_bcs::BCS2_MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ancestors_are_shortest_first_and_exclude_the_leaf() {
        assert_eq!(
            ancestors_of("a/b/c/file.edf"),
            vec!["a".to_string(), "a/b".to_string(), "a/b/c".to_string()]
        );
        assert!(ancestors_of("file.edf").is_empty());
    }

    #[test]
    fn capability_masks_match_the_documented_table() {
        assert_eq!(capabilities_for(Method::Store, false), 0);
        assert_eq!(capabilities_for(Method::Zstd, false), CAP_ZSTD);
        assert_eq!(capabilities_for(Method::Lml, false), CAP_LML_LOSSLESS_V1);
        assert_eq!(
            capabilities_for(Method::Lml, true),
            CAP_LML_LOSSLESS_V1 | CAP_LMA_SYNTHETIC_REEMIT,
            "a synthetic entry needs re-emit ON TOP OF decoding, not instead of it"
        );
    }

    #[test]
    fn a_reader_advertising_everything_covers_every_method() {
        for method in [Method::Store, Method::Zstd, Method::Lml] {
            for synthetic in [false, true] {
                let mask = capabilities_for(method, synthetic);
                assert_eq!(
                    mask & !READER_CAPABILITIES,
                    0,
                    "{method:?}/{synthetic} requires a capability this crate does not advertise"
                );
            }
        }
    }
}
