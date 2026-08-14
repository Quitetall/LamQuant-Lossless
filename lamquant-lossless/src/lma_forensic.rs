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
//! | current `Lml` | an LML bitstream | `CAP_LML_LOSSLESS_V1` |
//! | retired `LML1` | an LML1 bitstream | `CAP_LML_LOSSLESS_V1 | CAP_LML1_LEGACY_MATERIALIZE` |
//! | `Lml` + synthetic | an LML bitstream + template | method capabilities + `CAP_LMA_SYNTHETIC_REEMIT` |
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
//!
//! Exact extraction is Linux-only and requires a caller-created empty output
//! directory. Reader performs full metadata/capability/frame preflight before
//! creating children. Capsules packed directly from a directory preserve file
//! and directory permission bits. Historical LMA manifests omitted directory
//! modes and sometimes file modes; `lma-import-v1` therefore defines explicit
//! deterministic defaults of 0755 for directories and 0644 for files.

// `Method` is non-exhaustive for downstream users. Internal matches retain
// wildcard arms so future variants fail closed or remain explicitly classified.
// Current compiler sees those defensive arms as unreachable.
#![allow(unreachable_patterns)]

use crate::legacy_adapter_process;
pub use crate::legacy_adapter_process::LegacyAdapterConfig;
use crate::lma::{
    encode_archive_entry, ArchiveSummary, EncodedEntry, LmaArchive, Method, StoredExtent,
    SyntheticFromInfo,
};
use semantic_abir_bcs::{
    raw_content_id, write_forensic_tree_streaming, ForensicEntryMetadata, ForensicFileIndex,
    ForensicFileType, ForensicStoredForm, ForensicTimestamp, ForensicTreeMetadata,
    LmaSyntheticLineEnding, LmaSyntheticReemitParametersV1, ResourceBounds,
    CAP_LMA_SYNTHETIC_REEMIT, CAP_LML1_LEGACY_MATERIALIZE, CAP_LML_LOSSLESS_V1, CAP_ZSTD,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

type Error = Box<dyn std::error::Error + Send + Sync>;

/// Capabilities a fully equipped LamQuant reader offers.
pub const READER_CAPABILITIES: u64 = CAP_ZSTD
    | CAP_LML_LOSSLESS_V1
    | CAP_LMA_SYNTHETIC_REEMIT
    | if cfg!(target_os = "linux") {
        CAP_LML1_LEGACY_MATERIALIZE
    } else {
        0
    };

const FORENSIC_MAX_INDEX_ENTRIES: u32 = 1_000_000;
const FORENSIC_MAX_METADATA_BYTES: u32 = 512 * 1024 * 1024;
const LMA_IMPORT_PLATFORM: &str = "lma-import-v1";

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

fn timestamp_of(
    seconds: Option<u64>,
    nanos: Option<u32>,
) -> Result<Option<ForensicTimestamp>, Error> {
    let Some(seconds) = seconds else {
        return Ok(None);
    };
    let seconds = i64::try_from(seconds).map_err(|_| "lma capsule: mtime exceeds i64")?;
    let nanoseconds = nanos.unwrap_or(0);
    if nanoseconds >= 1_000_000_000 {
        return Err("lma capsule: mtime nanoseconds exceed timestamp domain".into());
    }
    Ok(Some(ForensicTimestamp {
        seconds,
        nanoseconds,
    }))
}

/// Capability mask required to recover a file stored under `method`.
fn capabilities_for(method: Method, synthetic: bool, retired_lml1: bool) -> u64 {
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
    if retired_lml1 {
        if !matches!(method, Method::Lml) {
            return u64::MAX;
        }
        mask |= CAP_LML1_LEGACY_MATERIALIZE;
    }
    mask
}

fn synthetic_parameters(info: Option<&SyntheticFromInfo>) -> Result<[u8; 32], Error> {
    let Some(info) = info else {
        return Ok([0; 32]);
    };
    if info.format != "ascii_int_lines" {
        return Err(format!(
            "lma capsule: unsupported synthetic descriptor `{}`",
            info.format
        )
        .into());
    }
    let template = crate::ingest::AsciiLinesTemplate::from_json(&info.template_json)
        .map_err(|error| format!("lma capsule: invalid synthetic descriptor: {error}"))?;
    let line_ending = match template.line_ending {
        crate::ingest::ascii_lines::LineEnding::Lf => LmaSyntheticLineEnding::Lf,
        crate::ingest::ascii_lines::LineEnding::CrLf => LmaSyntheticLineEnding::CrLf,
    };
    Ok(LmaSyntheticReemitParametersV1::new(
        line_ending,
        template.leading_whitespace,
        template.field_width,
        template.trailing_newline,
    )
    .encode())
}

fn method_for_capabilities(capabilities: u64) -> Result<Method, Error> {
    let storage = capabilities & (CAP_ZSTD | CAP_LML_LOSSLESS_V1);
    match storage {
        0 if capabilities == 0 => Ok(Method::Store),
        CAP_ZSTD => Ok(Method::Zstd),
        CAP_LML_LOSSLESS_V1 => Ok(Method::Lml),
        0 => Err("lma capsule: capability mask lacks a storage method".into()),
        _ => Err("lma capsule: capability mask declares conflicting storage methods".into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn file_metadata(
    rel_path: &str,
    stored_content_id: semantic_abir::ContentId,
    stored_len: u64,
    method: Method,
    original_size: u64,
    mtime: Option<u64>,
    mtime_nanos: Option<u32>,
    mode: Option<u32>,
    synthetic_from: Option<&SyntheticFromInfo>,
    logical_content_id: semantic_abir::ContentId,
    retired_lml1: bool,
) -> Result<ForensicEntryMetadata, Error> {
    let capabilities = capabilities_for(method, synthetic_from.is_some(), retired_lml1);
    if capabilities == u64::MAX {
        return Err(
            format!("lma capsule: entry `{rel_path}` uses an unknown compression method").into(),
        );
    }
    if capabilities == 0 && (stored_content_id != logical_content_id || stored_len != original_size)
    {
        return Err(format!(
            "lma capsule: verbatim entry `{rel_path}` has contradictory stored and logical identity"
        )
        .into());
    }

    let parameters = synthetic_parameters(synthetic_from)?;
    let stored_form = if capabilities == 0 {
        None
    } else {
        let mut stored = ForensicStoredForm::new(capabilities, stored_content_id, stored_len);
        stored.parameters = parameters;
        Some(stored)
    };

    Ok(ForensicEntryMetadata {
        path: rel_path.as_bytes().to_vec(),
        file_type: ForensicFileType::Regular,
        mode: mode.unwrap_or(0o644) & 0o7777,
        owner: None,
        timestamps: [None, timestamp_of(mtime, mtime_nanos)?, None, None],
        acl: None,
        xattrs: Vec::new(),
        hardlink_target: None,
        symlink_target: None,
        sparse_extents: Vec::new(),
        flags: 0,
        device: None,
        special_type: None,
        content_id: Some(logical_content_id),
        content_len: Some(original_size),
        stored_form,
    })
}

fn build_metadata(
    mut files: Vec<ForensicEntryMetadata>,
    dir_mtimes: &BTreeMap<String, (Option<u64>, Option<u32>, Option<u32>)>,
    platform: &str,
) -> Result<ForensicTreeMetadata, Error> {
    let mut directories = BTreeSet::new();
    for entry in &files {
        let path = std::str::from_utf8(&entry.path)
            .map_err(|_| "lma capsule: non-UTF-8 path is not portable")?;
        directories.extend(ancestors_of(path));
    }
    for path in dir_mtimes.keys() {
        directories.insert(path.clone());
        directories.extend(ancestors_of(path));
    }
    let mut entries: Vec<_> = directories
        .into_iter()
        .map(|path| -> Result<ForensicEntryMetadata, Error> {
            Ok(ForensicEntryMetadata {
                path: path.as_bytes().to_vec(),
                file_type: ForensicFileType::Directory,
                // Historical LMA manifests did not preserve directory modes.
                // `lma-import-v1` defines deterministic 0755 restoration.
                mode: dir_mtimes
                    .get(&path)
                    .and_then(|(_, _, mode)| *mode)
                    .unwrap_or(0o755)
                    & 0o7777,
                owner: None,
                timestamps: [
                    None,
                    match dir_mtimes.get(&path) {
                        Some((seconds, nanos, _)) => timestamp_of(*seconds, *nanos)?,
                        None => None,
                    },
                    None,
                    None,
                ],
                acl: None,
                xattrs: Vec::new(),
                hardlink_target: None,
                symlink_target: None,
                sparse_extents: Vec::new(),
                flags: 0,
                device: None,
                special_type: None,
                content_id: None,
                content_len: None,
                stored_form: None,
            })
        })
        .collect::<Result<_, _>>()?;
    entries.append(&mut files);
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(ForensicTreeMetadata {
        platform: platform.to_owned(),
        entries,
    })
}

fn platform_tag() -> String {
    // Constrained to the profile's charset: alphanumerics, dot, underscore, dash.
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Writer ceilings for one directory archive.
///
/// A directory archive is routinely larger than the conservative
/// single-artifact defaults, in both entry count and per-frame size — a single
/// multi-gigabyte EDF exceeds the 64 MiB default frame cap on its own. The
/// Payload frame ceiling grows from trusted source extents. Metadata and entry
/// ceilings remain fixed policy, independent of artifact-controlled lengths.
fn writer_bounds(largest_frame: u64) -> ResourceBounds {
    let headroom = |value: u64| -> u32 {
        u32::try_from(value.saturating_mul(2).saturating_add(1 << 20)).unwrap_or(u32::MAX)
    };
    let mut bounds = ResourceBounds::default();
    bounds.max_index_entries = FORENSIC_MAX_INDEX_ENTRIES;
    bounds.max_catalog_bytes = FORENSIC_MAX_METADATA_BYTES;
    bounds.max_frame_bytes = bounds
        .max_frame_bytes
        .max(FORENSIC_MAX_METADATA_BYTES)
        .max(headroom(largest_frame));
    bounds
}

/// Independent acceptance policy for untrusted capsules.
fn reader_bounds() -> ResourceBounds {
    ResourceBounds {
        max_index_entries: FORENSIC_MAX_INDEX_ENTRIES,
        max_catalog_bytes: FORENSIC_MAX_METADATA_BYTES,
        // Payloads may legitimately be multi-gigabyte. ForensicFileIndex checks
        // metadata against max_catalog_bytes before reading it.
        max_frame_bytes: u32::MAX,
        ..ResourceBounds::default()
    }
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

    let output_parent = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let spool_dir = tempfile::tempdir_in(output_parent)?;
    let total = files.len();
    let mut entries: Vec<ForensicEntryMetadata> = Vec::with_capacity(total);
    let mut frames = BTreeMap::<semantic_abir::ContentId, tempfile::TempPath>::new();
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
    let mut largest_frame: u64 = 0;

    for (index, (full_path, rel_path)) in files.iter().enumerate() {
        if let Some(report) = progress_fn {
            report(index, total, rel_path);
        }
        match encode_archive_entry(full_path, rel_path, zstd_level, spool_dir.path(), verbose) {
            EncodedEntry::Skipped { rel_path, msg } => summary.errors.push((rel_path, msg)),
            EncodedEntry::Ready {
                rel_path,
                compressed,
                method,
                original_size,
                logical_content_id,
                file_hash: _,
                mtime,
                mtime_nanos,
                mode,
                synthetic_from,
                warnings,
            } => {
                summary.errors.extend(warnings);
                let stored_content_id = raw_content_id(&compressed);
                let stored_len = compressed.len() as u64;
                let retired_lml1 = matches!(method, Method::Lml) && compressed.starts_with(b"LML1");
                largest_frame = largest_frame.max(compressed.len() as u64);
                summary.original_bytes += original_size;
                match method {
                    Method::Lml => summary.counts_lml += 1,
                    Method::Zstd => summary.counts_zstd += 1,
                    Method::Store => summary.counts_store += 1,
                    _ => {}
                }
                summary.n_files += 1;
                entries.push(file_metadata(
                    &rel_path,
                    stored_content_id,
                    stored_len,
                    method,
                    original_size,
                    mtime,
                    mtime_nanos,
                    mode,
                    synthetic_from.as_ref(),
                    logical_content_id,
                    retired_lml1,
                )?);
                if let std::collections::btree_map::Entry::Vacant(slot) =
                    frames.entry(stored_content_id)
                {
                    let mut staged = tempfile::NamedTempFile::new_in(spool_dir.path())?;
                    staged.write_all(&compressed)?;
                    staged.flush()?;
                    slot.insert(staged.into_temp_path());
                }
            }
        }
    }

    let dir_mtimes = collect_dir_mtimes(input_dir);
    let tree = build_metadata(entries, &dir_mtimes, &platform_tag())?;
    let bounds = writer_bounds(largest_frame);
    let mut staged_output = tempfile::NamedTempFile::new_in(output_parent)?;
    let receipt = write_forensic_tree_streaming(
        &mut staged_output,
        &tree,
        |content_id| {
            let path = frames.get(&content_id).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no staged frame for BCS2 content {content_id}"),
                )
            })?;
            File::open(path)
        },
        bounds,
    )
    .map_err(|error| format!("lma capsule: streaming encode failed: {error}"))?;
    staged_output.flush()?;
    staged_output.as_file().sync_all()?;
    staged_output.seek(SeekFrom::Start(0))?;
    let validated = ForensicFileIndex::open(&mut staged_output, READER_CAPABILITIES, bounds)
        .map_err(|error| format!("lma capsule: output validation failed: {error}"))?;
    if validated.root_content_id() != receipt.root_content_id()
        || staged_output.as_file().metadata()?.len() != receipt.artifact_len()
    {
        return Err("lma capsule: output receipt contradicts validated artifact".into());
    }
    staged_output
        .persist_noclobber(output_path)
        .map_err(|error| -> Error { error.error.into() })?;

    summary.archive_bytes = receipt.artifact_len();
    summary.cr = if summary.archive_bytes > 0 {
        summary.original_bytes as f64 / summary.archive_bytes as f64
    } else {
        0.0
    };
    Ok(summary)
}

fn collect_dir_mtimes(root: &Path) -> BTreeMap<String, (Option<u64>, Option<u32>, Option<u32>)> {
    let mut out = BTreeMap::new();
    for (rel, seconds) in crate::lma::walk_dirs(root) {
        #[cfg(unix)]
        let mode = {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::metadata(root.join(&rel)).ok();
            metadata.map(|value| value.permissions().mode() & 0o7777)
        };
        #[cfg(not(unix))]
        let mode = None;
        out.insert(rel, (Some(seconds), None, mode));
    }
    out
}

/// One entry as recovered from a capsule.
pub struct CapsuleEntry {
    pub path: String,
    pub original_size: u64,
    pub stored_size: u64,
    pub method: Method,
    pub content_id: Option<semantic_abir::ContentId>,
    pub required_capabilities: u64,
    pub synthetic_from: Option<SyntheticFromInfo>,
    pub mtime: Option<u64>,
    pub mtime_nanos: Option<u32>,
    pub mode: Option<u32>,
}

fn synthetic_of(
    entry: &semantic_abir_bcs::ForensicEntryMetadata,
) -> Result<Option<SyntheticFromInfo>, Error> {
    let capabilities = entry.required_capabilities();
    if capabilities & CAP_LMA_SYNTHETIC_REEMIT == 0 {
        return Ok(None);
    }
    let stored = entry
        .stored_form
        .ok_or("lma capsule: synthetic capability lacks stored form")?;
    let descriptor = LmaSyntheticReemitParametersV1::decode(stored.parameters)
        .map_err(|_| "lma capsule: invalid synthetic transform descriptor")?;
    let line_ending = match descriptor.line_ending {
        LmaSyntheticLineEnding::Lf => crate::ingest::ascii_lines::LineEnding::Lf,
        LmaSyntheticLineEnding::CrLf => crate::ingest::ascii_lines::LineEnding::CrLf,
    };
    let template_json = crate::ingest::AsciiLinesTemplate {
        line_ending,
        leading_whitespace: descriptor.leading_whitespace,
        field_width: descriptor.field_width,
        trailing_newline: descriptor.trailing_newline,
    }
    .to_json();
    Ok(Some(SyntheticFromInfo {
        format: "ascii_int_lines".to_owned(),
        // Historical sample rate never participates in exact re-emission.
        sample_rate: 0.0,
        template_json,
    }))
}

/// List a capsule's regular-file entries.
pub fn list_capsule(path: &Path) -> Result<Vec<CapsuleEntry>, Error> {
    let mut file = File::open(path)?;
    let bounds = reader_bounds();
    let index = ForensicFileIndex::open(&mut file, READER_CAPABILITIES, bounds)
        .map_err(|e| format!("lma capsule: parse failed: {e:?}"))?;
    let mut out = Vec::new();
    for entry in index.entries() {
        if entry.file_type != ForensicFileType::Regular {
            continue;
        }
        let method = method_for_capabilities(entry.required_capabilities())?;
        let path = std::str::from_utf8(&entry.path)
            .map_err(|_| "lma capsule: non-UTF-8 path is not portable")?
            .to_owned();
        let mtime = match entry.timestamps[1] {
            Some(timestamp) => Some(
                u64::try_from(timestamp.seconds)
                    .map_err(|_| "lma capsule: negative mtime is not representable in LMA")?,
            ),
            None => None,
        };
        out.push(CapsuleEntry {
            path,
            original_size: entry.content_len.unwrap_or(0),
            stored_size: entry.frame_len().unwrap_or(0),
            method,
            content_id: entry.content_id,
            required_capabilities: entry.required_capabilities(),
            synthetic_from: synthetic_of(entry)?,
            mtime,
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
    expected_sha256: Option<&str>,
    synthetic_from: Option<&SyntheticFromInfo>,
    tmp_dir: Option<&Path>,
    legacy_config: &LegacyAdapterConfig,
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
            if stored.starts_with(b"LML1") {
                let scratch = tempfile::Builder::new()
                    .prefix(".lml-legacy-materialize-")
                    .tempdir_in(tmp_dir.unwrap_or(Path::new(".")))?;
                let source = scratch.path().join("source.lml");
                let destination = scratch.path().join("original.bin");
                std::fs::write(&source, stored)?;
                match synthetic_from {
                    Some(info) => legacy_adapter_process::materialize_synthetic_exact(
                        legacy_config,
                        &source,
                        &destination,
                        expected_sha256,
                        original_size,
                        original_size.saturating_mul(16).saturating_add(16 << 20),
                        &info.format,
                        &info.template_json,
                    )?,
                    None => legacy_adapter_process::materialize_exact(
                        legacy_config,
                        &source,
                        &destination,
                        expected_sha256,
                        original_size,
                        original_size.saturating_mul(16).saturating_add(16 << 20),
                    )?,
                };
                std::fs::read(destination)?
            } else {
                let edf = crate::lma::decode_lml_to_edf(stored, Some(original_size), tmp_dir)?;
                match synthetic_from {
                    Some(info) => crate::lma::re_emit_synthetic(&edf, info)?,
                    None => edf,
                }
            }
        }
        _ => return Err("lma capsule: unknown compression method".into()),
    };
    Ok(bytes)
}

fn frame_extent(
    index: &ForensicFileIndex,
    entry: &ForensicEntryMetadata,
) -> Result<(u64, u64), Error> {
    let frame_id = entry
        .frame_content_id()
        .ok_or("lma capsule: regular entry has no frame")?;
    let frame = index
        .artifact()
        .frames()
        .binary_search_by_key(&frame_id, |candidate| candidate.content_id())
        .ok()
        .map(|position| &index.artifact().frames()[position])
        .ok_or("lma capsule: indexed frame is absent")?;
    Ok((frame.offset(), frame.len()))
}

fn preflight_logical_content(file: &mut File, index: &ForensicFileIndex) -> Result<(), Error> {
    for entry in index.entries() {
        let path = std::str::from_utf8(&entry.path)
            .map_err(|_| "lma capsule: non-UTF-8 path is not portable")?;
        match entry.file_type {
            ForensicFileType::Directory => continue,
            ForensicFileType::Regular => {}
            _ => continue,
        }

        let capabilities = entry.required_capabilities();
        let method = method_for_capabilities(capabilities)?;
        let synthetic = synthetic_of(entry)?;
        if synthetic.is_some() && !matches!(method, Method::Lml) {
            return Err(format!(
                "lma capsule: `{path}` declares synthetic re-emission without LML storage"
            )
            .into());
        }
        if capabilities & CAP_LML1_LEGACY_MATERIALIZE != 0 && !matches!(method, Method::Lml) {
            return Err(format!(
                "lma capsule: `{path}` declares retired LML1 recovery without LML storage"
            )
            .into());
        }

        let (offset, len) = frame_extent(index, entry)?;
        if matches!(method, Method::Lml) {
            let mut magic = [0_u8; 4];
            if len >= magic.len() as u64 {
                file.seek(SeekFrom::Start(offset))?;
                file.read_exact(&mut magic)?;
            }
            let is_retired = magic == *b"LML1";
            let declares_retired = capabilities & CAP_LML1_LEGACY_MATERIALIZE != 0;
            if is_retired != declares_retired {
                return Err(format!(
                    "lma capsule: `{path}` retired LML1 capability contradicts stored frame"
                )
                .into());
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn preflight_exact_restore(file: &mut File, index: &ForensicFileIndex) -> Result<(), Error> {
    let current_platform = platform_tag();
    if index.platform() != current_platform && index.platform() != LMA_IMPORT_PLATFORM {
        return Err(format!(
            "lma capsule: exact restore platform mismatch: capsule `{}`, reader `{current_platform}`",
            index.platform()
        )
        .into());
    }

    for entry in index.entries() {
        let path = std::str::from_utf8(&entry.path)
            .map_err(|_| "lma capsule: non-UTF-8 path is not portable")?;
        if entry.owner.is_some()
            || entry.timestamps[0].is_some()
            || entry.timestamps[2].is_some()
            || entry.timestamps[3].is_some()
            || entry.acl.is_some()
            || !entry.xattrs.is_empty()
            || entry.hardlink_target.is_some()
            || entry.symlink_target.is_some()
            || !entry.sparse_extents.is_empty()
            || entry.flags != 0
            || entry.device.is_some()
            || entry.special_type.is_some()
            || entry.mode & !0o7777 != 0
        {
            return Err(format!(
                "lma capsule: `{path}` carries metadata this exact restore path cannot reproduce"
            )
            .into());
        }

        match entry.file_type {
            ForensicFileType::Directory | ForensicFileType::Regular => {}
            other => {
                return Err(format!("lma capsule: `{path}` has unsupported type {other:?}").into())
            }
        }
    }
    preflight_logical_content(file, index)
}

/// Verify every logical file in a capsule without materializing an output tree.
pub fn verify_capsule(capsule_path: &Path, verbose: bool) -> Result<ArchiveSummary, Error> {
    let config = LegacyAdapterConfig::from_env()?;
    verify_capsule_with_legacy_config(capsule_path, verbose, &config)
}

/// Verify with an explicit retired-format Adapter supervision policy.
pub fn verify_capsule_with_legacy_config(
    capsule_path: &Path,
    verbose: bool,
    legacy_config: &LegacyAdapterConfig,
) -> Result<ArchiveSummary, Error> {
    let mut file = File::open(capsule_path)?;
    let artifact_len = file.metadata()?.len();
    let bounds = reader_bounds();
    let index = ForensicFileIndex::open(&mut file, READER_CAPABILITIES, bounds)
        .map_err(|error| format!("lma capsule: parse failed: {error:?}"))?;
    preflight_logical_content(&mut file, &index)?;
    if index
        .entries()
        .iter()
        .any(|entry| entry.required_capabilities() & CAP_LML1_LEGACY_MATERIALIZE != 0)
    {
        legacy_adapter_process::require_parent_verified_materialization(legacy_config)?;
    }

    let mut summary = ArchiveSummary {
        n_files: 0,
        original_bytes: 0,
        archive_bytes: artifact_len,
        cr: 0.0,
        counts_lml: 0,
        counts_zstd: 0,
        counts_store: 0,
        errors: Vec::new(),
    };
    for entry in index.entries() {
        if entry.file_type != ForensicFileType::Regular {
            continue;
        }
        let rel = std::str::from_utf8(&entry.path)
            .map_err(|_| "lma capsule: non-UTF-8 path is not portable")?;
        let method = method_for_capabilities(entry.required_capabilities())?;
        let synthetic = synthetic_of(entry)?;
        let (frame_offset, frame_len) = frame_extent(&index, entry)?;
        file.seek(SeekFrom::Start(frame_offset))?;
        let mut stored = Vec::new();
        stored
            .try_reserve_exact(
                usize::try_from(frame_len)
                    .map_err(|_| "lma capsule: frame too large for this platform")?,
            )
            .map_err(|_| "lma capsule: cannot reserve frame buffer")?;
        Read::by_ref(&mut file)
            .take(frame_len)
            .read_to_end(&mut stored)?;
        if stored.len() as u64 != frame_len {
            return Err(format!("lma capsule: entry `{rel}` frame is truncated").into());
        }
        let original_size = entry.content_len.unwrap_or(0);
        let recovered = recover(
            &stored,
            method,
            original_size,
            None,
            synthetic.as_ref(),
            capsule_path.parent(),
            legacy_config,
        )
        .map_err(|error| format!("lma capsule: cannot verify `{rel}`: {error}"))?;
        let expected = entry
            .content_id
            .ok_or_else(|| format!("lma capsule: `{rel}` lacks logical content identity"))?;
        if raw_content_id(&recovered) != expected || recovered.len() as u64 != original_size {
            return Err(format!(
                "lma capsule: verified `{rel}` bytes contradict logical identity or extent"
            )
            .into());
        }
        if verbose {
            eprintln!("  OK: {rel}");
        }
        summary.n_files += 1;
        summary.original_bytes += original_size;
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

/// Unpack a capsule into `out_dir`, restoring bytes, mtimes and modes.
pub fn unpack_capsule(
    capsule_path: &Path,
    out_dir: &Path,
    verbose: bool,
) -> Result<ArchiveSummary, Error> {
    let config = LegacyAdapterConfig::from_env()?;
    unpack_capsule_with_legacy_config(capsule_path, out_dir, verbose, &config)
}

/// Unpack with an explicit retired-format Adapter supervision policy.
#[cfg(not(target_os = "linux"))]
pub fn unpack_capsule_with_legacy_config(
    _capsule_path: &Path,
    _out_dir: &Path,
    _verbose: bool,
    _legacy_config: &LegacyAdapterConfig,
) -> Result<ArchiveSummary, Error> {
    Err("lma capsule: exact restore currently requires Linux".into())
}

/// Unpack with an explicit retired-format Adapter supervision policy.
#[cfg(target_os = "linux")]
pub fn unpack_capsule_with_legacy_config(
    capsule_path: &Path,
    out_dir: &Path,
    verbose: bool,
    legacy_config: &LegacyAdapterConfig,
) -> Result<ArchiveSummary, Error> {
    prepare_restore_directory(out_dir)?;
    let mut file = File::open(capsule_path)?;
    let artifact_len = file.metadata()?.len();
    let bounds = reader_bounds();
    let index = ForensicFileIndex::open(&mut file, READER_CAPABILITIES, bounds)
        .map_err(|e| format!("lma capsule: parse failed: {e:?}"))?;
    preflight_exact_restore(&mut file, &index)?;
    if index
        .entries()
        .iter()
        .any(|entry| entry.required_capabilities() & CAP_LML1_LEGACY_MATERIALIZE != 0)
    {
        legacy_adapter_process::require_parent_verified_materialization(legacy_config)?;
    }

    let mut summary = ArchiveSummary {
        n_files: 0,
        original_bytes: 0,
        archive_bytes: artifact_len,
        cr: 0.0,
        counts_lml: 0,
        counts_zstd: 0,
        counts_store: 0,
        errors: Vec::new(),
    };

    let mut directory_indices = Vec::new();
    for (entry_index, entry) in index.entries().iter().enumerate() {
        let rel = std::str::from_utf8(&entry.path)
            .map_err(|_| "lma capsule: non-UTF-8 path is not portable")?
            .to_owned();
        // The profile already rejects absolute paths and `..` components at
        // parse time, so this join cannot escape `out_dir`.
        let target = out_dir.join(&rel);
        match entry.file_type {
            ForensicFileType::Directory => {
                std::fs::create_dir_all(&target)?;
                directory_indices.push(entry_index);
                continue;
            }
            ForensicFileType::Regular => {}
            other => {
                return Err(format!("lma capsule: `{rel}` has unsupported type {other:?}").into());
            }
        }

        let method = method_for_capabilities(entry.required_capabilities())?;
        let synthetic = synthetic_of(entry)?;
        let (frame_offset, frame_len) = frame_extent(&index, entry)?;
        file.seek(SeekFrom::Start(frame_offset))?;
        let mut stored = Vec::new();
        stored
            .try_reserve_exact(
                usize::try_from(frame_len)
                    .map_err(|_| "lma capsule: frame too large for this platform")?,
            )
            .map_err(|_| "lma capsule: cannot reserve frame buffer")?;
        Read::by_ref(&mut file)
            .take(frame_len)
            .read_to_end(&mut stored)?;
        if stored.len() as u64 != frame_len {
            return Err(format!("lma capsule: entry `{rel}` frame is truncated").into());
        }
        let original_size = entry.content_len.unwrap_or(0);
        let recovered = match recover(
            &stored,
            method,
            original_size,
            None,
            synthetic.as_ref(),
            Some(out_dir),
            legacy_config,
        ) {
            Ok(bytes) => bytes,
            Err(e) => {
                if verbose {
                    eprintln!("  WARN: {rel}: {e}");
                }
                return Err(format!("lma capsule: cannot recover `{rel}`: {e}").into());
            }
        };

        // The capsule's content_id is the archived file's identity. Checking the
        // recovered bytes against it is what makes extraction verified rather
        // than merely attempted -- and it is the check the LMA path could only
        // do via the manifest's separate sha256 field.
        let expected = entry
            .content_id
            .ok_or_else(|| format!("lma capsule: `{rel}` lacks logical content identity"))?;
        let actual = raw_content_id(&recovered);
        if actual != expected {
            return Err(format!(
                "lma capsule: recovered `{rel}` bytes do not match archived content id"
            )
            .into());
        }

        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        publish_restored_file(&target, &recovered)?;
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

    directory_indices.sort_by(|left, right| {
        index.entries()[*right]
            .path
            .split(|byte| *byte == b'/')
            .count()
            .cmp(
                &index.entries()[*left]
                    .path
                    .split(|byte| *byte == b'/')
                    .count(),
            )
    });
    for entry_index in directory_indices {
        let entry = &index.entries()[entry_index];
        let rel = std::str::from_utf8(&entry.path)
            .map_err(|_| "lma capsule: non-UTF-8 path is not portable")?;
        restore_metadata(&out_dir.join(rel), entry)?;
    }

    summary.cr = if summary.archive_bytes > 0 {
        summary.original_bytes as f64 / summary.archive_bytes as f64
    } else {
        0.0
    };
    Ok(summary)
}

#[cfg(target_os = "linux")]
fn publish_restored_file(target: &Path, bytes: &[u8]) -> Result<(), Error> {
    let parent = target
        .parent()
        .ok_or("lma capsule: restore target has no parent")?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)?;
    staged.write_all(bytes)?;
    staged.as_file().sync_all()?;
    staged
        .persist_noclobber(target)
        .map_err(|error| -> Error { error.error.into() })?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn prepare_restore_directory(out_dir: &Path) -> Result<(), Error> {
    match std::fs::symlink_metadata(out_dir) {
        Ok(metadata) => {
            if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
                return Err(format!(
                    "lma capsule: restore destination is not a real directory: {}",
                    out_dir.display()
                )
                .into());
            }
            if std::fs::read_dir(out_dir)?.next().transpose()?.is_some() {
                return Err(format!(
                    "lma capsule: restore destination is not empty: {}",
                    out_dir.display()
                )
                .into());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "lma capsule: restore destination must already exist: {}",
                out_dir.display()
            )
            .into())
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn restore_metadata(
    target: &Path,
    entry: &semantic_abir_bcs::ForensicEntryMetadata,
) -> Result<(), Error> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(target, std::fs::Permissions::from_mode(entry.mode))?;
    }
    #[cfg(not(target_os = "linux"))]
    if entry.mode != 0 {
        return Err("lma capsule: exact Unix mode restore is unsupported on this platform".into());
    }
    if let Some(stamp) = entry.timestamps[1] {
        let mtime = filetime::FileTime::from_unix_time(stamp.seconds, stamp.nanoseconds);
        filetime::set_file_mtime(target, mtime)?;
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
/// Every stored frame is carried across verbatim rather than re-encoded. This
/// helper recovers original bytes only to verify the manifest's logical-file
/// identity before the independently indexed source extent is streamed into
/// BCS2.
fn recover_for_conversion(
    entry: &crate::lma::ArchiveEntry,
    stored: &[u8],
    tmp_dir: Option<&Path>,
    legacy_config: &LegacyAdapterConfig,
) -> Result<Vec<u8>, Error> {
    match entry.method {
        Method::Store => Ok(stored.to_vec()),
        Method::Zstd => crate::lma::decode_zstd_bounded(
            stored,
            usize::try_from(entry.original_size)
                .map_err(|_| "lma capsule: entry too large for this platform")?,
            &entry.path,
        ),
        Method::Lml => recover(
            stored,
            Method::Lml,
            entry.original_size,
            Some(&entry.sha256),
            entry.synthetic_from.as_ref(),
            tmp_dir,
            legacy_config,
        ),
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
    let config = LegacyAdapterConfig::from_env()?;
    convert_archive_with_legacy_config(lma_path, output_path, zstd_level, &config)
}

/// Convert an LMA archive with an explicit retired-format Adapter policy.
pub fn convert_archive_with_legacy_config(
    lma_path: &Path,
    output_path: &Path,
    // Retained for API/CLI compatibility. Exact migration preserves source
    // frame bytes, so no compression level is applied.
    _zstd_level: i32,
    legacy_config: &LegacyAdapterConfig,
) -> Result<ArchiveSummary, Error> {
    let mut archive = LmaArchive::open(lma_path)?;
    archive.verify_archive_hash()?;
    let manifest = archive.entries().to_vec();
    let directory_mtimes: BTreeMap<String, (Option<u64>, Option<u32>, Option<u32>)> = archive
        .directories()
        .iter()
        .map(|(path, seconds)| (path.clone(), (Some(*seconds), None, None)))
        .collect();
    let mut entries = Vec::with_capacity(manifest.len());
    let mut frame_extents = BTreeMap::<semantic_abir::ContentId, StoredExtent>::new();
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
    let mut largest_frame = 0_u64;

    for entry in &manifest {
        let extent = archive.stored_extent_for(entry)?;
        let stored = archive.read_stored_entry(entry)?;
        let stored_content_id = raw_content_id(&stored);
        let original = recover_for_conversion(entry, &stored, output_path.parent(), legacy_config)
            .map_err(|error| format!("lma capsule: cannot recover `{}`: {error}", entry.path))?;
        let actual_sha = crate::lma::sha256_hex(&original);
        if actual_sha != entry.sha256 {
            return Err(format!(
                "source archive integrity for `{}`: manifest records {} but entry recovers to {}",
                entry.path, entry.sha256, actual_sha
            )
            .into());
        }

        largest_frame = largest_frame.max(extent.len);
        summary.original_bytes += entry.original_size;
        summary.n_files += 1;
        match entry.method {
            Method::Lml => summary.counts_lml += 1,
            Method::Zstd => summary.counts_zstd += 1,
            Method::Store => summary.counts_store += 1,
            _ => {}
        }
        let logical_content_id = raw_content_id(&original);
        let retired_lml1 = matches!(entry.method, Method::Lml) && stored.starts_with(b"LML1");
        entries.push(file_metadata(
            &entry.path,
            stored_content_id,
            extent.len,
            entry.method,
            entry.original_size,
            entry.mtime,
            entry.mtime_nanos,
            entry.mode,
            entry.synthetic_from.as_ref(),
            logical_content_id,
            retired_lml1,
        )?);
        frame_extents.entry(stored_content_id).or_insert(extent);
    }

    let tree = build_metadata(entries, &directory_mtimes, LMA_IMPORT_PLATFORM)?;
    let bounds = writer_bounds(largest_frame);
    let mut output = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(output_path)?;
    let write_result = (|| -> Result<u64, Error> {
        let receipt = write_forensic_tree_streaming(
            &mut output,
            &tree,
            |content_id| {
                let extent = frame_extents.get(&content_id).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("no LMA extent for BCS2 frame {content_id}"),
                    )
                })?;
                let mut source = File::open(lma_path)?;
                source.seek(SeekFrom::Start(extent.offset))?;
                Ok(source.take(extent.len))
            },
            bounds,
        )
        .map_err(|error| format!("lma capsule: streaming encode failed: {error}"))?;
        output.flush()?;
        output.sync_all()?;
        output.seek(SeekFrom::Start(0))?;
        let validated = ForensicFileIndex::open(&mut output, READER_CAPABILITIES, bounds)
            .map_err(|error| format!("lma capsule: output validation failed: {error}"))?;
        if validated.root_content_id() != receipt.root_content_id()
            || output.metadata()?.len() != receipt.artifact_len()
        {
            return Err("lma capsule: output receipt contradicts validated artifact".into());
        }
        Ok(receipt.artifact_len())
    })();
    let archive_bytes = match write_result {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output);
            let _ = std::fs::remove_file(output_path);
            return Err(error);
        }
    };

    summary.archive_bytes = archive_bytes;
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
        assert_eq!(capabilities_for(Method::Store, false, false), 0);
        assert_eq!(capabilities_for(Method::Zstd, false, false), CAP_ZSTD);
        assert_eq!(
            capabilities_for(Method::Lml, false, false),
            CAP_LML_LOSSLESS_V1
        );
        assert_eq!(
            capabilities_for(Method::Lml, false, true),
            CAP_LML_LOSSLESS_V1 | CAP_LML1_LEGACY_MATERIALIZE
        );
        assert_eq!(
            capabilities_for(Method::Lml, true, false),
            CAP_LML_LOSSLESS_V1 | CAP_LMA_SYNTHETIC_REEMIT,
            "a synthetic entry needs re-emit ON TOP OF decoding, not instead of it"
        );
    }

    #[test]
    fn a_reader_advertising_everything_covers_every_method() {
        for method in [Method::Store, Method::Zstd, Method::Lml] {
            for synthetic in [false, true] {
                let mask = capabilities_for(method, synthetic, false);
                assert_eq!(
                    mask & !READER_CAPABILITIES,
                    0,
                    "{method:?}/{synthetic} requires a capability this crate does not advertise"
                );
            }
        }
    }
}
