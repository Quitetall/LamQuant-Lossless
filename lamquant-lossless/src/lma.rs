//! LMA — LamQuant Archive container format.
//!
//! A single `.lma` file preserves an entire directory tree:
//!   - EDF/BDF files → LML lossless codec (domain-specific, ~2.5x CR)
//!   - Text files (.csv, .tse, .lbl, .txt, .json) → zstd-9
//!   - Already-compressed files (.lml, .gz, .zst) → stored as-is
//!
//! Format layout (wire-compatible with Python `lamquant_codec/lma.py`):
//!     [4 bytes]   Magic: b"LMA1"
//!     [4 bytes]   Version: u32 LE (1)
//!     [4 bytes]   Number of entries: u32 LE
//!     [4 bytes]   Manifest length: u32 LE (after zstd compression)
//!     [variable]  Manifest: zstd-compressed JSON
//!     [variable]  Entry payloads (concatenated)
//!     [32 bytes]  Archive SHA-256 (of everything before this)

use sha2::{Digest, Sha256};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const LMA_MAGIC: &[u8; 4] = b"LMA1";
const LMA_VERSION: u32 = 1;
/// Maximum per-entry compressed size we'll trust before allocating
/// a decompression buffer. 4 GB — large enough for real LML containers,
/// small enough to refuse adversarial archives that claim 10 GB+ entries
/// to force OOM. Hoisted to module scope so both `unpack_archive` and
/// `extract_entry` share the same bound (Bible R7 — single source of
/// truth for safety guards).
const MAX_ENTRY_DECOMPRESS_SIZE: u64 = 4 * 1024 * 1024 * 1024;
/// Decompressed (original) entry size ceiling. Clinical EDF recordings
/// can reach 8-10 GB after decode; cap at 16 GB to refuse adversarial
/// archives that claim e.g. 10 MB compressed -> 100 GB decompressed
/// (zstd bomb pattern) before we attempt the decompression.
const MAX_ENTRY_ORIGINAL_SIZE: u64 = 16 * 1024 * 1024 * 1024;
const MAX_MANIFEST_SIZE: usize = 256 * 1024 * 1024; // 256 MB decompressed
/// Compressed-manifest alloc cap (refuse to allocate before decompress when the
/// manifest is zstd). Sits beside MAX_MANIFEST_SIZE as the single source of
/// truth for the chokepoint + any future caller.
const MAX_COMPRESSED_MANIFEST_SIZE: usize = 16 * 1024 * 1024;

// ---- LMA v2 (streaming, footer/EOCD) ----
// v1 writes the manifest at the FRONT (offset 16), so packing must stage every
// payload to a temp file (to learn the offsets the front manifest needs) and
// then COPY temp -> final after the manifest -- 2x the archive size in
// transient disk, which overflows on huge corpora (TUEG ~700 GB -> 1.4 TB
// transient). v2 puts the manifest in a trailing EOCD-style footer so payloads
// stream straight into the final file in one forward pass (offsets accrue as
// written), the manifest + footer go last, and the SHA-256 is computed
// incrementally over the whole stream -- no temp, no seek-back, 1x disk.
//
// v2 layout:
//   header(16):  magic "LMA2" | version(u32 LE = 2) | reserved(8, zero)
//   payloads:    concatenated entry payloads, starting at offset 16
//   manifest:    zstd (or stored) manifest JSON -- identical schema to v1
//   footer(12):  manifest_len_field(u32 LE, top bit = uncompressed)
//                | n_entries(u32 LE) | foot_magic "LFT2"
//   trailer(32): sha256 over [header .. footer], appended last
// The reader locates the manifest from the end: [-32..]=sha, [-36..-32]=
// foot_magic, [-40..-36]=n_entries, [-44..-40]=manifest_len_field, manifest at
// [-44-manifest_len .. -44]; payload section is [16 .. -44-manifest_len]. A
// half-written v2 archive has no valid footer -> reads fail cleanly as
// incomplete (v1 could falsely list a growing file).
const LMA_MAGIC_V2: &[u8; 4] = b"LMA2";
const LMA_VERSION_V2: u32 = 2;
const LMA_FOOT_MAGIC: &[u8; 4] = b"LFT2";
/// v2 fixed footer size, excluding the trailing 32-byte sha: manifest_len(4)
/// + n_entries(4) + foot_magic(4).
const LMA_V2_FOOTER_LEN: u64 = 12;

/// Monotonic counter for tempfile uniqueness across concurrent calls.
/// Combined with `std::process::id()` it gives a per-call tempname
/// that two concurrent processes / threads can't collide on. ADR
/// 0021 Tier 2 audit (N1) introduced this to close the `.tmp.extract`
/// / `.new` / `.bak` collision races identified by
/// defensive-code-validator at lma.rs:2305, 1867, 1923.
static APPEND_TMP_SEQ: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);

/// Returns true if `path` should be rejected as unsafe during archive
/// extraction (Audit-2026-05-11 Fix-#45).
///
/// Covers:
/// - Unix-style absolute paths (`/etc/passwd`)
/// - Windows-style backslash root (`\Windows\System32`)
/// - Parent-directory traversal (`../`, `..\\`, exact `..`)
/// - Windows drive letters (`C:\…`, `C:/…`)
/// - Windows UNC paths (`\\server\share`)
/// - Embedded NUL bytes (filesystem termination smuggling)
/// - Windows Alternate Data Streams (`file:stream`) on first segment
/// - Reserved DOS device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
fn path_is_unsafe(path: &str) -> bool {
    if path.is_empty() {
        return true;
    }
    // NUL byte anywhere.
    if path.contains('\0') {
        return true;
    }
    // Absolute / root.
    if path.starts_with('/') || path.starts_with('\\') {
        return true;
    }
    // UNC: leading "\\".
    if path.starts_with("\\\\") {
        return true;
    }
    // Parent traversal.
    if path == ".."
        || path.contains("../")
        || path.contains("..\\")
        || path.starts_with("..") && (path.len() == 2 || matches!(path.as_bytes()[2], b'/' | b'\\'))
    {
        return true;
    }
    // Windows drive letter (X: at start, where X is ASCII letter).
    let bytes = path.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
        return true;
    }
    // ADS / device names: check each path segment.
    for segment in path.split(['/', '\\']) {
        if segment.is_empty() {
            continue;
        }
        // ADS marker `:` inside a segment (Windows Alternate Data Streams).
        if segment.contains(':') {
            return true;
        }
        // Reserved DOS device names — case-insensitive, with or without ext.
        let stem_upper: String = segment.split('.').next().unwrap_or("").to_ascii_uppercase();
        const RESERVED: &[&str] = &[
            "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
            "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        ];
        if RESERVED.contains(&stem_upper.as_str()) {
            return true;
        }
    }
    false
}

/// Compression method for an archive entry.
///
/// `#[non_exhaustive]` forces every external matcher to add a wildcard
/// arm — adding a new variant in this crate will not silently break
/// downstream consumers, and downstream consumers cannot silently miss
/// handling a new variant. Internal matches in this crate are still
/// exhaustive (the attribute affects only out-of-crate usage).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    /// EDF/BDF → LML lossless codec
    Lml,
    /// Zstd secondary compression
    Zstd,
    /// Stored uncompressed (already-compressed files)
    Store,
}

impl Method {
    fn as_str(&self) -> &'static str {
        match self {
            Method::Lml => "lml",
            Method::Zstd => "secondary",
            Method::Store => "store",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "lml" => Some(Method::Lml),
            "secondary" | "zstd" => Some(Method::Zstd),
            "store" => Some(Method::Store),
            _ => None,
        }
    }
}

/// Choose compression method based on file extension.
pub fn choose_method(path: &Path) -> Method {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "edf" | "bdf" => Method::Lml,
        "lml" | "lmq" | "lma" | "gz" | "zst" | "zip" | "7z" | "png" | "jpg" | "jpeg" | "mp4"
        | "avi" => Method::Store,
        _ => Method::Zstd,
    }
}

/// A single file entry in the archive manifest.
#[derive(Debug, Clone)]
pub struct ArchiveEntry {
    pub path: String,
    pub original_size: u64,
    pub compressed_size: u64,
    pub method: Method,
    pub sha256: String,
    pub offset: u64,
    pub mtime: Option<u64>, // Unix timestamp (seconds since epoch)
    /// Sub-second portion of mtime (0..1e9 ns). Captured alongside
    /// mtime so high-resolution timestamps survive the roundtrip.
    /// Pre-fix archives (no field) deserialize as None and the
    /// extract path defaults to 0 ns.
    pub mtime_nanos: Option<u32>,
    /// Unix file permission bits (e.g. 0o644). None on Windows or when
    /// metadata read failed at archive time. Restored via chmod on
    /// extract (Unix only — ignored on Windows). Catches the
    /// regression where a 600 file extracted as 644.
    pub mode: Option<u32>,
    /// ADR 0023 Track A: when the original file wasn't an EEG container
    /// the codec understands natively (e.g. Bonn `.txt` ASCII), the
    /// ingest pipeline synthesises an EDF in-memory and routes it
    /// through the LML codec. To make the original file's bytes
    /// re-emittable bit-exactly on extract, this field carries the
    /// format tag + the format-specific template (e.g. line endings,
    /// leading whitespace, field width for ASCII int-per-line files).
    /// Missing on regular EDF / BDF / arbitrary-zstd entries.
    pub synthetic_from: Option<SyntheticFromInfo>,
}

/// Describes how an `ArchiveEntry` was synthesised from a non-EDF
/// source file. Carries the format tag + everything needed to
/// reconstruct the original bytes on extract.
///
/// `template_json` is intentionally a raw `serde_json::Value` so the
/// archive layer doesn't have to know every format's template shape —
/// it's opaque payload that the ingest module owns. The `format`
/// string is the discriminant the extract path matches on to pick
/// the right re-emitter.
#[derive(Debug, Clone)]
pub struct SyntheticFromInfo {
    /// Format tag, e.g. `"ascii_int_lines"`. Matches the
    /// `ingest::SyntheticFormat` enum's serialised name.
    pub format: String,
    /// Sample rate used when synthesising the intermediate EDF.
    /// Informational on extract.
    pub sample_rate: f64,
    /// Format-specific template (e.g. `AsciiLinesTemplate` serialised
    /// to JSON). Opaque to lma.rs; consumed by the ingest module.
    pub template_json: serde_json::Value,
}

/// Summary statistics from a pack or unpack operation.
#[derive(Debug)]
pub struct ArchiveSummary {
    pub n_files: usize,
    pub original_bytes: u64,
    pub archive_bytes: u64,
    pub cr: f64,
    pub counts_lml: usize,
    pub counts_zstd: usize,
    pub counts_store: usize,
    pub errors: Vec<(String, String)>,
}

/// Walk a directory tree and collect all files with relative paths.
///
/// Audit-2026-05-11 Fix-#52: emit a warning to stderr for every
/// entry that walkdir refuses (permission denied, broken symlink,
/// race-deleted file). Previously the `.filter_map(|e| e.ok())` chain
/// silently dropped the error — an archive could miss whole
/// subdirectories with no diagnostic to the operator.
fn walk_files(root: &Path) -> Vec<(PathBuf, String)> {
    let mut files = Vec::new();
    let walker = walkdir::WalkDir::new(root).sort_by_file_name();
    for entry_result in walker.into_iter() {
        let entry = match entry_result {
            Ok(e) => e,
            Err(e) => {
                eprintln!(
                    "WARNING: walk_files skipping entry at {}: {}",
                    e.path()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "<unknown>".into()),
                    e
                );
                continue;
            }
        };
        if entry.file_type().is_file() {
            let full = entry.path().to_path_buf();
            if let Ok(rel) = full.strip_prefix(root) {
                let rel_str = rel.to_string_lossy().into_owned();
                files.push((full, rel_str));
            }
        }
    }
    files
}

/// Collect all symlinks under root (recursive). Returned separately so
/// the archive caller can hard-error on their presence — silently
/// dropping symlinks would lose data, and following them would risk
/// duplicating content + escaping the input root.
fn walk_symlinks(root: &Path) -> Vec<(PathBuf, String)> {
    let mut links = Vec::new();
    let walker = walkdir::WalkDir::new(root).sort_by_file_name();
    for entry_result in walker.into_iter() {
        let entry = match entry_result {
            Ok(e) => e,
            Err(e) => {
                eprintln!(
                    "WARNING: walk_symlinks skipping entry at {}: {}",
                    e.path()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "<unknown>".into()),
                    e
                );
                continue;
            }
        };
        if entry.file_type().is_symlink() {
            let full = entry.path().to_path_buf();
            if let Ok(rel) = full.strip_prefix(root) {
                let rel_str = rel.to_string_lossy().into_owned();
                links.push((full, rel_str));
            }
        }
    }
    links
}

/// Collect all directories with relative paths and modification times.
fn walk_dirs(root: &Path) -> Vec<(String, u64)> {
    let mut dirs = Vec::new();
    let walker = walkdir::WalkDir::new(root).sort_by_file_name();
    for entry_result in walker.into_iter() {
        let entry = match entry_result {
            Ok(e) => e,
            Err(e) => {
                eprintln!(
                    "WARNING: walk_dirs skipping entry at {}: {}",
                    e.path()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "<unknown>".into()),
                    e
                );
                continue;
            }
        };
        if entry.file_type().is_dir() {
            let full = entry.path();
            if let Ok(rel) = full.strip_prefix(root) {
                let rel_str = rel.to_string_lossy().into_owned();
                if rel_str.is_empty() {
                    continue;
                } // skip root itself
                let mtime = std::fs::metadata(full)
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                dirs.push((rel_str, mtime));
            }
        }
    }
    dirs
}

/// Cast `bytes: u64` down to `usize` for allocation, refusing the
/// cast if the value doesn't fit. On 32-bit MCU targets `usize` is
/// 32-bit and any value > 2^32 silently truncates -- the audit
/// (Tier 2 N10) flagged that the existing `as usize` casts at
/// extract sites allow a 4 GB compressed_size manifest claim to
/// allocate `compressed_size - 2^32` bytes and the subsequent
/// `read_exact` overruns.
fn bounded_alloc_usize(
    bytes: u64,
    context: &str,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    usize::try_from(bytes).map_err(|_| {
        format!(
            "{} size {} doesn't fit in usize (32-bit MCU target?)",
            context, bytes
        )
        .into()
    })
}

/// Bounded zstd decompression — refuses to inflate past `max_size`.
///
/// `zstd::decode_all` allocates the full decompressed output before
/// any post-decode size check fires; a 16 MB compressed manifest
/// can claim to inflate to several GB and OOM-abort before any
/// caller-level check runs. This helper uses the streaming decoder
/// with a `Read::take(max_size + 1)` budget: if the decoder
/// produces more than `max_size` bytes the result is rejected with
/// a structured error.
///
/// ADR 0021 Tier 2 audit (N5): closes the zstd-bomb attack surface
/// at manifest decompression (lma.rs:2207/2607/2748) and per-entry
/// `Method::Zstd` extraction (lma.rs:1665/2373/2670).
fn decode_zstd_bounded(
    input: &[u8],
    max_size: usize,
    context: &str,
) -> Result<alloc::vec::Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Read;
    let mut dec = zstd::stream::read::Decoder::new(input)
        .map_err(|e| format!("zstd decode init ({}): {}", context, e))?;
    let mut out = alloc::vec::Vec::new();
    // Read at most max_size + 1 bytes. If the decoder produces
    // exactly max_size + 1 it means the original was longer than
    // the cap; reject.
    let limit = (max_size as u64).saturating_add(1);
    let mut limited = (&mut dec).take(limit);
    limited
        .read_to_end(&mut out)
        .map_err(|e| format!("zstd decode read ({}): {}", context, e))?;
    if out.len() > max_size {
        return Err(format!(
            "zstd decode ({}): decompressed size > {} byte cap; refusing as zstd bomb",
            context, max_size
        )
        .into());
    }
    Ok(out)
}

/// Resolved on-disk layout of an LMA archive: the parsed manifest
/// entries plus the byte bounds of the payload section. This is the
/// SINGLE place where the v1-vs-v2 layout difference lives — every
/// reader funnels through `read_lma_index` and then works purely in
/// terms of `entries`, `payload_base`, and `payload_end`. Entry
/// offsets in the manifest are relative to `payload_base`, so the
/// absolute byte position of an entry's payload is
/// `payload_base + entry.offset`. The payload section occupies
/// `[payload_base .. payload_end]`; an entry's
/// `offset + compressed_size` must fit within
/// `payload_end - payload_base`.
struct LmaIndex {
    entries: Vec<ArchiveEntry>,
    payload_base: u64,
    payload_end: u64,
    /// Directory (path, mtime-seconds) pairs from the manifest's
    /// `directories` array, used by `unpack_archive` to restore dir
    /// mtimes after extraction. Empty for manifests that carry no
    /// directory metadata (e.g. append-built archives).
    directories: Vec<(String, u64)>,
}

/// The ONE dispatch chokepoint for the LMA on-disk layout. Reads the
/// 16-byte header, sniffs the magic+version, and locates + decodes the
/// manifest for both wire formats:
///
///   - v2 (`LMA2` / version 2): streaming/EOCD layout. The manifest +
///     12-byte footer + 32-byte sha live at the END of the file. We
///     seek to `file_size - 32 - 12`, read the footer, validate its
///     `LFT2` magic, then read the compressed manifest immediately
///     before it. Payloads start at offset 16; the payload section
///     ends where the manifest begins.
///   - v1 (`LMA1` / version <= 1): legacy front-manifest layout. The
///     manifest sits right after the 16-byte header; payloads follow
///     it. payload_base = 16 + manifest_len, payload_end = file_size
///     - 32 (sha trailer). v1 read retained for migration; remove once
///     all archives are v2.
///
/// The manifest *decode* (alloc caps, top-bit uncompressed flag,
/// zstd-vs-raw, serde parse, `parse_manifest_entries`) is byte-
/// identical between the two versions once the manifest bytes + flag
/// are located, so it lives in a single shared block below — only the
/// positioning differs per version.
fn read_lma_index<R: Read + Seek>(
    f: &mut R,
    file_size: u64,
) -> Result<LmaIndex, Box<dyn std::error::Error + Send + Sync>> {
    // Minimum valid archive: 16 (header) + 0 (manifest) + 32 (sha) = 48.
    // (v2 also carries a 12-byte footer, but the 48-byte floor is the
    // shared guard kept from the v1 readers; a real v2 archive is
    // always >= 60 bytes and the footer read below bounds-checks
    // itself.)
    if file_size < 48 {
        return Err(format!(
            "Archive too small ({} bytes, minimum 48)",
            file_size
        )
        .into());
    }

    // Header: [4 magic][4 version][8 ...]. For v1 the trailing 8 bytes
    // are [n_entries][manifest_len_field]; for v2 they are reserved
    // (zero). We read magic + version first, then dispatch.
    f.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);

    // Per-version positioning yields (manifest_bytes, uncompressed_flag,
    // payload_base, payload_end). The manifest decode is shared below.
    // (alloc cap MAX_COMPRESSED_MANIFEST_SIZE is module-level.)
    let (manifest_payload, manifest_uncompressed, payload_base, payload_end): (
        Vec<u8>,
        bool,
        u64,
        u64,
    ) = if &magic == LMA_MAGIC_V2 && version == LMA_VERSION_V2 {
        // ── v2: trailing footer + manifest ──────────────────────────
        // Layout from the end: [-32..] sha, [-44..-32] footer (12),
        // manifest immediately before the footer.
        let foot_pos = file_size - 32 - LMA_V2_FOOTER_LEN;
        f.seek(SeekFrom::Start(foot_pos))?;
        let mut footer = [0u8; LMA_V2_FOOTER_LEN as usize];
        f.read_exact(&mut footer)?;
        if &footer[8..12] != LMA_FOOT_MAGIC {
            return Err("incomplete/corrupt v2 archive — truncated?".into());
        }
        let manifest_len_field = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
        let manifest_uncompressed = (manifest_len_field & 0x8000_0000) != 0;
        let manifest_len = (manifest_len_field & 0x7FFF_FFFF) as usize;

        let alloc_cap = if manifest_uncompressed {
            MAX_MANIFEST_SIZE
        } else {
            MAX_COMPRESSED_MANIFEST_SIZE
        };
        if manifest_len > alloc_cap {
            return Err(format!(
                "Manifest length {} (uncompressed={}) exceeds cap {} \
                 — refusing to allocate",
                manifest_len, manifest_uncompressed, alloc_cap,
            )
            .into());
        }
        // manifest occupies [foot_pos - manifest_len .. foot_pos].
        let manifest_start = foot_pos.checked_sub(manifest_len as u64).ok_or_else(
            || -> Box<dyn std::error::Error + Send + Sync> {
                "corrupt v2 archive: manifest_len larger than file".into()
            },
        )?;
        // payload_base = 16 (header). The manifest must not overlap the
        // header.
        if manifest_start < 16 {
            return Err(format!(
                "corrupt v2 archive: manifest starts at {} (< payload_base 16)",
                manifest_start
            )
            .into());
        }
        f.seek(SeekFrom::Start(manifest_start))?;
        let mut manifest_payload = vec![0u8; manifest_len];
        f.read_exact(&mut manifest_payload)?;
        (manifest_payload, manifest_uncompressed, 16u64, manifest_start)
    } else if &magic == LMA_MAGIC && version <= LMA_VERSION {
        // ── v1: front manifest ──────────────────────────────────────
        // v1 read retained for migration; remove once all archives are v2.
        // Header trailing 8 bytes = [n_entries][manifest_len_field].
        f.read_exact(&mut buf4)?; // n_entries — derivable from manifest
        f.read_exact(&mut buf4)?;
        let manifest_len_field = u32::from_le_bytes(buf4);
        let manifest_uncompressed = (manifest_len_field & 0x8000_0000) != 0;
        let manifest_len = (manifest_len_field & 0x7FFF_FFFF) as usize;

        let alloc_cap = if manifest_uncompressed {
            MAX_MANIFEST_SIZE
        } else {
            MAX_COMPRESSED_MANIFEST_SIZE
        };
        if manifest_len > alloc_cap {
            return Err(format!(
                "Manifest length {} (uncompressed={}) exceeds cap {} \
                 — refusing to allocate",
                manifest_len, manifest_uncompressed, alloc_cap,
            )
            .into());
        }
        // Manifest sits at offset 16 (right after the header).
        let mut manifest_payload = vec![0u8; manifest_len];
        f.read_exact(&mut manifest_payload)?;
        let payload_base = 16u64 + manifest_len as u64;
        let payload_end = file_size.saturating_sub(32);
        (manifest_payload, manifest_uncompressed, payload_base, payload_end)
    } else {
        return Err(format!(
            "Not an LMA archive / unsupported version (magic: {:?}, version: {})",
            magic, version
        )
        .into());
    };

    // ── shared manifest decode (identical for v1 + v2) ──────────────
    let manifest_raw = if manifest_uncompressed {
        manifest_payload
    } else {
        decode_zstd_bounded(&manifest_payload, MAX_MANIFEST_SIZE, "manifest")?
    };
    if manifest_raw.len() > MAX_MANIFEST_SIZE {
        return Err(format!(
            "Manifest too large after decompression ({} bytes, max {})",
            manifest_raw.len(),
            MAX_MANIFEST_SIZE
        )
        .into());
    }
    let manifest_json: serde_json::Value = serde_json::from_slice(&manifest_raw)?;
    let entries = parse_manifest_entries(&manifest_json)?;
    // Directory mtimes (used by unpack_archive). Absent in array-shaped
    // (oldest) manifests and in append-built manifests.
    let directories: Vec<(String, u64)> = manifest_json
        .get("directories")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|d| {
                    let path = d.get("path")?.as_str()?.to_string();
                    let mtime = d.get("mtime")?.as_u64()?;
                    Some((path, mtime))
                })
                .collect()
        })
        .unwrap_or_default();

    // Sanity: payload_base must not exceed payload_end for any
    // well-formed archive (an empty-payload archive has them equal).
    if payload_base > payload_end {
        return Err(format!(
            "Corrupt archive: payload_base ({}) exceeds payload_end ({})",
            payload_base, payload_end
        )
        .into());
    }

    Ok(LmaIndex {
        entries,
        payload_base,
        payload_end,
        directories,
    })
}

/// SHA-256 hex digest of a byte slice.
fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Decode LML bytes back to original EDF bytes (bit-exact reconstruction).
///
/// Writes the LML payload to a temp file, reads it with container::read_file,
/// parses the metadata to recover the original EDF header and non-EEG channels,
/// then interleaves everything back into the original EDF byte layout.
/// If `original_size` is provided, pads output to match (preserves trailing zeros).
fn decode_lml_to_edf(
    lml_bytes: &[u8],
    original_size: Option<u64>,
    tmp_dir_hint: Option<&Path>,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;

    // Write LML to temp, read back signal + metadata. Caller-supplied
    // `tmp_dir_hint` co-locates the tempfile on the output volume to
    // avoid the tmpfs ENOSPC failure mode that bit the corpus-wide
    // unpack path (a 30 GB tmpfs caps out before a multi-GB single
    // EDF finishes decoding). `None` falls back to $TMPDIR for
    // single-file flows where the tempfile is short-lived + small.
    let tmp = match tmp_dir_hint {
        Some(dir) => tempfile::Builder::new()
            .prefix(".lamquant-decode-")
            .tempfile_in(dir)?,
        None => tempfile::NamedTempFile::new()?,
    };
    let tmp_path = tmp.path().to_path_buf();
    std::fs::write(&tmp_path, lml_bytes)?;

    let (signal, meta_str) =
        crate::container::read_file(&tmp_path).map_err(|e| format!("LML decode failed: {}", e))?;
    drop(tmp);

    let meta: serde_json::Value = serde_json::from_str(&meta_str)?;

    // Extract the preserved EDF header
    let edf_header_b64 = meta
        .get("edf_header")
        .and_then(|v| v.as_str())
        .ok_or("LML metadata missing edf_header — cannot reconstruct EDF")?;

    if edf_header_b64.is_empty() {
        return Err("LML metadata has empty edf_header — cannot reconstruct EDF".into());
    }

    let hdr_compressed = b64.decode(edf_header_b64)?;
    let edf_header = zstd::decode_all(hdr_compressed.as_slice())?;

    if edf_header.len() < 256 {
        return Err(format!("EDF header too short ({} bytes)", edf_header.len()).into());
    }

    // Verify EDF header SHA-256 if provenance field is present.
    // ADR 0021 Tier 2 follow-up: legacy files without the field
    // are accepted but with a stderr WARNING so the operator
    // knows provenance is missing. Pre-fix accepted silently.
    if let Some(expected) = meta.get("edf_header_sha256").and_then(|v| v.as_str()) {
        let actual = {
            let mut h = sha2::Sha256::new();
            h.update(edf_header.as_slice());
            format!("{:x}", h.finalize())
        };
        if actual != expected {
            return Err(format!(
                "EDF header SHA-256 mismatch: expected {}, got {}. Header tampered or corrupted.",
                expected, actual
            )
            .into());
        }
    } else {
        eprintln!(
            "  WARNING: lma decode: this LML metadata lacks `edf_header_sha256` \
             (pre-provenance archive); EDF header is reconstructed but cannot be \
             tamper-verified against pack-time SHA."
        );
    }

    // Parse key fields from header
    let n_signals: usize = std::str::from_utf8(&edf_header[252..256])
        .unwrap_or("0")
        .trim()
        .parse()
        .unwrap_or(0);
    let is_bdf = edf_header[0] == 0xFF;
    let bps: usize = if is_bdf { 3 } else { 2 }; // bytes per sample

    // Get channel layout from metadata
    let n_data_records = meta
        .get("n_data_records")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let all_ns: Vec<usize> = meta
        .get("all_ns_per_rec")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default();

    if all_ns.len() != n_signals {
        return Err(format!(
            "all_ns_per_rec length ({}) != n_signals ({})",
            all_ns.len(),
            n_signals
        )
        .into());
    }

    let eeg_idx: Vec<usize> = meta
        .get("eeg_channel_indices")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default();

    // Build EEG index lookup
    let mut eeg_idx_map = std::collections::HashMap::new();
    for (j, &ch) in eeg_idx.iter().enumerate() {
        eeg_idx_map.insert(ch, j);
    }

    // Decompress non-EEG channel data — Audit-2026-05-11 Fix-#38:
    // propagate base64/zstd errors instead of silently dropping the
    // channel. Previously three nested `if let Ok` discarded any decode
    // failure, leaving non_eeg_data empty for that channel; downstream
    // the rebuild would zero-fill that slot, producing a clinical EDF
    // with silent data loss in annotation / status / triggers channels.
    let mut non_eeg_data: std::collections::HashMap<usize, Vec<u8>> =
        std::collections::HashMap::new();
    if let Some(non_eeg_obj) = meta.get("non_eeg_channels").and_then(|v| v.as_object()) {
        for (key, val) in non_eeg_obj {
            let ch_idx: usize =
                key.parse()
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        format!("non_eeg_channels key '{key}' is not a valid channel index: {e}")
                            .into()
                    })?;
            let encoded =
                val.as_str()
                    .ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                        format!("non_eeg_channels[{ch_idx}] is not a string").into()
                    })?;
            let ch_compressed =
                b64.decode(encoded)
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        format!("non_eeg_channels[{ch_idx}] base64 decode failed: {e}").into()
                    })?;
            let ch_bytes = zstd::decode_all(ch_compressed.as_slice()).map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("non_eeg_channels[{ch_idx}] zstd decode failed: {e}").into()
                },
            )?;
            non_eeg_data.insert(ch_idx, ch_bytes);
        }
    }

    // Rebuild data records
    let total_per_rec: usize = all_ns.iter().sum();
    let data_size = n_data_records * total_per_rec * bps;
    let mut data_block = vec![0u8; data_size];

    let mode_ns = if let Some(&first_eeg) = eeg_idx.first() {
        all_ns[first_eeg]
    } else {
        return Err("No EEG channels found".into());
    };

    for r in 0..n_data_records {
        let mut pos: usize = 0;
        for ch in 0..n_signals {
            let ns = all_ns[ch];
            let rec_offset = r * total_per_rec * bps + pos * bps;

            if let Some(&j) = eeg_idx_map.get(&ch) {
                // EEG channel: take samples from signal
                let sig_start = r * mode_ns;
                for s in 0..ns {
                    let sample_idx = sig_start + s;
                    let val = if sample_idx < signal[j].len() {
                        signal[j][sample_idx]
                    } else {
                        0
                    };

                    let byte_offset = rec_offset + s * bps;
                    if is_bdf {
                        // int24 LE
                        let v = if val < 0 { val + (1 << 24) } else { val } as u32;
                        data_block[byte_offset] = (v & 0xFF) as u8;
                        data_block[byte_offset + 1] = ((v >> 8) & 0xFF) as u8;
                        data_block[byte_offset + 2] = ((v >> 16) & 0xFF) as u8;
                    } else {
                        // int16 LE
                        let v = val as i16;
                        let bytes = v.to_le_bytes();
                        data_block[byte_offset] = bytes[0];
                        data_block[byte_offset + 1] = bytes[1];
                    }
                }
            } else {
                // Non-EEG channel: restore from preserved data
                if let Some(ch_bytes) = non_eeg_data.get(&ch) {
                    let chunk_size = ns * bps;
                    let start = r * chunk_size;
                    let end = start + chunk_size;
                    if end <= ch_bytes.len() && rec_offset + chunk_size <= data_block.len() {
                        data_block[rec_offset..rec_offset + chunk_size]
                            .copy_from_slice(&ch_bytes[start..end]);
                    }
                }
            }
            pos += ns;
        }
    }

    // Append trailing partial record data if present — Audit-2026-05-11
    // Fix-#39: propagate decode errors. Previously two nested `if let
    // Ok` left `trailing` empty on any decode failure, silently dropping
    // the last partial record. Clinical EDFs that end mid-record (very
    // common for live recordings stopped on event) lose data without
    // diagnostic.
    //
    // Encoder emits `"trailing_data": ""` (empty string) when there is
    // no trailing partial record. That MUST decode to empty Vec, not Err.
    // Only an actually-non-empty string that fails to decode is an error.
    let mut trailing = Vec::new();
    if let Some(trail_b64) = meta.get("trailing_data").and_then(|v| v.as_str()) {
        if !trail_b64.is_empty() {
            let trail_compressed =
                b64.decode(trail_b64)
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        format!("trailing_data base64 decode failed: {e}").into()
                    })?;
            trailing = zstd::decode_all(trail_compressed.as_slice()).map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("trailing_data zstd decode failed: {e}").into()
                },
            )?;
        }
    }

    // Assemble: header + data records + trailing partial record
    let mut edf_out = Vec::with_capacity(edf_header.len() + data_block.len() + trailing.len());
    edf_out.extend_from_slice(&edf_header);
    edf_out.extend_from_slice(&data_block);
    edf_out.extend_from_slice(&trailing);

    // Pad to original size if still short (e.g., trailing zeros not captured)
    if let Some(orig_sz) = original_size {
        let orig = orig_sz as usize;
        if edf_out.len() < orig {
            edf_out.resize(orig, 0);
        }
    }
    Ok(edf_out)
}

/// Encode a single EDF file to LML bytes in memory.
/// Returns Ok(lml_bytes) or Err with fallback to zstd.
///
/// `tmp_dir_hint` co-locates the LML scratch tempfile on the
/// output volume to dodge the tmpfs ENOSPC failure mode on
/// pathological single-EDF inputs (e.g. multi-day ICU monitor).
/// `None` falls back to $TMPDIR for single-file callers where
/// the file is small + the temp is short-lived.
///
/// ADR 0023 Track A-3: Sample rate hint for ingest-synthesised EDFs.
/// Bonn dataset stems match `[A-Z]\d{3}` (Z001, F042, N100, etc.) and
/// the original paper specifies 173.61 Hz. Other filename patterns
/// fall through to 256.0 Hz — a generic value the EDF reader will
/// accept without changing the synthesised file's record duration.
fn guess_synth_sample_rate(filename: Option<&str>) -> f64 {
    let Some(name) = filename else {
        return 256.0;
    };
    let stem = std::path::Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    // Match `[A-Z]\d{3}` — the Bonn filename pattern.
    let mut chars = stem.chars();
    let first_ok = chars
        .next()
        .map(|c| c.is_ascii_uppercase())
        .unwrap_or(false);
    let rest: Vec<char> = chars.collect();
    let rest_ok = rest.len() == 3 && rest.iter().all(|c| c.is_ascii_digit());
    if first_ok && rest_ok {
        173.61
    } else {
        256.0
    }
}

/// ADR 0023 Track A-3 — `pack_archive` calls this for files that
/// `choose_method` routed to `Method::Zstd` (the non-EDF cascade).
/// Returns `Some((lml_bytes, info))` only when ALL of the following:
///   1. A detector recognises the file's format.
///   2. Parse + synth + encode succeed.
///   3. The resulting LML bytes are **strictly smaller** than the
///      zstd-of-original output (`zstd_bytes_len`), so the ingest
///      path never makes the archive larger than the legacy zstd
///      fallback.
/// `None` means caller keeps the zstd path. Hard errors bubble through
/// `eprintln!` warnings — they're best-effort, not fatal.
fn try_ingest_to_lml(
    raw: &[u8],
    source_filename: Option<&str>,
    tmp_dir: &Path,
    zstd_bytes_len: usize,
) -> Option<(Vec<u8>, SyntheticFromInfo)> {
    // Detector 1: ASCII int-per-line (Bonn-class).
    if let Some(template) = crate::ingest::detect_ascii_int_lines(raw) {
        let samples = match crate::ingest::parse_ascii_int_lines(raw, &template) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "  lma ingest: ascii_int_lines parse failed ({}); falling back to zstd",
                    e
                );
                return None;
            }
        };
        if samples.is_empty() {
            return None;
        }
        let sample_rate = guess_synth_sample_rate(source_filename);
        let edf_bytes = crate::ingest::synth_single_channel_edf(&samples, sample_rate);

        // Spill to a tempfile co-located with the archive's tmp_dir
        // so the synth EDF doesn't land on /tmp tmpfs (commit 5769562).
        let tmp_file = match tempfile::Builder::new()
            .prefix("lma_ingest_synth_")
            .suffix(".edf")
            .tempfile_in(tmp_dir)
        {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  lma ingest: tempfile_in {}: {}", tmp_dir.display(), e);
                return None;
            }
        };
        if let Err(e) = std::fs::write(tmp_file.path(), &edf_bytes) {
            eprintln!("  lma ingest: write synth EDF: {}", e);
            return None;
        }

        let lml_bytes = match encode_edf_to_lml(tmp_file.path(), Some(tmp_dir)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "  lma ingest: encode_edf_to_lml on synth EDF failed: {}; falling back to zstd",
                    e
                );
                return None;
            }
        };

        // No-regression rule (ADR 0023): only adopt the ingest path
        // when it strictly beats zstd. Equal sizes fall back to zstd
        // so the archive byte-stream stays minimally different from
        // historical behaviour when both paths produce the same CR.
        if lml_bytes.len() >= zstd_bytes_len {
            return None;
        }

        let info = SyntheticFromInfo {
            format: "ascii_int_lines".into(),
            sample_rate,
            template_json: template.to_json(),
        };
        return Some((lml_bytes, info));
    }

    None
}

/// ADR 0023 Track A-3 — `unpack_archive` calls this for entries whose
/// `synthetic_from` field is populated. Takes the LML-reconstructed
/// EDF bytes and the format info, returns the original (non-EDF)
/// byte sequence. Roundtrip integrity is validated by the existing
/// SHA-256 check against `entry.sha256`.
fn re_emit_synthetic(
    edf_bytes: &[u8],
    info: &SyntheticFromInfo,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    if info.format != "ascii_int_lines" {
        return Err(format!(
            "lma re-emit: unknown synthetic_from.format `{}` (this LMA was written by a newer \
             codec; upgrade to extract)",
            info.format
        )
        .into());
    }
    let template = crate::ingest::AsciiLinesTemplate::from_json(&info.template_json)
        .map_err(|e| format!("lma re-emit: template parse: {}", e))?;
    // Extract i16 samples from the synthesised EDF. Layout is fixed
    // by `ingest::synth_single_channel_edf` (see SYNTH_EDF_HEADER_LEN
    // for the authoritative constant); changing the synth layout
    // means adding a new SyntheticFormat variant, not bumping this
    // offset.
    if edf_bytes.len() < crate::ingest::SYNTH_EDF_HEADER_LEN {
        return Err(format!(
            "lma re-emit: synth EDF only {} bytes; need >= {}",
            edf_bytes.len(),
            crate::ingest::SYNTH_EDF_HEADER_LEN
        )
        .into());
    }
    let sample_bytes = &edf_bytes[crate::ingest::SYNTH_EDF_HEADER_LEN..];
    if sample_bytes.len() % 2 != 0 {
        return Err("lma re-emit: synth EDF sample section has odd length".into());
    }
    let mut samples = Vec::with_capacity(sample_bytes.len() / 2);
    for chunk in sample_bytes.chunks_exact(2) {
        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(crate::ingest::render_ascii_int_lines(&samples, &template))
}

/// ADR 0023 Track B3 — pick `window_size` based on the signal length
/// + sample rate so short single-record sources (Bonn EEG, gait
/// snippets, EEG MMIDB tasks) land in a single window with full LPC
/// context, while long clinical clips keep the legacy chunking that
/// `cr_regression` baselines were measured against.
///
/// Reference-rate convention matches `container::write_file`:
/// `actual_window = window_size × sample_rate / 250.0`.
///
/// **Three regimes (no internal branching cliff at the LPC level —
/// the codec self-throttles via AIC + Burg's N/8 rule):**
///
///   1. `ref_signal ≤ DEFAULT` (≤ 2500 ref samples): one window of
///      `DEFAULT` is plenty; `DEFAULT` covers signals up to 10 s at
///      the 250 Hz reference. No tuning needed.
///   2. `DEFAULT < ref_signal ≤ MAX_AUTO_WINDOW` (≤ 16384 ref): one
///      window of `ref_signal` exactly, so LPC sees the entire
///      record in a single context. Bonn (ref ≈ 5900), EEG MMIDB
///      tasks, BCI segments all fit here.
///   3. `ref_signal > MAX_AUTO_WINDOW`: default chunking. This is the
///      regime cr_regression's three reference EDFs sit in
///      (chb01_01 ≈ 900 K ref samples, S013R01 / S106R05 ≈ 2500 ref
///      so they fit at the regime 1/2 boundary unchanged).
///
/// **Transition smoothness**: the boundary between regimes 2 and 3
/// (`ref_signal = MAX_AUTO_WINDOW = 16384`) is the only cliff. At
/// the boundary, regime 2 picks `ws = 16384` (single 16384-sample
/// window), regime 3 picks `ws = 2500` (chunks the signal into
/// `ceil(ref_signal / 2500) ≥ 7` windows). The CR delta across
/// that boundary is tested by `tests/window_size_transition.rs`,
/// which scans signal lengths 1k..32k samples and asserts the
/// encoded size doesn't dip pathologically at any transition.
fn auto_window_size(signal_len_per_ch: usize, sample_rate: f64) -> usize {
    const DEFAULT_WINDOW_SIZE: usize = 2500;
    /// Maximum signal length (in 250 Hz reference samples) that
    /// qualifies for single-window encoding. Bumped from 8192 → 16384
    /// to cover more single-record sources (BCI tasks, gait clips,
    /// 65 s @ 250 Hz). Larger thresholds cost peak per-window RAM
    /// but win CR on records that fit cleanly inside.
    const MAX_AUTO_WINDOW: usize = 16384;
    const REF_RATE: f64 = 250.0;

    if sample_rate <= 0.0 || !sample_rate.is_finite() || signal_len_per_ch == 0 {
        return DEFAULT_WINDOW_SIZE;
    }
    let ref_signal = ((signal_len_per_ch as f64) * REF_RATE / sample_rate).ceil() as usize;
    // `ref_signal == 0` is unreachable past the guards above (positive
    // signal_len_per_ch * positive REF_RATE / positive sample_rate
    // ceils to >= 1).
    if ref_signal > MAX_AUTO_WINDOW {
        // Long signal — preserve the pre-B3 default. cr_regression
        // reference EDFs (clinical multi-hour clips) land here and
        // produce byte-equal output to the pre-B3 codec.
        DEFAULT_WINDOW_SIZE
    } else {
        // Short or medium signal — pick the smallest window that
        // captures the entire signal in one packet, never below
        // `DEFAULT_WINDOW_SIZE` (a sub-2500 window risks under-
        // utilising LPC on tiny clips). Capped at `u16::MAX` so the
        // LML packet's `t` field doesn't overflow.
        ref_signal.max(DEFAULT_WINDOW_SIZE).min(u16::MAX as usize)
    }
}

fn encode_edf_to_lml(
    edf_path: &Path,
    tmp_dir_hint: Option<&Path>,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let edf_data = crate::edf::read_edf(edf_path)?;
    let sample_rate = edf_data.sample_rate;

    if !(sample_rate > 0.0 && sample_rate.is_finite()) {
        return Err(format!("Invalid sample_rate: {}", sample_rate).into());
    }
    // ADR 0023 Track B3 — auto-tune from signal length + sample rate.
    // Bonn (4097 samples at 173.61 Hz) now picks ~5900 instead of the
    // hard-coded 2500, so a single LPC pass sees the whole signal
    // instead of splitting it across 3 chunks. Long clinical EDFs
    // keep the pre-B3 default for byte-equality with the cr_regression
    // reference set.
    let signal_len = edf_data.signal.first().map(|ch| ch.len()).unwrap_or(0);
    let window_size = auto_window_size(signal_len, sample_rate);

    // Build metadata JSON (same as CLI encode)
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;

    // Compute signal SHA-256 for integrity verification
    let signal_sha = {
        let mut h = sha2::Sha256::new();
        for ch in &edf_data.signal {
            for &sample in ch {
                h.update(sample.to_le_bytes());
            }
        }
        format!("{:x}", h.finalize())
    };

    // Hash the raw EDF header for FDA provenance — SHA-256 of pre-zstd bytes.
    let edf_header_sha = {
        let mut h = sha2::Sha256::new();
        h.update(edf_data.raw_header.as_slice());
        format!("{:x}", h.finalize())
    };

    let hdr_compressed = zstd::encode_all(edf_data.raw_header.as_slice(), 9)
        .map_err(|e| format!("zstd compress EDF header: {}", e))?;
    let hdr_b64 = b64.encode(&hdr_compressed);

    let encoder_version = format!("lml/{}", env!("CARGO_PKG_VERSION"));

    let mut non_eeg_json = String::from("{");
    for (i, (ch_idx, ch_bytes)) in edf_data.non_eeg_data.iter().enumerate() {
        let compressed = zstd::encode_all(ch_bytes.as_slice(), 9)
            .map_err(|e| format!("zstd compress non-EEG ch {}: {}", ch_idx, e))?;
        let encoded = b64.encode(&compressed);
        if i > 0 {
            non_eeg_json.push(',');
        }
        non_eeg_json.push_str(&format!("\"{}\":\"{}\"", ch_idx, encoded));
    }
    non_eeg_json.push('}');

    // Preserve trailing partial record data
    let trailing_b64 = if !edf_data.trailing_data.is_empty() {
        let compressed = zstd::encode_all(edf_data.trailing_data.as_slice(), 9)
            .map_err(|e| format!("zstd compress trailing data: {}", e))?;
        b64.encode(&compressed)
    } else {
        String::new()
    };

    let channels_json: Vec<String> = edf_data
        .channels
        .iter()
        .map(|c| format!("\"{}\"", c.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect();
    let all_labels_json: Vec<String> = edf_data
        .all_labels
        .iter()
        .map(|c| format!("\"{}\"", c.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect();
    let eeg_idx_json: Vec<String> = edf_data.eeg_indices.iter().map(|i| i.to_string()).collect();
    // Audit-2026-05-11 Fix-#25: assert finiteness before serialising
    // float fields. `format!("{}", f64::NAN)` produces literal "NaN"
    // and `f64::INFINITY` produces "inf" — both invalid JSON that
    // would break any downstream consumer (Python decoder, manifest
    // viewer, future C reader). Upstream HIGH-G5 propagates phys_min/
    // max parse errors so non-finite values shouldn't reach here, but
    // defend explicitly so a future caller cannot inject NaN through
    // a different code path.
    for (i, &v) in edf_data.phys_min.iter().enumerate() {
        if !v.is_finite() {
            return Err(
                format!("phys_min[{i}] = {v} is not finite — cannot serialise to JSON").into(),
            );
        }
    }
    for (i, &v) in edf_data.phys_max.iter().enumerate() {
        if !v.is_finite() {
            return Err(
                format!("phys_max[{i}] = {v} is not finite — cannot serialise to JSON").into(),
            );
        }
    }
    let phys_min_json: Vec<String> = edf_data.phys_min.iter().map(|v| format!("{}", v)).collect();
    let phys_max_json: Vec<String> = edf_data.phys_max.iter().map(|v| format!("{}", v)).collect();
    let dig_min_json: Vec<String> = edf_data.dig_min.iter().map(|v| v.to_string()).collect();
    let dig_max_json: Vec<String> = edf_data.dig_max.iter().map(|v| v.to_string()).collect();
    let ns_json: Vec<String> = edf_data
        .all_ns_per_rec
        .iter()
        .map(|v| v.to_string())
        .collect();

    let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");

    let meta = format!(
        concat!(
            "{{",
            "\"source_file\":\"{}\",",
            "\"format\":\"{}\",",
            "\"channels\":[{}],",
            "\"n_channels\":{},",
            "\"n_signals_total\":{},",
            "\"sample_rate\":{},",
            "\"n_data_records\":{},",
            "\"record_duration\":{},",
            "\"phys_min\":[{}],",
            "\"phys_max\":[{}],",
            "\"dig_min\":[{}],",
            "\"dig_max\":[{}],",
            "\"phys_dim\":\"{}\",",
            "\"all_labels\":[{}],",
            "\"all_ns_per_rec\":[{}],",
            "\"eeg_channel_indices\":[{}],",
            "\"duration_s\":{},",
            "\"patient_id\":\"{}\",",
            "\"recording_info\":\"{}\",",
            "\"startdate\":\"{}\",",
            "\"edf_header\":\"{}\",",
            "\"edf_header_sha256\":\"{}\",",
            "\"encoder_version\":\"{}\",",
            "\"non_eeg_channels\":{},",
            "\"signal_sha256\":\"{}\",",
            "\"trailing_data\":\"{}\",",
            "\"trailing_data_size\":{}",
            "}}"
        ),
        escape(&edf_data.source_file),
        edf_data.format,
        channels_json.join(","),
        edf_data.n_channels,
        edf_data.n_signals_total,
        edf_data.sample_rate,
        edf_data.n_data_records,
        edf_data.record_duration,
        phys_min_json.join(","),
        phys_max_json.join(","),
        dig_min_json.join(","),
        dig_max_json.join(","),
        escape(&edf_data.phys_dim),
        all_labels_json.join(","),
        ns_json.join(","),
        eeg_idx_json.join(","),
        edf_data.duration_s,
        escape(&edf_data.patient_id),
        escape(&edf_data.recording_info),
        escape(&edf_data.startdate),
        hdr_b64,
        edf_header_sha,
        escape(&encoder_version),
        non_eeg_json,
        signal_sha,
        trailing_b64,
        edf_data.trailing_data.len(),
    );

    // Write LML to temp file, read back bytes. Caller-supplied
    // `tmp_dir_hint` keeps the tempfile on the output volume; see
    // function docstring for rationale.
    let tmp = match tmp_dir_hint {
        Some(dir) => tempfile::Builder::new()
            .prefix(".lamquant-encode-")
            .tempfile_in(dir)?,
        None => tempfile::NamedTempFile::new()?,
    };
    let tmp_path = tmp.path().to_path_buf();

    crate::container::write_file(
        &tmp_path,
        &edf_data.signal,
        edf_data.sample_rate,
        window_size,
        0,
        &meta,
    )?;

    let lml_bytes = std::fs::read(&tmp_path)?;
    // tmp drops here and cleans up automatically
    Ok(lml_bytes)
}

/// Archive a directory into a single .lma file.
///
/// - EDF/BDF files are encoded to LML
/// - Text/annotation files are zstd-9 compressed
/// - Already-compressed files (.lml, .gz) are stored as-is
/// - Payloads stream to a temp file (constant memory)
///
/// Returns summary with counts and sizes.
/// One file's fully-encoded payload + provenance metadata, computed OFF the
/// archive write thread so the per-file LML/zstd encode (the pack hot path) can
/// run on many rayon workers in parallel. The sequential writer consumes these
/// IN ORDER, so the archive stays byte-identical to the old single-threaded pack.
enum EncodedEntry {
    /// File read failed — skip it (logged) and record the error.
    Skipped { rel_path: String, msg: String },
    /// Encoded OK. `warnings` carries non-fatal cascade messages (e.g. an
    /// LML→zstd fallback) to fold into the summary in file order.
    Ready {
        rel_path: String,
        compressed: Vec<u8>,
        method: Method,
        original_size: u64,
        file_hash: String,
        mtime: Option<u64>,
        mtime_nanos: Option<u32>,
        mode: Option<u32>,
        synthetic_from: Option<SyntheticFromInfo>,
        warnings: Vec<(String, String)>,
    },
}

/// Pure per-file encode: read + the compression cascade (LML → zstd → store) +
/// provenance metadata (mtime/mode/sha). NO shared state — safe to call from
/// many rayon workers concurrently: the per-file LML/zstd tempfiles use unique
/// `tempfile::Builder` names, so parallel calls never collide. All ordered
/// side-effects (write, offset, counts, sha) are done by the sequential writer
/// in [`pack_archive`], keeping the output byte-identical.
fn encode_archive_entry(
    full_path: &Path,
    rel_path: &str,
    zstd_level: i32,
    tmp_dir: &Path,
    verbose: bool,
) -> EncodedEntry {
    let mut warnings: Vec<(String, String)> = Vec::new();
    let raw = match std::fs::read(full_path) {
        Ok(data) => data,
        Err(e) => {
            return EncodedEntry::Skipped {
                rel_path: rel_path.to_string(),
                msg: format!("read failed: {}", e),
            };
        }
    };
    let original_size = raw.len() as u64;
    let file_hash = sha256_hex(&raw);
    let mut method = choose_method(full_path);
    let mut synthetic_from: Option<SyntheticFromInfo> = None;

    // Compression cascade: preferred method → zstd fallback → store raw.
    let compressed: Vec<u8> = match method {
        Method::Lml => match encode_edf_to_lml(full_path, Some(tmp_dir)) {
            Ok(lml_bytes) => lml_bytes,
            Err(e) => {
                let msg = format!("{}", e);
                if verbose {
                    eprintln!(
                        "  WARN: LML failed for {}: {}, falling back to zstd",
                        rel_path, msg
                    );
                }
                warnings.push((rel_path.to_string(), msg));
                method = Method::Zstd;
                match zstd::encode_all(raw.as_slice(), zstd_level) {
                    Ok(zstd_bytes) => zstd_bytes,
                    Err(_) => {
                        method = Method::Store;
                        raw
                    }
                }
            }
        },
        Method::Zstd => match zstd::encode_all(raw.as_slice(), zstd_level) {
            Ok(zstd_bytes) => {
                // ADR 0023 Track A-3: try the ingest pipeline for non-EDF files.
                // Only commits to LML if strictly smaller than zstd, so ingest
                // never regresses the archive CR.
                let filename = full_path.file_name().and_then(|s| s.to_str());
                if let Some((lml_bytes, sf)) =
                    try_ingest_to_lml(&raw, filename, tmp_dir, zstd_bytes.len())
                {
                    method = Method::Lml;
                    synthetic_from = Some(sf);
                    lml_bytes
                } else {
                    zstd_bytes
                }
            }
            Err(_) => {
                method = Method::Store;
                raw
            }
        },
        Method::Store => raw,
    };

    // Capture mtime + Unix mode for exact restoration (best-effort; warn-not-fail).
    let meta = match std::fs::metadata(full_path) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!(
                "WARNING: cannot read metadata for {}: {} (mtime/mode lost)",
                full_path.display(),
                e
            );
            None
        }
    };
    let modified = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
    let mtime = modified.map(|d| d.as_secs());
    let mtime_nanos = modified.map(|d| d.subsec_nanos());
    let mode: Option<u32> = {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            meta.as_ref().map(|m| m.permissions().mode())
        }
        #[cfg(not(unix))]
        {
            None
        }
    };

    EncodedEntry::Ready {
        rel_path: rel_path.to_string(),
        compressed,
        method,
        original_size,
        file_hash,
        mtime,
        mtime_nanos,
        mode,
        synthetic_from,
        warnings,
    }
}

pub fn pack_archive(
    input_dir: &Path,
    output_path: &Path,
    zstd_level: i32,
    verbose: bool,
    progress_fn: Option<&dyn Fn(usize, usize, &str)>,
) -> Result<ArchiveSummary, Box<dyn std::error::Error + Send + Sync>> {
    // Audit-2026-05-11 Fix-#51: `is_dir()` follows symlinks. A symlink
    // pointing at `/` would silently pass this check, opening the door
    // for `walk_files(input_dir)` to traverse the whole filesystem.
    // Use `symlink_metadata` to test the inode itself, not the target.
    let meta = std::fs::symlink_metadata(input_dir)
        .map_err(|e| format!("Cannot stat {}: {}", input_dir.display(), e))?;
    if !meta.file_type().is_dir() {
        return Err(format!(
            "Not a directory (or is a symlink to one): {}",
            input_dir.display()
        )
        .into());
    }

    let all_files = walk_files(input_dir);
    // Hard-error on symlinks: silently dropping them is data loss, and
    // following them risks (a) double-archiving the same content if
    // the target is also under input_dir, (b) escaping the input root
    // and pulling in arbitrary filesystem content. Refuse loud; users
    // who want symlink semantics can resolve them before archiving
    // (e.g. `cp -L --parents` into a staging dir).
    let symlinks = walk_symlinks(input_dir);
    if !symlinks.is_empty() {
        let preview: Vec<String> = symlinks
            .iter()
            .take(5)
            .map(|(_, rel)| rel.clone())
            .collect();
        let extra = if symlinks.len() > 5 {
            format!(" (+{} more)", symlinks.len() - 5)
        } else {
            String::new()
        };
        return Err(format!(
            "{} symbolic link(s) found in input — archive refuses to silently \
             drop or follow them. Resolve to regular files first \
             (e.g. `cp -L --parents` into a staging directory). \
             First entries: {}{}",
            symlinks.len(),
            preview.join(", "),
            extra,
        )
        .into());
    }
    if all_files.is_empty() {
        return Err(format!("No files found in {}", input_dir.display()).into());
    }

    if verbose {
        eprintln!(
            "  Archiving {} files from {}",
            all_files.len(),
            input_dir.display()
        );
        eprintln!("  Compressor: zstd (level {})", zstd_level);
    }

    // Stream payloads to temp file (constant memory).
    //
    // CRITICAL: co-locate the payload tempfile on the OUTPUT volume,
    // NOT in $TMPDIR. tempfile::NamedTempFile::new() falls back to
    // /tmp which on Linux is often a tmpfs RAM-disk (32 GB on this
    // machine). For corpus-wide packs of multi-tens-of-GB EEG
    // datasets (CHB-MIT 43 GB, TUEG hundreds of GB) the payload
    // overflows tmpfs late in the run → ENOSPC → exit 1 after the
    // dashboard has already painted 100% per-file. Placing the
    // tempfile next to the final archive guarantees same-volume
    // free space (the user picked an output dir with room for the
    // archive itself, so the tempfile gets that same headroom).
    //
    // `_in(parent)` requires the parent dir to exist; create it
    // first if the caller pointed at e.g. /a/b/new_dir/out.lma.
    let tmp_dir = output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    // Propagate dir-create failure with full context (permissions,
    // existing-file-at-path, missing intermediate). `.ok()` would
    // swallow the root cause and surface only the downstream
    // tempfile_in IO error -- V4 Pro review of 5769562 caught this.
    std::fs::create_dir_all(&tmp_dir).map_err(|e| {
        format!(
            "Cannot create archive output dir {}: {}",
            tmp_dir.display(),
            e
        )
    })?;

    // v2 STREAMING: write directly into a same-dir `.partial` staging file and
    // rename to `output_path` on success (atomic; a half-written pack never
    // appears at the final path). This is 1x disk -- the staging file IS the
    // final archive, streamed once, then renamed. No payload temp + no
    // temp->final copy (the v1 2x-disk blowup). The 16-byte v2 header goes
    // first; payloads stream in during the loop at offset 16+; the manifest +
    // footer + sha are appended after the loop. The sha is computed
    // incrementally over the whole stream (header .. footer), no seek-back.
    let staging_path = {
        let mut s = output_path.as_os_str().to_os_string();
        s.push(".partial");
        PathBuf::from(s)
    };
    // CleanupGuard removes the staging file on any early-return / panic before
    // the successful rename, so a footer-less partial never lingers.
    struct CleanupGuard(PathBuf);
    impl Drop for CleanupGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let cleanup = CleanupGuard(staging_path.clone());
    let mut out = BufWriter::new(std::fs::File::create(&staging_path)?);
    let mut hasher = Sha256::new();
    {
        let mut header = [0u8; 16];
        header[0..4].copy_from_slice(LMA_MAGIC_V2);
        header[4..8].copy_from_slice(&LMA_VERSION_V2.to_le_bytes());
        // bytes 8..16 reserved, left zero
        out.write_all(&header)?;
        hasher.update(header);
    }
    let mut payload_offset: u64 = 0;

    let mut entries: Vec<ArchiveEntry> = Vec::with_capacity(all_files.len());
    let mut total_original: u64 = 0;
    let mut counts_lml: usize = 0;
    let mut counts_zstd: usize = 0;
    let mut counts_store: usize = 0;
    let mut errors: Vec<(String, String)> = Vec::new();

    use rayon::prelude::*;
    // Parallelize the per-file encode (the LML/zstd hot path) while keeping the
    // archive bytes IDENTICAL to the old single-threaded pack: encode a bounded
    // CHUNK of files on the rayon pool, then stream the results into `out` IN
    // ORDER. All ordered side-effects (write, sha, offset, counts) stay on this
    // one thread; only the pure `encode_archive_entry` runs in parallel.
    // `par_iter().collect()` preserves input order within a chunk, and chunks
    // are consumed in order — so payload offsets + the incremental sha match the
    // sequential pack byte-for-byte. Chunking caps the in-flight payloads held
    // in RAM (vs collecting all of them), keeping the v2 stream bounded-memory.
    let chunk_size = rayon::current_num_threads().saturating_mul(8).max(16);
    let mut processed = 0usize;
    for chunk in all_files.chunks(chunk_size) {
        let encoded: Vec<EncodedEntry> = chunk
            .par_iter()
            .map(|(full_path, rel_path)| {
                encode_archive_entry(full_path, rel_path, zstd_level, tmp_dir.as_path(), verbose)
            })
            .collect();
        for ent in encoded {
            processed += 1;
            let rel_path: String = match ent {
                EncodedEntry::Skipped { rel_path, msg } => {
                    eprintln!("  WARN: skipping {}: {}", rel_path, msg);
                    errors.push((rel_path.clone(), msg));
                    rel_path
                }
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
                    total_original += original_size;
                    match method {
                        Method::Lml => counts_lml += 1,
                        Method::Zstd => counts_zstd += 1,
                        Method::Store => counts_store += 1,
                    }
                    errors.extend(warnings);

                    let compressed_size = compressed.len() as u64;
                    let offset = payload_offset;
                    // Stream the payload straight into the final archive
                    // (offset 16+), hashing as we go. No payload temp.
                    out.write_all(&compressed)?;
                    hasher.update(&compressed);
                    // ADR 0021 Tier 2 audit (N11): checked_add. Won't fire in
                    // practice (> 18 EB) but matches the extract-site pattern.
                    payload_offset = payload_offset.checked_add(compressed_size).ok_or_else(
                        || -> Box<dyn std::error::Error + Send + Sync> {
                            format!(
                                "lma pack: payload offset overflowed u64 after {} bytes",
                                payload_offset
                            )
                            .into()
                        },
                    )?;

                    entries.push(ArchiveEntry {
                        path: rel_path.clone(),
                        original_size,
                        compressed_size,
                        method,
                        sha256: file_hash,
                        offset,
                        mtime,
                        mtime_nanos,
                        mode,
                        synthetic_from,
                    });
                    rel_path
                }
            };

            if verbose && processed % 500 == 0 {
                eprintln!("    {}/{} files...", processed, all_files.len());
            }
            if let Some(f) = progress_fn {
                f(processed, all_files.len(), &rel_path);
            }
        }
    }

    // (payloads already streamed into `out`; nothing to flush here)

    // Validate entry count fits in u32 footer field
    if entries.len() > u32::MAX as usize {
        return Err(format!(
            "Too many entries ({}) — LMA format max is {}",
            entries.len(),
            u32::MAX
        )
        .into());
    }

    // Collect directory modification times
    let dir_mtimes = walk_dirs(input_dir);

    // Build manifest JSON.
    let manifest_json = build_manifest_json(&entries, zstd_level, &dir_mtimes);

    // Manifest cascade — mirrors the per-file cascade so zero data is
    // ever lost. Try ZSTD first (the common path); on ZSTD failure
    // STORE the manifest uncompressed and flag the archive header.
    // The flag is the top bit of the 32-bit manifest-length field
    // (length capped at 2 GiB, still vastly more than any realistic
    // manifest). Readers detect the flag and skip zstd::decode_all.
    const MANIFEST_LEN_MAX: u32 = 0x7FFF_FFFF; // 2 GiB
    const MANIFEST_UNCOMPRESSED_FLAG: u32 = 0x8000_0000;
    let (manifest_payload, manifest_uncompressed): (Vec<u8>, bool) =
        match zstd::encode_all(manifest_json.as_bytes(), zstd_level) {
            Ok(bytes) => (bytes, false),
            Err(e) => {
                eprintln!(
                    "  ARCHIVE WARNING (HARD ERROR): ZSTD compression of manifest \
                     JSON failed ({}). Falling back to UNCOMPRESSED manifest. \
                     Archive remains usable; source files are untouched. \
                     Investigate zstd_level={} and zstd library health.",
                    e, zstd_level,
                );
                errors.push((
                    "<manifest>".into(),
                    format!("zstd manifest compression failed: {}", e),
                ));
                (manifest_json.into_bytes(), true)
            }
        };
    if manifest_payload.len() > MANIFEST_LEN_MAX as usize {
        return Err(format!(
            "Manifest too large ({} bytes) — LMA format max is {} bytes \
             (top bit of length field reserved for the uncompressed flag)",
            manifest_payload.len(),
            MANIFEST_LEN_MAX,
        )
        .into());
    }
    // Audit-2026-05-11 Fix-#43: enforce at write what the reader caps
    // at read. Without this check a 257 MB uncompressed manifest would
    // write successfully and then fail to load with a confusing
    // "exceeds MAX_MANIFEST_SIZE" error on every subsequent read.
    // Reader's MAX_MANIFEST_SIZE = 256 MB (decompressed).
    if manifest_uncompressed && manifest_payload.len() > MAX_MANIFEST_SIZE {
        return Err(format!(
            "Uncompressed manifest fallback {} bytes exceeds reader cap \
             MAX_MANIFEST_SIZE = {} bytes — archive would not load",
            manifest_payload.len(),
            MAX_MANIFEST_SIZE,
        )
        .into());
    }
    let manifest_len_field = (manifest_payload.len() as u32)
        | if manifest_uncompressed {
            MANIFEST_UNCOMPRESSED_FLAG
        } else {
            0
        };
    let manifest_compressed = manifest_payload;

    // v2 final write: the 16-byte header + all payloads are already streamed
    // into `out` (offset 16+). Append manifest -> 12-byte footer -> sha
    // trailer, then atomic-rename the staging file onto output_path.
    out.write_all(&manifest_compressed)?;
    hasher.update(&manifest_compressed);

    // Footer (12 bytes): manifest_len_field(u32, top bit = uncompressed)
    // | n_entries(u32) | foot_magic. Lets the reader locate the manifest
    // from the end of the file in one backward seek.
    let mut footer = [0u8; LMA_V2_FOOTER_LEN as usize];
    footer[0..4].copy_from_slice(&manifest_len_field.to_le_bytes());
    footer[4..8].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    footer[8..12].copy_from_slice(LMA_FOOT_MAGIC);
    out.write_all(&footer)?;
    hasher.update(footer);

    // SHA-256 trailer over [header .. footer], appended last.
    let archive_hash = hasher.finalize();
    out.write_all(&archive_hash)?;
    out.flush()?;
    drop(out);

    // Atomic publish: rename staging -> final, then disarm the cleanup guard
    // (the staging file no longer exists under its old name).
    std::fs::rename(&staging_path, output_path).map_err(|e| {
        format!(
            "Cannot publish archive {} -> {}: {}",
            staging_path.display(),
            output_path.display(),
            e
        )
    })?;
    std::mem::forget(cleanup);
    // _cleanup guard drops here and removes temp file

    let archive_bytes = std::fs::metadata(output_path)?.len();
    let cr = if archive_bytes > 0 {
        total_original as f64 / archive_bytes as f64
    } else {
        0.0
    };

    if verbose {
        eprintln!("  {} files archived", entries.len());
        eprintln!(
            "  Original:  {:.1} MiB",
            total_original as f64 / (1024.0 * 1024.0)
        );
        eprintln!(
            "  Archive:   {:.1} MiB  ({:.2}x)",
            archive_bytes as f64 / (1024.0 * 1024.0),
            cr
        );
        eprintln!(
            "  Methods:   {} LML, {} zstd, {} stored",
            counts_lml, counts_zstd, counts_store
        );
        if !errors.is_empty() {
            eprintln!("  Warnings:  {} files fell back to zstd", errors.len());
        }
    }

    Ok(ArchiveSummary {
        n_files: entries.len(),
        original_bytes: total_original,
        archive_bytes,
        cr,
        counts_lml,
        counts_zstd,
        counts_store,
        errors,
    })
}

// ── LML + siblings (per-file output, no archive) ────────────────────────────
//
// Same lossless codec as `pack_archive`, different container scope:
// EDFs become `.lml` next to where they lived; sidecars are copied
// verbatim (no zstd, no archive wrap). Preserves the directory tree
// 1:1. Use this when downstream tools need to see per-file `.lml` +
// loose sidecars (e.g. mne-bids, eeglab) instead of one opaque `.lma`.
//
// NOT a decomposition of pack_archive: separate top-level entry so
// the archive code path stays unchanged (and so its byte-equality
// invariants stay locked).

/// Per-file summary entry emitted by `pack_lml_with_siblings`.
#[derive(Debug, Clone)]
pub struct SiblingEntry {
    /// Path relative to the input root.
    pub src_rel: String,
    /// Path relative to the output root.
    pub dest_rel: String,
    pub kind: SiblingEntryKind,
    pub original_size: u64,
    pub written_size: u64,
    /// SHA-256 of the bytes written (LML bytes for `Lml`, original
    /// bytes for `Copied`). Round-trip integrity check.
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiblingEntryKind {
    /// EDF/BDF encoded to `.lml`.
    Lml,
    /// Non-EEG sibling copied verbatim.
    Copied,
}

/// Aggregate result of a `pack_lml_with_siblings` run.
#[derive(Debug, Default)]
pub struct SiblingPackSummary {
    pub entries: Vec<SiblingEntry>,
    pub counts_lml: usize,
    pub counts_copied: usize,
    pub original_bytes: u64,
    pub written_bytes: u64,
    pub errors: Vec<(String, String)>,
}

/// Encode every EDF/BDF under `input_dir` to `.lml` mirrored at
/// `output_dir`; copy every other file verbatim. Writes a
/// `MANIFEST.json` at `output_dir/MANIFEST.json` with per-entry
/// SHA-256 for tamper detection.
///
/// Same input safety as `pack_archive`: hard-error on symlinks
/// (refuse to silently follow or drop), hard-error if `input_dir`
/// isn't a real directory (symlinks-to-dirs rejected via
/// `symlink_metadata`).
///
/// `output_dir` must not exist OR must be an empty directory. If
/// it exists with content, the function refuses to clobber.
pub fn pack_lml_with_siblings(
    input_dir: &Path,
    output_dir: &Path,
    verbose: bool,
    progress_fn: Option<&dyn Fn(usize, usize, &str)>,
) -> Result<SiblingPackSummary, Box<dyn std::error::Error + Send + Sync>> {
    // Input safety — mirror pack_archive's checks.
    let meta = std::fs::symlink_metadata(input_dir)
        .map_err(|e| format!("Cannot stat {}: {}", input_dir.display(), e))?;
    if !meta.file_type().is_dir() {
        return Err(format!(
            "Not a directory (or is a symlink to one): {}",
            input_dir.display()
        )
        .into());
    }

    // Output safety: refuse to silently clobber a non-empty dir.
    // Allowed: dir missing (we create it) OR dir exists empty as a
    // real directory. Anything else (file at path, read denied,
    // dir-with-content) errors loud with the specific cause -- a
    // generic "already exists and is not empty" buried the root
    // error in the audit sweep.
    if output_dir.exists() {
        // `metadata` (follows symlinks) keeps the prior tolerance
        // for symlinked output directories. `symlink_metadata`
        // would silently regress: a user pointing at a symlinked
        // staging dir would suddenly hit "not a directory" when
        // the old `exists() + read_dir` chain followed the link.
        // V4 Pro review of db033e9 caught the regression.
        let meta = std::fs::metadata(output_dir).map_err(|e| {
            format!("Cannot stat output dir {}: {}", output_dir.display(), e)
        })?;
        if !meta.is_dir() {
            return Err(format!(
                "Output path exists but is not a directory: {}",
                output_dir.display()
            )
            .into());
        }
        let mut entries = std::fs::read_dir(output_dir).map_err(|e| {
            format!("Cannot read output dir {}: {}", output_dir.display(), e)
        })?;
        if entries.next().is_some() {
            return Err(format!(
                "Output dir already exists and is not empty: {}",
                output_dir.display()
            )
            .into());
        }
    } else {
        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("Cannot create {}: {}", output_dir.display(), e))?;
    }

    let all_files = walk_files(input_dir);
    let symlinks = walk_symlinks(input_dir);
    if !symlinks.is_empty() {
        let preview: Vec<String> = symlinks
            .iter()
            .take(5)
            .map(|(_, rel)| rel.clone())
            .collect();
        let extra = if symlinks.len() > 5 {
            format!(" (+{} more)", symlinks.len() - 5)
        } else {
            String::new()
        };
        return Err(format!(
            "{} symbolic link(s) found in input — pack_lml_with_siblings refuses \
             to silently drop or follow them. Resolve to regular files first \
             (e.g. `cp -L --parents` into a staging directory). \
             First entries: {}{}",
            symlinks.len(),
            preview.join(", "),
            extra,
        )
        .into());
    }
    if all_files.is_empty() {
        return Err(format!("No files found in {}", input_dir.display()).into());
    }

    if verbose {
        eprintln!(
            "  LML+siblings: {} files from {} -> {}",
            all_files.len(),
            input_dir.display(),
            output_dir.display(),
        );
    }

    let mut summary = SiblingPackSummary::default();
    let total = all_files.len();

    for (idx, (abs_path, rel_path)) in all_files.iter().enumerate() {
        if let Some(pf) = progress_fn {
            pf(idx, total, rel_path);
        }

        // Mirror the source path's directory under output_dir.
        let rel = Path::new(rel_path);
        let dest_parent = rel
            .parent()
            .map(|p| output_dir.join(p))
            .unwrap_or_else(|| output_dir.to_path_buf());
        std::fs::create_dir_all(&dest_parent).map_err(|e| {
            format!(
                "Cannot create output subdir {}: {}",
                dest_parent.display(),
                e
            )
        })?;

        let method = choose_method(abs_path);
        // Propagate the metadata error instead of silently recording
        // `original_size = 0`. A zero would make downstream CR / ratio
        // checks meaningless and could mask permission / race issues
        // (file deleted between walk and read). Audit-2026-05-20.
        let original_size = std::fs::metadata(abs_path)
            .map_err(|e| format!("Cannot stat {}: {}", abs_path.display(), e))?
            .len();
        summary.original_bytes += original_size;

        match method {
            // Same tempfile co-location reasoning as pack_archive --
            // keep LML scratch on the output volume.
            Method::Lml => match encode_edf_to_lml(abs_path, Some(output_dir)) {
                Ok(lml_bytes) => {
                    let stem = rel
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .ok_or_else(|| format!("Invalid file name: {}", rel_path))?;
                    let dest = dest_parent.join(format!("{}.lml", stem));
                    std::fs::write(&dest, &lml_bytes).map_err(|e| {
                        format!("Cannot write {}: {}", dest.display(), e)
                    })?;
                    let sha = sha256_hex(&lml_bytes);
                    let written = lml_bytes.len() as u64;
                    summary.written_bytes += written;
                    let dest_rel = dest
                        .strip_prefix(output_dir)
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| dest.to_string_lossy().into_owned());
                    summary.entries.push(SiblingEntry {
                        src_rel: rel_path.clone(),
                        dest_rel,
                        kind: SiblingEntryKind::Lml,
                        original_size,
                        written_size: written,
                        sha256: sha,
                    });
                    summary.counts_lml += 1;
                }
                Err(e) => {
                    // Fallback: copy the original EDF verbatim. NOT
                    // silent — record an error entry so the caller
                    // can surface it. Better than aborting the whole
                    // run mid-tree.
                    summary.errors.push((rel_path.clone(), e.to_string()));
                    copy_as_sibling(abs_path, &dest_parent, rel, &mut summary, original_size)?;
                }
            },
            Method::Zstd | Method::Store => {
                copy_as_sibling(abs_path, &dest_parent, rel, &mut summary, original_size)?;
            }
        }
    }

    // Manifest -- JSON next to outputs (NOT inside an archive). Same
    // SHA-256 per file as the archive flow; tooling can verify any
    // .lml in-place without unpacking anything.
    let manifest_json = build_sibling_manifest_json(&summary)?;
    let manifest_path = output_dir.join("MANIFEST.json");
    std::fs::write(&manifest_path, manifest_json.as_bytes()).map_err(|e| {
        format!("Cannot write {}: {}", manifest_path.display(), e)
    })?;

    if verbose {
        eprintln!(
            "  Done: {} LML, {} copied, {} errors",
            summary.counts_lml,
            summary.counts_copied,
            summary.errors.len(),
        );
    }

    Ok(summary)
}

/// Helper: copy `abs_path` to `dest_parent/<file_name>`, record entry.
fn copy_as_sibling(
    abs_path: &Path,
    dest_parent: &Path,
    rel: &Path,
    summary: &mut SiblingPackSummary,
    original_size: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let name = rel
        .file_name()
        .ok_or_else(|| format!("Invalid file name: {}", rel.display()))?;
    let dest = dest_parent.join(name);
    let data = std::fs::read(abs_path)
        .map_err(|e| format!("Cannot read {}: {}", abs_path.display(), e))?;
    std::fs::write(&dest, &data)
        .map_err(|e| format!("Cannot write {}: {}", dest.display(), e))?;
    let sha = sha256_hex(&data);
    let written = data.len() as u64;
    summary.written_bytes += written;
    let dest_rel = rel.to_string_lossy().into_owned();
    summary.entries.push(SiblingEntry {
        src_rel: dest_rel.clone(),
        dest_rel,
        kind: SiblingEntryKind::Copied,
        original_size,
        written_size: written,
        sha256: sha,
    });
    summary.counts_copied += 1;
    Ok(())
}

/// Build a JSON manifest via serde_json so paths containing
/// backslashes, quotes, unicode, or control chars are escaped per
/// the JSON spec. Earlier prototype used `{:?}` interpolation
/// which produces Rust-Debug output, NOT strict JSON, and would
/// silently emit invalid output for unusual paths (V4 Pro + V6 R
/// review of c34268d caught the risk before any real path hit it).
fn build_sibling_manifest_json(
    summary: &SiblingPackSummary,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let entries: Vec<serde_json::Value> = summary
        .entries
        .iter()
        .map(|e| {
            let kind = match e.kind {
                SiblingEntryKind::Lml => "lml",
                SiblingEntryKind::Copied => "copied",
            };
            serde_json::json!({
                "src": e.src_rel,
                "dest": e.dest_rel,
                "kind": kind,
                "original_size": e.original_size,
                "written_size": e.written_size,
                "sha256": e.sha256,
            })
        })
        .collect();
    let v = serde_json::json!({
        "version": 1,
        "counts_lml": summary.counts_lml,
        "counts_copied": summary.counts_copied,
        "original_bytes": summary.original_bytes,
        "written_bytes": summary.written_bytes,
        "entries": entries,
    });
    // serde_json::to_string_pretty errors only on stack overflow
    // for recursive value shapes. Our manifest is flat (one
    // entries array, no nesting) so this can't fail in practice,
    // but propagate the error instead of silently substituting an
    // empty "{}" manifest. Downstream tooling that opens the
    // manifest as source-of-truth must NOT see a sentinel-shaped
    // empty file masquerading as a real manifest (Bible R5).
    serde_json::to_string_pretty(&v)
        .map_err(|e| format!("manifest serialize failed: {}", e).into())
}

/// List contents of an LMA archive without extracting.
pub fn list_archive(
    archive_path: &Path,
) -> Result<Vec<ArchiveEntry>, Box<dyn std::error::Error + Send + Sync>> {
    let manifest = read_manifest(archive_path)?;
    Ok(manifest)
}

/// Phase 3.7 — extract one entry from an LMA archive to `output_path`
/// without unpacking the rest. Matches `entry_path` against the
/// manifest's `path` field by exact equality first, then by suffix
/// match (`endswith`) as a convenience for users who don't want to
/// type the archive's full relative path.
///
/// - LML entries reconstruct the original EDF/BDF, just like
///   `unpack_archive`. SHA-256 of the reconstructed bytes is verified
///   against the manifest.
/// - Zstd and Store entries are decompressed / written verbatim;
///   SHA-256 verified after decompression.
///
/// Returns the number of bytes written to `output_path`.
pub fn extract_entry(
    archive_path: &Path,
    entry_path: &str,
    output_path: &Path,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    // Single dispatch chokepoint: parsed entries + payload bounds for
    // v1/v2 in one read. We keep `f` open afterwards for the payload
    // seek+read (no separate header re-parse).
    let mut f = BufReader::new(std::fs::File::open(archive_path)?);
    let file_size = std::fs::metadata(archive_path)?.len();
    let idx = read_lma_index(&mut f, file_size)?;
    let entries = idx.entries;
    // Exact-match first; suffix-match second (helps when the user
    // types just the basename, e.g. `foo.edf`).
    //
    // ADR 0021 Tier 2 audit (N4): if the suffix-match falls
    // through, ensure it's unambiguous. Pre-fix: passing `foo.edf`
    // when both `train/foo.edf` and `test/foo.edf` exist silently
    // returned the first-listed entry. Now: detect multi-match
    // and refuse with the conflicting paths so the operator
    // disambiguates.
    let entry_opt: Option<&ArchiveEntry> =
        if let Some(exact) = entries.iter().find(|e| e.path == entry_path) {
            Some(exact)
        } else {
            let suffix_matches: Vec<&ArchiveEntry> = entries
                .iter()
                .filter(|e| e.path.ends_with(entry_path))
                .collect();
            if suffix_matches.len() > 1 {
                let preview: Vec<String> = suffix_matches
                    .iter()
                    .take(5)
                    .map(|e| e.path.clone())
                    .collect();
                return Err(format!(
                    "lma extract-entry: '{entry_path}' is ambiguous; {} entries match \
                     (first 5: {}). Use the full archive-relative path.",
                    suffix_matches.len(),
                    preview.join(", ")
                )
                .into());
            }
            suffix_matches.into_iter().next()
        };
    let entry = entry_opt
        .ok_or_else(|| {
            format!(
                "lma extract-entry: '{entry_path}' not found in archive ({} entries)",
                entries.len()
            )
        })?
        .clone();

    // payload_start / payload_end come from the chokepoint index
    // (v1/v2-aware). entry.offset is relative to payload_start.
    let payload_start = idx.payload_base;

    // Bounds guard before allocation -- adversarial archives could
    // claim absurd entry.compressed_size OR original_size. We check
    // both because a 10 MB compressed entry that decompresses to
    // 100 GB (zstd bomb) still OOMs even though the compressed
    // bound passes. Bible R30.
    if entry.compressed_size > MAX_ENTRY_DECOMPRESS_SIZE as u64 {
        return Err(format!(
            "lma extract-entry: entry '{entry_path}' compressed_size {} > MAX_ENTRY_DECOMPRESS_SIZE",
            entry.compressed_size
        )
        .into());
    }
    if entry.original_size > MAX_ENTRY_ORIGINAL_SIZE {
        return Err(format!(
            "lma extract-entry: entry '{entry_path}' original_size {} > MAX_ENTRY_ORIGINAL_SIZE {}",
            entry.original_size, MAX_ENTRY_ORIGINAL_SIZE
        )
        .into());
    }

    // Bound the seek before issuing it. unpack_archive (line ~2133)
    // and the read_entry sibling paths check entry.offset +
    // compressed_size against the payload section; extract_entry
    // historically did not, leaving an adversarial manifest free to
    // wrap `payload_start + entry.offset` past u64::MAX or land the
    // read inside the header/manifest region. Audit-2026-05-20.
    // Bound against `idx.payload_end` (the tighter, v2-correct end of
    // the payload section), not the whole file — for v2 the manifest +
    // footer + sha live AFTER the payloads, so the payload must not
    // extend into them.
    let payload_end = idx.payload_end;
    let abs_offset = payload_start.checked_add(entry.offset).ok_or_else(|| {
        format!(
            "lma extract-entry: entry '{entry_path}' offset overflow \
             (payload_start={} + entry.offset={})",
            payload_start, entry.offset,
        )
    })?;
    let entry_end = abs_offset.checked_add(entry.compressed_size).ok_or_else(|| {
        format!(
            "lma extract-entry: entry '{entry_path}' size overflow \
             (offset={} + compressed_size={})",
            abs_offset, entry.compressed_size,
        )
    })?;
    if entry_end > payload_end {
        return Err(format!(
            "lma extract-entry: entry '{entry_path}' extends past payload section end \
             (entry_end={} > payload_end={})",
            entry_end, payload_end,
        )
        .into());
    }
    f.seek(SeekFrom::Start(abs_offset))?;
    let mut compressed = vec![0u8; bounded_alloc_usize(entry.compressed_size, &entry.path)?];
    f.read_exact(&mut compressed)?;

    let data = match entry.method {
        // extract_entry handles one file at a time; tempfile is
        // short-lived. Co-locate on the destination volume so
        // decoding a pathologically large entry doesn't blow up
        // tmpfs when the caller picked a roomy output disk. When
        // output_path is a bare filename (no parent component),
        // `.parent()` returns None -- fall back to "." (cwd) so
        // we still get same-volume co-location rather than
        // sliding to $TMPDIR. V4 Pro review of 1923e44.
        Method::Lml => {
            let hint = output_path.parent().unwrap_or_else(|| Path::new("."));
            decode_lml_to_edf(&compressed, Some(entry.original_size), Some(hint))?
        }
        Method::Store => compressed,
        Method::Zstd => decode_zstd_bounded(
            &compressed,
            entry
                .original_size
                .min(MAX_ENTRY_ORIGINAL_SIZE) as usize,
            &entry.path,
        )?,
    };

    // SHA-256 verify against manifest.
    let hash = sha256_hex(&data);
    if hash != entry.sha256 {
        return Err(format!(
            "lma extract-entry: SHA-256 mismatch for '{entry_path}' (manifest {} vs reconstructed {})",
            &entry.sha256[..entry.sha256.len().min(12)],
            &hash[..hash.len().min(12)]
        )
        .into());
    }

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
            // ADR 0021 Tier 2 audit (N2): after create_dir_all,
            // refuse if `parent` is a symlink. `create_dir_all`
            // follows existing symlink components silently, so an
            // attacker who plants a symlink under the operator's
            // intended output_dir before extract runs can redirect
            // files outside the destination. `symlink_metadata`
            // tests the inode itself; if it reports a symlink we
            // bail rather than write through.
            let meta = std::fs::symlink_metadata(parent)?;
            if meta.file_type().is_symlink() {
                return Err(format!(
                    "lma extract-entry: refusing to write through symlinked parent {} \
                     (path traversal protection)",
                    parent.display()
                )
                .into());
            }
        }
    }
    // ADR 0021 Tier 2 audit (N3): atomic write via tmp + rename.
    // Pre-fix wrote directly to `output_path` -- a kill or power-
    // cut mid-write left a truncated file that LOOKS legitimate
    // (same path, partial bytes). unpack_archive already uses
    // this pattern; extract_entry should too. PID + atomic-seq
    // suffix keeps two concurrent extract calls from colliding
    // on the same tmp path (matches N1 pattern).
    let pid = std::process::id();
    let seq = APPEND_TMP_SEQ.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let tmp_path = {
        let mut p = output_path.to_path_buf();
        let fname = p.file_name().map(|n| n.to_os_string()).unwrap_or_default();
        let mut tmp_name = fname;
        tmp_name.push(alloc::format!(".tmp.extract.{}.{}", pid, seq).as_str());
        p.set_file_name(tmp_name);
        p
    };
    {
        let mut out_file = BufWriter::new(std::fs::File::create(&tmp_path)?);
        out_file.write_all(&data)?;
        out_file.into_inner()?.sync_all()?;
    }
    std::fs::rename(&tmp_path, output_path).map_err(|e| {
        // Best-effort cleanup of the leftover tmp on rename failure.
        let _ = std::fs::remove_file(&tmp_path);
        format!(
            "atomic rename {} -> {} failed: {}",
            tmp_path.display(),
            output_path.display(),
            e
        )
    })?;
    Ok(data.len() as u64)
}

/// Phase 3.8 — append one file to an existing LMA archive without
/// rewriting payload bytes of pre-existing entries.
///
/// Strategy (v2 streaming): byte-copy the existing prefix verbatim from
/// `[0 .. old payload_end]` (the 16-byte v2 header + all existing
/// payloads — their absolute positions and offsets don't move) into a
/// same-directory `<archive>.lma.new` tempfile, append the new entry's
/// compressed payload immediately after, then write a fresh combined
/// manifest (covering N+1 entries) + 12-byte v2 footer + SHA-256
/// trailer. The old manifest + footer + sha are dropped. The new
/// entry's offset (relative to payload_base 16) is `old payload_end -
/// 16`. A v1 source is normalised to v2 on append: its leading header
/// is replaced with a fresh v2 header and the existing entries' offsets
/// are re-based onto payload_base 16 (the dead v1 front manifest stays
/// inside the copied prefix, harmlessly superseded by the new footer
/// manifest). The old archive is moved to `<archive>.lma.bak` first so
/// an interrupted rename leaves both the old archive (`.bak`) and the
/// partial new archive (`.new`) — no data loss on power-cut. Bible R27.
///
/// `entry_path` is the relative path the new entry will carry inside
/// the manifest (e.g. `"sub-02.edf"`). `source_path` is the file on
/// disk to read bytes from. Defaults: `entry_path = source_path's
/// file name`, `zstd_level = 9`, `keep_bak = true`.
///
/// Idempotency (Bible R31): if `entry_path` already exists in the
/// archive and the new file's SHA-256 matches the recorded sha256,
/// the call is a no-op (returns the original `ArchiveSummary` shape
/// with no rewrite). If sha256 differs, error — never silently
/// overwrite. Pass `force_overwrite = true` to replace.
pub fn append_entry(
    archive_path: &Path,
    source_path: &Path,
    entry_path: Option<&str>,
    zstd_level: i32,
    force_overwrite: bool,
    keep_bak: bool,
) -> Result<ArchiveSummary, Box<dyn std::error::Error + Send + Sync>> {
    // 1. Read existing index (v1/v2-aware) + sniff archive layout.
    //
    // v2 rewrite: append = copy [0 .. old payload_end] verbatim (the
    // 16-byte header + all existing payloads, whose absolute positions
    // and offsets are unchanged), then stream the new entry's payload
    // immediately after, then write a fresh combined manifest + v2
    // footer + sha. The old manifest + footer + sha (which lived AFTER
    // the payloads in v2, or before them in v1) are dropped. The new
    // entry's offset (relative to payload_base 16) is
    // `old_payload_end - 16`.
    let archive_size = std::fs::metadata(archive_path)?.len();
    if archive_size < 48 {
        return Err(format!("lma append: archive too small ({archive_size} bytes, min 48)").into());
    }
    let idx = {
        let mut f = BufReader::new(std::fs::File::open(archive_path)?);
        read_lma_index(&mut f, archive_size)?
    };
    let entries_old = idx.entries;
    let payload_base_old = idx.payload_base; // 16 for v2, 16+mlen for v1
    let payload_end_old = idx.payload_end;
    // ADR 0021 Tier 2 audit (N7): refuse to append over an archive
    // whose manifest is stored uncompressed (top bit of the manifest_
    // len field set). The rebuild path below always writes a zstd-
    // compressed manifest, which would silently drop the "zstd encode
    // failed at pack time" recovery signal. The chokepoint index
    // doesn't surface that flag, so re-read it directly: for v2 it's in
    // the footer at [file-44 .. file-40]; for v1 it's in the header at
    // [12..16].
    {
        let mut f = BufReader::new(std::fs::File::open(archive_path)?);
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        let mlf: u32 = if &magic == LMA_MAGIC_V2 {
            let foot_pos = archive_size - 32 - LMA_V2_FOOTER_LEN;
            f.seek(SeekFrom::Start(foot_pos))?;
            let mut footer = [0u8; LMA_V2_FOOTER_LEN as usize];
            f.read_exact(&mut footer)?;
            u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]])
        } else if &magic == LMA_MAGIC {
            let mut hdr_tail = [0u8; 12];
            f.read_exact(&mut hdr_tail)?; // version(4) + n_entries(4) + manifest_len_field(4)
            u32::from_le_bytes([hdr_tail[8], hdr_tail[9], hdr_tail[10], hdr_tail[11]])
        } else {
            return Err("lma append: input is not an LMA archive (bad magic)".into());
        };
        if (mlf & 0x8000_0000) != 0 {
            return Err(format!(
                "lma append: source archive has uncompressed-manifest flag set \
                 (manifest_len_field high bit = 1); refusing to append because the \
                 rebuild path always re-compresses the manifest. Repack the archive \
                 first or use a future `--preserve-manifest-codec` flag."
            )
            .into());
        }
    }
    // Verbatim-copy region = [0 .. old payload_end] = header + old
    // payloads. New entry's offset (relative to payload_base 16) =
    // old_payload_end - 16. (For v1 archives the front manifest sits
    // inside this copied region; that's fine — the new v2 archive moves
    // the manifest to the footer and re-derives all offsets relative to
    // 16, so a v1→v2 append also normalises the layout.)
    let copy_prefix_len = payload_end_old; // bytes [0 .. payload_end_old]
    let new_entry_offset = payload_end_old.checked_sub(16).ok_or_else(
        || -> Box<dyn std::error::Error + Send + Sync> {
            format!(
                "lma append: corrupt archive — payload_end {} < header 16",
                payload_end_old
            )
            .into()
        },
    )?;
    // For v1 sources, existing manifest offsets are relative to
    // `payload_base_old` (16 + old manifest_len), NOT 16. Re-base them
    // onto payload_base 16 so the rewritten v2 manifest is correct.
    let v1_offset_shift = payload_base_old - 16;

    // 2. Build new entry. mtime/mode captured at append time.
    let resolved_entry_path = match entry_path {
        Some(s) => s.to_string(),
        None => source_path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("lma append: source_path has no filename — pass --as <path>")?
            .to_string(),
    };
    // ADR 0022 Group C: pre-check source size against
    // MAX_ENTRY_ORIGINAL_SIZE. Pre-fix `std::fs::read` loaded the
    // entire file into RAM with no cap; appending a 50 GB EDF
    // OOM'd the host even though the entry would be rejected
    // later by the manifest check.
    //
    // V4 Pro fast review of edd1c817 flagged a TOCTOU: stat()
    // and read() were separate syscalls; an attacker could swap
    // the file between them and bypass the cap. Use a single
    // File::open → file.metadata() → Read::take(cap+1) chain
    // so size + content come from the same handle, and bail if
    // the file grew past the cap during read.
    let mut f = std::fs::File::open(source_path)
        .map_err(|e| format!("lma append: open {}: {}", source_path.display(), e))?;
    let src_meta = f
        .metadata()
        .map_err(|e| format!("lma append: stat {}: {}", source_path.display(), e))?;
    if src_meta.len() > MAX_ENTRY_ORIGINAL_SIZE {
        return Err(format!(
            "lma append: source {} is {} bytes (> MAX_ENTRY_ORIGINAL_SIZE {}); refusing",
            source_path.display(),
            src_meta.len(),
            MAX_ENTRY_ORIGINAL_SIZE
        )
        .into());
    }
    let mut raw = Vec::with_capacity(src_meta.len() as usize);
    use std::io::Read as _;
    std::io::Read::take(&mut f, MAX_ENTRY_ORIGINAL_SIZE + 1)
        .read_to_end(&mut raw)
        .map_err(|e| format!("lma append: read {}: {}", source_path.display(), e))?;
    if raw.len() as u64 > MAX_ENTRY_ORIGINAL_SIZE {
        return Err(format!(
            "lma append: source {} grew past MAX_ENTRY_ORIGINAL_SIZE {} during read; refusing",
            source_path.display(),
            MAX_ENTRY_ORIGINAL_SIZE
        )
        .into());
    }
    let new_sha = sha256_hex(&raw);

    // R31 idempotency: if entry already exists with identical SHA, no-op.
    if let Some(existing) = entries_old.iter().find(|e| e.path == resolved_entry_path) {
        if existing.sha256 == new_sha {
            eprintln!(
                "lma append: '{resolved_entry_path}' already present with matching SHA-256 — no-op"
            );
            return Ok(ArchiveSummary {
                n_files: entries_old.len(),
                original_bytes: entries_old.iter().map(|e| e.original_size).sum(),
                archive_bytes: archive_size,
                cr: 0.0,
                counts_lml: 0,
                counts_zstd: 0,
                counts_store: 0,
                errors: Vec::new(),
            });
        }
        if !force_overwrite {
            return Err(format!(
                "lma append: '{resolved_entry_path}' already exists with different SHA-256 — \
                 pass --force to overwrite, or pick a different --as path"
            )
            .into());
        }
        // Overwrite path = remove existing entry from entries_old.
        // Note: payload bytes for the replaced entry remain orphaned in
        // the payload section (dead bytes). A future `lml repack` would
        // reclaim them. Not a data-loss bug, just space waste.
        eprintln!(
            "lma append: overwriting existing entry '{resolved_entry_path}' (old payload becomes dead space)"
        );
    }

    let original_size = raw.len() as u64;
    let mut counts_lml = 0usize;
    let mut counts_zstd = 0usize;
    let mut counts_store = 0usize;
    let mut method = choose_method(source_path);
    let compressed: Vec<u8> = match method {
        // Single-entry append: source file is small enough that
        // tmpfs is fine. Pass None to keep behavior unchanged.
        Method::Lml => match encode_edf_to_lml(source_path, None) {
            Ok(bytes) => {
                counts_lml += 1;
                bytes
            }
            Err(_) => match zstd::encode_all(raw.as_slice(), zstd_level) {
                Ok(bytes) => {
                    method = Method::Zstd;
                    counts_zstd += 1;
                    bytes
                }
                Err(_) => {
                    method = Method::Store;
                    counts_store += 1;
                    raw.clone()
                }
            },
        },
        Method::Zstd => match zstd::encode_all(raw.as_slice(), zstd_level) {
            Ok(bytes) => {
                counts_zstd += 1;
                bytes
            }
            Err(_) => {
                method = Method::Store;
                counts_store += 1;
                raw.clone()
            }
        },
        Method::Store => {
            counts_store += 1;
            raw.clone()
        }
    };
    let meta = std::fs::metadata(source_path).ok();
    let modified = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
    let mtime = modified.map(|d| d.as_secs());
    let mtime_nanos = modified.map(|d| d.subsec_nanos());
    let mode: Option<u32> = {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            meta.as_ref().map(|m| m.permissions().mode())
        }
        #[cfg(not(unix))]
        {
            None
        }
    };

    // 3. Compose new entries list. Existing entries keep their absolute
    //    byte positions (we copy the prefix verbatim), but the v2
    //    manifest expresses offsets relative to payload_base 16. For a
    //    v1 source the old offsets were relative to 16+manifest_len, so
    //    re-base them by `v1_offset_shift` (0 for v2 sources). The new
    //    entry sits at absolute `payload_end_old`, i.e. v2-relative
    //    offset `new_entry_offset` (= payload_end_old - 16).
    let mut entries_new: Vec<ArchiveEntry> = entries_old
        .into_iter()
        .filter(|e| e.path != resolved_entry_path)
        .map(|mut e| {
            e.offset += v1_offset_shift;
            e
        })
        .collect();
    entries_new.push(ArchiveEntry {
        path: resolved_entry_path.clone(),
        original_size,
        compressed_size: compressed.len() as u64,
        method,
        sha256: new_sha,
        offset: new_entry_offset,
        mtime,
        mtime_nanos,
        mode,
        // append_entry doesn't route through the ASCII ingest path
        // (it's used for single-file append, the caller already
        // chose the method). Always None.
        synthetic_from: None,
    });

    // 4. Build + compress new manifest. Reuse build_manifest_json with
    //    empty dir_mtimes (append doesn't carry directory metadata).
    let manifest_json = build_manifest_json(&entries_new, zstd_level, &[]);
    const MANIFEST_LEN_MAX: u32 = 0x7FFF_FFFF;
    const MANIFEST_UNCOMPRESSED_FLAG: u32 = 0x8000_0000;
    let (manifest_payload, manifest_uncompressed) =
        match zstd::encode_all(manifest_json.as_bytes(), zstd_level) {
            Ok(b) => (b, false),
            Err(_) => (manifest_json.into_bytes(), true),
        };
    if manifest_payload.len() > MANIFEST_LEN_MAX as usize {
        return Err(format!(
            "lma append: new manifest {} bytes exceeds MANIFEST_LEN_MAX {}",
            manifest_payload.len(),
            MANIFEST_LEN_MAX
        )
        .into());
    }
    let manifest_len_field_new = (manifest_payload.len() as u32)
        | if manifest_uncompressed {
            MANIFEST_UNCOMPRESSED_FLAG
        } else {
            0
        };

    // 5. Write to same-directory tempfile so atomic rename works across
    //    a single filesystem. Building `.lma.new` next to the archive
    //    is the WAL: if a power-cut hits mid-write, the partial `.lma.new`
    //    is recoverable manually and the original is intact.
    //
    // ADR 0021 Tier 2 audit (N1): suffix the temp paths with the
    // current PID + a per-call counter so two concurrent
    // `lml encode --lma --append` processes don't collide on
    // `.lma.new` / `.lma.bak` and silently overwrite each other.
    // PID alone is not enough (single process running this fn in
    // multiple threads would collide); add an atomic monotonic
    // counter for thread-safety.
    let archive_dir = archive_path.parent().unwrap_or(Path::new("."));
    let pid = std::process::id();
    let seq = APPEND_TMP_SEQ.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let base_ext = archive_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("lma");
    let new_path =
        archive_path.with_extension(format!("{}.new.{}.{}", base_ext, pid, seq));
    std::fs::create_dir_all(archive_dir)?;

    {
        let mut out = BufWriter::new(std::fs::File::create(&new_path)?);
        let mut hasher = Sha256::new();
        // v2 streaming layout:
        //   [0 .. payload_end_old]  copied verbatim from the source
        //                            (v2/v1 header + existing payloads)
        //   new entry payload
        //   manifest (zstd or raw)
        //   12-byte footer (manifest_len_field | n_entries | LFT2)
        //   32-byte sha over everything above.
        //
        // The verbatim prefix carries the SOURCE's header bytes. For a
        // v2 source that's already an LMA2/version-2 header; for a v1
        // source it's the LMA1 header + front manifest. Either way the
        // bytes are reproduced unchanged, and the NEW footer/manifest we
        // append below is what the reader uses — but a v1 prefix would
        // leave an `LMA1` magic at offset 0, which the v2 chokepoint
        // rejects. So when the source is v1, overwrite the leading
        // 16-byte header region with a fresh v2 header before copying
        // the rest. We detect v1 via `v1_offset_shift != 0` (v1 has a
        // front manifest, so payload_base > 16).
        let mut copy_from: u64 = 0;
        if v1_offset_shift != 0 {
            // Emit a fresh v2 header (16 bytes) in place of the v1
            // header; the v1 front manifest that follows it (bytes
            // 16..payload_base_old) becomes dead space inside the
            // copied prefix but is harmless — the new footer manifest
            // supersedes it and all offsets were re-based above.
            let mut header = [0u8; 16];
            header[0..4].copy_from_slice(LMA_MAGIC_V2);
            header[4..8].copy_from_slice(&LMA_VERSION_V2.to_le_bytes());
            // bytes 8..16 reserved, left zero
            out.write_all(&header)?;
            hasher.update(header);
            copy_from = 16;
        }
        // Copy [copy_from .. payload_end_old] verbatim, 8 MiB at a time.
        {
            let mut src = BufReader::new(std::fs::File::open(archive_path)?);
            src.seek(SeekFrom::Start(copy_from))?;
            let mut remaining = copy_prefix_len - copy_from;
            let mut buf = vec![0u8; 8 * 1024 * 1024];
            while remaining > 0 {
                let to_read = (remaining as usize).min(buf.len());
                let n = src.read(&mut buf[..to_read])?;
                if n == 0 {
                    return Err("lma append: short read on old prefix (header + payloads)".into());
                }
                out.write_all(&buf[..n])?;
                hasher.update(&buf[..n]);
                remaining -= n as u64;
            }
        }
        // New entry payload (sits at absolute payload_end_old).
        out.write_all(&compressed)?;
        hasher.update(&compressed);
        // New manifest.
        out.write_all(&manifest_payload)?;
        hasher.update(&manifest_payload);
        // v2 footer (12 bytes).
        let mut footer = [0u8; LMA_V2_FOOTER_LEN as usize];
        footer[0..4].copy_from_slice(&manifest_len_field_new.to_le_bytes());
        footer[4..8].copy_from_slice(&(entries_new.len() as u32).to_le_bytes());
        footer[8..12].copy_from_slice(LMA_FOOT_MAGIC);
        out.write_all(&footer)?;
        hasher.update(footer);
        // SHA-256 trailer over [0 .. footer].
        let archive_hash = hasher.finalize();
        out.write_all(&archive_hash)?;
        out.flush()?;
        let f = out.into_inner().map_err(|e| {
            let kind = e.error().kind();
            std::io::Error::new(kind, "lma append: BufWriter flush failed before sync_all")
        })?;
        f.sync_all()?;
    }
    // 6. Move old → .bak, then rename new → archive. Two renames are
    //    needed because rename can't replace a non-empty file across
    //    all filesystems without first moving the original aside.
    // Same PID + seq suffix as `.new` -- two concurrent appends
    // would otherwise both stomp on the same `.bak` and corrupt
    // each other's backup.
    let bak_path =
        archive_path.with_extension(format!("{}.bak.{}.{}", base_ext, pid, seq));
    if bak_path.exists() {
        std::fs::remove_file(&bak_path)?;
    }
    std::fs::rename(archive_path, &bak_path)?;
    // From here on, archive_path doesn't exist briefly. If the next
    // rename fails, recovery = move bak back.
    if let Err(e) = std::fs::rename(&new_path, archive_path) {
        // ADR 0021 Tier 2 audit (N8): propagate restore failure
        // instead of silently swallowing it. Pre-fix the
        // `let _ = ...` discarded the restore error, so if the
        // bak-restore ALSO failed (perms changed, target locked)
        // the user saw only the outer error while the original
        // archive was actually gone. Now we report both errors
        // so the operator can recover by hand.
        match std::fs::rename(&bak_path, archive_path) {
            Ok(()) => {
                return Err(format!(
                    "lma append: final atomic rename failed ({e}); restored from .bak — \
                     new archive is at {}",
                    new_path.display()
                )
                .into());
            }
            Err(restore_err) => {
                return Err(format!(
                    "lma append: final atomic rename failed ({e}); \
                     ALSO failed to restore from .bak ({restore_err}). \
                     Original archive at {} (.bak), new archive at {} (.new). \
                     Manual recovery required.",
                    bak_path.display(),
                    new_path.display()
                )
                .into());
            }
        }
    }
    // Fsync the directory so the rename is durable on power-cut.
    // ADR 0021 Tier 2 audit (N8): warn on fsync failure instead
    // of silently swallowing. A failed dir-fsync means the rename
    // may not survive a power-cut; the operator who never saw
    // the warning won't know the durability guarantee didn't
    // hold. We still don't error out -- some filesystems
    // (e.g. tmpfs, NFSv3) legitimately fail dir fsync -- but the
    // log line surfaces in the audit trail.
    #[cfg(unix)]
    {
        match std::fs::File::open(archive_dir) {
            Ok(dir) => {
                if let Err(e) = dir.sync_all() {
                    eprintln!(
                        "  WARNING: lma append: dir fsync of {} failed: {} \
                         (rename may not be durable on power-cut)",
                        archive_dir.display(),
                        e
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "  WARNING: lma append: cannot open dir {} for fsync: {} \
                     (rename may not be durable on power-cut)",
                    archive_dir.display(),
                    e
                );
            }
        }
    }
    if !keep_bak {
        if let Err(e) = std::fs::remove_file(&bak_path) {
            eprintln!(
                "  WARNING: lma append: failed to remove backup {}: {} \
                 (stale .bak left behind)",
                bak_path.display(),
                e
            );
        }
    }

    let archive_bytes_new = std::fs::metadata(archive_path)?.len();
    let cr = if archive_bytes_new > 0 {
        let total: u64 = entries_new.iter().map(|e| e.original_size).sum();
        total as f64 / archive_bytes_new as f64
    } else {
        0.0
    };
    Ok(ArchiveSummary {
        n_files: entries_new.len(),
        original_bytes: entries_new.iter().map(|e| e.original_size).sum(),
        archive_bytes: archive_bytes_new,
        cr,
        counts_lml,
        counts_zstd,
        counts_store,
        errors: Vec::new(),
    })
}

/// Extract an LMA archive, reconstructing the original directory tree.
///
/// - LML entries are written as .lml files (use Python `reconstruct_edf()`
///   for bit-exact EDF reconstruction)
/// - Zstd entries are decompressed
/// - Stored entries are written as-is
/// - Archive-level SHA-256 verifies overall integrity
/// - Per-entry SHA-256 verifies non-LML files after decompression
pub fn unpack_archive(
    archive_path: &Path,
    output_dir: &Path,
    verify: bool,
    verbose: bool,
    progress_fn: Option<&dyn Fn(usize, usize, &str)>,
) -> Result<ArchiveSummary, Box<dyn std::error::Error + Send + Sync>> {
    let mut f = BufReader::new(std::fs::File::open(archive_path)?);
    let file_size = std::fs::metadata(archive_path)?.len();

    // Guard: minimum valid archive is 16 (header) + 0 (manifest) + 32 (hash) = 48
    if file_size < 48 {
        return Err(format!(
            "Archive too small ({} bytes, minimum 48): {}",
            file_size,
            archive_path.display()
        )
        .into());
    }

    if verify {
        // Verify archive SHA-256.
        //
        // Audit-2026-05-11 Fix-#46: detect truncation BEFORE running
        // the SHA. Previously a truncated archive would hit the
        // read-loop `n == 0` break and then fail with the generic
        // "SHA-256 mismatch" error, obscuring the actual cause. Check
        // that we read the declared content_size; if short, return a
        // specific Truncated diagnostic.
        let mut hasher = Sha256::new();
        let content_size = file_size - 32;
        let mut remaining = content_size;
        let mut buf = vec![0u8; 8 * 1024 * 1024];
        f.seek(SeekFrom::Start(0))?;
        let mut read_total: u64 = 0;
        while remaining > 0 {
            let to_read = (remaining as usize).min(buf.len());
            let n = f.read(&mut buf[..to_read])?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            remaining -= n as u64;
            read_total += n as u64;
        }
        if read_total != content_size {
            return Err(format!(
                "Archive truncated: declared content {} bytes, only {} readable. \
                 SHA-256 not verified (truncation is the actual fault, not corruption).",
                content_size, read_total
            )
            .into());
        }
        let computed = hasher.finalize();
        let mut stored = [0u8; 32];
        f.read_exact(&mut stored)?;
        if computed.as_slice() != stored {
            return Err("Archive SHA-256 mismatch — file is corrupted".into());
        }
    }

    // BOM sniff: surface the friendlier "strip the BOM" diagnostic
    // before the generic chokepoint error. The chokepoint already
    // rejects non-LMA magic, but its message doesn't single out BOMs.
    {
        f.seek(SeekFrom::Start(0))?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != LMA_MAGIC && &magic != LMA_MAGIC_V2 {
            if magic[0] == 0xEF && magic[1] == 0xBB && magic[2] == 0xBF {
                return Err(
                    "File starts with UTF-8 BOM — not a valid LMA archive. Strip the BOM."
                        .into(),
                );
            }
            if (magic[0] == 0xFF && magic[1] == 0xFE) || (magic[0] == 0xFE && magic[1] == 0xFF) {
                return Err("File starts with UTF-16 BOM — not a valid LMA archive.".into());
            }
        }
    }

    // Single dispatch chokepoint: v1/v2 layout + parsed entries +
    // payload bounds. `read_lma_index` seeks internally, so the prior
    // verify-pass full read doesn't disturb it.
    let idx = read_lma_index(&mut f, file_size)?;
    let entries = idx.entries;
    let payload_start = idx.payload_base;
    let payload_end = idx.payload_end;
    let directories = idx.directories;

    std::fs::create_dir_all(output_dir)?;

    let mut extracted = 0usize;
    let mut verified_count = 0usize;
    let mut failed = 0usize;
    let mut total_original: u64 = 0;
    let mut errors: Vec<(String, String)> = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        // Path traversal guard — Audit-2026-05-11 Fix-#45: tighten beyond
        // the prior Unix-only check. Reject:
        //   - Unix-style absolute (`/foo`)
        //   - Windows-style backslash root (`\foo`)
        //   - Parent traversal (`../`, `..\\`, `..`)
        //   - Windows drive letters (`C:\foo`, `C:/foo`)
        //   - UNC paths (`\\server\share`)
        //   - NUL bytes (filesystem termination smuggling)
        //   - Windows Alternate Data Streams (`file:stream`) and reserved
        //     device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1..9`,
        //     `LPT1..9`) which the Win32 layer treats as device handles.
        if path_is_unsafe(&entry.path) {
            eprintln!("  SKIP {}: unsafe path (traversal or absolute)", entry.path);
            errors.push((entry.path.clone(), "unsafe path".into()));
            failed += 1;
            continue;
        }

        // Audit-2026-05-11 Fix-#8: checked_sub on the payload-section
        // size so a malformed archive with payload_start > payload_end
        // cannot underflow the bounds comparison.
        let payload_section_size = payload_end.checked_sub(payload_start).ok_or_else(
            || -> Box<dyn std::error::Error + Send + Sync> {
                "archive payload_start > payload_end (underflow)".into()
            },
        )?;

        // Bounds check: verify offset + size fits within archive payload section.
        if entry.offset.saturating_add(entry.compressed_size) > payload_section_size {
            eprintln!(
                "  FAIL {}: entry exceeds archive bounds (offset={}, size={}, max={})",
                entry.path, entry.offset, entry.compressed_size, payload_section_size
            );
            errors.push((entry.path.clone(), "exceeds archive bounds".into()));
            failed += 1;
            continue;
        }

        // Audit-2026-05-11 Fix-#9: cap per-entry allocation against
        // MAX_ENTRY_DECOMPRESS_SIZE. Without this an attacker can craft
        // a manifest claiming compressed_size = 10 GB, passing the
        // bounds check (which only verifies the offset+size fits in the
        // archive — irrelevant if the archive itself is enormous), and
        // force a 10 GB malloc per entry.
        if entry.compressed_size > MAX_ENTRY_DECOMPRESS_SIZE {
            eprintln!(
                "  FAIL {}: compressed_size {} exceeds MAX_ENTRY_DECOMPRESS_SIZE {}",
                entry.path, entry.compressed_size, MAX_ENTRY_DECOMPRESS_SIZE
            );
            errors.push((
                entry.path.clone(),
                format!(
                    "compressed_size {} > {}",
                    entry.compressed_size, MAX_ENTRY_DECOMPRESS_SIZE
                ),
            ));
            failed += 1;
            continue;
        }
        // Defense in depth: a 10 MB compressed entry could claim
        // original_size = 100 GB and OOM the decode path. The
        // compressed-size guard above doesn't catch that.
        if entry.original_size > MAX_ENTRY_ORIGINAL_SIZE {
            eprintln!(
                "  FAIL {}: original_size {} exceeds MAX_ENTRY_ORIGINAL_SIZE {}",
                entry.path, entry.original_size, MAX_ENTRY_ORIGINAL_SIZE
            );
            errors.push((
                entry.path.clone(),
                format!(
                    "original_size {} > {}",
                    entry.original_size, MAX_ENTRY_ORIGINAL_SIZE
                ),
            ));
            failed += 1;
            continue;
        }

        let out_path = output_dir.join(&entry.path);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
            // ADR 0021 Tier 2 audit (N2): refuse to write through
            // a symlinked parent directory. `create_dir_all`
            // follows existing symlinks silently; an attacker who
            // plants a symlink under `output_dir` before unpack
            // runs can redirect entries outside the destination.
            let meta = std::fs::symlink_metadata(parent).map_err(|e| {
                format!("Cannot stat parent {}: {}", parent.display(), e)
            })?;
            if meta.file_type().is_symlink() {
                errors.push((
                    entry.path.clone(),
                    format!(
                        "refusing to write through symlinked parent {} \
                         (path traversal protection)",
                        parent.display()
                    ),
                ));
                failed += 1;
                continue;
            }
        }

        // Seek to payload
        // ADR 0022 Group C: checked_add for symmetry with
        // extract_entry. Bounds-checked elsewhere via
        // payload_section_size math, but the raw `+` is fragile
        // against future refactors that might relax that math.
        f.seek(SeekFrom::Start(
            payload_start
                .checked_add(entry.offset)
                .ok_or("payload_start + entry.offset overflow")?,
        ))?;
        let mut compressed = vec![0u8; bounded_alloc_usize(entry.compressed_size, &entry.path)?];
        f.read_exact(&mut compressed)?;

        // Decompress based on method
        let data = match entry.method {
            Method::Lml => {
                // Reconstruct original EDF from LML payload. Hint
                // the scratch tempfile at the unpack destination
                // volume to dodge tmpfs ENOSPC on corpus-wide
                // unpacks of multi-GB archives.
                //
                // ADR 0023 Track A-3: For ingest-synthesised entries
                // (e.g. Bonn ASCII), the manifest's `original_size`
                // is the ASCII byte count, NOT the synthesised EDF
                // byte count. Pass `None` so decode_lml_to_edf
                // doesn't size-check against the wrong reference.
                let original_size_hint = if entry.synthetic_from.is_some() {
                    None
                } else {
                    Some(entry.original_size)
                };
                let edf_bytes = match decode_lml_to_edf(
                    &compressed,
                    original_size_hint,
                    Some(output_dir),
                ) {
                    Ok(edf_bytes) => edf_bytes,
                    Err(e) => {
                        eprintln!("  FAIL EDF reconstruct {}: {}", entry.path, e);
                        errors.push((entry.path.clone(), format!("EDF reconstruct failed: {}", e)));
                        failed += 1;
                        continue;
                    }
                };
                // If this entry was ingest-synthesised, re-emit the
                // original non-EDF bytes from the recovered samples
                // + manifest template. SHA-256 verify below catches
                // any mismatch.
                if let Some(sf) = &entry.synthetic_from {
                    match re_emit_synthetic(&edf_bytes, sf) {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            eprintln!(
                                "  FAIL re-emit {} (synthetic_from={}): {}",
                                entry.path, sf.format, e
                            );
                            errors.push((
                                entry.path.clone(),
                                format!("synthetic re-emit failed: {}", e),
                            ));
                            failed += 1;
                            continue;
                        }
                    }
                } else {
                    edf_bytes
                }
            }
            Method::Store => compressed,
            Method::Zstd => match decode_zstd_bounded(
                &compressed,
                entry
                    .original_size
                    .min(MAX_ENTRY_ORIGINAL_SIZE) as usize,
                &entry.path,
            ) {
                Ok(decompressed) => decompressed,
                Err(e) => {
                    eprintln!("  FAIL decompress {}: {}", entry.path, e);
                    errors.push((entry.path.clone(), format!("decompress failed: {}", e)));
                    failed += 1;
                    continue;
                }
            },
        };

        // SHA-256 verify: for all entries, compare against manifest hash
        // (LML entries now reconstruct the original EDF, so the hash matches)
        if verify {
            let hash = sha256_hex(&data);
            if hash != entry.sha256 {
                let expected = &entry.sha256[..entry.sha256.len().min(12)];
                let got = &hash[..hash.len().min(12)];
                eprintln!(
                    "  SHA-256 MISMATCH: {} (expected {}, got {})",
                    entry.path, expected, got
                );
                errors.push((entry.path.clone(), "SHA-256 mismatch".into()));
                failed += 1;
                continue;
            }
            verified_count += 1;
        }

        // Audit-2026-05-11 Fix-#23: atomic write via `.tmp` + rename so
        // a kill / power-loss mid-extract leaves either the previous
        // version of the file or no file at all — never a half-written
        // truncated file with the original path. SHA-256 is verified
        // pre-write (above); atomic rename closes the post-write window.
        // ADR 0021 Tier 2 audit (N1): suffix with PID + atomic
        // counter so two concurrent `unpack_archive` calls into
        // the same `output_dir` don't collide on `.tmp.extract`.
        // Previously a fixed suffix → second writer's bytes
        // landed at the same tmp_path, the rename winner ended
        // up with bytes from whichever process finished last.
        let pid = std::process::id();
        let seq = APPEND_TMP_SEQ.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let tmp_path = {
            let mut p = out_path.clone();
            let fname = p.file_name().map(|n| n.to_os_string()).unwrap_or_default();
            let mut tmp_name = fname;
            tmp_name.push(alloc::format!(".tmp.extract.{}.{}", pid, seq).as_str());
            p.set_file_name(tmp_name);
            p
        };
        std::fs::write(&tmp_path, &data)?;

        // Restore original Unix permission bits BEFORE mtime so the
        // chmod itself doesn't bump mtime. Unix-only; ignored on
        // Windows. Defends against the regression where 600 / 755
        // files all extracted as 644. Applied to the .tmp file so the
        // final rename atomically swaps in a file with correct perms.
        #[cfg(unix)]
        if let Some(mode_bits) = entry.mode {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(mode_bits));
        }

        // Restore original modification time (sec + nanosec precision).
        if let Some(mtime_secs) = entry.mtime {
            let nanos = entry.mtime_nanos.unwrap_or(0);
            let mtime = filetime::FileTime::from_unix_time(mtime_secs as i64, nanos);
            // Audit-2026-05-11 Fix-#37: warn on Windows non-admin or
            // similar mtime-set failures; previously silently swallowed
            // so provenance dropped without diagnostic.
            if let Err(e) = filetime::set_file_mtime(&tmp_path, mtime) {
                eprintln!(
                    "WARNING: set_file_mtime failed for {}: {} (provenance loss)",
                    tmp_path.display(),
                    e
                );
            }
        }

        std::fs::rename(&tmp_path, &out_path)?;

        extracted += 1;
        total_original += entry.original_size;

        if verbose && (i + 1) % 500 == 0 {
            eprintln!("    {}/{} files extracted...", i + 1, entries.len());
        }
        if let Some(prog) = progress_fn {
            prog(i + 1, entries.len(), &entry.path);
        }
    }

    // Restore directory modification times (must happen AFTER all files are written,
    // because writing files into a directory updates its mtime). The
    // `directories` list comes from the chokepoint index (manifest's
    // `directories` array), so this works identically for v1 + v2.
    {
        // Process deepest directories first so parent mtimes aren't overwritten
        let mut dir_entries: Vec<(&str, u64)> = directories
            .iter()
            .map(|(path, mtime)| (path.as_str(), *mtime))
            .collect();
        dir_entries.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // deepest first

        for (path, mtime_secs) in &dir_entries {
            let dir_path = output_dir.join(path);
            if dir_path.is_dir() {
                let mtime = filetime::FileTime::from_unix_time(*mtime_secs as i64, 0);
                // Audit-2026-05-11 Fix-#37: warn instead of swallow.
                if let Err(e) = filetime::set_file_mtime(&dir_path, mtime) {
                    eprintln!(
                        "WARNING: set_file_mtime failed for dir {}: {}",
                        dir_path.display(),
                        e
                    );
                }
            }
        }
    }

    if verbose {
        eprintln!(
            "  {} files extracted, {} verified, {} failed",
            extracted, verified_count, failed
        );
    }

    let archive_bytes = file_size;
    let cr = if archive_bytes > 0 {
        total_original as f64 / archive_bytes as f64
    } else {
        0.0
    };

    Ok(ArchiveSummary {
        n_files: extracted,
        original_bytes: total_original,
        archive_bytes,
        cr,
        counts_lml: entries.iter().filter(|e| e.method == Method::Lml).count(),
        counts_zstd: entries.iter().filter(|e| e.method == Method::Zstd).count(),
        counts_store: entries.iter().filter(|e| e.method == Method::Store).count(),
        errors,
    })
}

/// Read a single named entry from an LMA archive into RAM without
/// unpacking the entire archive to disk.
///
/// Returns the decompressed bytes for the named entry. Used by
/// training dataloaders that want random access to individual `.lml`
/// payloads (or sidecars) inside per-recording LMA bundles, avoiding
/// any tempfile round-trip.
///
/// Method dispatch:
///   - `Method::Store`  → raw stored bytes (no-op).
///   - `Method::Zstd`   → zstd-decompressed bytes.
///   - `Method::Lml`    → raw LML packet bytes (NOT reconstructed EDF).
///                        The original source for this entry was an EDF
///                        that was LML-encoded at pack time; the manifest's
///                        `sha256` is the EDF hash, so SHA verification of
///                        the LML payload alone is NOT performed here.
///                        Pass these bytes straight to `lml::decompress`.
///                        Use `unpack_archive` if you want EDF bytes back.
///
/// Errors:
///   - Archive too small / bad magic / unsupported version.
///   - Manifest length above cap (likely corrupt or malicious).
///   - Named entry not in manifest.
///   - Entry offset/size outside the payload section (corrupt manifest).
///   - I/O failure, zstd decompression failure.
pub fn read_entry(
    archive_path: &Path,
    entry_name: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let mut f = BufReader::new(std::fs::File::open(archive_path)?);
    let file_size = std::fs::metadata(archive_path)?.len();

    // Single dispatch chokepoint: resolves v1/v2 layout, returns parsed
    // entries + payload bounds. payload_base + entry.offset = absolute
    // byte position of the entry's payload.
    let idx = read_lma_index(&mut f, file_size)?;
    let entries = idx.entries;
    let payload_start = idx.payload_base;
    let payload_end = idx.payload_end;

    let entry = entries.iter().find(|e| e.path == entry_name).ok_or_else(
        || -> Box<dyn std::error::Error + Send + Sync> {
            format!(
                "Entry '{}' not found in archive {}",
                entry_name,
                archive_path.display()
            )
            .into()
        },
    )?;

    // Payload section bounds (same math as unpack_archive).
    let payload_section_size = payload_end.checked_sub(payload_start).ok_or_else(
        || -> Box<dyn std::error::Error + Send + Sync> {
            "archive payload_start > payload_end (underflow)".into()
        },
    )?;

    if entry.offset.saturating_add(entry.compressed_size) > payload_section_size {
        return Err(format!(
            "Entry '{}' exceeds archive bounds (offset={}, size={}, max={})",
            entry.path, entry.offset, entry.compressed_size, payload_section_size
        )
        .into());
    }

    // Same per-entry alloc cap as unpack_archive (Fix-#9).
    if entry.compressed_size > MAX_ENTRY_DECOMPRESS_SIZE {
        return Err(format!(
            "Entry '{}' compressed_size {} exceeds cap {}",
            entry.path, entry.compressed_size, MAX_ENTRY_DECOMPRESS_SIZE
        )
        .into());
    }
    if entry.original_size > MAX_ENTRY_ORIGINAL_SIZE {
        return Err(format!(
            "Entry '{}' original_size {} exceeds cap {}",
            entry.path, entry.original_size, MAX_ENTRY_ORIGINAL_SIZE
        )
        .into());
    }

    f.seek(SeekFrom::Start(payload_start + entry.offset))?;
    let mut compressed = vec![0u8; bounded_alloc_usize(entry.compressed_size, &entry.path)?];
    f.read_exact(&mut compressed)?;

    let data = match entry.method {
        Method::Store => compressed,
        Method::Zstd => decode_zstd_bounded(
            &compressed,
            entry
                .original_size
                .min(MAX_ENTRY_ORIGINAL_SIZE) as usize,
            &entry.path,
        )?,
        Method::Lml => compressed, // raw LML bytes, caller feeds to lml::decompress
    };

    // ADR 0021 Tier 2 audit (N6): SHA-256 verify the bytes
    // returned to the caller. Pre-fix `read_entry` skipped this
    // check for ALL methods, so silently-corrupted archive bytes
    // flowed straight to downstream tooling. For Store + Zstd the
    // returned bytes are the ORIGINAL file content and the
    // manifest's `entry.sha256` is the hash of that content --
    // direct compare. For Lml the manifest's sha256 is the
    // pre-encode EDF hash; verifying the LML payload alone is
    // structurally wrong (the bytes were never SHA'd at pack
    // time), so we skip the check for Lml -- callers must SHA
    // after `lml::decompress`. Documented in the function
    // docstring.
    if !matches!(entry.method, Method::Lml) {
        let hash = sha256_hex(&data);
        if hash != entry.sha256 {
            return Err(format!(
                "lma read_entry: SHA-256 mismatch for '{}' (manifest {} vs reconstructed {}); \
                 archive bytes are corrupted",
                entry.path,
                &entry.sha256[..entry.sha256.len().min(12)],
                &hash[..hash.len().min(12)]
            )
            .into());
        }
    }

    Ok(data)
}

/// Batch ranged-header read: parse the LMA index ONCE, then for each
/// requested name read only the first `n_bytes` of its payload.
///
/// Motivation (#229): callers that only need a recording's LML
/// container *header* (e.g. `duration_s` → training window count) were
/// going through `read_entry`, which (a) re-parses the whole footer
/// manifest on every call (~130 ms on the 70 K-entry TUEG LMA) and
/// (b) reads the ENTIRE entry (~6.67 MB) just to look at its first few
/// KB. On TUEG that is ~700 GB of reads + hours of repeated manifest
/// parsing. This helper amortises the index parse across all requested
/// names (one `read_lma_index` call) and reads only a prefix per entry.
///
/// Prefix validity: a prefix read is only meaningful for entries stored
/// **raw** — `Method::Lml` (raw LML packet bytes) and `Method::Store`
/// (stored uncompressed). For `Method::Zstd` the first `n_bytes` are a
/// truncated zstd stream that cannot be partially decoded into a usable
/// container, so those entries return `None` and the caller must fall
/// back to the full `read_entry`/`read_entry_decoded` path.
///
/// Returns a `Vec` aligned 1:1 with `names`:
///   - `Some(bytes)` — entry found, raw method, prefix read OK. `bytes`
///     length is `min(n_bytes, entry.compressed_size)`. The caller is
///     responsible for tolerating a too-short prefix (e.g. an LML
///     container whose metadata JSON exceeds `n_bytes`): parse the
///     prefix and, on a truncation error, fall back to the full read.
///   - `None` — name not in the manifest, OR a compressed
///     (`Method::Zstd`) entry, OR an out-of-bounds / oversized entry.
///     The caller falls back to the full read in every `None` case.
///
/// This function is purely additive and does NOT alter the behaviour of
/// `read_lma_index`, `read_entry`, or `read_entry_decoded`. It performs
/// NO SHA verification (it returns a partial payload by design); callers
/// that need integrity-checked full bytes must use `read_entry`.
pub fn read_entry_headers<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    names: &[String],
    n_bytes: usize,
) -> Result<Vec<Option<Vec<u8>>>, Box<dyn std::error::Error + Send + Sync>> {
    // Parse the on-disk index exactly ONCE for the whole batch.
    let idx = read_lma_index(reader, file_size)?;
    let payload_start = idx.payload_base;
    let payload_end = idx.payload_end;
    let payload_section_size = payload_end.checked_sub(payload_start).ok_or_else(
        || -> Box<dyn std::error::Error + Send + Sync> {
            "archive payload_start > payload_end (underflow)".into()
        },
    )?;

    // Name → entry lookup so each requested name is O(1) instead of a
    // linear scan over the (potentially 70 K-entry) manifest.
    let mut by_name: std::collections::HashMap<&str, &ArchiveEntry> =
        std::collections::HashMap::with_capacity(idx.entries.len());
    for e in &idx.entries {
        by_name.insert(e.path.as_str(), e);
    }

    let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(names.len());
    for name in names {
        let Some(entry) = by_name.get(name.as_str()).copied() else {
            out.push(None);
            continue;
        };
        // Only raw-stored tiers can be prefix-read meaningfully.
        if !matches!(entry.method, Method::Lml | Method::Store) {
            out.push(None);
            continue;
        }
        // Reject entries whose declared bounds fall outside the payload
        // section (corrupt manifest) — same guard as read_entry.
        if entry.offset.saturating_add(entry.compressed_size) > payload_section_size {
            out.push(None);
            continue;
        }
        // How many bytes to actually read: the smaller of the caller's
        // budget and the entry's stored size. `read_len` fits in usize
        // because n_bytes is already usize and we min() against it.
        let avail = entry.compressed_size.min(n_bytes as u64);
        let read_len = avail as usize;
        if read_len == 0 {
            // Empty entry (or n_bytes == 0): nothing to read; hand back
            // an empty buffer so the caller can decide (parse will fail
            // and trigger fallback).
            out.push(Some(Vec::new()));
            continue;
        }
        if reader
            .seek(SeekFrom::Start(payload_start + entry.offset))
            .is_err()
        {
            out.push(None);
            continue;
        }
        let mut buf = vec![0u8; read_len];
        match reader.read_exact(&mut buf) {
            Ok(()) => out.push(Some(buf)),
            Err(_) => out.push(None),
        }
    }
    Ok(out)
}

/// Path-based convenience wrapper over [`read_entry_headers`] that opens
/// the archive, stats its size, and runs the batch in one call. Used by
/// the PyO3 binding (`lma_entry_headers`). Additive — see
/// `read_entry_headers` for semantics.
pub fn read_entry_headers_path(
    archive_path: &Path,
    names: &[String],
    n_bytes: usize,
) -> Result<Vec<Option<Vec<u8>>>, Box<dyn std::error::Error + Send + Sync>> {
    let mut f = BufReader::new(std::fs::File::open(archive_path)?);
    let file_size = std::fs::metadata(archive_path)?.len();
    read_entry_headers(&mut f, file_size, names, n_bytes)
}

/// Like `read_entry`, but for `Method::Lml` entries the raw LML
/// payload is transparently decoded back to the byte-identical
/// EDF/BDF source via `decode_lml_to_edf`. The returned bytes match
/// what the original file produced when it was encoded into the
/// archive — i.e. `cat`ing the result equals `cat`ing the original
/// EDF off disk.
///
/// Use this when the caller wants the *file-shaped* payload (FUSE
/// mount, `lml cat`, in-process integrations) and shouldn't have to
/// know about the underlying storage tier. `read_entry` (raw) stays
/// available for callers that explicitly want the LML wire bytes
/// (re-pack, signature verification, low-level inspection).
///
/// Store / Zstd entries decode identically in both functions.
pub fn read_entry_decoded(
    archive_path: &Path,
    entry_name: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    // We can't call `read_entry` directly because it strips the
    // entry's storage method on the way out. Re-walk the header /
    // manifest so we know which tier the entry sits on.
    let mut f = BufReader::new(std::fs::File::open(archive_path)?);
    let file_size = std::fs::metadata(archive_path)?.len();
    // Single dispatch chokepoint: v1/v2 layout + parsed entries.
    let idx = read_lma_index(&mut f, file_size)?;
    let entries = idx.entries;
    let payload_start = idx.payload_base;
    let payload_end = idx.payload_end;
    let entry = entries.iter().find(|e| e.path == entry_name).ok_or_else(
        || -> Box<dyn std::error::Error + Send + Sync> {
            format!(
                "Entry '{}' not found in archive {}",
                entry_name,
                archive_path.display()
            )
            .into()
        },
    )?;
    let payload_section_size = payload_end.checked_sub(payload_start).ok_or_else(
        || -> Box<dyn std::error::Error + Send + Sync> {
            "archive payload_start > payload_end (underflow)".into()
        },
    )?;
    if entry.offset.saturating_add(entry.compressed_size) > payload_section_size {
        return Err(format!(
            "Entry '{}' exceeds archive bounds (offset={}, size={}, max={})",
            entry.path, entry.offset, entry.compressed_size, payload_section_size
        )
        .into());
    }
    if entry.compressed_size > MAX_ENTRY_DECOMPRESS_SIZE {
        return Err(format!(
            "Entry '{}' compressed_size {} exceeds cap {}",
            entry.path, entry.compressed_size, MAX_ENTRY_DECOMPRESS_SIZE
        )
        .into());
    }
    if entry.original_size > MAX_ENTRY_ORIGINAL_SIZE {
        return Err(format!(
            "Entry '{}' original_size {} exceeds cap {}",
            entry.path, entry.original_size, MAX_ENTRY_ORIGINAL_SIZE
        )
        .into());
    }
    f.seek(SeekFrom::Start(payload_start + entry.offset))?;
    let mut compressed = vec![0u8; bounded_alloc_usize(entry.compressed_size, &entry.path)?];
    f.read_exact(&mut compressed)?;

    let data = match entry.method {
        Method::Store => compressed,
        Method::Zstd => decode_zstd_bounded(
            &compressed,
            entry
                .original_size
                .min(MAX_ENTRY_ORIGINAL_SIZE) as usize,
            &entry.path,
        )?,
        // In-memory decode used by callers who want bytes back, not
        // a file output. ADR 0021 Tier 2 N6: pass `archive_path
        // .parent()` as the tmp-dir hint so the LML scratch
        // tempfile lands on the same volume as the input archive
        // (not /tmp/tmpfs which the audit found hits ENOSPC on
        // multi-GB extracts).
        Method::Lml => decode_lml_to_edf(
            &compressed,
            Some(entry.original_size),
            archive_path.parent(),
        )?,
    };

    // ADR 0021 Tier 2 audit (N6): SHA-256 verify the reconstructed
    // bytes against the manifest. read_entry_decoded was
    // documented as "returns reconstructed EDF" but never checked
    // the manifest's sha256 -- a tampered LML or Zstd payload
    // produced bytes other than the original and the caller
    // silently accepted them.
    let hash = sha256_hex(&data);
    if hash != entry.sha256 {
        return Err(format!(
            "lma read_entry_decoded: SHA-256 mismatch for '{}' (manifest {} vs reconstructed {}); \
             archive bytes are corrupted",
            entry.path,
            &entry.sha256[..entry.sha256.len().min(12)],
            &hash[..hash.len().min(12)]
        )
        .into());
    }
    Ok(data)
}

// ─── Internal helpers ──────────────────────────────────────────────

fn build_manifest_json(
    entries: &[ArchiveEntry],
    zstd_level: i32,
    dir_mtimes: &[(String, u64)],
) -> String {
    let mut json = String::with_capacity(entries.len() * 200);
    json.push_str("{\"compressor\":\"zstd\",\"compressor_level\":");
    json.push_str(&zstd_level.to_string());
    json.push_str(",\"files\":[");
    for (i, entry) in entries.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        // Escape path for JSON safety (backslashes, quotes, control chars)
        let safe_path = entry
            .path
            .replace('\\', "/")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        let mtime_str = entry
            .mtime
            .map(|t| format!(",\"mtime\":{}", t))
            .unwrap_or_default();
        let mtime_nanos_str = entry
            .mtime_nanos
            .filter(|&n| n != 0)
            .map(|n| format!(",\"mtime_nanos\":{}", n))
            .unwrap_or_default();
        // Mode emitted only when present (Unix archive). Reader treats
        // missing mode as None and skips the chmod step on extract.
        let mode_str = entry
            .mode
            .map(|m| format!(",\"mode\":{}", m))
            .unwrap_or_default();
        // ADR 0023 Track A: synthetic_from is serialised inline only
        // when present. serde_json::to_string can't fail on a Value;
        // unwrap_or("null") preserves the manifest's well-formedness
        // if it ever did (Bible R5 — never let a serialiser failure
        // produce an invalid JSON entry).
        let synthetic_from_str = entry
            .synthetic_from
            .as_ref()
            .map(|sf| {
                // serde_json::Value -> string is infallible; the
                // outer Option from to_string handles a hypothetical
                // recursion-depth limit that flat Value trees never
                // hit. Default to "null" if it ever did to keep the
                // manifest as well-formed JSON.
                let template_str = serde_json::to_string(&sf.template_json)
                    .unwrap_or_else(|_| "null".into());
                // V4 Pro review nit (A-2#1): NaN/Inf would serialise
                // to the literal `NaN`/`Inf` which is invalid JSON.
                // Floor to a JSON-safe finite number; preserve the
                // user-visible value otherwise. Practical impact is
                // diagnostic — synthesis caller only emits finite
                // sample_rate today.
                let sr = if sf.sample_rate.is_finite() {
                    sf.sample_rate
                } else {
                    0.0
                };
                format!(
                    ",\"synthetic_from\":{{\"format\":\"{}\",\"sample_rate\":{},\"template\":{}}}",
                    sf.format, sr, template_str
                )
            })
            .unwrap_or_default();
        json.push_str(&format!(
            concat!(
                "{{\"path\":\"{}\",",
                "\"original_size\":{},",
                "\"compressed_size\":{},",
                "\"method\":\"{}\",",
                "\"sha256\":\"{}\",",
                "\"offset\":{}{}{}{}{}}}",
            ),
            safe_path,
            entry.original_size,
            entry.compressed_size,
            entry.method.as_str(),
            entry.sha256,
            entry.offset,
            mtime_str,
            mtime_nanos_str,
            mode_str,
            synthetic_from_str,
        ));
    }
    json.push_str("],\"directories\":[");
    for (i, (path, mtime)) in dir_mtimes.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let safe_path = path.replace('\\', "/").replace('"', "\\\"");
        json.push_str(&format!(
            "{{\"path\":\"{}\",\"mtime\":{}}}",
            safe_path, mtime
        ));
    }
    json.push_str("]}");
    json
}

fn read_manifest(
    archive_path: &Path,
) -> Result<Vec<ArchiveEntry>, Box<dyn std::error::Error + Send + Sync>> {
    // Thin wrapper over the single dispatch chokepoint. `list_archive`,
    // `extract_entry`, and `append_entry` all source their entries here,
    // so routing through `read_lma_index` makes all three v2-correct for
    // free (they only need the parsed entries, not the payload bounds).
    let mut f = BufReader::new(std::fs::File::open(archive_path)?);
    let file_size = std::fs::metadata(archive_path)?.len();
    Ok(read_lma_index(&mut f, file_size)?.entries)
}

fn parse_manifest_entries(
    json: &serde_json::Value,
) -> Result<Vec<ArchiveEntry>, Box<dyn std::error::Error + Send + Sync>> {
    let files = if let Some(arr) = json.as_array() {
        arr // old format: manifest is directly a list
    } else if let Some(arr) = json.get("files").and_then(|v| v.as_array()) {
        arr // new format: manifest is { compressor, files: [...] }
    } else {
        return Ok(Vec::new());
    };

    // ADR 0021 Tier 2 audit (N9): pre-fix used `filter_map`, which
    // silently DROPPED any manifest entry whose required field was
    // missing or wrong-typed. An archive with one corrupt manifest
    // entry loaded with that entry invisibly missing -- operators
    // ran `lma list` and never saw the gap. Now: any malformed
    // entry returns Err with the offending entry's index + the
    // specific field that failed, so the audit trail captures
    // the corruption rather than hiding it.
    let mut out = Vec::with_capacity(files.len());
    for (idx, entry) in files.iter().enumerate() {
        let parse_err = |field: &str| -> Box<dyn std::error::Error + Send + Sync> {
            format!(
                "lma manifest entry [{}]: missing or malformed `{}` field",
                idx, field
            )
            .into()
        };
        let path = entry
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| parse_err("path"))?
            .to_string();
        let original_size = entry
            .get("original_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| parse_err("original_size"))?;
        let compressed_size = entry
            .get("compressed_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| parse_err("compressed_size"))?;
        let method_str = entry
            .get("method")
            .and_then(|v| v.as_str())
            .ok_or_else(|| parse_err("method"))?;
        let method = Method::from_str(method_str).ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
            format!(
                "lma manifest entry [{}]: unknown method '{}'",
                idx, method_str
            )
            .into()
        })?;
        let sha256 = entry
            .get("sha256")
            .and_then(|v| v.as_str())
            .ok_or_else(|| parse_err("sha256"))?
            .to_string();
        let offset = entry
            .get("offset")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| parse_err("offset"))?;
        out.push(ArchiveEntry {
            path,
            original_size,
            compressed_size,
            method,
            sha256,
            offset,
            mtime: entry.get("mtime").and_then(|v| v.as_u64()),
            // Sub-second mtime — pre-fix archives lack this field.
            mtime_nanos: entry
                .get("mtime_nanos")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32),
            // Missing on Windows-produced or pre-mode archives.
            // Casting from u64 → u32 is safe; Unix mode bits fit
            // in 16 bits (we only use the low 9 anyway).
            // ADR 0022 Group C: use try_from instead of `m as u32`
            // so a manifest claiming `mode = u64::MAX` doesn't
            // silently truncate to 0xFFFFFFFF and look like a
            // legitimate Unix mode. Out-of-range values are
            // dropped + warned to stderr.
            mode: entry.get("mode").and_then(|v| v.as_u64()).and_then(|m| {
                u32::try_from(m).ok().or_else(|| {
                    eprintln!(
                        "  WARNING: lma manifest entry mode {} doesn't fit in u32; dropping",
                        m
                    );
                    None
                })
            }),
            // ADR 0023 Track A — optional nested object.
            // Schema:
            //   "synthetic_from": {
            //       "format": "ascii_int_lines",
            //       "original_sha256": "<hex>",
            //       "sample_rate": 173.61,
            //       "template": { ... format-specific ... }
            //   }
            // Missing on regular archives. Refuse partial objects
            // (Bible R5) — half-populated synthetic_from is data
            // corruption, not a fallback case.
            // V4 Pro review nit (A-2#2): treat both absent key AND
            // explicit `null` as "no synthesis". Per ADR 0023 spec
            // synthetic_from is opt-in; explicit-null is the
            // strongest possible "no" and shouldn't error.
            synthetic_from: match entry.get("synthetic_from") {
                None | Some(serde_json::Value::Null) => None,
                Some(sf) => {
                    let format = sf
                        .get("format")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| parse_err("synthetic_from.format"))?
                        .to_string();
                    let sample_rate = sf
                        .get("sample_rate")
                        .and_then(|v| v.as_f64())
                        .ok_or_else(|| parse_err("synthetic_from.sample_rate"))?;
                    let template_json = sf
                        .get("template")
                        .cloned()
                        .ok_or_else(|| parse_err("synthetic_from.template"))?;
                    Some(SyntheticFromInfo {
                        format,
                        sample_rate,
                        template_json,
                    })
                }
            },
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ─── Method enum + dispatch ────────────────────────────────

    #[test]
    fn method_as_str_known_values() {
        assert_eq!(Method::Lml.as_str(), "lml");
        assert_eq!(Method::Zstd.as_str(), "secondary");
        assert_eq!(Method::Store.as_str(), "store");
    }

    #[test]
    fn method_from_str_known_aliases() {
        assert_eq!(Method::from_str("lml"), Some(Method::Lml));
        assert_eq!(Method::from_str("secondary"), Some(Method::Zstd));
        assert_eq!(Method::from_str("zstd"), Some(Method::Zstd));
        assert_eq!(Method::from_str("store"), Some(Method::Store));
    }

    #[test]
    fn method_from_str_unknown_returns_none() {
        assert_eq!(Method::from_str("future"), None);
        assert_eq!(Method::from_str(""), None);
        assert_eq!(Method::from_str("LML"), None); // case-sensitive
        assert_eq!(Method::from_str("brotli"), None);
    }

    /// Audit-2026-05-11 Fix-#45: path_is_unsafe rejects every known
    /// traversal / device-name attack vector across Unix + Windows.
    #[test]
    fn path_is_unsafe_rejects_attacks() {
        let attacks = [
            "/etc/passwd",           // unix abs
            "\\Windows\\System32",   // win backslash root
            "../escape",             // parent traversal
            "..\\escape",            // win parent traversal
            "..",                    // exact parent
            "C:\\Windows\\evil.exe", // drive letter
            "Z:/foo/bar",            // drive letter mixed slash
            "\\\\server\\share",     // UNC
            "file\0name",            // NUL byte
            "CON",                   // DOS device
            "PRN.txt",               // DOS device with extension
            "subdir/COM1",           // DOS device in subdir
            "file:stream",           // ADS
            "",                      // empty
        ];
        for p in &attacks {
            assert!(path_is_unsafe(p), "should reject: {p:?}");
        }
    }

    #[test]
    fn path_is_unsafe_accepts_normal_paths() {
        let safe = [
            "data.edf",
            "subdir/data.edf",
            "deep/nested/path/file.bin",
            "file_with_dots.0.5.lml",
            "name with spaces.edf",
            "..hidden", // not parent traversal — just a dot prefix
        ];
        for p in &safe {
            assert!(!path_is_unsafe(p), "should accept: {p:?}");
        }
    }

    #[test]
    fn method_roundtrip_str() {
        for m in [Method::Lml, Method::Zstd, Method::Store] {
            assert_eq!(Method::from_str(m.as_str()), Some(m));
        }
    }

    // ─── choose_method extension table ──────────────────────────

    #[test]
    fn choose_method_edf_bdf_is_lml() {
        assert_eq!(choose_method(Path::new("recording.edf")), Method::Lml);
        assert_eq!(choose_method(Path::new("recording.bdf")), Method::Lml);
        assert_eq!(choose_method(Path::new("X.EDF")), Method::Lml); // case-insensitive
    }

    #[test]
    fn choose_method_already_compressed_is_store() {
        for ext in [
            "lml", "lmq", "lma", "gz", "zst", "zip", "7z", "png", "jpg", "jpeg", "mp4", "avi",
        ] {
            let p = format!("file.{}", ext);
            assert_eq!(choose_method(Path::new(&p)), Method::Store, "ext={}", ext);
        }
    }

    #[test]
    fn choose_method_text_default_zstd() {
        assert_eq!(choose_method(Path::new("notes.csv")), Method::Zstd);
        assert_eq!(choose_method(Path::new("README.md")), Method::Zstd);
        assert_eq!(choose_method(Path::new("data.json")), Method::Zstd);
        assert_eq!(choose_method(Path::new("file_no_ext")), Method::Zstd);
    }

    // ─── magic + version constants ──────────────────────────────

    #[test]
    fn magic_and_version_constants_pinned() {
        assert_eq!(LMA_MAGIC, b"LMA1");
        assert_eq!(LMA_VERSION, 1);
        assert_eq!(MAX_MANIFEST_SIZE, 256 * 1024 * 1024);
    }

    // ─── sha256_hex known vector ────────────────────────────────

    #[test]
    fn sha256_hex_empty_input() {
        // RFC 6234 / known: SHA-256("") = e3b0c44298fc1c149afbf4c8996fb924...
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_hex_abc_known_vector() {
        // RFC 6234: SHA-256("abc") = ba7816bf...
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    // ─── pack/unpack roundtrip on tiny tree ─────────────────────

    fn make_tiny_tree() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("notes.csv"), b"a,b,c\n1,2,3\n").unwrap();
        std::fs::write(dir.path().join("blob.bin"), &[0xABu8; 64]).unwrap();
        dir
    }

    #[test]
    fn pack_archive_rejects_nondirectory() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let out = tempfile::NamedTempFile::new().unwrap();
        let r = pack_archive(tmp.path(), out.path(), 9, false, None);
        assert!(r.is_err(), "pack_archive should reject a file as input_dir");
    }

    #[test]
    fn pack_archive_rejects_empty_directory() {
        let dir = tempfile::tempdir().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let archive_path = out_dir.path().join("out.lma");
        let r = pack_archive(dir.path(), &archive_path, 9, false, None);
        assert!(r.is_err(), "pack_archive should reject empty input dir");
    }

    #[test]
    fn pack_then_list_then_unpack_roundtrip() {
        let src = make_tiny_tree();
        let archive = tempfile::NamedTempFile::new().unwrap();
        let summary = pack_archive(src.path(), archive.path(), 9, false, None).unwrap();
        assert_eq!(summary.n_files, 2);
        assert!(summary.archive_bytes > 0);

        let entries = list_archive(archive.path()).unwrap();
        assert_eq!(entries.len(), 2);
        let names: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert!(names.contains(&"notes.csv"));
        assert!(names.contains(&"blob.bin"));

        // Methods chosen by extension
        for e in &entries {
            match e.path.as_str() {
                "notes.csv" => assert_eq!(e.method, Method::Zstd),
                "blob.bin" => assert_eq!(e.method, Method::Zstd), // unknown ext
                other => panic!("unexpected entry {}", other),
            }
            assert!(!e.sha256.is_empty());
            assert_eq!(e.sha256.len(), 64); // hex SHA-256
        }

        let dst = tempfile::tempdir().unwrap();
        let unpack_summary = unpack_archive(archive.path(), dst.path(), true, false, None).unwrap();
        assert_eq!(unpack_summary.n_files, 2);

        // Byte-exact recovery of both files
        let notes = std::fs::read(dst.path().join("notes.csv")).unwrap();
        assert_eq!(notes, b"a,b,c\n1,2,3\n");
        let blob = std::fs::read(dst.path().join("blob.bin")).unwrap();
        assert_eq!(blob, &[0xABu8; 64]);
    }

    #[test]
    fn unpack_rejects_too_small_archive() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"shorty").unwrap(); // < 48 bytes
        let dst = tempfile::tempdir().unwrap();
        let r = unpack_archive(tmp.path(), dst.path(), false, false, None);
        assert!(r.is_err());
    }

    #[test]
    fn read_entry_returns_bytes_for_each_method() {
        // Build a tiny tree exercising all three methods:
        //   - text file → secondary (zstd)
        //   - binary file → secondary (zstd, default for unknown ext)
        //   - .lml file → store (extension list short-circuit)
        let src = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("notes.csv"), b"a,b,c\n1,2,3\n").unwrap();
        std::fs::write(src.path().join("blob.bin"), [0xABu8; 64]).unwrap();
        // Fake LML payload: only matters that the bytes round-trip; LML
        // parsing is not invoked because Store skips it.
        let fake_lml = (0u8..=255).cycle().take(2048).collect::<Vec<u8>>();
        std::fs::write(src.path().join("recording.lml"), &fake_lml).unwrap();

        let archive = tempfile::NamedTempFile::new().unwrap();
        let summary = pack_archive(src.path(), archive.path(), 9, false, None).unwrap();
        assert_eq!(summary.n_files, 3);

        // Confirm method classification
        let entries = list_archive(archive.path()).unwrap();
        for e in &entries {
            match e.path.as_str() {
                "notes.csv" => assert_eq!(e.method, Method::Zstd),
                "blob.bin" => assert_eq!(e.method, Method::Zstd),
                "recording.lml" => assert_eq!(e.method, Method::Store),
                other => panic!("unexpected entry {}", other),
            }
        }

        // Bit-exact roundtrip via read_entry
        let csv_bytes = read_entry(archive.path(), "notes.csv").unwrap();
        assert_eq!(csv_bytes, b"a,b,c\n1,2,3\n");

        let bin_bytes = read_entry(archive.path(), "blob.bin").unwrap();
        assert_eq!(bin_bytes, vec![0xABu8; 64]);

        let lml_bytes = read_entry(archive.path(), "recording.lml").unwrap();
        assert_eq!(lml_bytes, fake_lml);
    }

    #[test]
    fn read_entry_headers_prefix_matches_full_read_for_raw_tiers() {
        // #229: batch ranged-header read. Build a tree with a Store
        // (.lml) entry, a Zstd (.csv) entry, and a second Store (.lml)
        // so the batch path is exercised with > 1 raw entry.
        let src = tempfile::tempdir().unwrap();
        let fake_lml: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let fake_lml2: Vec<u8> = (0u8..=255).rev().cycle().take(1000).collect();
        std::fs::write(src.path().join("a.lml"), &fake_lml).unwrap();
        std::fs::write(src.path().join("b.lml"), &fake_lml2).unwrap();
        std::fs::write(src.path().join("notes.csv"), b"x,y,z\n9,8,7\n").unwrap();

        let archive = tempfile::NamedTempFile::new().unwrap();
        pack_archive(src.path(), archive.path(), 9, false, None).unwrap();

        // Confirm tiers as expected.
        let entries = list_archive(archive.path()).unwrap();
        for e in &entries {
            match e.path.as_str() {
                "a.lml" | "b.lml" => assert_eq!(e.method, Method::Store),
                "notes.csv" => assert_eq!(e.method, Method::Zstd),
                other => panic!("unexpected entry {}", other),
            }
        }

        let names = vec![
            "a.lml".to_string(),
            "notes.csv".to_string(),  // zstd -> None
            "missing.lml".to_string(), // absent -> None
            "b.lml".to_string(),
        ];

        // Generous prefix (>= every raw entry size).
        let out = read_entry_headers_path(archive.path(), &names, 8192).unwrap();
        assert_eq!(out.len(), 4);
        // a.lml: full prefix == full read (n_bytes > size -> min == size).
        assert_eq!(out[0].as_deref(), Some(fake_lml.as_slice()));
        // notes.csv is zstd -> None (prefix of a zstd stream is useless).
        assert!(out[1].is_none());
        // missing entry -> None.
        assert!(out[2].is_none());
        // b.lml: full bytes back.
        assert_eq!(out[3].as_deref(), Some(fake_lml2.as_slice()));

        // Truncating prefix: 100 bytes -> exactly the first 100 bytes of
        // the raw payload, which (Store tier) equals the original file.
        let out_short = read_entry_headers_path(
            archive.path(),
            &["a.lml".to_string()],
            100,
        )
        .unwrap();
        assert_eq!(out_short[0].as_deref(), Some(&fake_lml[..100]));

        // n_bytes == 0 -> empty buffer (caller-decides), not None.
        let out_zero =
            read_entry_headers_path(archive.path(), &["a.lml".to_string()], 0).unwrap();
        assert_eq!(out_zero[0].as_deref(), Some(&[][..]));

        // Index is parsed ONCE per call regardless of name count: a
        // batch of all names returns aligned results.
        let all: Vec<String> = entries.iter().map(|e| e.path.clone()).collect();
        let out_all = read_entry_headers_path(archive.path(), &all, 8192).unwrap();
        assert_eq!(out_all.len(), all.len());
    }

    #[test]
    fn read_entry_headers_rejects_corrupt_archive() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"shorty").unwrap(); // < 48 bytes
        let r = read_entry_headers_path(tmp.path(), &["anything".to_string()], 8192);
        assert!(r.is_err());
    }

    #[test]
    fn read_entry_missing_entry_returns_err() {
        let src = make_tiny_tree();
        let archive = tempfile::NamedTempFile::new().unwrap();
        pack_archive(src.path(), archive.path(), 9, false, None).unwrap();
        let r = read_entry(archive.path(), "no_such_file.bin");
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn read_entry_rejects_corrupt_archive() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"shorty").unwrap(); // < 48 bytes
        let r = read_entry(tmp.path(), "anything");
        assert!(r.is_err());
    }

    #[test]
    fn verify_archive_detects_byte_flip() {
        let src = make_tiny_tree();
        let archive = tempfile::NamedTempFile::new().unwrap();
        pack_archive(src.path(), archive.path(), 9, false, None).unwrap();

        // Flip a byte in the middle of the archive (after the 16-byte header
        // and into the manifest/payload region).
        let mut bytes = std::fs::read(archive.path()).unwrap();
        let flip_idx = bytes.len() / 2;
        bytes[flip_idx] ^= 0x01;
        std::fs::write(archive.path(), &bytes).unwrap();

        // Unpack with verify=true should report the corruption (either fail
        // outright or surface in summary.errors).
        let dst = tempfile::tempdir().unwrap();
        let result = unpack_archive(archive.path(), dst.path(), true, false, None);
        let detected = match result {
            Err(_) => true,
            Ok(summary) => !summary.errors.is_empty(),
        };
        assert!(
            detected,
            "byte flip in archive payload was NOT detected by verify=true"
        );
    }

    /// ADR 0023 Track A-2: an ArchiveEntry carrying a populated
    /// `synthetic_from` block survives the build_manifest_json →
    /// parse_manifest_entries roundtrip with every field preserved.
    /// Old archives (synthetic_from == None) still roundtrip cleanly.
    #[test]
    fn manifest_serde_roundtrip_synthetic_from() {
        let template = serde_json::json!({
            "line_ending": "CrLf",
            "leading_whitespace": 2,
            "field_width": 0,
            "trailing_newline": true,
        });
        let entries = vec![
            // Regular EDF entry — synthetic_from == None.
            ArchiveEntry {
                path: "regular.edf".into(),
                original_size: 1024,
                compressed_size: 512,
                method: Method::Lml,
                sha256: "a".repeat(64),
                offset: 0,
                mtime: Some(1700000000),
                mtime_nanos: Some(123_456_789),
                mode: Some(0o644),
                synthetic_from: None,
            },
            // Ingest-synthesised entry.
            ArchiveEntry {
                path: "Bonn/Z/Z001.txt".into(),
                original_size: 17433,
                compressed_size: 3950,
                method: Method::Lml,
                sha256: "b".repeat(64),
                offset: 512,
                mtime: None,
                mtime_nanos: None,
                mode: Some(0o600),
                synthetic_from: Some(SyntheticFromInfo {
                    format: "ascii_int_lines".into(),
                    sample_rate: 173.61,
                    template_json: template.clone(),
                }),
            },
        ];
        let json = build_manifest_json(&entries, 9, &[]);
        // Parse via the same path the reader uses.
        let v: serde_json::Value = serde_json::from_str(&json).expect("json valid");
        let parsed = parse_manifest_entries(&v).expect("parse ok");

        assert_eq!(parsed.len(), 2);
        // Regular entry's synthetic_from stays None.
        assert!(parsed[0].synthetic_from.is_none());
        assert_eq!(parsed[0].path, "regular.edf");
        // Synthesised entry preserves every field.
        let sf = parsed[1].synthetic_from.as_ref().expect("populated");
        assert_eq!(sf.format, "ascii_int_lines");
        assert!((sf.sample_rate - 173.61).abs() < 1e-9);
        assert_eq!(sf.template_json, template);
    }

    /// Half-populated synthetic_from objects (missing required field)
    /// are hard errors, not silent None. Per Bible R5 — no silent
    /// fallback on data corruption.
    #[test]
    fn manifest_serde_rejects_partial_synthetic_from() {
        let bad = serde_json::json!({
            "compressor": "zstd",
            "compressor_level": 9,
            "files": [{
                "path": "x.txt",
                "original_size": 1,
                "compressed_size": 1,
                "method": "lml",
                "sha256": "d".repeat(64),
                "offset": 0,
                "synthetic_from": {
                    "format": "ascii_int_lines",
                    // missing original_sha256, sample_rate, template
                }
            }]
        });
        let err = parse_manifest_entries(&bad).expect_err("should reject");
        let s = err.to_string();
        assert!(
            s.contains("synthetic_from"),
            "error should mention synthetic_from, got: {}",
            s
        );
    }

    /// V4 Pro nit (A-2#2): an explicit JSON null for synthetic_from
    /// should be treated as "no synthesis", same as the key being
    /// absent. Anything that's a non-null non-object is still a
    /// hard error (caught by the partial-object test above).
    #[test]
    fn manifest_serde_accepts_synthetic_from_null() {
        let v = serde_json::json!({
            "compressor": "zstd",
            "compressor_level": 9,
            "files": [{
                "path": "x.txt",
                "original_size": 1,
                "compressed_size": 1,
                "method": "zstd",
                "sha256": "e".repeat(64),
                "offset": 0,
                "synthetic_from": null,
            }]
        });
        let parsed = parse_manifest_entries(&v).expect("null should parse as None");
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].synthetic_from.is_none());
    }

    /// V4 Pro nit (A-2#1): a non-finite sample_rate must NOT
    /// produce invalid JSON in the manifest. The serialiser floors
    /// NaN/Inf to 0.0 (diagnostic-only — synthesis caller emits
    /// only finite values today).
    #[test]
    fn manifest_serde_handles_non_finite_sample_rate() {
        let entries = vec![ArchiveEntry {
            path: "x.txt".into(),
            original_size: 1,
            compressed_size: 1,
            method: Method::Lml,
            sha256: "f".repeat(64),
            offset: 0,
            mtime: None,
            mtime_nanos: None,
            mode: None,
            synthetic_from: Some(SyntheticFromInfo {
                format: "ascii_int_lines".into(),
                sample_rate: f64::NAN,
                template_json: serde_json::json!({
                    "line_ending": "Lf",
                    "leading_whitespace": 0,
                    "field_width": 0,
                    "trailing_newline": true,
                }),
            }),
        }];
        let json = build_manifest_json(&entries, 9, &[]);
        // Must parse back as valid JSON (no NaN/Infinity literals).
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("manifest must be valid JSON");
        let sr = parsed["files"][0]["synthetic_from"]["sample_rate"]
            .as_f64()
            .expect("sample_rate is a number");
        assert_eq!(sr, 0.0, "NaN should serialise as 0.0 floor");
    }

    // ─── Track B3 — auto_window_size correctness + transition  ─

    /// Degenerate inputs fall back to DEFAULT_WINDOW_SIZE without
    /// dividing by zero / NaN. NaN, 0, negative sample_rate, and
    /// zero signal length all land here.
    #[test]
    fn auto_window_size_handles_invalid_inputs() {
        assert_eq!(auto_window_size(0, 256.0), 2500);
        assert_eq!(auto_window_size(1000, 0.0), 2500);
        assert_eq!(auto_window_size(1000, -1.0), 2500);
        assert_eq!(auto_window_size(1000, f64::NAN), 2500);
        assert_eq!(auto_window_size(1000, f64::INFINITY), 2500);
    }

    /// Reference EDF cr_regression baselines: signal lengths sit
    /// at exactly the boundary between regimes 1 and 2 (ref ≈ 2500)
    /// or firmly inside regime 3 (chb01_01 ≈ 900 K ref). The window
    /// pick must match the pre-B3 default of 2500 in both cases.
    #[test]
    fn auto_window_size_preserves_cr_regression_baselines() {
        // S106R05 / S013R01 shape: 64ch motor-imagery, 256 Hz, ~10s
        // → 2560 samples per channel → ref ≈ 2500.
        assert_eq!(auto_window_size(2560, 256.0), 2500);
        // chb01_01 shape: 23ch clinical, 256 Hz, 1 hour
        // → 921600 samples → ref ≈ 900000 (long signal regime).
        assert_eq!(auto_window_size(921_600, 256.0), 2500);
    }

    /// Bonn dataset shape: 1 ch, 173.61 Hz, 4097 samples per file.
    /// Ref equivalent ≈ 5900 — squarely inside the single-window
    /// regime. The whole signal lives in one window.
    #[test]
    fn auto_window_size_bonn_single_window() {
        let ws = auto_window_size(4097, 173.61);
        // ref ≈ 5900; >= DEFAULT, ≤ MAX_AUTO_WINDOW.
        assert!(
            ws >= 2500 && ws <= 16384,
            "Bonn signal should pick a single window in [2500, 16384]; got {}",
            ws
        );
        // Sanity: window covers the actual signal length.
        let actual_window_samples = (ws as f64 * 173.61 / 250.0).ceil() as usize;
        assert!(
            actual_window_samples >= 4097,
            "auto_window_size for Bonn ({}) covers {} samples but signal has 4097",
            ws,
            actual_window_samples
        );
    }

    /// Boundary between regime 2 (single window) and regime 3
    /// (default chunking) is at ref_signal = MAX_AUTO_WINDOW = 16384.
    /// Verify both sides of the cliff explicitly.
    #[test]
    fn auto_window_size_regime_boundary() {
        // Exactly at the boundary at 250 Hz ref: signal_len = 16384.
        assert_eq!(auto_window_size(16_384, 250.0), 16384);
        // Just past the boundary: signal_len = 16385.
        assert_eq!(auto_window_size(16_385, 250.0), 2500);
        // Well inside regime 3: 32k samples at 250 Hz.
        assert_eq!(auto_window_size(32_768, 250.0), 2500);
    }

    /// Window pick is always within u16 (the LML packet's `t` field).
    /// A pathological caller passing a tiny sample_rate must not
    /// produce a window > 65535.
    #[test]
    fn auto_window_size_respects_u16_cap() {
        // signal_len_per_ch * REF_RATE / sample_rate could be huge if
        // sample_rate is tiny. e.g. signal=1e6 samples @ 0.001 Hz =>
        // ref = 1e6 * 250 / 0.001 = 2.5e11. Should NOT overflow t: u16.
        assert!(auto_window_size(1_000_000, 0.001) <= u16::MAX as usize);
        // Also at the upper edge of regime 2 with a low sample rate.
        let ws = auto_window_size(10_000, 100.0); // ref = 25000 > MAX → regime 3
        assert_eq!(ws, 2500);
    }

    /// Picking the maximum auto-window for a signal at exactly the
    /// boundary at a low sample rate. Verifies the actual window in
    /// samples (after the rate scaling in container::write_file)
    /// still covers the input.
    #[test]
    fn auto_window_size_low_sample_rate_short_signal() {
        // 100 Hz, 4000 samples ⇒ ref = 4000 * 250 / 100 = 10000.
        // Regime 2: ws = 10000.
        let ws = auto_window_size(4000, 100.0);
        assert_eq!(ws, 10000);
        // Actual window after rate scaling: 10000 * 100 / 250 = 4000.
        let actual = (ws as f64 * 100.0 / 250.0).ceil() as usize;
        assert_eq!(actual, 4000);
    }

    // ─── LMA v2 streaming (footer/EOCD) end-to-end roundtrip ────────
    //
    // Packs a tiny tree (empty file, binary file, deep/long-path text
    // file) into a v2 archive and exercises every reader through the
    // chokepoint: header magic/version, list_archive (untruncated
    // paths), read_entry (byte-equal), unpack_archive (byte-equal
    // tree), no leftover `.partial`, and an explicit sha-trailer check
    // over [0 .. len-32].
    #[test]
    fn v2_pack_list_read_unpack_roundtrip() {
        let src = tempfile::tempdir().unwrap();

        // 1) empty file
        let empty_rel = "empty.dat";
        std::fs::write(src.path().join(empty_rel), b"").unwrap();

        // 2) binary file (varied bytes, not just a constant)
        let bin_rel = "sub/data.bin";
        let bin_content: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        std::fs::create_dir_all(src.path().join("sub")).unwrap();
        std::fs::write(src.path().join(bin_rel), &bin_content).unwrap();

        // 3) text file at a deep, long relative path (> 60 chars total).
        //    The leaf filename alone is > 60 chars to catch any
        //    fixed-width path truncation in the manifest roundtrip.
        let deep_rel =
            "00_epilepsy/aaaa/s001/02_tcp_le/rec_long_name_over_sixty_chars_total.txt";
        assert!(
            deep_rel.len() > 60,
            "deep path must exceed 60 chars to exercise truncation; got {}",
            deep_rel.len()
        );
        let deep_content = b"col1,col2,col3\n1.0,2.0,3.0\nseizure,onset,offset\n";
        let deep_abs = src.path().join(deep_rel);
        std::fs::create_dir_all(deep_abs.parent().unwrap()).unwrap();
        std::fs::write(&deep_abs, deep_content).unwrap();

        // Pack to a v2 .lma. Signature: (input_dir, output_path,
        // zstd_level, verbose, progress_fn).
        let out_dir = tempfile::tempdir().unwrap();
        let archive = out_dir.path().join("roundtrip.lma");
        let summary = pack_archive(src.path(), &archive, 9, false, None).unwrap();
        assert_eq!(summary.n_files, 3, "expected 3 entries packed");

        // Header is v2: magic "LMA2", version u32 LE = 2.
        let raw = std::fs::read(&archive).unwrap();
        assert!(raw.len() >= 48, "archive too small");
        assert_eq!(&raw[0..4], LMA_MAGIC_V2, "header magic must be LMA2");
        assert_eq!(
            u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]),
            LMA_VERSION_V2,
            "header version must be 2"
        );

        // list_archive returns all 3 FULL paths, untruncated.
        let entries = list_archive(&archive).unwrap();
        assert_eq!(entries.len(), 3);
        let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&empty_rel), "empty path missing: {:?}", paths);
        assert!(paths.contains(&bin_rel), "bin path missing: {:?}", paths);
        assert!(
            paths.contains(&deep_rel),
            "deep/long path was truncated or missing: {:?}",
            paths
        );

        // read_entry returns byte-equal source bytes for each.
        assert_eq!(read_entry(&archive, empty_rel).unwrap(), b"");
        assert_eq!(read_entry(&archive, bin_rel).unwrap(), bin_content);
        assert_eq!(
            read_entry(&archive, deep_rel).unwrap(),
            deep_content.to_vec()
        );

        // extract_entry (separate reader: v2 seek = payload_base+offset,
        // bounded against payload_end) writes a byte-equal file for the
        // deep/long-path entry.
        let extracted = out_dir.path().join("extracted_deep.txt");
        let n = extract_entry(&archive, deep_rel, &extracted).unwrap();
        assert_eq!(n as usize, deep_content.len());
        assert_eq!(
            std::fs::read(&extracted).unwrap(),
            deep_content.to_vec(),
            "extract_entry produced non-byte-equal output on v2"
        );

        // unpack_archive reproduces all 3 files byte-equal (verify=true
        // also exercises the v2-agnostic archive-level sha pass).
        let dst = tempfile::tempdir().unwrap();
        let unpack = unpack_archive(&archive, dst.path(), true, false, None).unwrap();
        assert_eq!(unpack.n_files, 3);
        assert_eq!(std::fs::read(dst.path().join(empty_rel)).unwrap(), b"");
        assert_eq!(std::fs::read(dst.path().join(bin_rel)).unwrap(), bin_content);
        assert_eq!(
            std::fs::read(dst.path().join(deep_rel)).unwrap(),
            deep_content.to_vec()
        );

        // No `.partial` staging file lingers next to the output.
        let partial = {
            let mut s = archive.as_os_str().to_os_string();
            s.push(".partial");
            std::path::PathBuf::from(s)
        };
        assert!(
            !partial.exists(),
            "leftover .partial staging file at {}",
            partial.display()
        );

        // Explicit sha-trailer check: sha256 over [0 .. len-32] must
        // equal the trailing 32 bytes. (Independent of unpack's verify.)
        let (body, trailer) = raw.split_at(raw.len() - 32);
        let computed = {
            let mut h = Sha256::new();
            h.update(body);
            h.finalize()
        };
        assert_eq!(
            computed.as_slice(),
            trailer,
            "v2 sha trailer over [0..len-32] does not validate"
        );
    }

    // v2 append roundtrip: append a 4th entry to a v2 archive, then
    // confirm both the original three and the appended entry read back
    // byte-equal and the result is still a valid v2 archive with a
    // self-consistent sha trailer.
    #[test]
    fn v2_append_then_readback() {
        let src = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("a.bin"), [0x11u8; 100]).unwrap();
        std::fs::write(src.path().join("b.bin"), [0x22u8; 200]).unwrap();
        std::fs::write(src.path().join("c.bin"), [0x33u8; 300]).unwrap();

        let out_dir = tempfile::tempdir().unwrap();
        let archive = out_dir.path().join("app.lma");
        pack_archive(src.path(), &archive, 9, false, None).unwrap();

        // Source file for the appended entry.
        let appended = out_dir.path().join("d_source.bin");
        let d_content: Vec<u8> = (0u8..=255).cycle().take(1234).collect();
        std::fs::write(&appended, &d_content).unwrap();

        let summary = append_entry(
            &archive,
            &appended,
            Some("nested/d.bin"),
            9,
            false,
            false, // keep_bak=false so no .bak lingers
        )
        .unwrap();
        assert_eq!(summary.n_files, 4, "append should yield 4 entries");

        // Still a v2 archive.
        let raw = std::fs::read(&archive).unwrap();
        assert_eq!(&raw[0..4], LMA_MAGIC_V2);
        assert_eq!(
            u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]),
            LMA_VERSION_V2
        );

        // All four entries read back byte-equal — proves the existing
        // payload offsets stayed valid AND the new entry landed right.
        assert_eq!(read_entry(&archive, "a.bin").unwrap(), vec![0x11u8; 100]);
        assert_eq!(read_entry(&archive, "b.bin").unwrap(), vec![0x22u8; 200]);
        assert_eq!(read_entry(&archive, "c.bin").unwrap(), vec![0x33u8; 300]);
        assert_eq!(read_entry(&archive, "nested/d.bin").unwrap(), d_content);

        // sha trailer self-consistent after the rewrite.
        let (body, trailer) = raw.split_at(raw.len() - 32);
        let mut h = Sha256::new();
        h.update(body);
        assert_eq!(h.finalize().as_slice(), trailer, "v2 append sha invalid");

        // unpack reproduces the tree byte-equal too.
        let dst = tempfile::tempdir().unwrap();
        unpack_archive(&archive, dst.path(), true, false, None).unwrap();
        assert_eq!(std::fs::read(dst.path().join("a.bin")).unwrap(), vec![0x11u8; 100]);
        assert_eq!(std::fs::read(dst.path().join("nested/d.bin")).unwrap(), d_content);
    }

    /// Hand-build a legacy v1 archive (front manifest) so the v1 read
    /// path and the v1→v2 append branch can be exercised even though no
    /// v1 writer remains in-tree. Returns the on-disk path inside `dir`.
    /// v1 layout: header(16) | manifest(zstd) | payloads | sha256(32).
    fn build_v1_archive(dir: &std::path::Path, files: &[(&str, Vec<u8>)]) -> std::path::PathBuf {
        // Store all entries verbatim (Method::Store) — offsets are
        // relative to the v1 payload_base (16 + manifest_len).
        let mut entries: Vec<ArchiveEntry> = Vec::new();
        let mut payloads: Vec<u8> = Vec::new();
        let mut offset: u64 = 0;
        for (path, content) in files {
            let sha = sha256_hex(content);
            entries.push(ArchiveEntry {
                path: (*path).to_string(),
                original_size: content.len() as u64,
                compressed_size: content.len() as u64,
                method: Method::Store,
                sha256: sha,
                offset,
                mtime: None,
                mtime_nanos: None,
                mode: None,
                synthetic_from: None,
            });
            payloads.extend_from_slice(content);
            offset += content.len() as u64;
        }
        let manifest_json = build_manifest_json(&entries, 9, &[]);
        let manifest_payload = zstd::encode_all(manifest_json.as_bytes(), 9).unwrap();
        let manifest_len_field = manifest_payload.len() as u32; // compressed, top bit 0

        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(LMA_MAGIC); // "LMA1"
        out.extend_from_slice(&LMA_VERSION.to_le_bytes()); // 1
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes()); // n_entries
        out.extend_from_slice(&manifest_len_field.to_le_bytes());
        out.extend_from_slice(&manifest_payload);
        out.extend_from_slice(&payloads);
        // sha256 over everything before the trailer.
        let mut h = Sha256::new();
        h.update(&out);
        let digest = h.finalize();
        out.extend_from_slice(&digest);

        let path = dir.join("legacy_v1.lma");
        std::fs::write(&path, &out).unwrap();
        path
    }

    // The v1 read path: a hand-built LMA1 archive lists + reads back
    // byte-equal through the chokepoint (proves the v1 positioning +
    // front-manifest decode survive the conversion).
    #[test]
    fn v1_archive_reads_through_chokepoint() {
        let dir = tempfile::tempdir().unwrap();
        let a = (0u8..=255).cycle().take(500).collect::<Vec<u8>>();
        let b = vec![0x7Eu8; 321];
        let archive = build_v1_archive(dir.path(), &[("x/one.bin", a.clone()), ("two.bin", b.clone())]);

        let entries = list_archive(&archive).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(read_entry(&archive, "x/one.bin").unwrap(), a);
        assert_eq!(read_entry(&archive, "two.bin").unwrap(), b);

        let dst = tempfile::tempdir().unwrap();
        unpack_archive(&archive, dst.path(), true, false, None).unwrap();
        assert_eq!(std::fs::read(dst.path().join("x/one.bin")).unwrap(), a);
        assert_eq!(std::fs::read(dst.path().join("two.bin")).unwrap(), b);
    }

    // The v1→v2 append branch (offset re-base + fresh v2 header +
    // dead-v1-manifest-inside-payload). Append to a hand-built v1
    // archive; assert the result is a valid v2 archive and EVERY entry
    // (the two original v1 entries + the appended one) reads back
    // byte-equal with a self-consistent sha trailer.
    #[test]
    fn v1_to_v2_append_rebases_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let a = (0u8..=255).cycle().take(500).collect::<Vec<u8>>();
        let b = vec![0x7Eu8; 321];
        let archive = build_v1_archive(dir.path(), &[("x/one.bin", a.clone()), ("two.bin", b.clone())]);

        let appended = dir.path().join("d_source.bin");
        let d_content: Vec<u8> = (0..999u32).map(|i| (i.wrapping_mul(37) & 0xFF) as u8).collect();
        std::fs::write(&appended, &d_content).unwrap();

        let summary = append_entry(
            &archive,
            &appended,
            Some("deep/three.bin"),
            9,
            false,
            false,
        )
        .unwrap();
        assert_eq!(summary.n_files, 3);

        // Result is now v2.
        let raw = std::fs::read(&archive).unwrap();
        assert_eq!(&raw[0..4], LMA_MAGIC_V2, "v1→v2 append must emit an LMA2 header");
        assert_eq!(
            u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]),
            LMA_VERSION_V2
        );

        // Every entry reads back byte-equal — this is the load-bearing
        // proof that the v1 offsets were correctly re-based onto
        // payload_base 16.
        assert_eq!(read_entry(&archive, "x/one.bin").unwrap(), a, "v1 entry one corrupted after append");
        assert_eq!(read_entry(&archive, "two.bin").unwrap(), b, "v1 entry two corrupted after append");
        assert_eq!(read_entry(&archive, "deep/three.bin").unwrap(), d_content, "appended entry corrupted");

        // sha trailer self-consistent.
        let (body, trailer) = raw.split_at(raw.len() - 32);
        let mut h = Sha256::new();
        h.update(body);
        assert_eq!(h.finalize().as_slice(), trailer, "v1→v2 append sha invalid");

        // unpack reproduces the full tree byte-equal.
        let dst = tempfile::tempdir().unwrap();
        unpack_archive(&archive, dst.path(), true, false, None).unwrap();
        assert_eq!(std::fs::read(dst.path().join("x/one.bin")).unwrap(), a);
        assert_eq!(std::fs::read(dst.path().join("two.bin")).unwrap(), b);
        assert_eq!(std::fs::read(dst.path().join("deep/three.bin")).unwrap(), d_content);
    }
}
