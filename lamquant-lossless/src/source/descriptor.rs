//! Format-description DSL — "reader-as-data" (ADR 0069 Pillar 3, S5
//! Increment 3, task #20).
//!
//! Every reader under `source/` today (`RawReader`, `EdfReader`,
//! `BrainVisionReader`, ...) is hand-written Rust: a struct that owns a
//! byte source and a `read_bundle`/`lower_to_abir` pair of methods that
//! walk the format's byte layout by hand. That's the right shape for a
//! format with real structure (EDF's per-channel headers, BrainVision's
//! three-file split, DICOM's nested TLV dataset — see "Limits" below).
//! But a large class of real acquisition pipelines produce something much
//! simpler: a flat grid of fixed-width samples (one dtype, one byte
//! order, one of two orientations) plus a small JSON sidecar giving the
//! channel count / sample rate / labels / calibration. `RawReader` (see
//! `super::raw`) is the canonical example.
//!
//! For that whole class, hand-writing a new Rust reader per acquisition
//! rig is unnecessary ceremony. This module declares [`FormatDescriptor`]
//! — a `serde`-derivable **struct** that describes the fixed layout as
//! DATA — and an interpreter, [`read_bundle_from_descriptor`] /
//! [`lower_to_abir_from_descriptor`], that walks that description exactly
//! the way a hand-written reader's `read_bundle`/`lower_to_abir` would.
//! The proof obligation for this increment is narrow and load-bearing:
//! a `FormatDescriptor` built to describe the RAW format must produce
//! **byte-identical** output to `RawReader` — see
//! `raw_format_as_descriptor_matches_hand_written` below.
//!
//! Two design-pass gotchas (referred to as "G5" throughout this module,
//! after the design review that flagged them) had to be closed before a
//! descriptor could stand in for a hand-written reader at all:
//!
//!   - **Endian is not implicit.** Every hand-written binary reader in
//!     this crate hardcodes `from_le_bytes` (see `RawReader`,
//!     `BrainVisionReader`, `bitstream::read_i24_le`). A descriptor that
//!     didn't make [`Endian`] a first-class field would silently bake in
//!     that assumption instead of describing it — so `endian` is
//!     mandatory here, and a big-endian descriptor decodes differently
//!     from an otherwise-identical little-endian one (see the `endian`
//!     gate test).
//!   - **`F32` cannot be a descriptor dtype.** `Column::window_i64` casts
//!     `f32 as i64` — meaningful only for integer-valued floats, lossy
//!     (truncates toward zero) in general. A hand-written reader that
//!     wants a float lane makes that an explicit, reviewed decision (see
//!     `EeglabReader`'s float-bitcast lossless path, which deliberately
//!     stays off the default `lower_to_abir`). A *data-driven* descriptor
//!     has no such review gate, so [`FormatDescriptor::validate`] refuses
//!     `F32` unconditionally — see `validate_refuses_f32`.
//!
//! Sample-rate can also be *derived* rather than declared outright:
//! BrainVision's `.vhdr` stores a sampling **interval** in microseconds
//! and the reader computes `Hz = 1_000_000.0 / sample_interval_us` (see
//! `super::brainvision`). [`SampleRateSpec::Reciprocal`] expresses that
//! same `numerator / sidecar[field]` transform as data instead of code.
//!
//! # What this is deliberately NOT
//!
//! [`FormatDescriptor`] does not implement [`super::reader::SignalSourceReader`].
//! That trait owns its byte source at construction time (a file path); a
//! descriptor interpreter instead takes `data` / `sidecar_bytes` slices
//! explicitly, so the exact same descriptor works over file bytes,
//! in-memory buffers, or (a later increment) archive-embedded blobs
//! without caring which. A thin adapter (`FormatDescriptor` + `PathBuf`)
//! implementing `SignalSourceReader` by reading the two files and
//! delegating here is a natural follow-up, not part of this increment.
//!
//! # Limits — what a `FormatDescriptor` CANNOT express
//!
//! This DSL only covers "flat grid, one dtype, one endian, one
//! orientation, scalar/array sidecar lookups." It cannot, and is not
//! meant to, replace a hand-written reader for a format with real
//! internal structure:
//!
//!   - **EDF/BDF** (`edf_reader.rs`): each signal has its OWN physical
//!     range, digital range, and samples-per-record in the shared header
//!     — a per-channel-varying layout, not a single fixed grid. EDF+D's
//!     discontinuous data records add conditional structure on top.
//!   - **BrainVision** (`brainvision.rs`): a three-FILE bundle
//!     (`.vhdr` text header + `.eeg` binary + `.vmrk` text markers) held
//!     together by cross-references inside an INI-style text format that
//!     needs real parsing (sections, per-channel resolution lines), not
//!     scalar field lookups.
//!   - **DICOM** (`dicom.rs`): a nested, VR-typed, transfer-syntax-
//!     dependent TLV dataset — the opposite of a flat byte grid.
//!   - Anything requiring **conditional parsing** (format version
//!     branches inside the byte stream), **per-channel-varying width**,
//!     or **nested/variable-length containers** in general.
//!
//! A `FormatDescriptor`'s honest scope is "the RAW-shaped tail of the
//! format distribution": fixed dtype, fixed endian, fixed orientation,
//! scalar/array sidecar fields. Formats outside that shape stay
//! hand-written Rust readers.

use std::sync::Arc;

use abir::{Abir, Channel, Column};
use serde::{Deserialize, Serialize};

use crate::error::{LmlError, LmlResult};

use super::bundle::{SidecarBlob, SignalBundle, SourceMetadata};

// ─── Schema ──────────────────────────────────────────────────────────────

/// Sample element type a fixed-layout format may declare. **No lossless
/// path admits `F32`** — see the module docs' G5 note and
/// [`DescriptorError::FloatDtypeRefused`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DescriptorDtype {
    /// 16-bit integer samples (EDF, most EEG AFEs) — 2 bytes/sample.
    I16,
    /// 24-bit integer samples carried in `i32` lanes (BDF / 24-bit ADCs)
    /// — 3 bytes/sample on the wire.
    I24,
    /// 32-bit integer samples — 4 bytes/sample.
    I32,
    /// 32-bit float samples. Refused by [`FormatDescriptor::validate`]
    /// (G5): `Column::F32::window_i64` casts toward zero, which is lossy
    /// for a non-integer-valued float and silently wrong for a
    /// data-driven descriptor with no human review step.
    F32,
}

impl DescriptorDtype {
    /// Wire width in bytes. `F32` is included for completeness even
    /// though `validate()` refuses it before any decode is attempted.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::I16 => 2,
            Self::I24 => 3,
            Self::I32 => 4,
            Self::F32 => 4,
        }
    }
}

/// Byte order of multi-byte samples on the wire (G5: first-class, never
/// assumed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Endian {
    Little,
    Big,
}

/// Sample-major (`Multiplexed`) vs channel-major (`Vectorized`) layout —
/// same vocabulary as `RawReader`'s sidecar schema and
/// `BrainVisionReader`'s `DataOrientation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DescriptorOrientation {
    Multiplexed,
    Vectorized,
}

/// How many channels the recording has.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChannelCount {
    /// Baked into the descriptor itself.
    Fixed(usize),
    /// Read as an integer scalar from the sidecar JSON object's named
    /// top-level field (e.g. `"n_channels"`, mirroring `RawReader`).
    FromSidecarField(String),
}

/// How the sampling rate (Hz) is determined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SampleRateSpec {
    /// Baked into the descriptor itself.
    Fixed(f64),
    /// Read as a float scalar directly from the sidecar JSON object's
    /// named field (e.g. `"sample_rate"`, mirroring `RawReader`).
    FromSidecarField(String),
    /// Derived: `Hz = numerator / sidecar[sidecar_field]`. BrainVision's
    /// `Hz = 1_000_000.0 / SamplingInterval` (interval in microseconds)
    /// is the motivating case (G5) — see `reciprocal_sample_rate_*`.
    Reciprocal { sidecar_field: String, numerator: f64 },
}

/// One label-pattern → modality rule. `pattern` is matched
/// case-insensitively as a substring against each channel label; the
/// first rule (in list order) matched by ANY channel's label wins and
/// its `modality` becomes the format-hint string passed to
/// [`Abir::with_inferred_modality`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelModalityRule {
    pub pattern: String,
    pub modality: String,
}

/// Format-hint resolution for [`Abir::with_inferred_modality`]'s
/// `format: Option<&str>` argument. An empty `rules` list and `default:
/// None` (the [`Default`] impl) reproduces `RawReader`'s behavior exactly
/// — pure per-channel-label inference, no format-declared override.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChannelModality {
    #[serde(default)]
    pub rules: Vec<ChannelModalityRule>,
    /// Format-hint used when no rule matches (or the recording has no
    /// channels). `None` defers entirely to label-based inference.
    #[serde(default)]
    pub default: Option<String>,
}

/// A fixed-layout physiology format, described as data instead of as a
/// hand-written [`super::reader::SignalSourceReader`] impl. See the
/// module docs for the schema rationale, the G5 gotchas, and the explicit
/// scope limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormatDescriptor {
    /// Becomes `SignalBundle.metadata.format` (e.g. `"RAW"`).
    pub format_name: String,
    pub dtype: DescriptorDtype,
    pub endian: Endian,
    pub orientation: DescriptorOrientation,
    pub channel_count: ChannelCount,
    pub sample_rate: SampleRateSpec,
    #[serde(default)]
    pub channel_modality: ChannelModality,
    /// Which interpreter-recognized sidecar blobs to copy into the
    /// resulting `SignalBundle.sidecar`. Recognized values today:
    /// `"sidecar_json"` (the sidecar bytes verbatim) and `"payload"`
    /// (the sample `data` bytes verbatim). Any other value is a
    /// `validate()`-time error — fail-closed rather than silently
    /// dropping a preservation request.
    #[serde(default)]
    pub sidecar_keys: Vec<String>,
}

impl FormatDescriptor {
    /// Fail-closed structural check. Catches everything that's wrong
    /// regardless of what sidecar data the interpreter will later see:
    /// a refused dtype (G5), a statically-zero channel count, a
    /// non-positive fixed sample rate, a degenerate reciprocal
    /// numerator, or an unrecognized `sidecar_keys` entry.
    ///
    /// What it CANNOT catch (because the value isn't known until a
    /// sidecar is in hand): a `FromSidecarField` channel count that
    /// resolves to 0, or a `Reciprocal` transform whose sidecar-supplied
    /// divisor is 0. Both of those are checked at interpret time by
    /// [`read_bundle_from_descriptor`] / [`lower_to_abir_from_descriptor`]
    /// (via `resolve_channel_count` / `resolve_sample_rate`), which also
    /// call this method first — so every entry point is fully fail-closed
    /// end to end, just not from a single static call.
    pub fn validate(&self) -> Result<(), DescriptorError> {
        if self.dtype == DescriptorDtype::F32 {
            return Err(DescriptorError::FloatDtypeRefused);
        }
        if let ChannelCount::Fixed(0) = self.channel_count {
            return Err(DescriptorError::ZeroChannels);
        }
        match &self.sample_rate {
            SampleRateSpec::Fixed(hz) => {
                if !hz.is_finite() || *hz <= 0.0 {
                    return Err(DescriptorError::NonPositiveSampleRate);
                }
            }
            SampleRateSpec::Reciprocal { numerator, .. } => {
                if !numerator.is_finite() || *numerator == 0.0 {
                    return Err(DescriptorError::ReciprocalNumeratorInvalid);
                }
            }
            SampleRateSpec::FromSidecarField(_) => {}
        }
        for key in &self.sidecar_keys {
            if key != "sidecar_json" && key != "payload" {
                return Err(DescriptorError::UnknownSidecarKey(key.clone()));
            }
        }
        Ok(())
    }
}

// ─── Errors ──────────────────────────────────────────────────────────────

/// Typed error for descriptor validation AND interpretation. Local to
/// this crate (house style, like `ir::IrParseError` /
/// `pipeline_dsl::PipelineDslError`) — `crate::error::LmlError` is the
/// no_std-sibling wire-format error type shared with firmware and stays
/// out of scope; the interpreter entry points convert a `DescriptorError`
/// into `LmlError::InvalidHeader` at their boundary (`to_lml_err`) so
/// callers keep working with the crate's usual `LmlResult`.
///
/// Every variant is reachable without a panic — see
/// `truncation_sweep_never_panics` for the adversarial-input proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorError {
    /// G5: a descriptor declared `dtype: F32`.
    FloatDtypeRefused,
    /// `channel_count` is (or resolved to) zero.
    ZeroChannels,
    /// A `Fixed` or resolved sample rate is non-finite or `<= 0`.
    NonPositiveSampleRate,
    /// `SampleRateSpec::Reciprocal`'s `numerator` is non-finite or `0.0`
    /// — a schema-static bug, caught by `validate()`.
    ReciprocalNumeratorInvalid,
    /// `SampleRateSpec::Reciprocal`'s sidecar-supplied divisor resolved
    /// to `0.0` or non-finite (G5's "reciprocal transform with a zero
    /// divisor" case) — data-dependent, caught at interpret time.
    ReciprocalZeroDivisor { field: String },
    /// A sidecar field the descriptor references by name is absent from
    /// the sidecar JSON object.
    MissingSidecarField { field: String },
    /// A sidecar field exists but isn't the JSON type the descriptor
    /// needs (e.g. a string where a number was expected).
    BadSidecarFieldType { field: String },
    /// The sidecar's `channels` / `phys_min` / `phys_max` array length
    /// disagrees with the resolved channel count.
    ChannelArrayLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// `sidecar_keys` names a blob this interpreter doesn't know how to
    /// produce (only `"sidecar_json"` / `"payload"` are recognized).
    UnknownSidecarKey(String),
    /// `n_channels * bytes_per_sample` (or an orientation stride derived
    /// from it) overflowed `usize` — an adversarial/garbage
    /// sidecar-declared channel count.
    StrideOverflow,
    /// The sample payload's length isn't a whole multiple of the
    /// per-sample-row stride — truncated or malformed input.
    TruncatedPayload { stride: usize, actual: usize },
    /// A per-sample byte read fell outside the payload. Defense in depth:
    /// given the `TruncatedPayload` precondition check, this should be
    /// unreachable, but the interpreter never trusts that arithmetic
    /// blindly (Bible R30) — an out-of-range read is always a typed
    /// error here, never a slice-index panic.
    SampleOutOfBounds { offset: usize, width: usize, len: usize },
    /// The sidecar bytes aren't valid UTF-8, aren't valid JSON, or the
    /// top-level JSON value isn't an object.
    BadSidecarJson(String),
}

impl core::fmt::Display for DescriptorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FloatDtypeRefused => write!(
                f,
                "descriptor dtype F32 is refused (Column::window_i64 lossily \
                 truncates float lanes toward zero; see ADR 0069 S5 G5)"
            ),
            Self::ZeroChannels => write!(f, "descriptor channel count is 0"),
            Self::NonPositiveSampleRate => {
                write!(f, "descriptor sample rate must be finite and > 0")
            }
            Self::ReciprocalNumeratorInvalid => write!(
                f,
                "reciprocal sample-rate numerator must be finite and non-zero"
            ),
            Self::ReciprocalZeroDivisor { field } => write!(
                f,
                "reciprocal sample-rate divisor (sidecar field {field:?}) is zero or non-finite"
            ),
            Self::MissingSidecarField { field } => {
                write!(f, "sidecar: missing field {field:?}")
            }
            Self::BadSidecarFieldType { field } => {
                write!(f, "sidecar: field {field:?} has the wrong JSON type")
            }
            Self::ChannelArrayLengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "sidecar: `{field}`.len {actual} != resolved channel count {expected}"
            ),
            Self::UnknownSidecarKey(key) => {
                write!(f, "descriptor: unrecognized sidecar_keys entry {key:?}")
            }
            Self::StrideOverflow => {
                write!(f, "descriptor: channel-count * bytes-per-sample overflowed")
            }
            Self::TruncatedPayload { stride, actual } => write!(
                f,
                "payload length {actual} is not a multiple of the per-row stride {stride}"
            ),
            Self::SampleOutOfBounds { offset, width, len } => write!(
                f,
                "sample read at offset {offset} (width {width}) exceeds payload length {len}"
            ),
            Self::BadSidecarJson(msg) => write!(f, "sidecar: {msg}"),
        }
    }
}

impl std::error::Error for DescriptorError {}

/// Convert an interpreter-internal `DescriptorError` into the crate's
/// standard `LmlError` at the [`read_bundle_from_descriptor`] /
/// [`lower_to_abir_from_descriptor`] boundary — mirrors every other
/// reader's `LmlError::InvalidHeader(format!(...))` convention (see
/// `RawReader::read_bundle`).
fn to_lml_err(e: DescriptorError) -> LmlError {
    LmlError::InvalidHeader(e.to_string())
}

// ─── Sidecar field resolution ──────────────────────────────────────────

type JsonObject = serde_json::Map<String, serde_json::Value>;

fn resolve_channel_count(
    cc: &ChannelCount,
    obj: &JsonObject,
) -> Result<usize, DescriptorError> {
    let n = match cc {
        ChannelCount::Fixed(n) => *n,
        ChannelCount::FromSidecarField(field) => {
            let v = obj
                .get(field)
                .ok_or_else(|| DescriptorError::MissingSidecarField {
                    field: field.clone(),
                })?;
            v.as_u64()
                .ok_or_else(|| DescriptorError::BadSidecarFieldType {
                    field: field.clone(),
                })? as usize
        }
    };
    if n == 0 {
        return Err(DescriptorError::ZeroChannels);
    }
    Ok(n)
}

fn resolve_sample_rate(spec: &SampleRateSpec, obj: &JsonObject) -> Result<f64, DescriptorError> {
    let hz = match spec {
        SampleRateSpec::Fixed(hz) => *hz,
        SampleRateSpec::FromSidecarField(field) => {
            let v = obj
                .get(field)
                .ok_or_else(|| DescriptorError::MissingSidecarField {
                    field: field.clone(),
                })?;
            v.as_f64()
                .ok_or_else(|| DescriptorError::BadSidecarFieldType {
                    field: field.clone(),
                })?
        }
        SampleRateSpec::Reciprocal {
            sidecar_field,
            numerator,
        } => {
            let v = obj
                .get(sidecar_field)
                .ok_or_else(|| DescriptorError::MissingSidecarField {
                    field: sidecar_field.clone(),
                })?;
            let divisor = v
                .as_f64()
                .ok_or_else(|| DescriptorError::BadSidecarFieldType {
                    field: sidecar_field.clone(),
                })?;
            if !divisor.is_finite() || divisor == 0.0 {
                return Err(DescriptorError::ReciprocalZeroDivisor {
                    field: sidecar_field.clone(),
                });
            }
            numerator / divisor
        }
    };
    if !hz.is_finite() || hz <= 0.0 {
        return Err(DescriptorError::NonPositiveSampleRate);
    }
    Ok(hz)
}

/// Reads the fixed sidecar keys every descriptor-backed recording needs:
/// `channels` (array of strings), `phys_min` / `phys_max` (arrays of
/// numbers, length == `n_channels`), and `phys_dim` (string, default
/// `"uV"`). Mirrors `raw::parse_raw_sidecar`'s field extraction exactly —
/// the load-bearing byte-identity test depends on that.
fn resolve_channel_arrays(
    obj: &JsonObject,
    n_channels: usize,
) -> Result<(Vec<String>, Vec<f64>, Vec<f64>, String), DescriptorError> {
    let channels: Vec<String> = obj
        .get("channels")
        .and_then(|v| v.as_array())
        .ok_or_else(|| DescriptorError::MissingSidecarField {
            field: "channels".into(),
        })?
        .iter()
        .map(|c| c.as_str().unwrap_or("").to_string())
        .collect();
    if channels.len() != n_channels {
        return Err(DescriptorError::ChannelArrayLengthMismatch {
            field: "channels",
            expected: n_channels,
            actual: channels.len(),
        });
    }
    let phys_min: Vec<f64> = obj
        .get("phys_min")
        .and_then(|v| v.as_array())
        .ok_or_else(|| DescriptorError::MissingSidecarField {
            field: "phys_min".into(),
        })?
        .iter()
        .filter_map(|c| c.as_f64())
        .collect();
    if phys_min.len() != n_channels {
        return Err(DescriptorError::ChannelArrayLengthMismatch {
            field: "phys_min",
            expected: n_channels,
            actual: phys_min.len(),
        });
    }
    let phys_max: Vec<f64> = obj
        .get("phys_max")
        .and_then(|v| v.as_array())
        .ok_or_else(|| DescriptorError::MissingSidecarField {
            field: "phys_max".into(),
        })?
        .iter()
        .filter_map(|c| c.as_f64())
        .collect();
    if phys_max.len() != n_channels {
        return Err(DescriptorError::ChannelArrayLengthMismatch {
            field: "phys_max",
            expected: n_channels,
            actual: phys_max.len(),
        });
    }
    let phys_dim = obj
        .get("phys_dim")
        .and_then(|v| v.as_str())
        .unwrap_or("uV")
        .to_string();
    Ok((channels, phys_min, phys_max, phys_dim))
}

/// First-match label-pattern → modality resolution feeding
/// `Abir::with_inferred_modality`'s `format` hint. Empty rules + no
/// default (the `ChannelModality::default()` used by every RAW-as-
/// descriptor test) returns `None`, exactly matching `RawReader`'s own
/// `with_inferred_modality(&labels, None)` call.
fn resolve_format_hint(cm: &ChannelModality, labels: &[String]) -> Option<String> {
    for rule in &cm.rules {
        let pattern = rule.pattern.to_ascii_lowercase();
        if labels
            .iter()
            .any(|l| l.to_ascii_lowercase().contains(&pattern))
        {
            return Some(rule.modality.clone());
        }
    }
    cm.default.clone()
}

fn parse_sidecar_json(sidecar_bytes: &[u8]) -> Result<serde_json::Value, DescriptorError> {
    let text = std::str::from_utf8(sidecar_bytes)
        .map_err(|e| DescriptorError::BadSidecarJson(format!("invalid UTF-8 ({e})")))?;
    serde_json::from_str(text)
        .map_err(|e| DescriptorError::BadSidecarJson(format!("not valid JSON ({e})")))
}

fn sidecar_object(json: &serde_json::Value) -> Result<&JsonObject, DescriptorError> {
    json.as_object()
        .ok_or_else(|| DescriptorError::BadSidecarJson("must be a top-level JSON object".into()))
}

// ─── Byte-grid decode (panic-free) ──────────────────────────────────────

fn read_i16(b: &[u8], endian: Endian) -> i16 {
    let arr = [b[0], b[1]];
    match endian {
        Endian::Little => i16::from_le_bytes(arr),
        Endian::Big => i16::from_be_bytes(arr),
    }
}

fn read_i32(b: &[u8], endian: Endian) -> i32 {
    let arr = [b[0], b[1], b[2], b[3]];
    match endian {
        Endian::Little => i32::from_le_bytes(arr),
        Endian::Big => i32::from_be_bytes(arr),
    }
}

/// 24-bit two's-complement sign extension to `i32`, generalized over
/// endian from `bitstream::read_i24_le`'s little-endian-only algorithm
/// (that helper also documents an OOB panic contract this module cannot
/// accept, so it is not reused directly here).
fn read_i24(b: &[u8], endian: Endian) -> i32 {
    let (b0, b1, b2) = match endian {
        Endian::Little => (b[0], b[1], b[2]),
        Endian::Big => (b[2], b[1], b[0]),
    };
    let val = (b0 as i32) | ((b1 as i32) << 8) | ((b2 as i32) << 16);
    if val >= 0x0080_0000 {
        val - 0x0100_0000
    } else {
        val
    }
}

/// Generic fixed-width, fixed-orientation byte-grid decode: `n_channels`
/// columns of `T`, `width` bytes each, read via `read_elem` (which always
/// receives an exactly-`width`-byte slice — never fewer, so `read_elem`
/// impls may index it directly without a bounds check of their own).
///
/// `n_samples` is DERIVED from `data.len()` (never taken from a declared
/// count), so every computed offset is provably in-bounds — the same
/// invariant `RawReader` relies on. The `.get(..)` read below is
/// defense-in-depth on top of that proof, not a substitute for it (Bible
/// R30: never trust arithmetic blindly).
fn decode_grid<T: Copy>(
    orientation: DescriptorOrientation,
    n_channels: usize,
    width: usize,
    data: &[u8],
    read_elem: impl Fn(&[u8]) -> T,
) -> Result<(Vec<Vec<T>>, usize), DescriptorError> {
    let stride = n_channels
        .checked_mul(width)
        .ok_or(DescriptorError::StrideOverflow)?;
    if data.len() % stride != 0 {
        return Err(DescriptorError::TruncatedPayload {
            stride,
            actual: data.len(),
        });
    }
    let n_samples = data.len() / stride;
    let mut cols: Vec<Vec<T>> = (0..n_channels)
        .map(|_| Vec::with_capacity(n_samples))
        .collect();
    let read_at = |data: &[u8], off: usize| -> Result<T, DescriptorError> {
        let bytes = data
            .get(off..off + width)
            .ok_or(DescriptorError::SampleOutOfBounds {
                offset: off,
                width,
                len: data.len(),
            })?;
        Ok(read_elem(bytes))
    };
    match orientation {
        DescriptorOrientation::Multiplexed => {
            for s in 0..n_samples {
                let base = s * stride;
                for (ch, col) in cols.iter_mut().enumerate() {
                    col.push(read_at(data, base + ch * width)?);
                }
            }
        }
        DescriptorOrientation::Vectorized => {
            for (ch, col) in cols.iter_mut().enumerate() {
                let ch_base = ch
                    .checked_mul(n_samples)
                    .and_then(|v| v.checked_mul(width))
                    .ok_or(DescriptorError::StrideOverflow)?;
                for s in 0..n_samples {
                    col.push(read_at(data, ch_base + s * width)?);
                }
            }
        }
    }
    Ok((cols, n_samples))
}

/// Decode straight to the codec's `i64` working currency — the
/// `SignalBundle` path, mirroring every hand-written reader's
/// `read_bundle` (always widens, regardless of native width).
fn decode_signal_i64(
    descriptor: &FormatDescriptor,
    n_channels: usize,
    data: &[u8],
) -> Result<(Vec<Vec<i64>>, usize), DescriptorError> {
    let endian = descriptor.endian;
    match descriptor.dtype {
        DescriptorDtype::I16 => decode_grid(descriptor.orientation, n_channels, 2, data, move |b| {
            read_i16(b, endian) as i64
        }),
        DescriptorDtype::I24 => decode_grid(descriptor.orientation, n_channels, 3, data, move |b| {
            read_i24(b, endian) as i64
        }),
        DescriptorDtype::I32 => decode_grid(descriptor.orientation, n_channels, 4, data, move |b| {
            read_i32(b, endian) as i64
        }),
        DescriptorDtype::F32 => Err(DescriptorError::FloatDtypeRefused),
    }
}

/// Native-width columns for the Abir lowering path — the memory-win
/// specialization every hand-written reader's `lower_to_abir` override
/// does (see `RawReader::lower_to_abir`'s doc comment).
enum NativeColumns {
    I16(Vec<Vec<i16>>),
    I24(Vec<Vec<i32>>),
    I32(Vec<Vec<i32>>),
}

fn decode_native_columns(
    descriptor: &FormatDescriptor,
    n_channels: usize,
    data: &[u8],
) -> Result<(NativeColumns, usize), DescriptorError> {
    let endian = descriptor.endian;
    match descriptor.dtype {
        DescriptorDtype::I16 => {
            let (cols, n) =
                decode_grid(descriptor.orientation, n_channels, 2, data, move |b| {
                    read_i16(b, endian)
                })?;
            Ok((NativeColumns::I16(cols), n))
        }
        DescriptorDtype::I24 => {
            let (cols, n) =
                decode_grid(descriptor.orientation, n_channels, 3, data, move |b| {
                    read_i24(b, endian)
                })?;
            Ok((NativeColumns::I24(cols), n))
        }
        DescriptorDtype::I32 => {
            let (cols, n) =
                decode_grid(descriptor.orientation, n_channels, 4, data, move |b| {
                    read_i32(b, endian)
                })?;
            Ok((NativeColumns::I32(cols), n))
        }
        DescriptorDtype::F32 => Err(DescriptorError::FloatDtypeRefused),
    }
}

fn build_channels<T: Copy>(
    cols: Vec<Vec<T>>,
    labels: &[String],
    phys_min: &[f64],
    phys_max: &[f64],
    wrap: impl Fn(Arc<[T]>) -> Column,
) -> Vec<Channel> {
    cols.into_iter()
        .enumerate()
        .map(|(j, col)| Channel {
            label: Arc::from(labels[j].as_str()),
            data: wrap(Arc::from(col)),
            phys_min: phys_min[j],
            phys_max: phys_max[j],
        })
        .collect()
}

// ─── Interpreter (public entry points) ──────────────────────────────────

/// Interpret `descriptor` over `data` (the sample payload) +
/// `sidecar_bytes` (a JSON sidecar, same schema convention `RawReader`
/// uses: `channels` / `phys_min` / `phys_max` / `phys_dim` plus whatever
/// scalar fields `descriptor.channel_count` / `descriptor.sample_rate`
/// name), producing the codec-agnostic [`SignalBundle`]. Panic-free on
/// any input — malformed JSON, missing fields, wrong JSON types, a
/// truncated payload, or an oversized declared channel count are all
/// typed errors, never an unwrap/index/slice panic (see
/// `truncation_sweep_never_panics`).
pub fn read_bundle_from_descriptor(
    descriptor: &FormatDescriptor,
    data: &[u8],
    sidecar_bytes: &[u8],
) -> LmlResult<SignalBundle> {
    descriptor.validate().map_err(to_lml_err)?;
    let json = parse_sidecar_json(sidecar_bytes).map_err(to_lml_err)?;
    let obj = sidecar_object(&json).map_err(to_lml_err)?;

    let n_channels = resolve_channel_count(&descriptor.channel_count, obj).map_err(to_lml_err)?;
    let sample_rate = resolve_sample_rate(&descriptor.sample_rate, obj).map_err(to_lml_err)?;
    let (channels, phys_min, phys_max, phys_dim) =
        resolve_channel_arrays(obj, n_channels).map_err(to_lml_err)?;
    let (signal, n_samples) =
        decode_signal_i64(descriptor, n_channels, data).map_err(to_lml_err)?;

    let duration_s = if sample_rate > 0.0 {
        n_samples as f64 / sample_rate
    } else {
        0.0
    };

    let mut sidecar = Vec::new();
    for key in &descriptor.sidecar_keys {
        match key.as_str() {
            "sidecar_json" => sidecar.push(SidecarBlob {
                key: key.clone(),
                bytes: sidecar_bytes.to_vec(),
                aux: None,
            }),
            "payload" => sidecar.push(SidecarBlob {
                key: key.clone(),
                bytes: data.to_vec(),
                aux: None,
            }),
            other => {
                return Err(to_lml_err(DescriptorError::UnknownSidecarKey(
                    other.to_string(),
                )));
            }
        }
    }

    let bundle = SignalBundle {
        signal,
        sample_rate,
        channels,
        phys_min,
        phys_max,
        duration_s,
        metadata: SourceMetadata {
            source_file: String::new(),
            format: descriptor.format_name.clone(),
            patient_id: String::new(),
            recording_info: String::new(),
            startdate: String::new(),
            phys_dim,
        },
        sidecar,
    };
    bundle.validate()?;
    Ok(bundle)
}

/// The Abir lowering counterpart to [`read_bundle_from_descriptor`] —
/// decodes straight into a native-width `Column` (`I16`/`I24`/`I32`)
/// instead of always widening to `i64`, mirroring every hand-written
/// reader's specialized `lower_to_abir` override (ADR 0069 L7). `F32` is
/// unreachable here because `descriptor.validate()` (called first)
/// refuses it.
pub fn lower_to_abir_from_descriptor(
    descriptor: &FormatDescriptor,
    data: &[u8],
    sidecar_bytes: &[u8],
) -> LmlResult<Abir> {
    descriptor.validate().map_err(to_lml_err)?;
    let json = parse_sidecar_json(sidecar_bytes).map_err(to_lml_err)?;
    let obj = sidecar_object(&json).map_err(to_lml_err)?;

    let n_channels = resolve_channel_count(&descriptor.channel_count, obj).map_err(to_lml_err)?;
    let sample_rate = resolve_sample_rate(&descriptor.sample_rate, obj).map_err(to_lml_err)?;
    let (labels, phys_min, phys_max, _phys_dim) =
        resolve_channel_arrays(obj, n_channels).map_err(to_lml_err)?;
    let (native, n_samples) =
        decode_native_columns(descriptor, n_channels, data).map_err(to_lml_err)?;

    let channels: Vec<Channel> = match native {
        NativeColumns::I16(cols) => build_channels(cols, &labels, &phys_min, &phys_max, Column::I16),
        NativeColumns::I24(cols) => build_channels(cols, &labels, &phys_min, &phys_max, Column::I24),
        NativeColumns::I32(cols) => build_channels(cols, &labels, &phys_min, &phys_max, Column::I32),
    };

    let format_hint = resolve_format_hint(&descriptor.channel_modality, &labels);
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    Ok(Abir::from_parts(channels, sample_rate, n_samples)
        .with_inferred_modality(&label_refs, format_hint.as_deref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::raw::RawReader;
    use crate::source::reader::SignalSourceReader;

    // ─── Shared synthetic-input helpers ─────────────────────────────

    fn raw_sidecar_json(n_ch: usize, sample_rate: f64, dtype: &str, orient: &str) -> String {
        let chans: Vec<String> = (0..n_ch).map(|i| format!("\"ch{i}\"")).collect();
        let pmin: Vec<String> = (0..n_ch).map(|i| format!("{}", -200.0 - i as f64)).collect();
        let pmax: Vec<String> = (0..n_ch).map(|i| format!("{}", 200.0 + i as f64)).collect();
        format!(
            "{{\"n_channels\":{n_ch},\"sample_rate\":{sample_rate},\"dtype\":\"{dtype}\",\
             \"orientation\":\"{orient}\",\"channels\":[{}],\
             \"phys_min\":[{}],\"phys_max\":[{}],\"phys_dim\":\"uV\"}}",
            chans.join(","),
            pmin.join(","),
            pmax.join(","),
        )
    }

    fn synth_value(s: usize, ch: usize) -> i32 {
        (s as i32).wrapping_mul(37).wrapping_add((ch as i32).wrapping_mul(911)) % 60_000 - 30_000
    }

    fn raw_payload_int16(n_ch: usize, n_samples: usize, multiplexed: bool) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(n_ch * n_samples * 2);
        if multiplexed {
            for s in 0..n_samples {
                for ch in 0..n_ch {
                    bytes.extend_from_slice(&(synth_value(s, ch) as i16).to_le_bytes());
                }
            }
        } else {
            for ch in 0..n_ch {
                for s in 0..n_samples {
                    bytes.extend_from_slice(&(synth_value(s, ch) as i16).to_le_bytes());
                }
            }
        }
        bytes
    }

    fn raw_payload_int32(n_ch: usize, n_samples: usize, multiplexed: bool) -> Vec<u8> {
        let big = |s: usize, ch: usize| -> i32 {
            (s as i32).wrapping_mul(70_000).wrapping_add(ch as i32) - 500_000
        };
        let mut bytes = Vec::with_capacity(n_ch * n_samples * 4);
        if multiplexed {
            for s in 0..n_samples {
                for ch in 0..n_ch {
                    bytes.extend_from_slice(&big(s, ch).to_le_bytes());
                }
            }
        } else {
            for ch in 0..n_ch {
                for s in 0..n_samples {
                    bytes.extend_from_slice(&big(s, ch).to_le_bytes());
                }
            }
        }
        bytes
    }

    fn descriptor_for(dtype: DescriptorDtype, orientation: DescriptorOrientation) -> FormatDescriptor {
        FormatDescriptor {
            format_name: "RAW".to_string(),
            dtype,
            endian: Endian::Little,
            orientation,
            channel_count: ChannelCount::FromSidecarField("n_channels".to_string()),
            sample_rate: SampleRateSpec::FromSidecarField("sample_rate".to_string()),
            channel_modality: ChannelModality::default(),
            sidecar_keys: vec![],
        }
    }

    // ─── S5 Increment 3 gate 1 (load-bearing): raw-as-descriptor ────

    /// Builds a `FormatDescriptor` equivalent to the RAW format and runs
    /// it over the SAME synthetic inputs a `RawReader` would see (a real
    /// `.raw` + sidecar JSON on disk), then asserts the descriptor path's
    /// `SignalBundle` AND lowered `Abir` are byte-identical to
    /// `RawReader`'s: sample_rate, channels, phys_min/phys_max, the full
    /// `signal` matrix, the lowered `Column` variant, and its widened
    /// `window_i64` — across both dtype arms RAW supports (int16, int32)
    /// and both orientations (multiplexed, vectorized).
    #[test]
    fn raw_format_as_descriptor_matches_hand_written() {
        fn check(
            n_ch: usize,
            _n_samples: usize,
            dtype_str: &str,
            orient_str: &str,
            descriptor_dtype: DescriptorDtype,
            descriptor_orient: DescriptorOrientation,
            payload: Vec<u8>,
        ) {
            let tmp = tempfile::tempdir().unwrap();
            let raw_path = tmp.path().join("data.raw");
            let json_path = tmp.path().join("data.json");
            let sidecar_text = raw_sidecar_json(n_ch, 273.5, dtype_str, orient_str);
            std::fs::write(&raw_path, &payload).unwrap();
            std::fs::write(&json_path, &sidecar_text).unwrap();

            let hand_bundle = RawReader::new(&raw_path).read_bundle().unwrap();
            let hand_abir = RawReader::new(&raw_path).lower_to_abir().unwrap();

            let descriptor = descriptor_for(descriptor_dtype, descriptor_orient);
            let desc_bundle =
                read_bundle_from_descriptor(&descriptor, &payload, sidecar_text.as_bytes())
                    .unwrap();
            let desc_abir =
                lower_to_abir_from_descriptor(&descriptor, &payload, sidecar_text.as_bytes())
                    .unwrap();

            // SignalBundle byte-identity.
            assert_eq!(desc_bundle.sample_rate, hand_bundle.sample_rate);
            assert_eq!(desc_bundle.channels, hand_bundle.channels);
            assert_eq!(desc_bundle.phys_min, hand_bundle.phys_min);
            assert_eq!(desc_bundle.phys_max, hand_bundle.phys_max);
            assert_eq!(desc_bundle.signal, hand_bundle.signal);
            assert_eq!(desc_bundle.duration_s, hand_bundle.duration_s);

            // Lowered Abir byte-identity.
            assert_eq!(desc_abir.n_channels(), hand_abir.n_channels());
            assert_eq!(desc_abir.n_samples, hand_abir.n_samples);
            assert_eq!(desc_abir.sample_rate, hand_abir.sample_rate);
            for (d_ch, h_ch) in desc_abir.channels.iter().zip(hand_abir.channels.iter()) {
                assert_eq!(d_ch.label, h_ch.label);
                assert_eq!(d_ch.phys_min, h_ch.phys_min);
                assert_eq!(d_ch.phys_max, h_ch.phys_max);
                match (&d_ch.data, &h_ch.data) {
                    (Column::I16(_), Column::I16(_)) => {}
                    (Column::I32(_), Column::I32(_)) => {}
                    (a, b) => panic!("Column variant diverged: {a:?} vs {b:?}"),
                }
                let d_w = d_ch.data.window_i64(0, desc_abir.n_samples);
                let h_w = h_ch.data.window_i64(0, hand_abir.n_samples);
                assert_eq!(d_w.as_ref(), h_w.as_ref());
            }
            assert_eq!(desc_abir.provenance().tag, hand_abir.provenance().tag);
            assert_eq!(desc_abir.provenance().source, hand_abir.provenance().source);
        }

        let n_ch = 3usize;
        let n_samples = 200usize;
        check(
            n_ch,
            n_samples,
            "int16",
            "multiplexed",
            DescriptorDtype::I16,
            DescriptorOrientation::Multiplexed,
            raw_payload_int16(n_ch, n_samples, true),
        );
        check(
            n_ch,
            n_samples,
            "int16",
            "vectorized",
            DescriptorDtype::I16,
            DescriptorOrientation::Vectorized,
            raw_payload_int16(n_ch, n_samples, false),
        );

        let n_ch2 = 2usize;
        let n_samples2 = 150usize;
        check(
            n_ch2,
            n_samples2,
            "int32",
            "multiplexed",
            DescriptorDtype::I32,
            DescriptorOrientation::Multiplexed,
            raw_payload_int32(n_ch2, n_samples2, true),
        );
        check(
            n_ch2,
            n_samples2,
            "int32",
            "vectorized",
            DescriptorDtype::I32,
            DescriptorOrientation::Vectorized,
            raw_payload_int32(n_ch2, n_samples2, false),
        );
    }

    // ─── Gate 2: G5 F32 refusal ──────────────────────────────────────

    #[test]
    fn validate_refuses_f32() {
        let d = FormatDescriptor {
            format_name: "RAW".into(),
            dtype: DescriptorDtype::F32,
            endian: Endian::Little,
            orientation: DescriptorOrientation::Multiplexed,
            channel_count: ChannelCount::Fixed(1),
            sample_rate: SampleRateSpec::Fixed(250.0),
            channel_modality: ChannelModality::default(),
            sidecar_keys: vec![],
        };
        assert_eq!(d.validate(), Err(DescriptorError::FloatDtypeRefused));

        // Also refused at both interpreter entry points, not just
        // validate() in isolation — defense in depth.
        let err = read_bundle_from_descriptor(&d, &[0u8; 8], b"{}").unwrap_err();
        assert!(err.to_string().to_lowercase().contains("f32"), "got: {err}");
        let err = lower_to_abir_from_descriptor(&d, &[0u8; 8], b"{}").unwrap_err();
        assert!(err.to_string().to_lowercase().contains("f32"), "got: {err}");
    }

    // ─── Gate 3: endian is first-class (G5) ─────────────────────────

    #[test]
    fn endian_big_decodes_differently_from_little() {
        // A 2-channel, 2-sample int16 multiplexed payload whose bytes are
        // not palindromic, so LE and BE decode to different values.
        let bytes: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let sidecar = b"{\"n_channels\":2,\"sample_rate\":250.0,\"channels\":[\"a\",\"b\"],\
                          \"phys_min\":[-1.0,-1.0],\"phys_max\":[1.0,1.0]}";

        let mut le = descriptor_for(DescriptorDtype::I16, DescriptorOrientation::Multiplexed);
        le.channel_count = ChannelCount::Fixed(2);
        le.sample_rate = SampleRateSpec::Fixed(250.0);
        let mut be = le.clone();
        be.endian = Endian::Big;
        assert_eq!(le.endian, Endian::Little);

        let le_bundle = read_bundle_from_descriptor(&le, &bytes, sidecar).unwrap();
        let be_bundle = read_bundle_from_descriptor(&be, &bytes, sidecar).unwrap();

        assert_ne!(
            le_bundle.signal, be_bundle.signal,
            "LE and BE descriptors must decode the same bytes differently"
        );
        assert_eq!(le_bundle.signal[0][0], i16::from_le_bytes([0x01, 0x02]) as i64);
        assert_eq!(be_bundle.signal[0][0], i16::from_be_bytes([0x01, 0x02]) as i64);
        assert_eq!(le_bundle.signal[1][0], i16::from_le_bytes([0x03, 0x04]) as i64);
        assert_eq!(be_bundle.signal[1][0], i16::from_be_bytes([0x03, 0x04]) as i64);
    }

    // ─── Gate 4: reciprocal sample-rate transform (G5) ──────────────

    #[test]
    fn reciprocal_sample_rate_derives_hz_and_rejects_zero_divisor() {
        // BrainVision's own case: Hz = 1e6 / SamplingInterval(us).
        // interval=4000us -> 250Hz, matching brainvision.rs's synth fixture.
        let mut d = descriptor_for(DescriptorDtype::I16, DescriptorOrientation::Multiplexed);
        d.channel_count = ChannelCount::Fixed(1);
        d.sample_rate = SampleRateSpec::Reciprocal {
            sidecar_field: "SamplingInterval".to_string(),
            numerator: 1_000_000.0,
        };
        let sidecar = b"{\"SamplingInterval\":4000,\"channels\":[\"a\"],\
                          \"phys_min\":[-1.0],\"phys_max\":[1.0]}";
        let bundle = read_bundle_from_descriptor(&d, &[0, 0, 0, 0], sidecar).unwrap();
        assert!(
            (bundle.sample_rate - 250.0).abs() < 1e-9,
            "sr={}",
            bundle.sample_rate
        );

        // Zero divisor -> typed Err, never a panic (division by zero on
        // f64 would silently produce inf/NaN if unchecked).
        let zero_sidecar = b"{\"SamplingInterval\":0,\"channels\":[\"a\"],\
                               \"phys_min\":[-1.0],\"phys_max\":[1.0]}";
        let err = read_bundle_from_descriptor(&d, &[0, 0, 0, 0], zero_sidecar).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("divisor"),
            "got: {err}"
        );
    }

    // ─── Gate 5: panic-free on malformed/truncated input ────────────

    /// Mirrors `ir.rs`'s / `pipeline_dsl.rs`'s truncation-sweep
    /// convention: truncating a fully valid payload (or its sidecar JSON
    /// text) at every byte offset must never panic, regardless of
    /// whether the result is `Ok` or a typed `Err`.
    #[test]
    fn truncation_sweep_never_panics() {
        let n_ch = 3usize;
        let n_samples = 50usize;
        let payload = raw_payload_int16(n_ch, n_samples, true);
        let sidecar = raw_sidecar_json(n_ch, 250.0, "int16", "multiplexed");
        let descriptor = descriptor_for(DescriptorDtype::I16, DescriptorOrientation::Multiplexed);

        for k in 0..=payload.len() {
            let _ = read_bundle_from_descriptor(&descriptor, &payload[..k], sidecar.as_bytes());
            let _ = lower_to_abir_from_descriptor(&descriptor, &payload[..k], sidecar.as_bytes());
        }

        assert!(sidecar.is_ascii(), "sweep corpus must be ASCII-safe to slice");
        for k in 0..sidecar.len() {
            let _ = read_bundle_from_descriptor(&descriptor, &payload, sidecar[..k].as_bytes());
        }

        // Also sweep a handful of adversarial channel-count magnitudes
        // that could overflow a naive `n_channels * bytes_per_sample`.
        let mut huge = descriptor.clone();
        huge.channel_count = ChannelCount::Fixed(usize::MAX / 2);
        let _ = read_bundle_from_descriptor(&huge, &payload, sidecar.as_bytes());
        let _ = lower_to_abir_from_descriptor(&huge, &payload, sidecar.as_bytes());
    }

    // ─── Extra coverage (schema completeness, not gate-numbered) ────

    #[test]
    fn validate_rejects_unknown_sidecar_key() {
        let mut d = descriptor_for(DescriptorDtype::I16, DescriptorOrientation::Multiplexed);
        d.channel_count = ChannelCount::Fixed(1);
        d.sidecar_keys = vec!["not_a_real_key".to_string()];
        assert_eq!(
            d.validate(),
            Err(DescriptorError::UnknownSidecarKey("not_a_real_key".into()))
        );
    }

    #[test]
    fn sidecar_keys_preserve_payload_and_sidecar_bytes() {
        let mut d = descriptor_for(DescriptorDtype::I16, DescriptorOrientation::Multiplexed);
        d.channel_count = ChannelCount::Fixed(1);
        d.sample_rate = SampleRateSpec::Fixed(100.0);
        d.sidecar_keys = vec!["sidecar_json".to_string(), "payload".to_string()];
        let payload = vec![1u8, 0, 2, 0];
        let sidecar =
            b"{\"channels\":[\"a\"],\"phys_min\":[-1.0],\"phys_max\":[1.0]}".to_vec();
        let bundle = read_bundle_from_descriptor(&d, &payload, &sidecar).unwrap();
        assert_eq!(bundle.sidecar.len(), 2);
        assert_eq!(bundle.sidecar_first("sidecar_json").unwrap().bytes, sidecar);
        assert_eq!(bundle.sidecar_first("payload").unwrap().bytes, payload);
    }

    #[test]
    fn i24_dtype_decodes_correctly() {
        // Not a RAW-supported dtype, but part of the descriptor schema —
        // sanity-check the primitive independent of any hand-written
        // reader comparison. 0x7FFFFF / -1 / min-i24, little-endian.
        let mut d = descriptor_for(DescriptorDtype::I24, DescriptorOrientation::Multiplexed);
        d.channel_count = ChannelCount::Fixed(1);
        d.sample_rate = SampleRateSpec::Fixed(100.0);
        let bytes: Vec<u8> = vec![0xFF, 0xFF, 0x7F, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x80];
        let sidecar = b"{\"channels\":[\"a\"],\"phys_min\":[-1.0],\"phys_max\":[1.0]}";
        let bundle = read_bundle_from_descriptor(&d, &bytes, sidecar).unwrap();
        assert_eq!(bundle.signal[0], vec![0x7F_FFFF_i64, -1, -8_388_608]);

        let abir = lower_to_abir_from_descriptor(&d, &bytes, sidecar).unwrap();
        assert!(matches!(abir.channels[0].data, Column::I24(_)));
        assert_eq!(
            abir.channels[0].data.window_i64(0, 3).as_ref(),
            &[0x7F_FFFF_i64, -1, -8_388_608]
        );
    }

    #[test]
    fn channel_modality_rule_selects_format_hint() {
        use abir::{Ecg, Modality};

        let mut d = descriptor_for(DescriptorDtype::I16, DescriptorOrientation::Multiplexed);
        d.channel_count = ChannelCount::Fixed(3);
        d.sample_rate = SampleRateSpec::Fixed(250.0);
        d.channel_modality = ChannelModality {
            rules: vec![ChannelModalityRule {
                pattern: "lead".to_string(),
                modality: "ECG".to_string(),
            }],
            default: None,
        };
        let bytes = raw_payload_int16(3, 10, true);
        let sidecar = b"{\"channels\":[\"Lead I\",\"Lead II\",\"V1\"],\
                          \"phys_min\":[-1.0,-1.0,-1.0],\"phys_max\":[1.0,1.0,1.0]}";
        let abir = lower_to_abir_from_descriptor(&d, &bytes, sidecar).unwrap();
        assert_eq!(abir.provenance().tag, Ecg::TAG);
    }
}
