//! Textual biosignal-IR form — the LLVM `.ll` analogue for [`SignalBundle`]
//! (ADR 0069). A **deterministic manifest** of the IR a frontend produced:
//! recording metadata, per-channel facts, a signal digest, and sidecar
//! keys/sizes/digests. Its purpose is **golden-diffing and debugging** — you
//! dump what a reader produced and prove a refactor (or a pass) is byte-faithful.
//! It is NOT a user authoring language, and (deliberately) it digests the large
//! tensors rather than dumping every sample.
//!
//! Determinism contract: same `SignalBundle` ⇒ byte-identical text, on every
//! run and platform. No map iteration; floats via `{:?}` (round-trippable);
//! sidecar order is the reader's (already deterministic).

use crate::source::SignalBundle;
use core::fmt::Write as _;
use sha2::{Digest, Sha256};

fn sha_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::new().chain_update(bytes).finalize())
}

/// sha256 over channel-major signal: count, then per-channel (len + i64-LE).
/// Matches `tests/front_end_bit_exact.rs::sha_signal` so the two goldens agree.
fn signal_digest(signal: &[Vec<i64>]) -> String {
    let mut h = Sha256::new();
    h.update((signal.len() as u64).to_le_bytes());
    for ch in signal {
        h.update((ch.len() as u64).to_le_bytes());
        for &s in ch {
            h.update(s.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

/// Render a [`SignalBundle`] to the deterministic textual IR (v1).
pub fn to_ir_text(b: &SignalBundle) -> String {
    let m = &b.metadata;
    let mut s = String::new();
    // `write!` to a String is infallible; ignore the Result.
    let _ = writeln!(s, "; lamquant biosignal IR v1");
    let _ = writeln!(s, "recording {{");
    let _ = writeln!(s, "  format = {:?}", m.format);
    let _ = writeln!(s, "  source = {:?}", m.source_file);
    let _ = writeln!(s, "  patient = {:?}", m.patient_id);
    let _ = writeln!(s, "  start = {:?}", m.startdate);
    let _ = writeln!(s, "  info = {:?}", m.recording_info);
    let _ = writeln!(s, "  phys_dim = {:?}", m.phys_dim);
    let _ = writeln!(s, "  sample_rate = {:?}", b.sample_rate);
    let _ = writeln!(s, "  duration_s = {:?}", b.duration_s);
    let _ = writeln!(s, "}}");

    let _ = writeln!(s, "channels [{}] {{", b.channels.len());
    for (i, name) in b.channels.iter().enumerate() {
        let pmin = b.phys_min.get(i).copied().unwrap_or(0.0);
        let pmax = b.phys_max.get(i).copied().unwrap_or(0.0);
        let n = b.signal.get(i).map(Vec::len).unwrap_or(0);
        let _ = writeln!(s, "  {i}: {name:?} phys[{pmin:?},{pmax:?}] n={n}");
    }
    let _ = writeln!(s, "}}");

    let _ = writeln!(
        s,
        "signal {{ channels={} digest=sha256:{} }}",
        b.signal.len(),
        signal_digest(&b.signal)
    );

    let _ = writeln!(s, "sidecar [{}] {{", b.sidecar.len());
    for sc in &b.sidecar {
        let _ = writeln!(
            s,
            "  {:?} aux={:?} {} bytes digest=sha256:{}",
            sc.key,
            sc.aux,
            sc.bytes.len(),
            sha_hex(&sc.bytes)
        );
    }
    let _ = writeln!(s, "}}");
    s
}

// ─── S5 Increment 1: the textual-IR parser (ADR 0069, task #20) ──────────
//
// `to_ir_text` is lossy by construction (it digests samples + sidecar bytes
// and never emits provenance beyond `SourceMetadata`), so a literal
// `parse(text) -> SignalBundle` inverse is impossible: the bytes aren't in
// the text to recover. What IS recoverable is exactly what the text
// carries — that's `IrManifest`, and the contract this module proves is a
// **projection** round-trip:
//
//     parse_ir_text(&to_ir_text(b)) == project(b)
//
// `project` builds the manifest from a `SignalBundle` using the same
// digest functions `to_ir_text` reads (`signal_digest`, `sha_hex`);
// `parse_ir_text` is the hand-written recursive-descent inverse of the
// `to_ir_text` grammar (magic line, `recording { }`, `channels [N] { }`,
// `signal { }`, `sidecar [K] { }`). It is panic-free on arbitrary input —
// every malformed token is a typed `IrParseError`, never an unwrap/index
// panic (see `negative_battery` + the truncation sweep below).

/// Bit-pattern equality for `f64` — NaN-safe (`NaN.to_bits() ==
/// NaN.to_bits()` for identical payloads), unlike `==`. Used by
/// [`IrManifest`]'s `PartialEq`/`Eq` impls so the round-trip check holds
/// even if a bundle ever carries a non-finite phys/sample-rate/duration
/// value. The corpus this module tests against uses finite values only —
/// `to_ir_text`'s `{:?}` float rendering collapses all NaN payloads to the
/// single canonical `"NaN"` token, so a NaN *specifically* does not survive
/// the projection round-trip bit-for-bit; this is a documented, inherent
/// property of the text format, not a parser bug.
fn f64_bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// `recording { }` block of the projected IR manifest — the
/// `SourceMetadata` + shape scalars `to_ir_text` renders.
#[derive(Debug, Clone)]
pub struct IrRecording {
    pub format: String,
    pub source: String,
    pub patient: String,
    pub start: String,
    pub info: String,
    pub phys_dim: String,
    pub sample_rate: f64,
    pub duration_s: f64,
}

impl PartialEq for IrRecording {
    fn eq(&self, other: &Self) -> bool {
        self.format == other.format
            && self.source == other.source
            && self.patient == other.patient
            && self.start == other.start
            && self.info == other.info
            && self.phys_dim == other.phys_dim
            && f64_bits_eq(self.sample_rate, other.sample_rate)
            && f64_bits_eq(self.duration_s, other.duration_s)
    }
}
impl Eq for IrRecording {}

/// One `channels [N] { }` body line, projected.
#[derive(Debug, Clone)]
pub struct IrChannel {
    pub index: usize,
    pub name: String,
    pub phys_min: f64,
    pub phys_max: f64,
    pub n: usize,
}

impl PartialEq for IrChannel {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
            && self.name == other.name
            && f64_bits_eq(self.phys_min, other.phys_min)
            && f64_bits_eq(self.phys_max, other.phys_max)
            && self.n == other.n
    }
}
impl Eq for IrChannel {}

/// The `signal { }` line, projected: channel count + the digest
/// `signal_digest` computes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrSignal {
    pub n_channels: usize,
    pub digest: String,
}

/// One `sidecar [K] { }` body line, projected: the blob's key/aux/length +
/// its digest (never the bytes themselves — those are not in the text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrSidecarEntry {
    pub key: String,
    pub aux: Option<i64>,
    pub len: usize,
    pub digest: String,
}

/// The full projected manifest — exactly what [`to_ir_text`] carries.
/// This is the round-trip target: `parse_ir_text(&to_ir_text(b)) ==
/// project(b)`. It is a *projection* of [`SignalBundle`], not the bundle
/// itself (see module-level note above).
#[derive(Debug, Clone)]
pub struct IrManifest {
    pub recording: IrRecording,
    pub channels: Vec<IrChannel>,
    pub signal: IrSignal,
    pub sidecar: Vec<IrSidecarEntry>,
}

impl PartialEq for IrManifest {
    fn eq(&self, other: &Self) -> bool {
        self.recording == other.recording
            && self.channels == other.channels
            && self.signal == other.signal
            && self.sidecar == other.sidecar
    }
}
impl Eq for IrManifest {}

/// Build the [`IrManifest`] projection of a [`SignalBundle`] — the same
/// fields, read the same way (including the `get(i).unwrap_or` shape
/// tolerance for a not-yet-`validate`d bundle), that [`to_ir_text`] reads.
pub fn project(b: &SignalBundle) -> IrManifest {
    let m = &b.metadata;
    let channels = b
        .channels
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let phys_min = b.phys_min.get(i).copied().unwrap_or(0.0);
            let phys_max = b.phys_max.get(i).copied().unwrap_or(0.0);
            let n = b.signal.get(i).map(Vec::len).unwrap_or(0);
            IrChannel {
                index: i,
                name: name.clone(),
                phys_min,
                phys_max,
                n,
            }
        })
        .collect();
    let sidecar = b
        .sidecar
        .iter()
        .map(|sc| IrSidecarEntry {
            key: sc.key.clone(),
            aux: sc.aux,
            len: sc.bytes.len(),
            digest: sha_hex(&sc.bytes),
        })
        .collect();
    IrManifest {
        recording: IrRecording {
            format: m.format.clone(),
            source: m.source_file.clone(),
            patient: m.patient_id.clone(),
            start: m.startdate.clone(),
            info: m.recording_info.clone(),
            phys_dim: m.phys_dim.clone(),
            sample_rate: b.sample_rate,
            duration_s: b.duration_s,
        },
        channels,
        signal: IrSignal {
            n_channels: b.signal.len(),
            digest: signal_digest(&b.signal),
        },
        sidecar,
    }
}

/// Parse error for the [`to_ir_text`] grammar. One variant per grammar
/// violation; the parser never panics on malformed/truncated/adversarial
/// input — see `negative_battery` (one crafted input per variant) and the
/// byte-offset truncation sweep in the test module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrParseError {
    /// The first line isn't `; lamquant biosignal IR v1`.
    BadMagic,
    /// A required fixed-literal line (`recording {`, a closing `}`, or a
    /// `channels [N] {` / `sidecar [K] {` header) didn't match.
    MissingBlock { expected: String },
    /// A `recording { }` `  key = value` line didn't match the expected
    /// key/` = `/quoted-value shape (wrong key, missing separator, or
    /// trailing content after the value).
    BadRecordingField { field: &'static str },
    /// A numeric token (channel index, `phys_min`/`phys_max`, `n`,
    /// `sample_rate`, `duration_s`, a `[N]`/`[K]` count, `aux`, or a
    /// sidecar `len`) failed `FromStr`.
    BadNumber { context: &'static str },
    /// A `channels [N] {` body line doesn't match `  {i}: {name:?}
    /// phys[{pmin},{pmax}] n={n}`. `line` is the 0-based index within the
    /// channels block (not the file line number).
    BadChannelLine { line: usize },
    /// `signal { channels=M ... }`'s `M` disagrees with the number of
    /// lines actually parsed in the `channels [N] { }` block.
    ChannelCountMismatch,
    /// A `sha256:` digest isn't exactly 64 lowercase hex chars, or its
    /// surrounding literal framing (`signal { ... }` / `... bytes
    /// digest=sha256:`) didn't match.
    BadDigest,
    /// A `{:?}`-quoted string's closing (unescaped) quote was never found
    /// before the line ended.
    UnterminatedString,
    /// A `\` inside a quoted string wasn't followed by one of `" \ n t r
    /// 0 u{HEX}` (or the `u{...}` body was empty/non-hex/unterminated, or
    /// encoded a value outside the Unicode scalar range).
    BadEscape,
    /// A `sidecar [K] {` body line doesn't match `  {key:?} aux={aux:?}
    /// {len} bytes digest=sha256:{hex}`. `line` is the 0-based index
    /// within the sidecar block.
    BadSidecarLine { line: usize },
    /// Non-empty content follows the document's final `}`.
    TrailingGarbage,
    /// Input ended while a line/block/token was still expected.
    Truncated { context: &'static str },
}

impl core::fmt::Display for IrParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => write!(
                f,
                "not a lamquant biosignal IR v1 document (bad magic line)"
            ),
            Self::MissingBlock { expected } => write!(f, "expected line {expected:?}"),
            Self::BadRecordingField { field } => write!(f, "malformed recording field `{field}`"),
            Self::BadNumber { context } => write!(f, "malformed number ({context})"),
            Self::BadChannelLine { line } => write!(f, "malformed channel line at index {line}"),
            Self::ChannelCountMismatch => {
                write!(
                    f,
                    "signal{{channels=..}} disagrees with the channels block count"
                )
            }
            Self::BadDigest => write!(f, "malformed sha256 digest"),
            Self::UnterminatedString => write!(f, "unterminated quoted string"),
            Self::BadEscape => write!(f, "invalid escape sequence in quoted string"),
            Self::BadSidecarLine { line } => write!(f, "malformed sidecar line at index {line}"),
            Self::TrailingGarbage => write!(f, "trailing content after the document end"),
            Self::Truncated { context } => write!(f, "truncated input: expected {context}"),
        }
    }
}

impl std::error::Error for IrParseError {}

/// Parse a Rust-`{:?}`-quoted string starting at byte 0 of `s` (must begin
/// with `"`). Returns the unescaped value and the remainder of `s` after
/// the closing quote. Walks `s.chars()` (never raw byte indexing), so
/// multi-byte UTF-8 is never split and this can never panic. `structural`
/// is the error to return if `s` doesn't start with `"` at all (the
/// caller knows whether that means a bad recording field, channel line,
/// or sidecar line).
fn parse_quoted(s: &str, structural: IrParseError) -> Result<(String, &str), IrParseError> {
    let mut chars = s.chars();
    if chars.next() != Some('"') {
        return Err(structural);
    }
    let mut out = String::new();
    loop {
        match chars.next() {
            None => return Err(IrParseError::UnterminatedString),
            Some('"') => return Ok((out, chars.as_str())),
            Some('\\') => match chars.next() {
                None => return Err(IrParseError::UnterminatedString),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('0') => out.push('\0'),
                Some('u') => {
                    if chars.next() != Some('{') {
                        return Err(IrParseError::BadEscape);
                    }
                    let mut hex = String::new();
                    loop {
                        match chars.next() {
                            Some('}') => break,
                            Some(c) if c.is_ascii_hexdigit() && hex.len() < 6 => hex.push(c),
                            _ => return Err(IrParseError::BadEscape),
                        }
                    }
                    if hex.is_empty() {
                        return Err(IrParseError::BadEscape);
                    }
                    let cp = u32::from_str_radix(&hex, 16).map_err(|_| IrParseError::BadEscape)?;
                    let ch = char::from_u32(cp).ok_or(IrParseError::BadEscape)?;
                    out.push(ch);
                }
                Some(_) => return Err(IrParseError::BadEscape),
            },
            Some(c) => out.push(c),
        }
    }
}

/// Pull the next line, or `Truncated { context }` if the input is
/// exhausted.
fn next_line<'a>(
    lines: &mut std::str::Lines<'a>,
    context: &'static str,
) -> Result<&'a str, IrParseError> {
    lines.next().ok_or(IrParseError::Truncated { context })
}

/// Pull the next line and require it to equal `expected` exactly.
fn expect_line(lines: &mut std::str::Lines, expected: &'static str) -> Result<(), IrParseError> {
    let line = next_line(lines, expected)?;
    if line != expected {
        return Err(IrParseError::MissingBlock {
            expected: expected.to_string(),
        });
    }
    Ok(())
}

/// Parse a `  {key} = ` prefix off a `recording { }` field line, returning
/// the remainder (the still-to-be-parsed value).
fn parse_kv_prefix<'a>(line: &'a str, key: &'static str) -> Result<&'a str, IrParseError> {
    line.strip_prefix("  ")
        .and_then(|r| r.strip_prefix(key))
        .and_then(|r| r.strip_prefix(" = "))
        .ok_or(IrParseError::BadRecordingField { field: key })
}

/// Parse a `  {key} = "..."` string field.
fn parse_str_field(line: &str, key: &'static str) -> Result<String, IrParseError> {
    let rest = parse_kv_prefix(line, key)?;
    let (val, rest) = parse_quoted(rest, IrParseError::BadRecordingField { field: key })?;
    if !rest.is_empty() {
        return Err(IrParseError::BadRecordingField { field: key });
    }
    Ok(val)
}

/// Parse a `  {key} = {float}` field (`f64::from_str` accepts every
/// `{:?}` form: `NaN`, `inf`, `-inf`, and ordinary decimals).
fn parse_float_field(line: &str, key: &'static str) -> Result<f64, IrParseError> {
    let rest = parse_kv_prefix(line, key)?;
    rest.parse::<f64>()
        .map_err(|_| IrParseError::BadNumber { context: key })
}

/// Parse a `{name} [{count}] {{` block-opening header line (`channels [N]
/// {` / `sidecar [K] {`).
fn parse_count_header(
    lines: &mut std::str::Lines,
    name: &'static str,
) -> Result<usize, IrParseError> {
    let line = next_line(lines, name)?;
    let bad = || IrParseError::MissingBlock {
        expected: format!("{name} [N] {{"),
    };
    let rest = line
        .strip_prefix(name)
        .and_then(|r| r.strip_prefix(" ["))
        .ok_or_else(bad)?;
    let digits = rest.strip_suffix("] {").ok_or_else(bad)?;
    digits
        .parse::<usize>()
        .map_err(|_| IrParseError::BadNumber { context: name })
}

/// Parse one `channels [N] { }` body line: `  {i}: {name:?}
/// phys[{pmin},{pmax}] n={n}`.
fn parse_channel_line(line: &str, idx: usize) -> Result<IrChannel, IrParseError> {
    let bad = || IrParseError::BadChannelLine { line: idx };
    let rest = line.strip_prefix("  ").ok_or_else(bad)?;
    let (idx_str, rest) = rest.split_once(':').ok_or_else(bad)?;
    let index: usize = idx_str.parse().map_err(|_| IrParseError::BadNumber {
        context: "channel index",
    })?;
    let rest = rest.strip_prefix(' ').ok_or_else(bad)?;
    let (name, rest) = parse_quoted(rest, bad())?;
    let rest = rest.strip_prefix(" phys[").ok_or_else(bad)?;
    let (pmin_str, rest) = rest.split_once(',').ok_or_else(bad)?;
    let phys_min: f64 = pmin_str.parse().map_err(|_| IrParseError::BadNumber {
        context: "phys_min",
    })?;
    let (pmax_str, rest) = rest.split_once(']').ok_or_else(bad)?;
    let phys_max: f64 = pmax_str.parse().map_err(|_| IrParseError::BadNumber {
        context: "phys_max",
    })?;
    let rest = rest.strip_prefix(" n=").ok_or_else(bad)?;
    let n: usize = rest.parse().map_err(|_| IrParseError::BadNumber {
        context: "channel n",
    })?;
    Ok(IrChannel {
        index,
        name,
        phys_min,
        phys_max,
        n,
    })
}

/// Validate a digest string is exactly 64 lowercase hex chars (the shape
/// `sha_hex`/`{:x}` on a `Sha256` always produces).
fn valid_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Parse the `signal { channels=M digest=sha256:{hex} }` line.
fn parse_signal_line(line: &str) -> Result<(usize, String), IrParseError> {
    let bad_block = || IrParseError::MissingBlock {
        expected: "signal { channels=N digest=sha256:<64 hex> }".to_string(),
    };
    let rest = line
        .strip_prefix("signal { channels=")
        .ok_or_else(bad_block)?;
    let (chan_str, rest) = rest.split_once(" digest=sha256:").ok_or_else(bad_block)?;
    let n_channels: usize = chan_str.parse().map_err(|_| IrParseError::BadNumber {
        context: "signal channels",
    })?;
    let digest = rest.strip_suffix(" }").ok_or(IrParseError::BadDigest)?;
    if !valid_hex64(digest) {
        return Err(IrParseError::BadDigest);
    }
    Ok((n_channels, digest.to_string()))
}

/// Parse one `sidecar [K] { }` body line: `  {key:?} aux={aux:?} {len}
/// bytes digest=sha256:{hex}`.
fn parse_sidecar_line(line: &str, idx: usize) -> Result<IrSidecarEntry, IrParseError> {
    let bad = || IrParseError::BadSidecarLine { line: idx };
    let rest = line.strip_prefix("  ").ok_or_else(bad)?;
    let (key, rest) = parse_quoted(rest, bad())?;
    let rest = rest.strip_prefix(" aux=").ok_or_else(bad)?;
    let (aux, rest) = if let Some(r) = rest.strip_prefix("None") {
        (None, r)
    } else if let Some(r) = rest.strip_prefix("Some(") {
        let (num_str, r) = r.split_once(')').ok_or_else(bad)?;
        let v: i64 = num_str.parse().map_err(|_| IrParseError::BadNumber {
            context: "sidecar aux",
        })?;
        (Some(v), r)
    } else {
        return Err(bad());
    };
    let rest = rest.strip_prefix(' ').ok_or_else(bad)?;
    let (len_str, rest) = rest.split_once(" bytes digest=sha256:").ok_or_else(bad)?;
    let len: usize = len_str.parse().map_err(|_| IrParseError::BadNumber {
        context: "sidecar len",
    })?;
    if !valid_hex64(rest) {
        return Err(IrParseError::BadDigest);
    }
    Ok(IrSidecarEntry {
        key,
        aux,
        len,
        digest: rest.to_string(),
    })
}

/// Parse the textual IR form ([`to_ir_text`]'s output) back into an
/// [`IrManifest`] — the exact inverse of that grammar. Panic-free on any
/// input (malformed, truncated, or adversarial): every failure is a typed
/// [`IrParseError`], never an unwrap/index panic.
///
/// This is a *projection* parse, not a `SignalBundle` reconstruction —
/// `to_ir_text` never emits the samples or sidecar bytes (only their
/// digests), so there is nothing to parse them back from. See the
/// module-level note above and `project`.
pub fn parse_ir_text(input: &str) -> Result<IrManifest, IrParseError> {
    let mut lines = input.lines();

    let magic = next_line(&mut lines, "magic line")?;
    if magic != "; lamquant biosignal IR v1" {
        return Err(IrParseError::BadMagic);
    }

    expect_line(&mut lines, "recording {")?;
    let format = parse_str_field(next_line(&mut lines, "format field")?, "format")?;
    let source = parse_str_field(next_line(&mut lines, "source field")?, "source")?;
    let patient = parse_str_field(next_line(&mut lines, "patient field")?, "patient")?;
    let start = parse_str_field(next_line(&mut lines, "start field")?, "start")?;
    let info = parse_str_field(next_line(&mut lines, "info field")?, "info")?;
    let phys_dim = parse_str_field(next_line(&mut lines, "phys_dim field")?, "phys_dim")?;
    let sample_rate =
        parse_float_field(next_line(&mut lines, "sample_rate field")?, "sample_rate")?;
    let duration_s = parse_float_field(next_line(&mut lines, "duration_s field")?, "duration_s")?;
    expect_line(&mut lines, "}")?;

    let n_channels_hdr = parse_count_header(&mut lines, "channels")?;
    let mut channels = Vec::new();
    for i in 0..n_channels_hdr {
        let line = next_line(&mut lines, "channel line")?;
        channels.push(parse_channel_line(line, i)?);
    }
    expect_line(&mut lines, "}")?;

    let signal_line = next_line(&mut lines, "signal line")?;
    let (n_sig_channels, digest) = parse_signal_line(signal_line)?;
    if n_sig_channels != channels.len() {
        return Err(IrParseError::ChannelCountMismatch);
    }

    let n_sidecar_hdr = parse_count_header(&mut lines, "sidecar")?;
    let mut sidecar = Vec::new();
    for i in 0..n_sidecar_hdr {
        let line = next_line(&mut lines, "sidecar line")?;
        sidecar.push(parse_sidecar_line(line, i)?);
    }
    expect_line(&mut lines, "}")?;

    if lines.next().is_some() {
        return Err(IrParseError::TrailingGarbage);
    }

    Ok(IrManifest {
        recording: IrRecording {
            format,
            source,
            patient,
            start,
            info,
            phys_dim,
            sample_rate,
            duration_s,
        },
        channels,
        signal: IrSignal {
            n_channels: n_sig_channels,
            digest,
        },
        sidecar,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{SidecarBlob, SourceMetadata};

    fn sample_bundle() -> SignalBundle {
        SignalBundle {
            signal: vec![vec![0, 1, 2, 3], vec![10, 11, 12, 13]],
            sample_rate: 256.0,
            channels: vec!["Fp1".into(), "Fp2".into()],
            phys_min: vec![-200.0, -200.0],
            phys_max: vec![200.0, 200.0],
            duration_s: 0.015625,
            metadata: SourceMetadata {
                source_file: "rec.edf".into(),
                format: "EDF+C".into(),
                patient_id: "X".into(),
                recording_info: "demo".into(),
                startdate: "2026-06-30".into(),
                phys_dim: "uV".into(),
            },
            sidecar: vec![SidecarBlob {
                key: "raw_header".into(),
                bytes: vec![1, 2, 3],
                aux: None,
            }],
        }
    }

    #[test]
    fn ir_text_is_deterministic() {
        let b = sample_bundle();
        assert_eq!(to_ir_text(&b), to_ir_text(&b));
    }

    /// Frozen golden for the textual IR form — locks the v1 layout so a refactor
    /// that changes the rendering (or a reader's output) is caught.
    #[test]
    fn ir_text_golden() {
        let expected = "\
; lamquant biosignal IR v1
recording {
  format = \"EDF+C\"
  source = \"rec.edf\"
  patient = \"X\"
  start = \"2026-06-30\"
  info = \"demo\"
  phys_dim = \"uV\"
  sample_rate = 256.0
  duration_s = 0.015625
}
channels [2] {
  0: \"Fp1\" phys[-200.0,200.0] n=4
  1: \"Fp2\" phys[-200.0,200.0] n=4
}
signal { channels=2 digest=sha256:CHANNELDIGEST }
sidecar [1] {
  \"raw_header\" aux=None 3 bytes digest=sha256:039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81
}
";
        let got = to_ir_text(&sample_bundle());
        // Replace the data-derived signal digest with a placeholder for the
        // structural comparison, then assert the digest line is present + stable.
        let digest = signal_digest(&sample_bundle().signal);
        let normalized = got.replace(&digest, "CHANNELDIGEST");
        assert_eq!(normalized, expected, "IR text layout drifted");
    }

    // ─── S5 Increment 1: parser tests ─────────────────────────────────

    /// Multi-channel corpus shape (4 channels, mixed sidecar aux/None,
    /// negative + fractional phys bounds) — a second, richer shape beyond
    /// `sample_bundle` for the round-trip proof.
    fn multi_channel_bundle() -> SignalBundle {
        SignalBundle {
            signal: vec![
                vec![0, 1, 2, 3, 4],
                vec![-5, -4, -3, -2, -1],
                vec![100; 5],
                vec![7, 7, 8, 9, 10],
            ],
            sample_rate: 512.5,
            channels: vec!["Fp1".into(), "Fp2".into(), "Cz".into(), "Oz".into()],
            phys_min: vec![-200.0, -350.25, 0.0, -1.0],
            phys_max: vec![200.0, 350.25, 1.0, 1.0],
            duration_s: 5.0 / 512.5,
            metadata: SourceMetadata {
                source_file: "multi/rec 01.edf".into(),
                format: "BDF".into(),
                patient_id: "anon-0007".into(),
                recording_info: "ICU bed 3; leads re-taped 02:14".into(),
                startdate: "2026-01-02T03:04:05".into(),
                phys_dim: "mV".into(),
            },
            sidecar: vec![
                SidecarBlob {
                    key: "raw_header".into(),
                    bytes: vec![0xAA; 512],
                    aux: None,
                },
                SidecarBlob {
                    key: "non_eeg_chunk".into(),
                    bytes: vec![1, 2, 3, 4],
                    aux: Some(-5),
                },
                SidecarBlob {
                    key: "non_eeg_chunk".into(),
                    bytes: vec![],
                    aux: Some(0),
                },
                SidecarBlob {
                    key: "edf_meta".into(),
                    bytes: b"{}".to_vec(),
                    aux: Some(i64::MAX),
                },
            ],
        }
    }

    /// Odd-name corpus shape: a channel name AND a sidecar key containing a
    /// quote, a `]`, and a `,` — the exact lexical hazard the channel-line
    /// grammar warns about (`phys[` must be found only after the quoted
    /// name is fully consumed by the unescaper, not by naive splitting).
    /// Also exercises backslash/newline/tab/CR/NUL escapes and a `\u{..}`
    /// control-char escape, all in the same string.
    fn odd_name_bundle() -> SignalBundle {
        let hazard = "Fp1 \"we\\ird\"[1,2],3]\ttab\nline\r\0\u{7}";
        SignalBundle {
            signal: vec![vec![0, -1, 2], vec![3, 4, 5]],
            sample_rate: 250.0,
            channels: vec![hazard.to_string(), "normal".into()],
            phys_min: vec![-1.5, -2.0],
            phys_max: vec![1.5, 2.0],
            duration_s: 0.012,
            metadata: SourceMetadata {
                source_file: hazard.to_string(),
                format: "RAW".into(),
                patient_id: "".into(),
                recording_info: hazard.to_string(),
                startdate: "".into(),
                phys_dim: "uV".into(),
            },
            sidecar: vec![SidecarBlob {
                key: hazard.to_string(),
                bytes: vec![9, 9],
                aux: None,
            }],
        }
    }

    fn corpus() -> Vec<SignalBundle> {
        vec![sample_bundle(), multi_channel_bundle(), odd_name_bundle()]
    }

    /// The projection round-trip contract: `parse_ir_text(&to_ir_text(b))
    /// == project(b)` for every corpus shape, including the escape/hazard
    /// corpus.
    #[test]
    fn round_trip() {
        for b in corpus() {
            let text = to_ir_text(&b);
            let parsed = parse_ir_text(&text)
                .unwrap_or_else(|e| panic!("parse_ir_text failed on:\n{text}\nerror: {e}"));
            assert_eq!(parsed, project(&b), "round-trip mismatch for:\n{text}");
        }
    }

    /// The parsed `signal.digest` is exactly what `signal_digest`
    /// recomputes from the bundle — the round-trip isn't just structurally
    /// equal, the hash itself made it through unchanged.
    #[test]
    fn digest_matches() {
        for b in corpus() {
            let text = to_ir_text(&b);
            let parsed = parse_ir_text(&text).expect("parse_ir_text");
            assert_eq!(parsed.signal.digest, signal_digest(&b.signal));
            for (sc, entry) in b.sidecar.iter().zip(parsed.sidecar.iter()) {
                assert_eq!(entry.digest, sha_hex(&sc.bytes));
            }
        }
    }

    /// One crafted malformed input per `IrParseError` variant — each must
    /// fail with EXACTLY that variant (not some other Err, not a panic).
    #[test]
    fn negative_battery() {
        let magic = "; lamquant biosignal IR v1";

        // BadMagic: first line present but wrong.
        assert!(matches!(
            parse_ir_text("not the magic\n"),
            Err(IrParseError::BadMagic)
        ));

        // Truncated: nothing at all.
        assert!(matches!(
            parse_ir_text(""),
            Err(IrParseError::Truncated { .. })
        ));
        // Also truncated mid-document (magic ok, then EOF).
        assert!(matches!(
            parse_ir_text(&format!("{magic}\n")),
            Err(IrParseError::Truncated { .. })
        ));

        // MissingBlock: magic ok, but "recording {" replaced.
        assert!(matches!(
            parse_ir_text(&format!("{magic}\nrecords {{\n")),
            Err(IrParseError::MissingBlock { .. })
        ));

        // BadRecordingField: wrong key name on the first field line.
        assert!(matches!(
            parse_ir_text(&format!("{magic}\nrecording {{\n  fmt = \"x\"\n")),
            Err(IrParseError::BadRecordingField { .. })
        ));

        // BadNumber: sample_rate isn't a float.
        {
            let doc = format!(
                "{magic}\nrecording {{\n  format = \"a\"\n  source = \"b\"\n  patient = \"c\"\n  start = \"d\"\n  info = \"e\"\n  phys_dim = \"f\"\n  sample_rate = not_a_number\n"
            );
            assert!(matches!(
                parse_ir_text(&doc),
                Err(IrParseError::BadNumber { .. })
            ));
        }

        // Helper: a fully valid recording block + closing brace, for the
        // remaining cases that need to reach the channels/signal/sidecar
        // sections.
        fn recording_block() -> String {
            "recording {\n  format = \"a\"\n  source = \"b\"\n  patient = \"c\"\n  start = \"d\"\n  info = \"e\"\n  phys_dim = \"f\"\n  sample_rate = 1.0\n  duration_s = 1.0\n}\n".to_string()
        }

        // BadChannelLine: missing the `:` after the index.
        assert!(matches!(
            parse_ir_text(&format!(
                "{magic}\n{}channels [1] {{\n  0 \"Fp1\" phys[-1.0,1.0] n=4\n}}\n",
                recording_block()
            )),
            Err(IrParseError::BadChannelLine { .. })
        ));

        // UnterminatedString: recording field's quoted value never closes.
        assert!(matches!(
            parse_ir_text(&format!(
                "{magic}\nrecording {{\n  format = \"unterminated\n"
            )),
            Err(IrParseError::UnterminatedString)
        ));

        // BadEscape: unknown escape char `\q`.
        assert!(matches!(
            parse_ir_text(&format!(
                "{magic}\nrecording {{\n  format = \"bad\\qescape\"\n"
            )),
            Err(IrParseError::BadEscape)
        ));

        // ChannelCountMismatch: signal{channels=..} disagrees with the
        // channels block (1 channel declared and parsed, signal says 2).
        assert!(matches!(
            parse_ir_text(&format!(
                "{magic}\n{}channels [1] {{\n  0: \"Fp1\" phys[-1.0,1.0] n=4\n}}\nsignal {{ channels=2 digest=sha256:{} }}\n",
                recording_block(),
                "0".repeat(64)
            )),
            Err(IrParseError::ChannelCountMismatch)
        ));

        // BadDigest: right length but non-hex characters.
        assert!(matches!(
            parse_ir_text(&format!(
                "{magic}\n{}channels [0] {{\n}}\nsignal {{ channels=0 digest=sha256:{} }}\n",
                recording_block(),
                "z".repeat(64)
            )),
            Err(IrParseError::BadDigest)
        ));

        // BadSidecarLine: missing the `aux=` literal.
        assert!(matches!(
            parse_ir_text(&format!(
                "{magic}\n{}channels [0] {{\n}}\nsignal {{ channels=0 digest=sha256:{} }}\nsidecar [1] {{\n  \"key\" wrong=None 0 bytes digest=sha256:{}\n}}\n",
                recording_block(),
                "0".repeat(64),
                "0".repeat(64)
            )),
            Err(IrParseError::BadSidecarLine { .. })
        ));

        // TrailingGarbage: a fully valid, empty-shape document plus one
        // extra trailing line.
        assert!(matches!(
            parse_ir_text(&format!(
                "{magic}\n{}channels [0] {{\n}}\nsignal {{ channels=0 digest=sha256:{} }}\nsidecar [0] {{\n}}\nEXTRA\n",
                recording_block(),
                "0".repeat(64)
            )),
            Err(IrParseError::TrailingGarbage)
        ));
    }

    /// Bounds-safety sweep (mirrors `oracle_diff.rs`'s `container::read_bytes`
    /// truncation sweep): truncating a fully valid rendering at every byte
    /// offset `0..len` must never panic. The corpus is ASCII-only (see
    /// `odd_name_bundle`'s hazard string — quote/bracket/comma/backslash/
    /// control chars are all single-byte), so every offset is a valid `str`
    /// char boundary and the harness itself can't panic on the slice.
    #[test]
    fn truncation_sweep_never_panics() {
        for b in corpus() {
            let text = to_ir_text(&b);
            assert!(
                text.is_ascii(),
                "truncation sweep corpus must be ASCII-safe to slice"
            );
            for k in 0..text.len() {
                let _ = parse_ir_text(&text[..k]);
            }
        }
    }
}
