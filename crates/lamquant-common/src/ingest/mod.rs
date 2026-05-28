//! Ingest pipeline — convert non-EDF EEG file representations into a
//! form the LML codec can compress.
//!
//! ADR 0023 Track A: LMA should read every EEG file shape. The
//! container's `pack_archive` calls into this module BEFORE falling
//! through to the zstd compressor. The module:
//!
//!   1. Sniffs candidate non-EDF formats (`detect_*` functions).
//!   2. Parses the file's signal data into `i16` samples.
//!   3. Captures a "format template" so the original bytes can be
//!      re-emitted bit-exactly on extract.
//!   4. Synthesizes a minimal valid EDF that the existing codec can
//!      consume.
//!
//! Decode side: the manifest entry carries the format template; the
//! decoder runs the LML codec to recover samples then re-emits the
//! original file via the template.
//!
//! Roundtrip is verified by SHA-256 — any byte that doesn't match
//! the original is a hard error, not a silent loss.

pub mod ascii_lines;
pub mod edf_synth;

pub use ascii_lines::{
    detect_ascii_int_lines, parse_ascii_int_lines, render_ascii_int_lines, AsciiLinesTemplate,
};
pub use edf_synth::{synth_single_channel_edf, SYNTH_EDF_HEADER_LEN};

/// Tag for the originating format. Carried in the manifest's
/// `synthetic_from.format` field. Decoders match on this to pick
/// the right re-emitter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyntheticFormat {
    /// One integer per line (Bonn EEG dataset style). Template carries
    /// the line ending, leading-zero policy, field width, and trailing-
    /// newline policy.
    AsciiIntLines(AsciiLinesTemplate),
}
