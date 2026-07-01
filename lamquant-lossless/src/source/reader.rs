//! `SignalSourceReader` — the trait every physiology reader implements.
//!
//! Today the only consumer is `bin/lml.rs` indirectly (via the legacy
//! `crate::edf::read_edf`); Phase 0.4+ will migrate the codec
//! pipeline to consume `SignalBundle` directly. Plug-in readers
//! (BrainVision, CNT, DICOM, custom raw) arrive in Phase 4 with no
//! changes required here.
//!
//! Bible alignment:
//! - R1  Each impl does ONE format. Composition over inheritance.
//! - R6  `SignalBundle` is the strongly-typed boundary contract.
//! - R23 Validate at both ends: reader checks its input bytes, caller
//!   checks the returned `SignalBundle` invariants.

use crate::error::LmlResult;
use lamquant_abir::Abir;

use super::bundle::SignalBundle;

/// Read a physiology recording into the codec-agnostic `SignalBundle`.
///
/// Implementations:
///   - own the byte source (path, stream, …) at construction time
///   - consume that source exactly once when `read_bundle` is called
///   - produce errors via `LmlResult` (no panics on malformed input)
///
/// Phase 0.5 will add a generic-over-`R: Read` variant; for now the
/// per-source ownership style keeps the surface tight.
pub trait SignalSourceReader {
    fn read_bundle(&mut self) -> LmlResult<SignalBundle>;

    /// Lower this reader's recording directly into the ABIR atom
    /// currency (ADR 0069 L7).
    ///
    /// Default: go through `read_bundle` and wrap the resulting
    /// `Vec<Vec<i64>>` as an all-`I64` `Abir` (`Abir::from_channels_i64`)
    /// — always correct, no memory win. Readers whose native sample
    /// width is a signed integer that was widened to `i64` via a plain
    /// `as i64` (no calibration/transform folded into the sample lane)
    /// override this to decode straight into a narrower `Column`
    /// (`I16`/`I24`/`I32`) — byte-exact because narrowing back with
    /// `as i16`/`as i32` exactly inverts the widen. See `EdfReader`,
    /// `BrainVisionReader`, `CntReader`, `RawReader`, and
    /// `DicomWaveformReader` for the specialized paths; `EeglabReader`
    /// stays on this default (its lossless path float-bitcasts, not
    /// numeric-widens).
    ///
    /// Every path — this default and every specialized override — calls
    /// `Abir::with_inferred_modality` before returning (ADR 0069 S3b:
    /// born-typed lowering). `SignalBundle::metadata.format` is a
    /// container-format identifier (`"EDF"`, `"EEGLAB_LOSSY_I16"`, …),
    /// NOT a modality declaration, so the default path passes `None` for
    /// the format hint and relies purely on channel-label inference; see
    /// `crate::source::dicom` for the one reader that DOES have a real
    /// declared-modality field to pass through.
    fn lower_to_abir(&mut self) -> LmlResult<Abir> {
        let b = self.read_bundle()?;
        let labels: Vec<&str> = b.channels.iter().map(String::as_str).collect();
        Ok(Abir::from_channels_i64(b.signal, b.sample_rate).with_inferred_modality(&labels, None))
    }
}
