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
}
