//! `SignalSourceReader` — the trait every physiology reader implements.
//!
//! ABIR is the public source boundary. Readers may use native parser layouts or
//! the temporary `SignalBundle` carrier internally, but every implementation
//! must return one validated semantic dataset plus payload resolver.
//!
//! Bible alignment:
//! - R1  Each impl does ONE format. Composition over inheritance.
//! - R6  `AbirDataset` is the strongly-typed boundary contract.
//! - R23 Validate at both ends: reader checks its input bytes, caller
//!   receives only validated ABIR.

use super::bundle::SignalBundle;
use super::semantic::SemanticRead;
use crate::error::{LmlError, LmlResult};

/// Lower a physiology recording into canonical semantic ABIR.
///
/// Implementations:
///   - own the byte source (path, stream, …) at construction time
///   - consume that source exactly once when `lower_to_abir` is called
///   - produce errors via `LmlResult` (no panics on malformed input)
///
/// Phase 0.5 will add a generic-over-`R: Read` variant; for now the
/// per-source ownership style keeps the surface tight.
pub trait SignalSourceReader {
    /// Read this source into a validated `AbirDataset` plus owned payload
    /// resolver. Every source implementation must define this semantic seam.
    fn lower_to_abir(&mut self) -> LmlResult<SemanticRead>;

    /// Transitional carrier used only by callers not yet migrated to ABIR.
    ///
    /// Package 26 removes this method with `SignalBundle`. New readers must not
    /// implement it and new callers must use [`Self::lower_to_abir`].
    #[doc(hidden)]
    fn read_bundle(&mut self) -> LmlResult<SignalBundle> {
        Err(LmlError::InvalidHeader(
            "legacy SignalBundle projection is unavailable; use lower_to_abir".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{
        bundle::{SignalBundle, SourceMetadata},
        from_signal_bundle,
    };

    struct SemanticOnlyReader;

    impl SignalSourceReader for SemanticOnlyReader {
        fn lower_to_abir(&mut self) -> LmlResult<SemanticRead> {
            from_signal_bundle(
                SignalBundle {
                    signal: vec![vec![1, 2, 3]],
                    sample_rate: 250.0,
                    channels: vec!["Cz".into()],
                    phys_min: vec![-1.0],
                    phys_max: vec![1.0],
                    duration_s: 3.0 / 250.0,
                    metadata: SourceMetadata {
                        source_file: "fixture.raw".into(),
                        format: "RAW".into(),
                        patient_id: String::new(),
                        recording_info: String::new(),
                        startdate: String::new(),
                        phys_dim: "digital".into(),
                    },
                    sidecar: Vec::new(),
                },
                semantic_abir::ValidationLimits::default(),
            )
        }
    }

    #[test]
    fn semantic_reader_trait_requires_only_abir_lowering() {
        let read = SemanticOnlyReader.lower_to_abir().unwrap();
        assert_eq!(read.mapping.channel_count, 1);
    }
}
