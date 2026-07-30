//! `SignalSourceReader` — the trait every physiology reader implements.
//!
//! ABIR is the public source boundary. Readers may use private native parser
//! layouts internally, but every implementation returns one validated semantic
//! dataset plus payload resolver.
//!
//! Bible alignment:
//! - R1  Each impl does ONE format. Composition over inheritance.
//! - R6  `AbirDataset` is the strongly-typed boundary contract.
//! - R23 Validate at both ends: reader checks its input bytes, caller
//!   receives only validated ABIR.

use super::semantic::SemanticRead;
use crate::error::LmlResult;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{
        bundle::{ParsedUniformSignal, SourceMetadata},
        semantic::{lower_parsed_uniform_signal, SemanticLoweringOptions},
    };

    struct SemanticOnlyReader;

    impl SignalSourceReader for SemanticOnlyReader {
        fn lower_to_abir(&mut self) -> LmlResult<SemanticRead> {
            lower_parsed_uniform_signal(
                ParsedUniformSignal {
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
                SemanticLoweringOptions::default(),
            )
        }
    }

    #[test]
    fn semantic_reader_trait_requires_only_abir_lowering() {
        let read = SemanticOnlyReader.lower_to_abir().unwrap();
        assert_eq!(read.mapping.channel_count, 1);
    }
}
