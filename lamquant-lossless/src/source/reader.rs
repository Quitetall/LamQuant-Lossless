//! `SignalSourceReader` — ABIR seam every physiology reader implements.
//!
//! Bible alignment:
//! - R1  Each impl does ONE format. Composition over inheritance.
//! - R6  validated semantic ABIR is the strongly-typed boundary contract.
//! - R23 Validate at both ends: reader checks source bytes and ABIR construction
//!   validates the returned dataset before exposure.

use super::semantic::SemanticRead;
use crate::error::LmlResult;

/// Read a physiology source into validated semantic ABIR.
///
/// Implementations:
///   - own the byte source (path, stream, …) at construction time
///   - consume that source exactly once when `lower_to_abir` is called
///   - produce errors via `LmlResult` (no panics on malformed input)
pub trait SignalSourceReader {
    /// Canonical public module seam. Reader-private native layouts may remain
    /// inside this call, but validated ABIR is the only trait-level result.
    fn lower_to_abir(&mut self) -> LmlResult<SemanticRead>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LmlError;

    struct SemanticOnlyReader;

    impl SignalSourceReader for SemanticOnlyReader {
        fn lower_to_abir(&mut self) -> LmlResult<SemanticRead> {
            Err(LmlError::InvalidHeader("semantic-only reader probe".into()))
        }
    }

    #[test]
    fn implementors_are_required_to_supply_only_the_abir_seam() {
        let mut reader = SemanticOnlyReader;
        assert!(reader.lower_to_abir().is_err());
    }
}
