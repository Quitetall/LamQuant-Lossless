//! Unified error type for the LSL integration. Bridges
//! `lamquant_core::error::LmlError` (codec layer) and `lsl::Error`
//! (transport layer) so callers see one consistent enum at the API
//! boundary.

use lamquant_core::error::LmlError;

#[derive(Debug)]
pub enum LslIntegrationError {
    /// Failed to read or decode the source `.lml` file.
    LmlDecode(LmlError),
    /// liblsl returned an error (typically network / outlet creation).
    /// Only constructible when the `liblsl` feature is enabled.
    #[cfg(feature = "liblsl")]
    Lsl(lsl::Error),
    /// I/O error reading the source file or writing the target.
    Io(std::io::Error),
    /// Source EDF metadata is missing a required field (channel
    /// labels, sample rate, etc.) needed to build a well-formed
    /// `StreamInfo`.
    MissingMetadata(String),
    /// LSL functionality was requested but the `liblsl` feature is
    /// not compiled in. Contains an install hint pointing at the
    /// system liblsl install steps.
    FeatureDisabled,
    /// Generic / wrap-up of an arbitrary error message.
    Other(String),
}

impl std::fmt::Display for LslIntegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::LmlDecode(e) => write!(f, "lml decode: {}", e),
            #[cfg(feature = "liblsl")]
            Self::Lsl(e) => write!(f, "lsl: {:?}", e),
            Self::Io(e) => write!(f, "io: {}", e),
            Self::MissingMetadata(s) => write!(f, "missing metadata: {}", s),
            Self::FeatureDisabled => write!(
                f,
                "lamquant-lsl: built without the `liblsl` feature. \
                 Rebuild with `--features liblsl` after installing \
                 the system liblsl library (apt install liblsl-dev, \
                 brew install lsl, or build from \
                 https://github.com/sccn/liblsl)."
            ),
            Self::Other(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for LslIntegrationError {}

impl From<LmlError> for LslIntegrationError {
    fn from(e: LmlError) -> Self {
        Self::LmlDecode(e)
    }
}

#[cfg(feature = "liblsl")]
impl From<lsl::Error> for LslIntegrationError {
    fn from(e: lsl::Error) -> Self {
        Self::Lsl(e)
    }
}

impl From<std::io::Error> for LslIntegrationError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
