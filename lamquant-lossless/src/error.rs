//! LML error types — structured errors for all codec operations.
//!
//! `no_std` compatible. The `Io` variant + `std::error::Error` impl are
//! gated behind the `std` feature.

use alloc::string::String;
use core::fmt;

/// Error type for all LML codec operations.
#[derive(Debug)]
pub enum LmlError {
    /// I/O error (file read/write). Only available with `std` feature.
    #[cfg(feature = "std")]
    Io(std::io::Error),
    /// Magic bytes don't match LML format.
    InvalidMagic([u8; 4]),
    /// File version is newer than this reader supports.
    UnsupportedVersion(u8),
    /// Data is truncated (expected vs actual bytes).
    Truncated {
        expected: usize,
        actual: usize,
        context: &'static str,
    },
    /// CRC-32 mismatch (corruption detected).
    CrcMismatch { expected: u32, actual: u32 },
    /// Header field has invalid value.
    InvalidHeader(String),
    /// Decompression failed.
    DecompressFailure(String),
    /// Golomb-Rice entropy coder rejected the input/output.
    /// Surfaces typed errors from `codec_errors::GolombError` so
    /// callers can match on the specific failure (i64::MIN, OversizeQ,
    /// truncated bitstream, etc.) instead of stringly-typed lookups.
    Golomb(crate::codec_errors::GolombError),
}

impl fmt::Display for LmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::InvalidMagic(m) => {
                write!(f, "Not an LML file (magic: {:?}). Check file integrity.", m)
            }
            Self::UnsupportedVersion(v) => write!(
                f,
                "LML version {} is newer than this reader (version 1). Update LamQuant.",
                v
            ),
            Self::Truncated {
                expected,
                actual,
                context,
            } => write!(
                f,
                "Truncated {}: expected {} bytes, got {}. File is incomplete.",
                context, expected, actual
            ),
            Self::CrcMismatch { expected, actual } => write!(
                f,
                "CRC-32 mismatch: expected 0x{:08X}, got 0x{:08X}. Data is corrupted.",
                expected, actual
            ),
            Self::InvalidHeader(msg) => write!(f, "Invalid header: {}", msg),
            Self::DecompressFailure(msg) => write!(f, "Decompression failed: {}", msg),
            Self::Golomb(e) => write!(f, "Golomb: {}", e),
        }
    }
}

impl From<crate::codec_errors::GolombError> for LmlError {
    fn from(e: crate::codec_errors::GolombError) -> Self {
        Self::Golomb(e)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LmlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for LmlError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Convenience alias.
pub type LmlResult<T> = Result<T, LmlError>;
