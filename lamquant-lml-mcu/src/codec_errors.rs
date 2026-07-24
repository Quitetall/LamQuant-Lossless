//! Typed errors for entropy coders + EDF reader (ADR 0021).
//!
//! Three module-scoped error enums replace the prior pattern of
//! `debug_assert!(false, ...)` (release-stripped) + silent
//! saturation. Each variant carries diagnostic context so callers
//! can surface `path:offset:reason` instead of opaque panics.
//!
//! `no_std + alloc` compatible. The `std::error::Error` impls + the
//! `From<std::io::Error>` blanket are gated behind `feature = "std"`.
//!
//! Three enums (NOT a unified `CodecError`) per the design-review
//! decision in `decisions/0021-entropy-edf-hardening-design-
//! review.md`: each module owns its error space, callers import
//! only what they need.
//!
//! `#[non_exhaustive]` on every enum: future variants don't break
//! callers.

use alloc::string::String;
use core::fmt;

// ─── Golomb-Rice ────────────────────────────────────────────────────

/// Errors from the Golomb-Rice entropy coder.
///
/// Covers both encode + decode paths. Encoder errors fire when the
/// input would force a silent saturation (e.g. `i64::MIN`, `q >
/// MAX_Q`); decoder errors fire when the bitstream is malformed,
/// truncated, or claims sizes the payload cannot satisfy.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GolombError {
    /// Input contained `i64::MIN`, which the zigzag map can't
    /// safely negate without overflow. Encoder rejects rather
    /// than silently saturating to `MIN + 1`.
    I64Min { index: usize },
    /// A residual produced a quotient `q > MAX_Q` (~2^40 × k).
    /// Encoder rejects; downstream would silently saturate and
    /// the decoder would emit different samples than the input.
    OversizeQ { index: usize, q_estimate: u64 },
    /// Decoder header declared k outside the supported range
    /// (k > MAX_K = 31). Refuses rather than silently zero-
    /// filling the subband.
    OversizeK { k: u32 },
    /// `n_total` from the header exceeds `u16::MAX` or the
    /// payload byte count × 8. Refuses pre-allocation.
    HeaderOverflow {
        n_total: usize,
        payload_bytes: usize,
    },
    /// Decoder ran out of bits during the unary prefix scan.
    /// Distinct from a clean stream end after a complete value.
    TruncatedUnary { at_byte: usize, partial_q: u64 },
    /// Decoder ran out of bits during the remainder read.
    TruncatedRemainder {
        at_byte: usize,
        k: u32,
        bits_short: u32,
    },
    /// Decoder encountered `q > max_q` mid-stream. Carries the
    /// actual byte offset reached so callers do NOT double-count
    /// payload bytes into the next subband (the bug fixed at
    /// `golomb.rs:229-230`).
    QuotientExceedsCeiling {
        q: u64,
        max_q: u64,
        bytes_consumed: usize,
    },
}

impl fmt::Display for GolombError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I64Min { index } => write!(
                f,
                "golomb: encoder refused i64::MIN at index {}; cannot zigzag-encode without \
                 overflow",
                index
            ),
            Self::OversizeQ { index, q_estimate } => write!(
                f,
                "golomb: encoder refused q ≈ {} at index {} (exceeds MAX_Q); silent saturation \
                 would desync the bitstream",
                q_estimate, index
            ),
            Self::OversizeK { k } => {
                write!(f, "golomb: decoder header declares k = {} (MAX_K = 31)", k)
            }
            Self::HeaderOverflow {
                n_total,
                payload_bytes,
            } => write!(
                f,
                "golomb: header claims {} symbols but payload has only {} bytes ({} bits)",
                n_total,
                payload_bytes,
                payload_bytes.saturating_mul(8)
            ),
            Self::TruncatedUnary { at_byte, partial_q } => write!(
                f,
                "golomb: bitstream truncated mid-unary at byte {}; partial q = {}",
                at_byte, partial_q
            ),
            Self::TruncatedRemainder {
                at_byte,
                k,
                bits_short,
            } => write!(
                f,
                "golomb: bitstream truncated mid-remainder at byte {} (k = {}, missing {} bits)",
                at_byte, k, bits_short
            ),
            Self::QuotientExceedsCeiling {
                q,
                max_q,
                bytes_consumed,
            } => write!(
                f,
                "golomb: decoded q = {} exceeds max_q = {}; aborting after {} payload bytes",
                q, max_q, bytes_consumed
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GolombError {}

// ─── rANS ───────────────────────────────────────────────────────────

/// Errors from the rANS entropy coder.
///
/// Encoder errors fire when the symbol distribution is degenerate
/// (zero divisor) or the state arithmetic would underflow.
/// Decoder errors fire on truncated streams or attacker-supplied
/// `n_symbols` that would force an unbounded allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RansError {
    /// Symbol table has cumulative frequency `m == 0`. Both
    /// encoder and decoder use `m` as a divisor; degenerate
    /// table can't be coded.
    ZeroDivisor,
    /// Decoder header claims more symbols than a sane bound
    /// (`MAX_RANS_SYMBOLS`). Refuses to allocate the requested
    /// `Vec<...>` capacity. Was the OOM-abort risk at
    /// `rans.rs:104`.
    HeaderOverflow { claimed: usize, max_allowed: usize },
    /// Decoder state would underflow during the `state - s_val`
    /// subtraction; bitstream is corrupt.
    StateUnderflow { state: u32, s_val: u32 },
    /// Decoder ran out of bytes mid-state-renormalization.
    TruncatedStream { at_byte: usize, bytes_needed: usize },
    /// Encoder symbol-count overflowed u16 max (the rANS wire
    /// format encodes the symbol count as u16).
    SymbolCountOverflow { count: usize },
}

impl fmt::Display for RansError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDivisor => write!(
                f,
                "rans: symbol table has zero cumulative frequency (m = 0); cannot encode/decode"
            ),
            Self::HeaderOverflow {
                claimed,
                max_allowed,
            } => write!(
                f,
                "rans: header claims {} symbols (MAX_RANS_SYMBOLS = {})",
                claimed, max_allowed
            ),
            Self::StateUnderflow { state, s_val } => write!(
                f,
                "rans: decoder state underflow (state = {}, s_val = {}); bitstream corrupt",
                state, s_val
            ),
            Self::TruncatedStream {
                at_byte,
                bytes_needed,
            } => write!(
                f,
                "rans: bitstream truncated at byte {} (needed {} more bytes for renormalize)",
                at_byte, bytes_needed
            ),
            Self::SymbolCountOverflow { count } => write!(
                f,
                "rans: encoder symbol count {} exceeds u16 wire-format max ({})",
                count,
                u16::MAX
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RansError {}

// ─── EDF / BDF reader ───────────────────────────────────────────────

/// Errors from the EDF/BDF reader.
///
/// Covers header validation, magic-byte rejection, sample-rate
/// finiteness, and arithmetic-overflow guards on adversarial
/// header fields.
///
/// `PartialEq` only (not `Eq`) because `InvalidSampleRate(f64)`
/// carries a float, and f64 doesn't impl `Eq`. PartialEq is
/// sufficient for `assert_eq!` in tests; downstream code that
/// needs `Eq` should match on the discriminant.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EdfError {
    /// Magic byte / BIOSEMI signature mismatch. The current
    /// `edf.rs:74` BDF check is a 3-byte-vs-1-byte slice
    /// comparison that's always false; this variant fires
    /// after the fix so any 0xFF-prefix file without BIOSEMI
    /// proof gets rejected.
    BadMagic { first_eight: [u8; 8] },
    /// EDF/BDF version field non-numeric or out of range.
    UnsupportedVersion(String),
    /// `phys_min` / `phys_max` parsed as NaN or Inf. Non-finite
    /// scale factors propagate through LML metadata and break
    /// downstream consumers.
    NonFiniteScale { field: &'static str, value: String },
    /// `mode_ns * usable_records` (or any analogous product)
    /// overflowed `usize`. On 32-bit MCU targets a malicious
    /// header can wrap this and force an undersized allocation.
    HeaderOverflow { reason: &'static str },
    /// `n_signals` exceeds the sane channel-count ceiling.
    /// Hard-bounded to refuse 64 MiB+ raw_header clones from a
    /// hostile header.
    TooManySignals { claimed: usize, max_allowed: usize },
    /// A data record was shorter than the declared layout
    /// implies. Non-EEG channels currently silently skip; this
    /// variant fires after the `edf.rs:448` hardening so the
    /// truncation is loud.
    TruncatedRecord {
        record_index: usize,
        channel: String,
        bytes_short: usize,
    },
    /// `sample_rate` parsed as zero, negative, or non-finite.
    InvalidSampleRate(f64),
    /// Required header field absent or unparseable.
    InvalidHeader(String),
    /// I/O error reading the EDF/BDF file from disk.
    #[cfg(feature = "std")]
    Io(String),
}

impl fmt::Display for EdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic { first_eight } => write!(
                f,
                "edf: bad magic bytes (first 8: {:02X?}); not a valid EDF or BDF file",
                first_eight
            ),
            Self::UnsupportedVersion(v) => write!(f, "edf: unsupported version {:?}", v),
            Self::NonFiniteScale { field, value } => write!(
                f,
                "edf: {} parsed as non-finite value {:?}; NaN/Inf scale factors are rejected",
                field, value
            ),
            Self::HeaderOverflow { reason } => {
                write!(f, "edf: header field arithmetic overflowed ({})", reason)
            }
            Self::TooManySignals {
                claimed,
                max_allowed,
            } => write!(
                f,
                "edf: header claims {} signal channels (max {})",
                claimed, max_allowed
            ),
            Self::TruncatedRecord {
                record_index,
                channel,
                bytes_short,
            } => write!(
                f,
                "edf: record {} truncated in channel {:?} ({} bytes short)",
                record_index, channel, bytes_short
            ),
            Self::InvalidSampleRate(sr) => write!(
                f,
                "edf: invalid sample_rate {} (must be finite + positive)",
                sr
            ),
            Self::InvalidHeader(msg) => write!(f, "edf: invalid header: {}", msg),
            #[cfg(feature = "std")]
            Self::Io(msg) => write!(f, "edf: I/O error: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EdfError {}

#[cfg(feature = "std")]
impl From<std::io::Error> for EdfError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(alloc::format!("{}", e))
    }
}

// stdlib provides a blanket `impl<E: Error + Send + Sync + 'static>
// From<E> for Box<dyn Error + Send + Sync>` so the `?` operator
// auto-converts the typed errors above via the existing
// `std::error::Error` impl. No explicit From<...> for Box<...>
// needed -- adding one collides with the blanket impl.

#[cfg(test)]
mod tests {
    use super::*;

    /// Display impls produce strings with the diagnostic context
    /// from the enum -- catches future variants that forget to
    /// include path:offset:reason fields.
    #[test]
    fn golomb_display_includes_context() {
        let e = GolombError::OversizeQ {
            index: 42,
            q_estimate: 1_000_000_000_000,
        };
        let s = alloc::format!("{}", e);
        assert!(s.contains("42"), "Display missing index: {}", s);
        assert!(
            s.contains("1000000000000") || s.contains("1_000_000_000_000"),
            "Display missing q_estimate: {}",
            s
        );
    }

    #[test]
    fn rans_display_includes_context() {
        let e = RansError::HeaderOverflow {
            claimed: 1_000_000,
            max_allowed: 65536,
        };
        let s = alloc::format!("{}", e);
        assert!(s.contains("1000000"), "Display missing claimed: {}", s);
        assert!(s.contains("65536"), "Display missing max_allowed: {}", s);
    }

    #[test]
    fn edf_display_includes_context() {
        let e = EdfError::TooManySignals {
            claimed: 65537,
            max_allowed: 256,
        };
        let s = alloc::format!("{}", e);
        assert!(s.contains("65537"));
        assert!(s.contains("256"));
    }

    /// Box conversion compiles + preserves the Display message.
    /// Catches future blanket-impl regressions.
    #[cfg(feature = "std")]
    #[test]
    fn box_conversion_preserves_display() {
        let e = GolombError::OversizeK { k: 99 };
        let boxed: alloc::boxed::Box<dyn std::error::Error + Send + Sync> = e.into();
        let s = alloc::format!("{}", boxed);
        assert!(s.contains("99"));
    }
}
