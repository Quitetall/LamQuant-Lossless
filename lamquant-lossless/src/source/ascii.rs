//! ASCII fixed-field parse helpers for header-based signal formats.
//!
//! EDF/BDF/BDF+ and BrainVision all store integers and floats as
//! space-padded ASCII inside fixed-width header slices. These helpers
//! are the validation primitives every reader uses; they:
//!   - reject non-UTF-8 bytes loudly (Bible R7 graceful fail)
//!   - trim whitespace (formats pad with spaces)
//!   - propagate `core::num` parse errors with the offending lexeme
//!
//! Returning `LmlError::InvalidHeader` keeps the error variant count
//! down; per-format context wraps with `format!(...)` at the call site
//! where useful.

use crate::error::{LmlError, LmlResult};

/// Parse a space-padded ASCII unsigned integer field.
///
/// Empty after trim → `Err` ("expected integer, got empty field"); a
/// caller that wants to treat empty as zero should check explicitly.
#[inline]
pub fn parse_usize(bytes: &[u8]) -> LmlResult<usize> {
    let s = std::str::from_utf8(bytes)
        .map_err(|e| LmlError::InvalidHeader(format!("ASCII header non-UTF-8: {e}")))?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(LmlError::InvalidHeader(
            "ASCII header: expected unsigned integer, got empty field".into(),
        ));
    }
    trimmed
        .parse()
        .map_err(|e| LmlError::InvalidHeader(format!("bad unsigned integer {trimmed:?}: {e}")))
}

/// Parse a space-padded ASCII signed 64-bit integer field.
#[inline]
pub fn parse_i64(bytes: &[u8]) -> LmlResult<i64> {
    let s = std::str::from_utf8(bytes)
        .map_err(|e| LmlError::InvalidHeader(format!("ASCII header non-UTF-8: {e}")))?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(LmlError::InvalidHeader(
            "ASCII header: expected signed integer, got empty field".into(),
        ));
    }
    trimmed
        .parse()
        .map_err(|e| LmlError::InvalidHeader(format!("bad signed integer {trimmed:?}: {e}")))
}

/// Parse a space-padded ASCII f64 field. Accepts plain, scientific, and
/// signed forms — anything Rust's `str::parse::<f64>` accepts.
///
/// NaN / Inf are **rejected** explicitly (Bible R30 hostile-caller —
/// no clinical format should ever encode them and admitting them
/// silently leads to downstream pow/log divergences in the codec).
#[inline]
pub fn parse_float(bytes: &[u8]) -> LmlResult<f64> {
    let s = std::str::from_utf8(bytes)
        .map_err(|e| LmlError::InvalidHeader(format!("ASCII header non-UTF-8: {e}")))?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(LmlError::InvalidHeader(
            "ASCII header: expected float, got empty field".into(),
        ));
    }
    let v: f64 = trimmed
        .parse()
        .map_err(|e| LmlError::InvalidHeader(format!("bad float {trimmed:?}: {e}")))?;
    if !v.is_finite() {
        return Err(LmlError::InvalidHeader(format!(
            "non-finite float in ASCII header: {v}"
        )));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_usize_simple() {
        assert_eq!(parse_usize(b"42").unwrap(), 42);
    }

    #[test]
    fn parse_usize_trims_padding() {
        assert_eq!(parse_usize(b"   7  ").unwrap(), 7);
    }

    #[test]
    fn parse_usize_rejects_empty() {
        assert!(parse_usize(b"        ").is_err());
    }

    #[test]
    fn parse_usize_rejects_negative() {
        assert!(parse_usize(b"-3").is_err());
    }

    #[test]
    fn parse_usize_rejects_non_utf8() {
        assert!(parse_usize(&[0xFF, 0xFE]).is_err());
    }

    #[test]
    fn parse_i64_handles_sign() {
        assert_eq!(parse_i64(b"-42").unwrap(), -42);
        assert_eq!(parse_i64(b"+42").unwrap(), 42);
    }

    #[test]
    fn parse_i64_rejects_empty() {
        assert!(parse_i64(b"   ").is_err());
    }

    #[test]
    fn parse_float_plain() {
        assert!((parse_float(b"3.14").unwrap() - 314.0 / 100.0).abs() < 1e-9);
    }

    #[test]
    fn parse_float_scientific() {
        assert!((parse_float(b"1.5e2").unwrap() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn parse_float_rejects_nan_string() {
        let r = parse_float(b"nan");
        assert!(r.is_err(), "ASCII headers must not admit NaN — got {r:?}");
    }

    #[test]
    fn parse_float_rejects_inf_string() {
        assert!(parse_float(b"inf").is_err());
        assert!(parse_float(b"-inf").is_err());
    }

    #[test]
    fn parse_float_rejects_empty() {
        assert!(parse_float(b"").is_err());
    }
}
