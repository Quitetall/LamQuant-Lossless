//! Projecting LamQuant electrode montage onto ABIR's exact numeric model.
//!
//! The `.lmq` container stores electrode positions as `f32` metres. ABIR stores
//! geometry as [`ExactNumber`] — integers and rationals, no floating point
//! anywhere — because a coordinate that changes value depending on who parsed it
//! is not a coordinate you can seal into a content-addressed artifact.
//!
//! Bridging those needs two decisions, and neither is arithmetic.
//!
//! # Exact conversion, not rounding
//!
//! Every finite `f32` IS a rational: mantissa over a power of two. So the
//! conversion is lossless, and this module does it losslessly.
//!
//! Rounding to a "sensible" precision — micrometres, say — was the tempting
//! alternative and it is wrong twice over. It discards information the source
//! actually had, and it conflates two different things:
//! *representation* precision with *measurement* precision. ABIR already models
//! the second explicitly, as [`CoordinateFrame`](abir_coordinate_frame)'s
//! `uncertainty` field. An electrode position known to ±2 mm should say so
//! there, while still carrying the exact number it was given. Rounding would
//! silently claim a precision the measurement does not have, and lose the
//! original besides.
//!
//! [abir_coordinate_frame]: semantic_abir::CoordinateFrame
//!
//! # NaN is absence, not a number
//!
//! The `.lmq` montage block uses `NaN` to mean *this electrode's position is
//! unknown*. That is a sentinel, not a value.
//!
//! ABIR models absence structurally — `ChannelSpec::coordinate_frame_id` is an
//! `Option` — so the projection maps the sentinel onto the absence, rather than
//! trying to encode a non-number as a number. Any conversion that tried to
//! preserve `NaN` numerically would be inventing a position for an electrode
//! nobody located.

use semantic_abir::{ExactNumber, Rational};

/// Why a coordinate could not be projected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MontageError {
    /// The value is infinite. Unlike `NaN` this is not a documented sentinel, so
    /// it is a corrupt montage rather than an unknown position, and is refused
    /// rather than quietly turned into an absence.
    NotFinite,
    /// Exact representation needs a denominator larger than `i128` holds.
    ///
    /// Only reachable for magnitudes far below any physical electrode position
    /// (roughly 1e-30 m and smaller), where `f32`'s exponent drives the
    /// denominator past 2^127. Refused rather than rounded, because rounding
    /// here would silently substitute a different position.
    DenominatorTooLarge,
}

/// One electrode position, or its documented absence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Coordinate {
    /// A position, exactly as the source gave it.
    Known([ExactNumber; 3]),
    /// The source recorded `NaN`: this electrode was never located.
    Unknown,
}

/// Convert one finite `f32` to the rational it exactly equals.
///
/// `f32` is sign × mantissa × 2^exponent, so the exact value is either an
/// integer (non-negative exponent) or mantissa over a power of two.
pub fn exact_from_f32(value: f32) -> Result<ExactNumber, MontageError> {
    if !value.is_finite() {
        return Err(MontageError::NotFinite);
    }
    if value == 0.0 {
        // Covers -0.0 too, which is exactly zero and must not become a signed
        // rational with a negative zero numerator.
        return Ok(ExactNumber::Integer(0));
    }

    let bits = value.to_bits();
    let sign: i128 = if bits >> 31 == 1 { -1 } else { 1 };
    let biased_exponent = ((bits >> 23) & 0xFF) as i32;
    let raw_mantissa = (bits & 0x007F_FFFF) as i128;

    // Subnormals have no implicit leading one and a fixed exponent.
    let (mantissa, exponent) = if biased_exponent == 0 {
        (raw_mantissa, -149_i32)
    } else {
        (raw_mantissa | (1 << 23), biased_exponent - 150)
    };

    if exponent >= 0 {
        // Large enough that the value is a whole number.
        let shift = u32::try_from(exponent).map_err(|_| MontageError::DenominatorTooLarge)?;
        let scaled = mantissa
            .checked_shl(shift)
            .ok_or(MontageError::DenominatorTooLarge)?;
        // checked_shl does not report lost high bits, so verify the shift was
        // reversible rather than trusting it.
        if scaled >> shift != mantissa {
            return Err(MontageError::DenominatorTooLarge);
        }
        return Ok(ExactNumber::Integer(sign * scaled));
    }

    let shift = u32::try_from(-exponent).map_err(|_| MontageError::DenominatorTooLarge)?;
    // i128 holds 2^126 comfortably; beyond that the denominator does not fit.
    if shift > 126 {
        return Err(MontageError::DenominatorTooLarge);
    }
    let denominator = 1_i128 << shift;
    // A value like 3.0 arrives here with mantissa 12582912 and shift 22, and
    // `Rational::new` would reduce it to 3/1 -- numerically right, but a second
    // spelling of `Integer(3)`. Canonical forms have to be unique, so detect
    // wholeness before constructing rather than trying to inspect afterwards.
    if mantissa % denominator == 0 {
        return Ok(ExactNumber::Integer(sign * (mantissa / denominator)));
    }
    Rational::new(sign * mantissa, denominator)
        .map(ExactNumber::Rational)
        .map_err(|_| MontageError::DenominatorTooLarge)
}

/// Project one electrode's `[x, y, z]` metres onto ABIR geometry.
///
/// All three components must agree about whether the position is known: a
/// partially-`NaN` triple describes an electrode located in some axes but not
/// others, which is not a position and not a documented absence.
pub fn coordinate_from_f32(xyz: [f32; 3]) -> Result<Coordinate, MontageError> {
    let unknown = xyz.iter().filter(|value| value.is_nan()).count();
    match unknown {
        3 => return Ok(Coordinate::Unknown),
        0 => {}
        _ => return Err(MontageError::NotFinite),
    }
    Ok(Coordinate::Known([
        exact_from_f32(xyz[0])?,
        exact_from_f32(xyz[1])?,
        exact_from_f32(xyz[2])?,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recover the f32 a projected coordinate stands for, so exactness is
    /// checked by round trip rather than by trusting the arithmetic above.
    ///
    /// `Rational` keeps its parts private, so this reads them back through the
    /// canonical `Display` form. Going through the public surface is the point:
    /// it proves the value a consumer can actually observe is the original,
    /// not merely that some internal field holds the right bits.
    fn back_to_f32(value: &ExactNumber) -> f64 {
        match value {
            ExactNumber::Integer(n) => *n as f64,
            ExactNumber::Rational(r) => {
                let text = alloc::format!("{r}");
                let (numerator, denominator) =
                    text.split_once('/').expect("rationals display as n/d");
                numerator.parse::<f64>().expect("numerator")
                    / denominator.parse::<f64>().expect("denominator")
            }
        }
    }

    #[test]
    fn realistic_electrode_positions_convert_exactly() {
        // Scalp coordinates in metres: centimetres from the head centre.
        for value in [0.081_f32, -0.0725, 0.0, 0.1234567, -0.001] {
            let exact = exact_from_f32(value).expect("converts");
            assert_eq!(
                back_to_f32(&exact),
                f64::from(value),
                "{value} did not survive exactly; rounding crept in"
            );
        }
    }

    #[test]
    fn whole_numbers_stay_integers() {
        // A rational with denominator one would be a needless second spelling
        // of the same value, and canonical forms must be unique.
        assert_eq!(exact_from_f32(3.0).unwrap(), ExactNumber::Integer(3));
        assert_eq!(exact_from_f32(-64.0).unwrap(), ExactNumber::Integer(-64));
    }

    #[test]
    fn both_zeroes_are_the_same_exact_zero() {
        // -0.0 is a distinct f32 bit pattern but the same number. Letting it
        // through as a signed rational would give one position two encodings.
        assert_eq!(exact_from_f32(0.0).unwrap(), ExactNumber::Integer(0));
        assert_eq!(exact_from_f32(-0.0).unwrap(), ExactNumber::Integer(0));
    }

    #[test]
    fn an_all_nan_triple_is_absence_not_a_number() {
        // The `.lmq` sentinel for "this electrode was never located". Encoding
        // it numerically would invent a position nobody measured.
        assert_eq!(
            coordinate_from_f32([f32::NAN; 3]).unwrap(),
            Coordinate::Unknown
        );
    }

    #[test]
    fn a_partially_nan_triple_is_refused() {
        // Located in x and y but not z is neither a position nor the documented
        // absence. Silently treating it as either would fabricate meaning.
        assert_eq!(
            coordinate_from_f32([0.08, 0.02, f32::NAN]),
            Err(MontageError::NotFinite)
        );
    }

    #[test]
    fn infinities_are_refused_rather_than_treated_as_unknown() {
        // Infinity is not a documented sentinel, so it means the montage is
        // corrupt -- a different fact from "position unknown".
        assert_eq!(exact_from_f32(f32::INFINITY), Err(MontageError::NotFinite));
        assert_eq!(
            coordinate_from_f32([f32::NEG_INFINITY, 0.0, 0.0]),
            Err(MontageError::NotFinite)
        );
    }

    #[test]
    fn a_known_triple_round_trips_componentwise() {
        let xyz = [0.081_f32, -0.0725, 0.0341];
        let Coordinate::Known(exact) = coordinate_from_f32(xyz).unwrap() else {
            panic!("a finite triple is a known position");
        };
        for (component, original) in exact.iter().zip(xyz) {
            assert_eq!(back_to_f32(component), f64::from(original));
        }
    }

    #[test]
    fn magnitudes_below_physical_scale_are_refused_not_rounded() {
        // Far below any electrode position. Rounding would substitute a
        // different point; refusing says so.
        assert_eq!(
            exact_from_f32(1e-38),
            Err(MontageError::DenominatorTooLarge)
        );
    }
}
