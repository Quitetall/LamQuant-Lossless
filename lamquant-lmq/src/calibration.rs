//! Exact conversion between ABIR digital samples and neural model units.
//!
//! Source payloads remain digital integers. Neural backends may instead declare
//! signed Q47.16 microvolts. Conversion stays deterministic, allocation-free,
//! and checked so host and firmware shells share identical semantics.

use semantic_abir::{Calibration, ConceptId, Rational};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CalibrationDomainError {
    Range,
    UnsupportedUnit,
}

#[derive(Clone, Copy)]
struct CheckedRational {
    numerator: i128,
    denominator: i128,
}

impl CheckedRational {
    fn new(numerator: i128, denominator: i128) -> Result<Self, CalibrationDomainError> {
        if denominator == 0 || denominator == i128::MIN {
            return Err(calibration_range_error());
        }
        if numerator == 0 {
            return Ok(Self {
                numerator: 0,
                denominator: 1,
            });
        }
        let mut numerator = numerator;
        let mut denominator = denominator;
        if denominator < 0 {
            numerator = numerator
                .checked_neg()
                .ok_or_else(calibration_range_error)?;
            denominator = denominator
                .checked_neg()
                .ok_or_else(calibration_range_error)?;
        }
        let divisor = gcd_u128(numerator.unsigned_abs(), denominator as u128);
        let divisor = i128::try_from(divisor).map_err(|_| calibration_range_error())?;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    fn from_abir(value: Rational) -> Result<Self, CalibrationDomainError> {
        let (numerator, denominator) = value.parts();
        Self::new(numerator, denominator)
    }

    const fn integer(value: i128) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    fn add(self, other: Self) -> Result<Self, CalibrationDomainError> {
        let divisor = gcd_u128(self.denominator as u128, other.denominator as u128);
        let divisor = i128::try_from(divisor).map_err(|_| calibration_range_error())?;
        let left_factor = other.denominator / divisor;
        let right_factor = self.denominator / divisor;
        let numerator = self
            .numerator
            .checked_mul(left_factor)
            .and_then(|left| {
                other
                    .numerator
                    .checked_mul(right_factor)
                    .and_then(|right| left.checked_add(right))
            })
            .ok_or_else(calibration_range_error)?;
        let denominator = self
            .denominator
            .checked_mul(left_factor)
            .ok_or_else(calibration_range_error)?;
        Self::new(numerator, denominator)
    }

    fn subtract(self, other: Self) -> Result<Self, CalibrationDomainError> {
        let numerator = other
            .numerator
            .checked_neg()
            .ok_or_else(calibration_range_error)?;
        self.add(Self::new(numerator, other.denominator)?)
    }

    fn multiply(self, other: Self) -> Result<Self, CalibrationDomainError> {
        let left_divisor = gcd_u128(self.numerator.unsigned_abs(), other.denominator as u128);
        let right_divisor = gcd_u128(other.numerator.unsigned_abs(), self.denominator as u128);
        let left_divisor = i128::try_from(left_divisor).map_err(|_| calibration_range_error())?;
        let right_divisor = i128::try_from(right_divisor).map_err(|_| calibration_range_error())?;
        let numerator = (self.numerator / left_divisor)
            .checked_mul(other.numerator / right_divisor)
            .ok_or_else(calibration_range_error)?;
        let denominator = (self.denominator / right_divisor)
            .checked_mul(other.denominator / left_divisor)
            .ok_or_else(calibration_range_error)?;
        Self::new(numerator, denominator)
    }

    fn divide(self, other: Self) -> Result<Self, CalibrationDomainError> {
        if other.numerator == 0 {
            return Err(calibration_range_error());
        }
        self.multiply(Self::new(other.denominator, other.numerator)?)
    }

    fn round_ties_even_i64(self) -> Result<i64, CalibrationDomainError> {
        debug_assert!(self.denominator > 0);
        let quotient = self.numerator / self.denominator;
        let remainder = self.numerator % self.denominator;
        let magnitude = remainder.unsigned_abs();
        let complement = (self.denominator as u128)
            .checked_sub(magnitude)
            .ok_or_else(calibration_range_error)?;
        let round_away =
            magnitude > complement || (magnitude == complement && quotient.rem_euclid(2) != 0);
        let rounded = if round_away {
            quotient
                .checked_add(if self.numerator < 0 { -1 } else { 1 })
                .ok_or_else(calibration_range_error)?
        } else {
            quotient
        };
        i64::try_from(rounded).map_err(|_| calibration_range_error())
    }
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

const fn calibration_range_error() -> CalibrationDomainError {
    CalibrationDomainError::Range
}

fn unit_to_microvolt(unit: &ConceptId) -> Result<CheckedRational, CalibrationDomainError> {
    match unit.as_str() {
        "ucum:V" | "abir:unit/volt" => Ok(CheckedRational::integer(1_000_000)),
        "ucum:mV" | "abir:unit/millivolt" => Ok(CheckedRational::integer(1_000)),
        "ucum:uV" | "abir:unit/microvolt" => Ok(CheckedRational::integer(1)),
        "ucum:nV" | "abir:unit/nanovolt" => CheckedRational::new(1, 1_000),
        _ => Err(CalibrationDomainError::UnsupportedUnit),
    }
}

/// Per-channel affine conversion compiled once from exact ABIR calibration.
///
/// Forward: `q16 = digital * forward_scale + forward_offset`.
/// Inverse: `digital = q16 * inverse_scale + inverse_offset`.
#[derive(Clone, Copy)]
pub(crate) struct AffineDomainTransform {
    forward_scale: CheckedRational,
    forward_offset: CheckedRational,
    inverse_scale: CheckedRational,
    inverse_offset: CheckedRational,
}

impl AffineDomainTransform {
    pub(crate) fn compile(calibration: &Calibration) -> Result<Self, CalibrationDomainError> {
        let scale = CheckedRational::from_abir(calibration.scale())?;
        let offset = CheckedRational::from_abir(calibration.offset())?;
        let model_units =
            unit_to_microvolt(calibration.unit())?.multiply(CheckedRational::integer(65_536))?;
        let forward_scale = scale.multiply(model_units)?;
        let forward_offset = offset.multiply(model_units)?;
        let inverse_scale = CheckedRational::integer(1)
            .divide(model_units)?
            .divide(scale)?;
        let inverse_offset = CheckedRational::integer(0)
            .subtract(offset)?
            .divide(scale)?;
        Ok(Self {
            forward_scale,
            forward_offset,
            inverse_scale,
            inverse_offset,
        })
    }

    pub(crate) fn digital_to_model(self, sample: i64) -> Result<i64, CalibrationDomainError> {
        CheckedRational::integer(i128::from(sample))
            .multiply(self.forward_scale)?
            .add(self.forward_offset)?
            .round_ties_even_i64()
    }

    pub(crate) fn model_to_digital(self, sample: i64) -> Result<i64, CalibrationDomainError> {
        CheckedRational::integer(i128::from(sample))
            .multiply(self.inverse_scale)?
            .add(self.inverse_offset)?
            .round_ties_even_i64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn concept(value: &str) -> ConceptId {
        ConceptId::new(value).unwrap()
    }

    fn calibration(scale: (i128, i128), offset: (i128, i128), unit: &str) -> Calibration {
        Calibration::new(
            Rational::new(scale.0, scale.1).unwrap(),
            Rational::new(offset.0, offset.1).unwrap(),
            concept(unit),
        )
        .unwrap()
    }

    #[test]
    fn exact_conversion_rounds_positive_and_negative_ties_to_even() {
        let half_quantum = calibration((1, 131_072), (0, 1), "ucum:uV");
        let transform = AffineDomainTransform::compile(&half_quantum).unwrap();
        assert_eq!(transform.digital_to_model(1).unwrap(), 0);
        assert_eq!(transform.digital_to_model(3).unwrap(), 2);
        assert_eq!(transform.digital_to_model(-1).unwrap(), 0);
        assert_eq!(transform.digital_to_model(-3).unwrap(), -2);
        assert_eq!(transform.model_to_digital(2).unwrap(), 4);
    }

    #[test]
    fn conversion_normalizes_supported_voltage_units() {
        let volts = calibration((1, 1_000_000), (0, 1), "ucum:V");
        let millivolts = calibration((1, 1_000), (0, 1), "ucum:mV");
        let microvolts = calibration((1, 1), (0, 1), "abir:unit/microvolt");
        assert_eq!(
            AffineDomainTransform::compile(&volts)
                .unwrap()
                .digital_to_model(1)
                .unwrap(),
            65_536
        );
        assert_eq!(
            AffineDomainTransform::compile(&millivolts)
                .unwrap()
                .digital_to_model(1)
                .unwrap(),
            65_536
        );
        assert_eq!(
            AffineDomainTransform::compile(&microvolts)
                .unwrap()
                .digital_to_model(1)
                .unwrap(),
            65_536
        );
    }

    #[test]
    fn conversion_rejects_unknown_units_and_overflow() {
        let unknown = calibration((1, 1), (0, 1), "example:unit/count");
        assert!(AffineDomainTransform::compile(&unknown).is_err());
        let oversized = calibration((i128::MAX, 1), (0, 1), "ucum:V");
        assert!(AffineDomainTransform::compile(&oversized).is_err());
    }
}
