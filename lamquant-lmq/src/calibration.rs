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

    #[cfg(test)]
    fn round_ties_even_i64(self) -> Result<i64, CalibrationDomainError> {
        round_ratio_ties_even_i64(self.numerator, self.denominator)
    }
}

fn round_ratio_ties_even_i64(
    numerator: i128,
    denominator: i128,
) -> Result<i64, CalibrationDomainError> {
    debug_assert!(denominator > 0);
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let magnitude = remainder.unsigned_abs();
    let complement = (denominator as u128)
        .checked_sub(magnitude)
        .ok_or_else(calibration_range_error)?;
    let round_away =
        magnitude > complement || (magnitude == complement && quotient.rem_euclid(2) != 0);
    let rounded = if round_away {
        quotient
            .checked_add(if numerator < 0 { -1 } else { 1 })
            .ok_or_else(calibration_range_error)?
    } else {
        quotient
    };
    i64::try_from(rounded).map_err(|_| calibration_range_error())
}

/// Pre-reduced `sample * scale + offset` with one shared denominator.
///
/// Compilation performs every GCD and rational reduction. Applying the kernel
/// to one sample needs only one checked multiply, one checked add, division,
/// and ties-to-even rounding.
#[derive(Clone, Copy)]
struct CheckedAffine {
    multiplier: i128,
    addend: i128,
    denominator: i128,
}

impl CheckedAffine {
    fn compile(
        scale: CheckedRational,
        offset: CheckedRational,
    ) -> Result<Self, CalibrationDomainError> {
        let divisor = gcd_u128(scale.denominator as u128, offset.denominator as u128);
        let divisor = i128::try_from(divisor).map_err(|_| calibration_range_error())?;
        let scale_factor = offset.denominator / divisor;
        let offset_factor = scale.denominator / divisor;
        let multiplier = scale
            .numerator
            .checked_mul(scale_factor)
            .ok_or_else(calibration_range_error)?;
        let addend = offset
            .numerator
            .checked_mul(offset_factor)
            .ok_or_else(calibration_range_error)?;
        let denominator = scale
            .denominator
            .checked_mul(scale_factor)
            .ok_or_else(calibration_range_error)?;
        let common_divisor = gcd_u128(
            gcd_u128(multiplier.unsigned_abs(), addend.unsigned_abs()),
            denominator as u128,
        );
        let common_divisor =
            i128::try_from(common_divisor).map_err(|_| calibration_range_error())?;
        Ok(Self {
            multiplier: multiplier / common_divisor,
            addend: addend / common_divisor,
            denominator: denominator / common_divisor,
        })
    }

    fn apply(self, sample: i64) -> Result<i64, CalibrationDomainError> {
        let numerator = i128::from(sample)
            .checked_mul(self.multiplier)
            .and_then(|scaled| scaled.checked_add(self.addend))
            .ok_or_else(calibration_range_error)?;
        round_ratio_ties_even_i64(numerator, self.denominator)
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
    forward: CheckedAffine,
    inverse: CheckedAffine,
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
            forward: CheckedAffine::compile(forward_scale, forward_offset)?,
            inverse: CheckedAffine::compile(inverse_scale, inverse_offset)?,
        })
    }

    pub(crate) fn digital_to_model(self, sample: i64) -> Result<i64, CalibrationDomainError> {
        self.forward.apply(sample)
    }

    pub(crate) fn model_to_digital(self, sample: i64) -> Result<i64, CalibrationDomainError> {
        self.inverse.apply(sample)
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

    #[test]
    fn compiled_affine_matches_exact_rational_reference() {
        let cases = [
            calibration((3, 7), (-11, 5), "ucum:V"),
            calibration((-5, 13), (17, 19), "ucum:mV"),
            calibration((23, 29), (-31, 37), "ucum:uV"),
            calibration((41, 43), (47, 53), "ucum:nV"),
        ];
        let samples = [
            i32::MIN as i64,
            -65_537,
            -3,
            -1,
            0,
            1,
            3,
            65_537,
            i32::MAX as i64,
        ];
        for calibration in cases {
            let scale = CheckedRational::from_abir(calibration.scale()).unwrap();
            let offset = CheckedRational::from_abir(calibration.offset()).unwrap();
            let model_units = unit_to_microvolt(calibration.unit())
                .unwrap()
                .multiply(CheckedRational::integer(65_536))
                .unwrap();
            let forward_scale = scale.multiply(model_units).unwrap();
            let forward_offset = offset.multiply(model_units).unwrap();
            let inverse_scale = CheckedRational::integer(1)
                .divide(model_units)
                .unwrap()
                .divide(scale)
                .unwrap();
            let inverse_offset = CheckedRational::integer(0)
                .subtract(offset)
                .unwrap()
                .divide(scale)
                .unwrap();
            let transform = AffineDomainTransform::compile(&calibration).unwrap();
            for sample in samples {
                let expected_forward = CheckedRational::integer(i128::from(sample))
                    .multiply(forward_scale)
                    .and_then(|scaled| scaled.add(forward_offset))
                    .and_then(CheckedRational::round_ties_even_i64);
                assert_eq!(transform.digital_to_model(sample), expected_forward);
                let expected_inverse = CheckedRational::integer(i128::from(sample))
                    .multiply(inverse_scale)
                    .and_then(|scaled| scaled.add(inverse_offset))
                    .and_then(CheckedRational::round_ties_even_i64);
                assert_eq!(transform.model_to_digital(sample), expected_inverse);
            }
        }
    }

    #[test]
    #[ignore = "manual calibration hot-path benchmark"]
    fn calibration_hot_path_benchmark() {
        let transform =
            AffineDomainTransform::compile(&calibration((3, 7), (-11, 5), "ucum:uV")).unwrap();
        let iterations = 20_000_000_i64;
        let started = std::time::Instant::now();
        let mut checksum = 0_i64;
        for sample in 0..iterations {
            checksum ^= std::hint::black_box(transform)
                .digital_to_model(std::hint::black_box(sample % 65_537 - 32_768))
                .unwrap();
        }
        let elapsed = started.elapsed();
        eprintln!(
            "calibration_hot_path iterations={iterations} elapsed_ns={} ns_per_sample={:.3} checksum={checksum}",
            elapsed.as_nanos(),
            elapsed.as_nanos() as f64 / iterations as f64
        );
    }
}
