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
//! the second explicitly, as [`CoordinateFrame`]'s
//! `uncertainty` field. An electrode position known to ±2 mm should say so
//! there, while still carrying the exact number it was given. Rounding would
//! silently claim a precision the measurement does not have, and lose the
//! original besides.
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

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::vec::Vec;
use semantic_abir::{
    ChannelSpec, ConceptId, CoordinateFrame, CoordinateFrameTag, ExactNumber, ObjectId, Rational,
};
use sha2::{Digest, Sha256};

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

/// Domain separator for montage object ids.
const ID_DOMAIN: &[u8] = b"org.quitetall.lamquant.montage-v1\0";
/// Domain separator for the shared coordinate-space root.
const ROOT_ID_DOMAIN: &[u8] = b"org.quitetall.lamquant.montage-root-v1\0";

/// Concept naming an electrode-position frame.
const ELECTRODE_FRAME_CONCEPT: &str = "lamquant:electrode-position-frame";
/// Concept naming the common coordinate system carried by one LMQC montage.
const MONTAGE_ROOT_CONCEPT: &str = "lamquant:lmqc-montage-coordinate-frame-v1";

/// Derive the shared coordinate-space id for one montage.
pub fn root_frame_id_for(montage_digest: &[u8; 32]) -> ObjectId<CoordinateFrameTag> {
    let mut hasher = Sha256::new();
    hasher.update(ROOT_ID_DOMAIN);
    hasher.update(montage_digest);
    let digest: [u8; 32] = hasher.finalize().into();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    ObjectId::from_bytes(bytes)
}

/// Declare the common coordinate system against which electrode translations
/// are expressed.
///
/// LMQC records no transform from its montage coordinates into another frame.
/// The root therefore has no parent or transform. This preserves that absence
/// while still declaring that every located electrode shares one basis.
pub fn montage_root_frame(montage_digest: &[u8; 32]) -> Result<CoordinateFrame, MontageError> {
    let concept = ConceptId::new(MONTAGE_ROOT_CONCEPT).map_err(|_| MontageError::NotFinite)?;
    let zero = Rational::new(0, 1).map_err(|_| MontageError::NotFinite)?;
    Ok(CoordinateFrame::new(
        root_frame_id_for(montage_digest),
        concept,
        None,
        None,
        zero,
    ))
}

/// Derive a coordinate-frame id from the montage content itself.
///
/// Deterministic on purpose. A random id would make two byte-identical montages
/// produce different artifacts, which defeats content addressing: the same
/// recording archived twice would no longer deduplicate, and comparing two
/// captures of one montage would report a difference that does not exist.
///
/// The channel index is included because two electrodes may legitimately sit at
/// the same measured position (a reference tied to another site, say) and still
/// need distinct frames.
pub fn frame_id_for(montage_digest: &[u8; 32], channel_index: u32) -> ObjectId<CoordinateFrameTag> {
    let mut hasher = Sha256::new();
    hasher.update(ID_DOMAIN);
    hasher.update(montage_digest);
    hasher.update(channel_index.to_le_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    ObjectId::from_bytes(bytes)
}

/// A 4x4 row-major homogeneous transform placing an electrode at `position`.
///
/// Identity rotation with the position in the translation column: an electrode
/// frame is a displacement from its parent, not a reorientation. Encoding it as
/// a transform rather than three loose numbers is what lets a consumer compose
/// it with the head frame without knowing it came from a `.lmq` file.
fn translation_transform(position: [ExactNumber; 3]) -> [ExactNumber; 16] {
    let zero = ExactNumber::Integer(0);
    let one = ExactNumber::Integer(1);
    [
        one,
        zero,
        zero,
        position[0],
        zero,
        one,
        zero,
        position[1],
        zero,
        zero,
        one,
        position[2],
        zero,
        zero,
        zero,
        one,
    ]
}

/// Build the coordinate frame for one electrode, or `None` when unlocated.
///
/// `uncertainty` is the MEASUREMENT precision of the montage — how well the
/// electrode position is actually known. It is deliberately a caller input
/// rather than something inferred from the numbers, because nothing in an `f32`
/// says how carefully it was measured, and inferring it would manufacture a
/// confidence the data never carried.
pub fn coordinate_frame_for(
    montage_digest: &[u8; 32],
    channel_index: u32,
    coordinate: Coordinate,
    parent: Option<ObjectId<CoordinateFrameTag>>,
    uncertainty: Rational,
) -> Result<Option<CoordinateFrame>, MontageError> {
    let Coordinate::Known(position) = coordinate else {
        return Ok(None);
    };
    let concept = ConceptId::new(ELECTRODE_FRAME_CONCEPT).map_err(|_| MontageError::NotFinite)?;
    Ok(Some(CoordinateFrame::new(
        frame_id_for(montage_digest, channel_index),
        concept,
        parent,
        Some(translation_transform(position)),
        uncertainty,
    )))
}

/// Build the channel spec for one electrode.
///
/// An unlocated electrode gets a spec with no coordinate frame rather than no
/// spec at all: the channel exists and carries signal regardless of whether
/// anyone measured where it sat.
pub fn channel_spec_for(
    concept: ConceptId,
    frame: Option<ObjectId<CoordinateFrameTag>>,
) -> ChannelSpec {
    let spec = ChannelSpec::new(concept);
    match frame {
        Some(id) => spec.with_coordinate_frame(id),
        None => spec,
    }
}

/// Project a whole montage: one frame per located electrode, in channel order.
pub fn frames_for_montage(
    montage_digest: &[u8; 32],
    coordinates: &[[f32; 3]],
    parent: Option<ObjectId<CoordinateFrameTag>>,
    uncertainty: Rational,
) -> Result<Vec<Option<CoordinateFrame>>, MontageError> {
    let mut frames = Vec::with_capacity(coordinates.len());
    for (index, xyz) in coordinates.iter().enumerate() {
        let index = u32::try_from(index).map_err(|_| MontageError::NotFinite)?;
        frames.push(coordinate_frame_for(
            montage_digest,
            index,
            coordinate_from_f32(*xyz)?,
            parent,
            uncertainty,
        )?);
    }
    Ok(frames)
}

#[cfg(test)]
mod entity_tests {
    use super::*;

    fn micrometre() -> Rational {
        // 1e-6 m: a realistic digitiser precision, and the point is that it is
        // DECLARED rather than implied by how the numbers were rounded.
        Rational::new(1, 1_000_000).expect("valid rational")
    }

    #[test]
    fn identical_montages_derive_identical_frame_ids() {
        // Content addressing depends on this: a random id would make the same
        // recording archived twice fail to deduplicate.
        let digest = [0x7a_u8; 32];
        assert_eq!(frame_id_for(&digest, 3), frame_id_for(&digest, 3));
    }

    #[test]
    fn different_channels_and_montages_derive_different_ids() {
        let digest = [0x7a_u8; 32];
        let other = [0x7b_u8; 32];
        assert_ne!(
            frame_id_for(&digest, 3),
            frame_id_for(&digest, 4),
            "two electrodes may share a position and still need distinct frames"
        );
        assert_ne!(frame_id_for(&digest, 3), frame_id_for(&other, 3));
        assert_ne!(root_frame_id_for(&digest), root_frame_id_for(&other));
        assert_ne!(root_frame_id_for(&digest), frame_id_for(&digest, 3));
    }

    #[test]
    fn an_unlocated_electrode_gets_no_frame() {
        let frame = coordinate_frame_for(&[0_u8; 32], 0, Coordinate::Unknown, None, micrometre())
            .expect("projects");
        assert!(frame.is_none(), "absence must stay absence");
    }

    #[test]
    fn a_located_electrode_carries_its_position_in_the_transform() {
        let coordinate = coordinate_from_f32([0.081, -0.0725, 0.0341]).unwrap();
        let Coordinate::Known(expected) = coordinate else {
            panic!("finite triple");
        };
        let frame = coordinate_frame_for(&[0_u8; 32], 0, coordinate, None, micrometre())
            .expect("projects")
            .expect("located electrodes get a frame");
        let transform = frame.transform().expect("a position is a transform");
        // Translation column of a row-major 4x4.
        assert_eq!(transform[3], expected[0]);
        assert_eq!(transform[7], expected[1]);
        assert_eq!(transform[11], expected[2]);
        assert_eq!(transform[15], ExactNumber::Integer(1));
    }

    #[test]
    fn an_unlocated_electrode_still_gets_a_channel_spec() {
        // The channel carries signal whether or not anyone measured where it sat.
        let concept = ConceptId::new("lamquant:eeg-channel").expect("valid concept");
        let spec = channel_spec_for(concept, None);
        assert!(spec.coordinate_frame_id().is_none());
    }

    #[test]
    fn a_montage_projects_one_frame_per_channel_in_order() {
        let coordinates = [[0.081_f32, 0.0, 0.0], [f32::NAN; 3], [0.0, 0.05, 0.0]];
        let frames =
            frames_for_montage(&[0_u8; 32], &coordinates, None, micrometre()).expect("projects");
        assert_eq!(frames.len(), 3);
        assert!(frames[0].is_some());
        assert!(frames[1].is_none(), "the NaN sentinel must stay an absence");
        assert!(frames[2].is_some());
    }

    #[test]
    fn located_electrodes_share_declared_montage_root() {
        let digest = [0x42_u8; 32];
        let root = montage_root_frame(&digest).unwrap();
        let frames = frames_for_montage(
            &digest,
            &[[0.081_f32, 0.0, 0.0], [0.0, 0.05, 0.0]],
            Some(root.id()),
            micrometre(),
        )
        .unwrap();
        assert!(root.parent_id().is_none());
        assert!(root.transform().is_none());
        assert!(frames
            .iter()
            .flatten()
            .all(|frame| frame.parent_id() == Some(root.id())));
    }
}
