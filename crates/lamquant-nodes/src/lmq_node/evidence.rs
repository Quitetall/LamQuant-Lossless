//! Exact, structurally complete PCCP gate-evidence verification.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::String;
use core::cmp::Ordering;
use core::fmt;

use semantic_abir::{payload_content_id, ContentId, ElementType, Rational};
use serde::de::{DeserializeSeed, Deserializer, Error as _, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

use super::LmqNodeProfileError;

const MAX_PCCP_EVIDENCE_BYTES: usize = 16 * 1024 * 1024;
const REQUIRED_ACCEPTANCE_METRICS: [&str; 7] = [
    "activation_memory_kb",
    "cr_avg",
    "cr_worst",
    "latency_rp2350_ms",
    "param_count",
    "pearson_r",
    "weight_memory_kb",
];

/// Structurally verified PCCP gate result bound to one trusted registry digest.
///
/// This proves record completeness and immutable identities. Current promotion
/// authorization and signer trust remain external ledger decisions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedPccpEvidence {
    pub(super) evidence_id: ContentId,
    pub(super) checkpoint_sha256: [u8; 32],
    pub(super) change_id: String,
    pub(super) pearson_floor: Rational,
}

pub fn verify_pccp_gate_evidence(
    evidence: &[u8],
    trusted_registry_sha256: [u8; 32],
) -> Result<VerifiedPccpEvidence, LmqNodeProfileError> {
    if evidence.is_empty() || evidence.len() > MAX_PCCP_EVIDENCE_BYTES {
        return Err(LmqNodeProfileError::InvalidPccpEvidence);
    }
    if !has_unique_object_members(evidence) {
        return Err(LmqNodeProfileError::InvalidPccpEvidence);
    }
    let document: Value =
        serde_json::from_slice(evidence).map_err(|_| LmqNodeProfileError::InvalidPccpEvidence)?;
    let root = document
        .as_object()
        .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?;
    if root.get("model").and_then(Value::as_str) != Some("encoder")
        || root.get("passed").and_then(Value::as_bool) != Some(true)
    {
        return Err(LmqNodeProfileError::InvalidPccpEvidence);
    }
    let change_id = root
        .get("change_id")
        .and_then(Value::as_str)
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 128
                && value.trim() == *value
                && !value.chars().any(char::is_control)
        })
        .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?;
    let checkpoint_sha256 = root
        .get("candidate_sha256")
        .and_then(Value::as_str)
        .and_then(parse_sha256_hex)
        .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?;
    let registry_sha256 = root
        .get("registry_sha256")
        .and_then(Value::as_str)
        .and_then(parse_sha256_hex)
        .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?;
    if registry_sha256 != trusted_registry_sha256 {
        return Err(LmqNodeProfileError::InvalidPccpEvidence);
    }

    let criteria = root
        .get("criteria")
        .and_then(Value::as_array)
        .filter(|criteria| !criteria.is_empty())
        .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?;
    let mut acceptance_names = BTreeSet::new();
    let mut pearson = None;
    for criterion in criteria {
        let Some(criterion) = criterion.as_object() else {
            return Err(LmqNodeProfileError::InvalidPccpEvidence);
        };
        if criterion.get("kind").and_then(Value::as_str) != Some("acceptance") {
            continue;
        }
        let name = criterion
            .get("name")
            .and_then(Value::as_str)
            .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?;
        if !acceptance_names.insert(name) {
            return Err(LmqNodeProfileError::InvalidPccpEvidence);
        }
        if criterion.get("skipped").and_then(Value::as_bool) != Some(false)
            || criterion.get("passed").and_then(Value::as_bool) != Some(true)
            || !criterion.get("measured").is_some_and(Value::is_number)
        {
            return Err(LmqNodeProfileError::InvalidPccpEvidence);
        }
        if name == "pearson_r" {
            if pearson.is_some() {
                return Err(LmqNodeProfileError::InvalidPccpEvidence);
            }
            let floor = exact_json_rational(
                criterion
                    .get("floor")
                    .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?,
            )?;
            let measured = exact_json_rational(
                criterion
                    .get("measured")
                    .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?,
            )?;
            let (floor_numerator, floor_denominator) = floor.parts();
            if floor_numerator <= 0
                || floor_numerator > floor_denominator
                || rational_cmp(measured, floor) == Ordering::Less
            {
                return Err(LmqNodeProfileError::InvalidPccpEvidence);
            }
            pearson = Some((floor, measured));
        }
    }
    let required = REQUIRED_ACCEPTANCE_METRICS
        .into_iter()
        .collect::<BTreeSet<_>>();
    if acceptance_names != required {
        return Err(LmqNodeProfileError::InvalidPccpEvidence);
    }
    let (pearson_floor, measured) = pearson.ok_or(LmqNodeProfileError::InvalidPccpEvidence)?;
    let recorded = exact_json_rational(
        root.get("measurements")
            .and_then(Value::as_object)
            .and_then(|measurements| measurements.get("pearson_r"))
            .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?,
    )?;
    if recorded != measured {
        return Err(LmqNodeProfileError::InvalidPccpEvidence);
    }
    Ok(VerifiedPccpEvidence {
        evidence_id: payload_content_id(ElementType::Bytes, evidence),
        checkpoint_sha256,
        change_id: change_id.into(),
        pearson_floor,
    })
}

pub(super) fn has_unique_object_members(document: &[u8]) -> bool {
    let mut deserializer = serde_json::Deserializer::from_slice(document);
    DuplicateRejectingSeed
        .deserialize(&mut deserializer)
        .is_ok()
        && deserializer.end().is_ok()
}

#[derive(Clone, Copy)]
struct DuplicateRejectingSeed;

impl<'de> DeserializeSeed<'de> for DuplicateRejectingSeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(DuplicateRejectingVisitor)
    }
}

struct DuplicateRejectingVisitor;

impl<'de> Visitor<'de> for DuplicateRejectingVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object member names")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        DuplicateRejectingSeed.deserialize(deserializer)
    }

    fn visit_newtype_struct<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        DuplicateRejectingSeed.deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence
            .next_element_seed(DuplicateRejectingSeed)?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key) {
                return Err(A::Error::custom("duplicate JSON object member"));
            }
            map.next_value_seed(DuplicateRejectingSeed)?;
        }
        Ok(())
    }
}

fn exact_json_rational(value: &Value) -> Result<Rational, LmqNodeProfileError> {
    let number = value
        .as_number()
        .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?
        .to_string();
    decimal_rational(&number)
}

fn decimal_rational(value: &str) -> Result<Rational, LmqNodeProfileError> {
    let (mantissa, exponent) = match value.find(['e', 'E']) {
        Some(index) => {
            let exponent = value[index + 1..]
                .parse::<i32>()
                .map_err(|_| LmqNodeProfileError::InvalidPccpEvidence)?;
            (&value[..index], exponent)
        }
        None => (value, 0),
    };
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.strip_prefix(['-', '+']).unwrap_or(mantissa);
    let (integer, fraction) = match mantissa.split_once('.') {
        Some(parts) => parts,
        None => (mantissa, ""),
    };
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(LmqNodeProfileError::InvalidPccpEvidence);
    }
    let digits = format!("{integer}{fraction}");
    let mut numerator = digits
        .parse::<i128>()
        .map_err(|_| LmqNodeProfileError::InvalidPccpEvidence)?;
    if negative {
        numerator = numerator
            .checked_neg()
            .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?;
    }
    let scale = i32::try_from(fraction.len())
        .map_err(|_| LmqNodeProfileError::InvalidPccpEvidence)?
        .checked_sub(exponent)
        .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?;
    let (numerator, denominator) = if scale >= 0 {
        (
            numerator,
            checked_power_of_ten(scale as u32).ok_or(LmqNodeProfileError::InvalidPccpEvidence)?,
        )
    } else {
        (
            numerator
                .checked_mul(
                    checked_power_of_ten(scale.unsigned_abs())
                        .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?,
                )
                .ok_or(LmqNodeProfileError::InvalidPccpEvidence)?,
            1,
        )
    };
    Rational::new(numerator, denominator).map_err(|_| LmqNodeProfileError::InvalidPccpEvidence)
}

fn checked_power_of_ten(exponent: u32) -> Option<i128> {
    (0..exponent).try_fold(1_i128, |value, _| value.checked_mul(10))
}

fn rational_cmp(left: Rational, right: Rational) -> Ordering {
    let (left_numerator, left_denominator) = left.parts();
    let (right_numerator, right_denominator) = right.parts();
    match (left_numerator.cmp(&0), right_numerator.cmp(&0)) {
        (Ordering::Less, Ordering::Less) => compare_positive_rationals(
            right_numerator.unsigned_abs(),
            right_denominator as u128,
            left_numerator.unsigned_abs(),
            left_denominator as u128,
        ),
        (Ordering::Less, _) => Ordering::Less,
        (_, Ordering::Less) => Ordering::Greater,
        (Ordering::Equal, Ordering::Equal) => Ordering::Equal,
        (Ordering::Equal, Ordering::Greater) => Ordering::Less,
        (Ordering::Greater, Ordering::Equal) => Ordering::Greater,
        (Ordering::Greater, Ordering::Greater) => compare_positive_rationals(
            left_numerator as u128,
            left_denominator as u128,
            right_numerator as u128,
            right_denominator as u128,
        ),
    }
}

fn compare_positive_rationals(
    mut left_numerator: u128,
    mut left_denominator: u128,
    mut right_numerator: u128,
    mut right_denominator: u128,
) -> Ordering {
    let mut reversed = false;
    loop {
        let left_quotient = left_numerator / left_denominator;
        let right_quotient = right_numerator / right_denominator;
        if left_quotient != right_quotient {
            let ordering = left_quotient.cmp(&right_quotient);
            return if reversed {
                ordering.reverse()
            } else {
                ordering
            };
        }
        let left_remainder = left_numerator % left_denominator;
        let right_remainder = right_numerator % right_denominator;
        let ordering = match (left_remainder == 0, right_remainder == 0) {
            (true, true) => return Ordering::Equal,
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        };
        if let Some(ordering) = ordering {
            return if reversed {
                ordering.reverse()
            } else {
                ordering
            };
        }
        left_numerator = left_denominator;
        left_denominator = left_remainder;
        right_numerator = right_denominator;
        right_denominator = right_remainder;
        reversed = !reversed;
    }
}

fn parse_sha256_hex(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    let mut digest = [0_u8; 32];
    for (index, output) in digest.iter_mut().enumerate() {
        *output = u8::from_str_radix(value.get(index * 2..index * 2 + 2)?, 16).ok()?;
    }
    Some(digest)
}
