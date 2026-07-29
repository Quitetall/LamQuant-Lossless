//! ADR 0074 Track N — the `NeuralBackend` seam.
//!
//! The shell owns everything wire-critical (Rust, stable); a **backend** owns
//! ONLY the neural network. The trait is object-safe (no generic dataset type)
//! so the shell can hold a `&dyn NeuralBackend` and swap a Python backend now for
//! a fully-Rust one later WITHOUT a wire change — the wire is the [`crate::body`]
//! format either way.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;
use semantic_abir::{ContentId, Rational};
use semantic_abir_bcs::{ModelProvenance, PccpStatus};

/// Execution realm supplied by one neural backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BackendTarget {
    /// Deterministic model-free test implementation.
    Reference,
    /// Model executes in a bounded host subprocess.
    HostSubprocess,
    /// Model executes in the current native process.
    HostNative,
    /// Model executes in an allocation-free MCU runtime.
    McuNative,
}

/// Typed input/output envelope enforced before and after neural inference.
///
/// Resource bounds on the outer shell remain independent. This contract states
/// what the backend can interpret; [`crate::shell::LmqResourceBounds`] states
/// what one invocation may allocate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NeuralBackendCapabilities {
    pub target: BackendTarget,
    pub operational: bool,
    pub minimum_channels: u16,
    pub maximum_channels: u16,
    pub minimum_samples: u32,
    pub maximum_samples: u32,
    pub minimum_sample_rate: Rational,
    pub maximum_sample_rate: Rational,
    pub maximum_tokens: u32,
    pub maximum_schedule_bytes: u32,
    pub maximum_backend_metadata_bytes: u32,
    pub minimum_alphabet: u16,
    pub maximum_alphabet: u16,
}

/// Exact backend-contract violation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BackendCapabilityError {
    Unavailable,
    ChannelCount,
    SampleCount,
    SampleRate,
    TokenCount,
    ScheduleBytes,
    BackendMetadataBytes,
    Alphabet,
    TokenValue,
}

impl NeuralBackendCapabilities {
    pub fn validate_input(
        self,
        channels: u16,
        samples: u32,
        sample_rate: Rational,
    ) -> Result<(), BackendCapabilityError> {
        if !self.operational {
            return Err(BackendCapabilityError::Unavailable);
        }
        if !(self.minimum_channels..=self.maximum_channels).contains(&channels) {
            return Err(BackendCapabilityError::ChannelCount);
        }
        if !(self.minimum_samples..=self.maximum_samples).contains(&samples) {
            return Err(BackendCapabilityError::SampleCount);
        }
        if sample_rate.parts().0 <= 0 {
            return Err(BackendCapabilityError::SampleRate);
        }
        if rational_cmp(sample_rate, self.minimum_sample_rate) == Ordering::Less
            || rational_cmp(sample_rate, self.maximum_sample_rate) == Ordering::Greater
        {
            return Err(BackendCapabilityError::SampleRate);
        }
        Ok(())
    }

    pub fn validate_output(self, tokens: &NeuralTokens) -> Result<(), BackendCapabilityError> {
        if !(self.minimum_channels..=self.maximum_channels).contains(&tokens.n_channels) {
            return Err(BackendCapabilityError::ChannelCount);
        }
        if !(self.minimum_samples..=self.maximum_samples).contains(&tokens.n_samples) {
            return Err(BackendCapabilityError::SampleCount);
        }
        if u32::try_from(tokens.tokens.len()).map_or(true, |count| count > self.maximum_tokens) {
            return Err(BackendCapabilityError::TokenCount);
        }
        if u32::try_from(tokens.schedule.len())
            .map_or(true, |count| count > self.maximum_schedule_bytes)
        {
            return Err(BackendCapabilityError::ScheduleBytes);
        }
        if u32::try_from(tokens.backend_meta.len())
            .map_or(true, |count| count > self.maximum_backend_metadata_bytes)
        {
            return Err(BackendCapabilityError::BackendMetadataBytes);
        }
        if !(self.minimum_alphabet..=self.maximum_alphabet).contains(&tokens.alphabet) {
            return Err(BackendCapabilityError::Alphabet);
        }
        if tokens
            .tokens
            .iter()
            .any(|token| *token < 0 || *token >= i32::from(tokens.alphabet))
        {
            return Err(BackendCapabilityError::TokenValue);
        }
        Ok(())
    }
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

/// Compare non-negative rationals without cross multiplication. Continued
/// fractions keep all intermediates within the original `u128` values.
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

/// Neural tokens + the shape/model metadata the shell needs to wire them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeuralTokens {
    /// FSQ symbols, unsigned in `[0, alphabet)`, flattened in a backend-defined
    /// order (the backend is the sole interpreter of the layout).
    pub tokens: Vec<i32>,
    /// Per-timestep FSQ level schedule (the adaptive-CR state) — opaque to the
    /// shell, round-tripped verbatim through the body.
    pub schedule: Vec<u8>,
    /// The FSQ alphabet size L (symbols are in `[0, alphabet)`); drives the rANS
    /// frequency model the shell builds.
    pub alphabet: u16,
    /// Channel count of the source recording (reconstruction shape).
    pub n_channels: u16,
    /// Per-channel sample count of the source recording (reconstruction length).
    pub n_samples: u32,
    /// Opaque backend state the DECODER needs but that isn't a token — e.g. the
    /// Python codec's per-channel LPC/lifting preprocessing metadata, or latent
    /// normalization. The shell carries it verbatim in the LMQ packet metadata
    /// section (never interprets it). Empty for backends that need none (Stub).
    pub backend_meta: Vec<u8>,
}

/// A backend failure — a Python inference error, a shape mismatch, a missing
/// checkpoint, etc. Textual so the seam stays decoupled from any backend's own
/// error type.
#[derive(Debug)]
pub struct BackendError(pub String);

/// The swappable neural inference seam. Object-safe by design.
pub trait NeuralBackend {
    /// Exact shape/rate/output envelope and execution realm.
    fn capabilities(&self) -> NeuralBackendCapabilities;

    /// Immutable identity of the exact checkpoint and PCCP evidence this
    /// backend executes. The shell seals and verifies this value; callers
    /// cannot claim provenance independently of the inference implementation.
    fn model_provenance(&self) -> ModelProvenance;

    /// Encode a modality-blind signal (`[n_channels][n_samples]`, sampled at
    /// `sample_rate` Hz) into tokens.
    fn encode(
        &self,
        signal: &[Vec<i64>],
        sample_rate: Rational,
    ) -> Result<NeuralTokens, BackendError>;

    /// Reconstruct the signal (`[n_channels][n_samples]`) from tokens. LOSSY:
    /// `decode(encode(x)) ≈ x`, never `== x`.
    fn decode(&self, tokens: &NeuralTokens) -> Result<Vec<Vec<i64>>, BackendError>;
}

/// A deterministic, model-free reference backend for shell/DAG tests (ADR 0074
/// N1). NOT a real codec: it "encodes" each sample as its residue mod `alphabet`
/// (a trivial uniform quantizer, channel-major) and "decodes" the residues back,
/// so the shell round-trip is exercised without a trained model.
/// `decode(encode(x)) == x mod alphabet` (lossy), deterministically.
pub struct StubBackend {
    /// FSQ alphabet size the stub quantizes to (`2..=255`).
    pub alphabet: u16,
}

impl Default for StubBackend {
    fn default() -> Self {
        Self { alphabet: 5 }
    }
}

impl NeuralBackend for StubBackend {
    fn capabilities(&self) -> NeuralBackendCapabilities {
        NeuralBackendCapabilities {
            target: BackendTarget::Reference,
            operational: true,
            minimum_channels: 1,
            maximum_channels: u16::MAX,
            minimum_samples: 1,
            maximum_samples: u32::MAX,
            minimum_sample_rate: Rational::new(1, i64::MAX.into()).expect("positive denominator"),
            maximum_sample_rate: Rational::new(i64::MAX.into(), 1).expect("positive denominator"),
            maximum_tokens: u32::MAX,
            maximum_schedule_bytes: u32::MAX,
            maximum_backend_metadata_bytes: 0,
            minimum_alphabet: 2,
            maximum_alphabet: u8::MAX.into(),
        }
    }

    fn model_provenance(&self) -> ModelProvenance {
        ModelProvenance {
            checkpoint_content_id: ContentId::from_bytes([0x51; 32]),
            checkpoint_sha256: [0x52; 32],
            pccp_change_id: String::from("LMQ-STUB-REFERENCE"),
            pccp_evidence_id: ContentId::from_bytes([0x53; 32]),
            pccp_status: PccpStatus::Candidate,
        }
    }

    fn encode(
        &self,
        signal: &[Vec<i64>],
        sample_rate: Rational,
    ) -> Result<NeuralTokens, BackendError> {
        if signal.is_empty() {
            return Err(BackendError(String::from("stub: empty signal")));
        }
        if !(2..=u16::from(u8::MAX)).contains(&self.alphabet) {
            return Err(BackendError(String::from(
                "stub: alphabet must be in 2..=255",
            )));
        }
        let n_channels = u16::try_from(signal.len())
            .map_err(|_| BackendError(String::from("stub: too many channels")))?;
        let n_samples = u32::try_from(signal[0].len())
            .map_err(|_| BackendError(String::from("stub: too many samples")))?;
        if signal
            .iter()
            .any(|channel| u32::try_from(channel.len()) != Ok(n_samples))
        {
            return Err(BackendError(String::from("stub: ragged channels")));
        }
        self.capabilities()
            .validate_input(n_channels, n_samples, sample_rate)
            .map_err(|error| BackendError(format!("stub capability mismatch: {error:?}")))?;
        let l = self.alphabet as i64;
        let tokens: Vec<i32> = signal
            .iter()
            .flat_map(|ch| ch.iter().map(move |&s| s.rem_euclid(l) as i32))
            .collect();
        // One schedule entry per timestep (stand-in; the shell just carries it).
        let schedule = alloc::vec![self.alphabet as u8; n_samples as usize];
        let tokens = NeuralTokens {
            tokens,
            schedule,
            alphabet: self.alphabet,
            n_channels,
            n_samples,
            backend_meta: Vec::new(), // the stub is self-contained; no metadata
        };
        self.capabilities()
            .validate_output(&tokens)
            .map_err(|error| BackendError(format!("stub capability mismatch: {error:?}")))?;
        Ok(tokens)
    }

    fn decode(&self, t: &NeuralTokens) -> Result<Vec<Vec<i64>>, BackendError> {
        self.capabilities()
            .validate_output(t)
            .map_err(|error| BackendError(format!("stub capability mismatch: {error:?}")))?;
        let n_ch = t.n_channels as usize;
        let n_s = t.n_samples as usize;
        if t.tokens.len() != n_ch.saturating_mul(n_s) {
            return Err(BackendError(String::from(
                "stub: token count != n_channels * n_samples",
            )));
        }
        let mut out = Vec::with_capacity(n_ch);
        for c in 0..n_ch {
            out.push(
                t.tokens[c * n_s..(c + 1) * n_s]
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_encode_decode_is_deterministic_mod_alphabet() {
        let signal = alloc::vec![
            alloc::vec![0i64, 6, 12, -1, 20],
            alloc::vec![3, 3, 3, 8, 100]
        ];
        let b = StubBackend { alphabet: 5 };
        let t = b.encode(&signal, Rational::new(250, 1).unwrap()).unwrap();
        assert_eq!(t.n_channels, 2);
        assert_eq!(t.n_samples, 5);
        assert_eq!(t.tokens.len(), 10);
        let recon = b.decode(&t).unwrap();
        // decode(encode(x)) == x mod alphabet.
        let expect: Vec<Vec<i64>> = signal
            .iter()
            .map(|ch| ch.iter().map(|&s| s.rem_euclid(5)).collect())
            .collect();
        assert_eq!(recon, expect);
    }

    #[test]
    fn stub_rejects_empty_and_ragged() {
        let b = StubBackend::default();
        assert!(b.encode(&[], Rational::new(250, 1).unwrap()).is_err());
        assert!(b
            .encode(
                &alloc::vec![alloc::vec![0i64, 1], alloc::vec![0i64]],
                Rational::new(250, 1).unwrap(),
            )
            .is_err());
        assert!(b
            .encode(
                &alloc::vec![alloc::vec![0i64, 1]],
                Rational::new(-250, 1).unwrap(),
            )
            .is_err());
    }

    #[test]
    fn capabilities_reject_out_of_envelope_values() {
        let caps = NeuralBackendCapabilities {
            target: BackendTarget::HostNative,
            operational: true,
            minimum_channels: 2,
            maximum_channels: 4,
            minimum_samples: 10,
            maximum_samples: 20,
            minimum_sample_rate: Rational::new(100, 1).unwrap(),
            maximum_sample_rate: Rational::new(500, 1).unwrap(),
            maximum_tokens: 8,
            maximum_schedule_bytes: 4,
            maximum_backend_metadata_bytes: 2,
            minimum_alphabet: 2,
            maximum_alphabet: 8,
        };
        assert_eq!(
            caps.validate_input(1, 10, Rational::new(250, 1).unwrap()),
            Err(BackendCapabilityError::ChannelCount)
        );
        assert_eq!(
            caps.validate_input(2, 9, Rational::new(250, 1).unwrap()),
            Err(BackendCapabilityError::SampleCount)
        );
        assert_eq!(
            caps.validate_input(2, 10, Rational::new(1_000, 1).unwrap()),
            Err(BackendCapabilityError::SampleRate)
        );
        assert_eq!(
            caps.validate_input(2, 10, Rational::new(-250, 1).unwrap()),
            Err(BackendCapabilityError::SampleRate)
        );
        let mut tokens = NeuralTokens {
            tokens: alloc::vec![0; 9],
            schedule: alloc::vec![2; 4],
            alphabet: 2,
            n_channels: 2,
            n_samples: 10,
            backend_meta: alloc::vec![0; 2],
        };
        assert_eq!(
            caps.validate_output(&tokens),
            Err(BackendCapabilityError::TokenCount)
        );
        tokens.tokens.truncate(8);
        tokens.backend_meta.push(0);
        assert_eq!(
            caps.validate_output(&tokens),
            Err(BackendCapabilityError::BackendMetadataBytes)
        );
    }

    #[test]
    fn rational_capability_comparison_never_cross_multiplies() {
        let maximum = i128::MAX;
        let mut caps = StubBackend::default().capabilities();
        caps.minimum_sample_rate = Rational::new(maximum - 2, maximum - 1).unwrap();
        caps.maximum_sample_rate = Rational::new(1, 1).unwrap();
        assert_eq!(
            caps.validate_input(1, 1, Rational::new(maximum - 1, maximum).unwrap()),
            Ok(())
        );
    }

    #[test]
    fn stub_rejects_alphabet_that_cannot_fit_schedule() {
        let backend = StubBackend { alphabet: 256 };
        assert!(backend
            .encode(&[alloc::vec![1]], Rational::new(250, 1).unwrap())
            .is_err());
        assert_eq!(backend.capabilities().maximum_alphabet, 255);
    }
}
