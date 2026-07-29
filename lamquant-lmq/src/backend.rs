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
use semantic_abir::{ConceptId, ContentId, Rational};
use semantic_abir_bcs::{ModelProvenance, PccpStatus};

const MODEL_INPUT_CONTRACT_DOMAIN: &[u8] = b"lamquant.lmq.model-input-contract.v1";
const TRAINED_MODEL_ARTIFACT_DOMAIN: &[u8] = b"lamquant.lmq.trained-model-artifact.v1";

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

/// Numeric meaning of every `i64` sample crossing [`NeuralBackend`].
///
/// `PhysicalMicrovoltQ16` is signed Q47.16 microvolts: `65_536` represents
/// exactly `1 µV`. ABIR calibration stays in the Rust shell; host helpers never
/// interpret source-format digital counts or unit identifiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SignalDomain {
    DigitalInteger,
    PhysicalMicrovoltQ16,
}

impl SignalDomain {
    /// Stable identifier sealed into bundle parameters and subprocess requests.
    pub const fn protocol_name(self) -> &'static str {
        match self {
            Self::DigitalInteger => "digital-integer",
            Self::PhysicalMicrovoltQ16 => "physical-microvolt-q16",
        }
    }
}

/// Immutable semantic contract accepted by one neural model.
///
/// Shape bounds remain in [`NeuralBackendCapabilities`]. This contract binds
/// meanings that shape alone cannot express: modality, ordered channel basis,
/// exact weighted reference construction, upstream derivation proof, model
/// domain, and the preprocessing executed inside the backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInputContract {
    modality: ConceptId,
    channel_concepts: Vec<ConceptId>,
    model_channel_basis_content_id: ContentId,
    sample_rate: Rational,
    samples: u32,
    signal_domain: SignalDomain,
    upstream_derivation: ConceptId,
    upstream_claim_kind: ConceptId,
    backend_pipeline: ConceptId,
    content_id: ContentId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ModelInputContractDefinitionError {
    EmptyChannelBasis,
    InvalidSampleRate,
    ZeroSamples,
    TooManyChannels,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ModelInputContractError {
    Modality,
    MissingChannelBasis,
    MissingChannelBasisConstruction,
    MissingChannelBasisSource,
    AmbiguousChannelBasisSource,
    ChannelCount,
    ChannelOrder,
    ChannelBasis,
    MissingDerivation,
    Derivation,
    MissingDerivationClaim,
    SampleRate,
    SampleCount,
    SignalDomain,
}

impl ModelInputContract {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        modality: ConceptId,
        channel_concepts: Vec<ConceptId>,
        model_channel_basis_content_id: ContentId,
        sample_rate: Rational,
        samples: u32,
        signal_domain: SignalDomain,
        upstream_derivation: ConceptId,
        upstream_claim_kind: ConceptId,
        backend_pipeline: ConceptId,
    ) -> Result<Self, ModelInputContractDefinitionError> {
        if channel_concepts.is_empty() {
            return Err(ModelInputContractDefinitionError::EmptyChannelBasis);
        }
        if u16::try_from(channel_concepts.len()).is_err() {
            return Err(ModelInputContractDefinitionError::TooManyChannels);
        }
        if !sample_rate.is_positive() {
            return Err(ModelInputContractDefinitionError::InvalidSampleRate);
        }
        if samples == 0 {
            return Err(ModelInputContractDefinitionError::ZeroSamples);
        }
        let mut value = Self {
            modality,
            channel_concepts,
            model_channel_basis_content_id,
            sample_rate,
            samples,
            signal_domain,
            upstream_derivation,
            upstream_claim_kind,
            backend_pipeline,
            content_id: ContentId::from_bytes([0; 32]),
        };
        value.content_id = value.compute_content_id();
        Ok(value)
    }

    pub fn modality(&self) -> &ConceptId {
        &self.modality
    }

    pub fn channel_concepts(&self) -> &[ConceptId] {
        &self.channel_concepts
    }

    /// Portable model-role identity of exact weighted channel construction.
    ///
    /// Unlike ABIR object identity, this projection uses unique source-channel
    /// role concepts so one model contract can apply across recordings. Dataset
    /// contract validation rejects ambiguity among basis-referenced source roles
    /// before inference.
    pub const fn model_channel_basis_content_id(&self) -> ContentId {
        self.model_channel_basis_content_id
    }

    pub const fn sample_rate(&self) -> Rational {
        self.sample_rate
    }

    pub const fn samples(&self) -> u32 {
        self.samples
    }

    pub const fn signal_domain(&self) -> SignalDomain {
        self.signal_domain
    }

    pub fn upstream_derivation(&self) -> &ConceptId {
        &self.upstream_derivation
    }

    /// Required ABIR typed claim over the selected input derivation.
    ///
    /// ABIR Proof records are semantic claims, not runtime authorization or
    /// signature-verification results. Model-pack trust remains an external
    /// gate and is bound separately by [`TrainedModelArtifact`].
    pub fn upstream_claim_kind(&self) -> &ConceptId {
        &self.upstream_claim_kind
    }

    /// Pipeline applied by the backend after validated ABIR input is leased.
    ///
    /// This is not a claim about prior dataset processing. Prior processing is
    /// separately bound by [`Self::upstream_derivation`] and
    /// [`Self::upstream_claim_kind`].
    pub fn backend_pipeline(&self) -> &ConceptId {
        &self.backend_pipeline
    }

    pub const fn content_id(&self) -> ContentId {
        self.content_id
    }

    fn compute_content_id(&self) -> ContentId {
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, MODEL_INPUT_CONTRACT_DOMAIN);
        hash_field(&mut hasher, self.modality.as_str().as_bytes());
        hash_field(&mut hasher, self.model_channel_basis_content_id.as_bytes());
        let (sample_rate_numerator, sample_rate_denominator) = self.sample_rate.parts();
        hash_field(&mut hasher, &sample_rate_numerator.to_le_bytes());
        hash_field(&mut hasher, &sample_rate_denominator.to_le_bytes());
        hash_field(&mut hasher, &self.samples.to_le_bytes());
        hash_field(&mut hasher, self.signal_domain.protocol_name().as_bytes());
        hash_field(&mut hasher, self.upstream_derivation.as_str().as_bytes());
        hash_field(&mut hasher, self.upstream_claim_kind.as_str().as_bytes());
        hash_field(&mut hasher, self.backend_pipeline.as_str().as_bytes());
        let channel_count =
            u32::try_from(self.channel_concepts.len()).expect("constructor checked channel count");
        hash_field(&mut hasher, &channel_count.to_le_bytes());
        for concept in &self.channel_concepts {
            hash_field(&mut hasher, concept.as_str().as_bytes());
        }
        ContentId::from_bytes(*hasher.finalize().as_bytes())
    }
}

/// Immutable binding between executable model identity and accepted input.
///
/// A trained backend owns this value as one unit. Checkpoint hashes, PCCP
/// evidence, preprocessing identity, and input semantics therefore cannot be
/// supplied through independent backend methods or silently recombined.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainedModelArtifact {
    provenance: ModelProvenance,
    input_contract: ModelInputContract,
    content_id: ContentId,
}

impl TrainedModelArtifact {
    pub fn new(provenance: ModelProvenance, input_contract: ModelInputContract) -> Self {
        let mut value = Self {
            provenance,
            input_contract,
            content_id: ContentId::from_bytes([0; 32]),
        };
        value.content_id = value.compute_content_id();
        value
    }

    pub const fn provenance(&self) -> &ModelProvenance {
        &self.provenance
    }

    pub const fn input_contract(&self) -> &ModelInputContract {
        &self.input_contract
    }

    pub const fn content_id(&self) -> ContentId {
        self.content_id
    }

    fn compute_content_id(&self) -> ContentId {
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, TRAINED_MODEL_ARTIFACT_DOMAIN);
        hash_field(
            &mut hasher,
            self.provenance.checkpoint_content_id.as_bytes(),
        );
        hash_field(&mut hasher, &self.provenance.checkpoint_sha256);
        hash_field(&mut hasher, self.provenance.pccp_change_id.as_bytes());
        hash_field(&mut hasher, self.provenance.pccp_evidence_id.as_bytes());
        hash_field(
            &mut hasher,
            &[match self.provenance.pccp_status {
                PccpStatus::Candidate => 0,
                PccpStatus::GatePass => 1,
                PccpStatus::Rejected => 2,
            }],
        );
        hash_field(&mut hasher, self.input_contract.content_id().as_bytes());
        ContentId::from_bytes(*hasher.finalize().as_bytes())
    }
}

pub(crate) fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Complete model binding supplied by every neural backend.
///
/// No default exists. Trained model provenance and input semantics travel as
/// one immutable [`TrainedModelArtifact`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendModel<'a> {
    /// Deterministic backend with no trained weights or input contract.
    ///
    /// This variant intentionally skips trained-model semantic validation and
    /// must not describe an executable learned artifact.
    ModelFree(ModelProvenance),
    /// Learned artifact whose provenance and input semantics are inseparable.
    Trained(&'a TrainedModelArtifact),
}

impl<'a> BackendModel<'a> {
    pub const fn trained(artifact: &'a TrainedModelArtifact) -> Self {
        Self::Trained(artifact)
    }

    pub const fn provenance(&self) -> &ModelProvenance {
        match self {
            Self::ModelFree(provenance) => provenance,
            Self::Trained(artifact) => artifact.provenance(),
        }
    }

    pub const fn input_contract(&self) -> Option<&ModelInputContract> {
        match self {
            Self::ModelFree(_) => None,
            Self::Trained(artifact) => Some(artifact.input_contract()),
        }
    }

    pub const fn trained_artifact(&self) -> Option<&TrainedModelArtifact> {
        match self {
            Self::ModelFree(_) => None,
            Self::Trained(artifact) => Some(artifact),
        }
    }
}

/// Typed input/output envelope enforced before and after neural inference.
///
/// Resource bounds on the outer shell remain independent. This contract states
/// what the backend can interpret; [`crate::shell::LmqResourceBounds`] states
/// what one invocation may allocate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NeuralBackendCapabilities {
    pub target: BackendTarget,
    pub signal_domain: SignalDomain,
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
    SignalDomain,
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

    pub fn validate_signal(
        self,
        signal: &NeuralSignal,
        sample_rate: Rational,
    ) -> Result<(), BackendCapabilityError> {
        if signal.domain != self.signal_domain {
            return Err(BackendCapabilityError::SignalDomain);
        }
        let channels = u16::try_from(signal.channels.len())
            .map_err(|_| BackendCapabilityError::ChannelCount)?;
        let samples = signal.channels.first().map_or(0, Vec::len);
        let samples = u32::try_from(samples).map_err(|_| BackendCapabilityError::SampleCount)?;
        if signal
            .channels
            .iter()
            .any(|channel| u32::try_from(channel.len()) != Ok(samples))
        {
            return Err(BackendCapabilityError::SampleCount);
        }
        self.validate_input(channels, samples, sample_rate)
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

/// Compare non-negative rationals without cross multiplication. Euclidean
/// quotients and strictly decreasing remainders keep every operand bounded by
/// the original numerator and denominator values.
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

/// Rectangular signal in one backend-declared numeric domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeuralSignal {
    pub domain: SignalDomain,
    pub channels: Vec<Vec<i64>>,
}

impl NeuralSignal {
    pub fn digital(channels: Vec<Vec<i64>>) -> Self {
        Self {
            domain: SignalDomain::DigitalInteger,
            channels,
        }
    }

    pub fn physical_microvolt_q16(channels: Vec<Vec<i64>>) -> Self {
        Self {
            domain: SignalDomain::PhysicalMicrovoltQ16,
            channels,
        }
    }
}

/// Neural tokens + shape/model metadata needed by shell.
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

    /// Complete immutable model binding beyond rectangular shape.
    ///
    /// Required rather than defaulted: trained implementations cannot supply
    /// checkpoint provenance independently from their input contract.
    fn model(&self) -> BackendModel<'_>;

    /// Encode a rectangular signal in the exact domain declared by
    /// [`NeuralBackendCapabilities::signal_domain`].
    fn encode(
        &self,
        signal: &NeuralSignal,
        sample_rate: Rational,
    ) -> Result<NeuralTokens, BackendError>;

    /// Reconstruct in the same declared domain. LOSSY:
    /// `decode(encode(x)) ≈ x`, never `== x`.
    fn decode(&self, tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError>;
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
            signal_domain: SignalDomain::DigitalInteger,
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

    fn model(&self) -> BackendModel<'_> {
        BackendModel::ModelFree(ModelProvenance {
            checkpoint_content_id: ContentId::from_bytes([0x51; 32]),
            checkpoint_sha256: [0x52; 32],
            pccp_change_id: String::from("LMQ-STUB-REFERENCE"),
            pccp_evidence_id: ContentId::from_bytes([0x53; 32]),
            pccp_status: PccpStatus::Candidate,
        })
    }

    fn encode(
        &self,
        signal: &NeuralSignal,
        sample_rate: Rational,
    ) -> Result<NeuralTokens, BackendError> {
        if signal.channels.is_empty() {
            return Err(BackendError(String::from("stub: empty signal")));
        }
        if !(2..=u16::from(u8::MAX)).contains(&self.alphabet) {
            return Err(BackendError(String::from(
                "stub: alphabet must be in 2..=255",
            )));
        }
        let n_channels = u16::try_from(signal.channels.len())
            .map_err(|_| BackendError(String::from("stub: too many channels")))?;
        let n_samples = u32::try_from(signal.channels[0].len())
            .map_err(|_| BackendError(String::from("stub: too many samples")))?;
        if signal
            .channels
            .iter()
            .any(|channel| u32::try_from(channel.len()) != Ok(n_samples))
        {
            return Err(BackendError(String::from("stub: ragged channels")));
        }
        self.capabilities()
            .validate_signal(signal, sample_rate)
            .map_err(|error| BackendError(format!("stub capability mismatch: {error:?}")))?;
        let l = self.alphabet as i64;
        let tokens: Vec<i32> = signal
            .channels
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

    fn decode(&self, t: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
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
        Ok(NeuralSignal::digital(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> ModelProvenance {
        ModelProvenance {
            checkpoint_content_id: ContentId::from_bytes([11; 32]),
            checkpoint_sha256: [12; 32],
            pccp_change_id: String::from("LMQ-TEST-ARTIFACT"),
            pccp_evidence_id: ContentId::from_bytes([13; 32]),
            pccp_status: PccpStatus::GatePass,
        }
    }

    #[test]
    fn model_input_contract_identity_binds_channel_order() {
        let first = ModelInputContract::new(
            ConceptId::new("abir:modality/eeg").unwrap(),
            ["lamquant:test-channel/c0", "lamquant:test-channel/c1"]
                .map(|value| ConceptId::new(value).unwrap())
                .to_vec(),
            ContentId::from_bytes([3; 32]),
            Rational::new(250, 1).unwrap(),
            2_500,
            SignalDomain::PhysicalMicrovoltQ16,
            ConceptId::new("lamquant:operation/model-input-v1").unwrap(),
            ConceptId::new("lamquant:proof/model-input-v1").unwrap(),
            ConceptId::new("lamquant:backend-pipeline/subband-v1").unwrap(),
        )
        .unwrap();
        let mut reversed = first.channel_concepts().to_vec();
        reversed.reverse();
        let second = ModelInputContract::new(
            first.modality().clone(),
            reversed,
            first.model_channel_basis_content_id(),
            first.sample_rate(),
            first.samples(),
            first.signal_domain(),
            first.upstream_derivation().clone(),
            first.upstream_claim_kind().clone(),
            first.backend_pipeline().clone(),
        )
        .unwrap();
        assert_ne!(first.content_id(), second.content_id());
        assert_eq!(
            ModelInputContract::new(
                first.modality().clone(),
                Vec::new(),
                first.model_channel_basis_content_id(),
                first.sample_rate(),
                first.samples(),
                first.signal_domain(),
                first.upstream_derivation().clone(),
                first.upstream_claim_kind().clone(),
                first.backend_pipeline().clone(),
            ),
            Err(ModelInputContractDefinitionError::EmptyChannelBasis)
        );
        assert_eq!(
            first.content_id().to_string(),
            "9eb5c58a26861ce5bbc129b1e3c2e5367307d740b11ae771d1d0a86cc72e0361"
        );
        assert_eq!(
            ModelInputContract::new(
                first.modality().clone(),
                alloc::vec![
                    ConceptId::new("lamquant:test-channel/repeated").unwrap();
                    usize::from(u16::MAX) + 1
                ],
                first.model_channel_basis_content_id(),
                first.sample_rate(),
                first.samples(),
                first.signal_domain(),
                first.upstream_derivation().clone(),
                first.upstream_claim_kind().clone(),
                first.backend_pipeline().clone(),
            ),
            Err(ModelInputContractDefinitionError::TooManyChannels)
        );
    }

    #[test]
    fn stub_encode_decode_is_deterministic_mod_alphabet() {
        let signal = alloc::vec![
            alloc::vec![0i64, 6, 12, -1, 20],
            alloc::vec![3, 3, 3, 8, 100]
        ];
        let b = StubBackend { alphabet: 5 };
        let signal = NeuralSignal::digital(signal);
        let t = b.encode(&signal, Rational::new(250, 1).unwrap()).unwrap();
        assert_eq!(t.n_channels, 2);
        assert_eq!(t.n_samples, 5);
        assert_eq!(t.tokens.len(), 10);
        let recon = b.decode(&t).unwrap();
        // decode(encode(x)) == x mod alphabet.
        let expect: Vec<Vec<i64>> = signal
            .channels
            .iter()
            .map(|ch| ch.iter().map(|&s| s.rem_euclid(5)).collect())
            .collect();
        assert_eq!(recon.domain, SignalDomain::DigitalInteger);
        assert_eq!(recon.channels, expect);
    }

    #[test]
    fn stub_rejects_empty_and_ragged() {
        let b = StubBackend::default();
        assert!(b
            .encode(
                &NeuralSignal::digital(alloc::vec![]),
                Rational::new(250, 1).unwrap()
            )
            .is_err());
        assert!(b
            .encode(
                &NeuralSignal::digital(alloc::vec![alloc::vec![0i64, 1], alloc::vec![0i64]]),
                Rational::new(250, 1).unwrap(),
            )
            .is_err());
        assert!(b
            .encode(
                &NeuralSignal::digital(alloc::vec![alloc::vec![0i64, 1]]),
                Rational::new(-250, 1).unwrap(),
            )
            .is_err());
    }

    #[test]
    fn capabilities_reject_out_of_envelope_values() {
        let caps = NeuralBackendCapabilities {
            target: BackendTarget::HostNative,
            signal_domain: SignalDomain::DigitalInteger,
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
            .encode(
                &NeuralSignal::digital(alloc::vec![alloc::vec![1]]),
                Rational::new(250, 1).unwrap()
            )
            .is_err());
        assert_eq!(backend.capabilities().maximum_alphabet, 255);
    }

    #[test]
    fn backend_capabilities_bind_sample_domain() {
        assert_eq!(
            StubBackend::default().capabilities().signal_domain,
            SignalDomain::DigitalInteger
        );
    }

    #[test]
    fn trained_model_artifact_identity_binds_checkpoint_and_contract() {
        let contract = ModelInputContract::new(
            ConceptId::new("abir:modality/eeg").unwrap(),
            alloc::vec![ConceptId::new("lamquant:test-channel/c0").unwrap()],
            ContentId::from_bytes([3; 32]),
            Rational::new(250, 1).unwrap(),
            2_500,
            SignalDomain::PhysicalMicrovoltQ16,
            ConceptId::new("lamquant:operation/model-input-v1").unwrap(),
            ConceptId::new("lamquant:proof/model-input-v1").unwrap(),
            ConceptId::new("lamquant:backend-pipeline/subband-v1").unwrap(),
        )
        .unwrap();
        let first = TrainedModelArtifact::new(provenance(), contract.clone());
        let mut changed_provenance = provenance();
        changed_provenance.checkpoint_sha256[0] ^= 1;
        let changed_checkpoint = TrainedModelArtifact::new(changed_provenance, contract.clone());
        let changed_contract = TrainedModelArtifact::new(
            provenance(),
            ModelInputContract::new(
                contract.modality().clone(),
                contract.channel_concepts().to_vec(),
                contract.model_channel_basis_content_id(),
                contract.sample_rate(),
                contract.samples(),
                contract.signal_domain(),
                contract.upstream_derivation().clone(),
                contract.upstream_claim_kind().clone(),
                ConceptId::new("lamquant:backend-pipeline/changed-v1").unwrap(),
            )
            .unwrap(),
        );

        assert_ne!(first.content_id(), changed_checkpoint.content_id());
        assert_ne!(first.content_id(), changed_contract.content_id());
        assert_eq!(
            BackendModel::trained(&first).provenance(),
            first.provenance()
        );
        assert_eq!(
            BackendModel::trained(&first).input_contract(),
            Some(first.input_contract())
        );
        assert_eq!(
            first.content_id().to_string(),
            "ff0b52d79f4b0e7b87b5c47f8354fd2c46adf4d7932884e6bae7c08f879128ee"
        );
    }
}
