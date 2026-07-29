//! ADR 0074 Track N — the `RustBackend` **scaffold** (N3).
//!
//! The deployable / MCU endgame: a fully-Rust neural forward pass with embedded
//! ternary weights (no Python at runtime). Wired behind [`NeuralBackend`] NOW so
//! the swap is a drop-in with **no wire change**; the forward pass itself is the
//! deferred, multi-week port and is intentionally not implemented here — it
//! returns a clear deferred error rather than panicking (a shipped codec must not
//! `unimplemented!()`).
//!
//! Deferred (N3, weeks): the packed-ternary conv1d encoder + INT8 bottleneck +
//! 32×32 rotation + int16 CDF-LUT + `ScalarFSQ`, the `MambaSNN` selective-scan
//! (~57K params — the delicate softplus/exp/cumsum), and Vocos via a Rust ML
//! runtime (`candle`/`tract`) or the STM32N6 NPU. Weights come from the C-export
//! packed-ternary + Q31-alpha layout via a `.ckpt→pack` converter, registry-SHA
//! pinned, `include_bytes!`'d into [`EmbeddedWeights`].

use alloc::string::String;
use semantic_abir::Rational;

use crate::backend::{
    BackendError, BackendErrorKind, BackendModel, BackendTarget, NeuralBackend,
    NeuralBackendCapabilities, NeuralSignal, NeuralTokens, SignalDomain, TrainedModelArtifact,
};

/// Registry-SHA-pinned ternary weights baked into the binary. Placeholder: the
/// real packed-ternary + Q31-alpha blob is `include_bytes!`'d here once the model
/// architecture freezes (until then a Rust engine would lock to a moving target).
#[derive(Debug, Default)]
pub struct EmbeddedWeights {
    // Future: `pub packed_ternary: &'static [u8], pub alphas_q31: &'static [i32], …`
    _private: (),
}

impl EmbeddedWeights {
    /// The weights baked into this binary. Deferred → an empty placeholder.
    pub const fn embedded() -> Self {
        Self { _private: () }
    }
}

/// The fully-Rust neural backend (ADR 0074 N3). **Scaffold only** — the forward
/// pass is deferred; it is wired behind [`NeuralBackend`] so it drops in with no
/// wire change when the architecture freezes. Use `PyBackend` (N2) until then.
#[derive(Debug)]
pub struct RustBackend {
    #[allow(dead_code)]
    weights: EmbeddedWeights,
    model: TrainedModelArtifact,
}

impl RustBackend {
    /// Construct with the embedded weights.
    pub fn new(model: TrainedModelArtifact) -> Self {
        Self {
            weights: EmbeddedWeights::embedded(),
            model,
        }
    }
}

const DEFERRED: &str =
    "RustBackend forward-pass port is deferred (ADR 0074 N3) — use PyBackend for now";

impl NeuralBackend for RustBackend {
    fn capabilities(&self) -> NeuralBackendCapabilities {
        NeuralBackendCapabilities {
            target: BackendTarget::McuNative,
            signal_domain: SignalDomain::PhysicalMicrovoltQ16,
            operational: false,
            minimum_channels: 21,
            maximum_channels: 21,
            minimum_samples: 2_500,
            maximum_samples: 2_500,
            minimum_sample_rate: Rational::new(250, 1).expect("valid model sample rate"),
            maximum_sample_rate: Rational::new(250, 1).expect("valid model sample rate"),
            maximum_tokens: 32 * 79,
            maximum_schedule_bytes: 79,
            maximum_backend_metadata_bytes: 16 * 1024 * 1024,
            minimum_alphabet: 32,
            maximum_alphabet: 32,
        }
    }

    fn model(&self) -> BackendModel<'_> {
        BackendModel::trained(&self.model)
    }

    fn encode(
        &self,
        _signal: &NeuralSignal,
        _sample_rate: Rational,
    ) -> Result<NeuralTokens, BackendError> {
        Err(BackendError::new(
            BackendErrorKind::Deferred,
            String::from(DEFERRED),
        ))
    }

    fn decode(&self, _tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
        Err(BackendError::new(
            BackendErrorKind::Deferred,
            String::from(DEFERRED),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ModelInputContract;
    use semantic_abir::{ConceptId, ContentId};
    use semantic_abir_bcs::{ModelProvenance, PccpStatus};

    fn model_input_contract() -> ModelInputContract {
        ModelInputContract::new(
            ConceptId::new("abir:modality/eeg").unwrap(),
            alloc::vec![ConceptId::new("lamquant:test-channel/c0").unwrap()],
            ContentId::from_bytes([4; 32]),
            Rational::new(250, 1).unwrap(),
            2_500,
            SignalDomain::PhysicalMicrovoltQ16,
            ConceptId::new("lamquant:operation/model-input-v1").unwrap(),
            ConceptId::new("lamquant:proof/model-input-v1").unwrap(),
            ConceptId::new("lamquant:backend-pipeline/subband-v1").unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn rust_backend_is_wired_but_deferred() {
        let contract = model_input_contract();
        let model = TrainedModelArtifact::new(
            ModelProvenance {
                checkpoint_content_id: ContentId::from_bytes([0x61; 32]),
                checkpoint_sha256: [0x62; 32],
                pccp_change_id: String::from("LMQ-RUST-DEFERRED"),
                pccp_evidence_id: ContentId::from_bytes([0x63; 32]),
                pccp_status: PccpStatus::Candidate,
            },
            contract.clone(),
        );
        let b = RustBackend::new(model);
        assert_eq!(b.model().input_contract(), Some(&contract));
        // Wired behind the trait (so the swap is a drop-in), but the forward pass
        // returns a clear deferred error — never a panic.
        assert!(b
            .encode(
                &NeuralSignal::physical_microvolt_q16(alloc::vec![alloc::vec![0i64, 1]]),
                Rational::new(250, 1).unwrap()
            )
            .is_err());
        assert!(b
            .decode(&NeuralTokens {
                tokens: alloc::vec![0],
                schedule: alloc::vec![2],
                alphabet: 2,
                n_channels: 1,
                n_samples: 1,
                backend_meta: alloc::vec::Vec::new(),
            })
            .is_err());
    }
}
