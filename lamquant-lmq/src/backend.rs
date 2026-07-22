//! ADR 0074 Track N — the `NeuralBackend` seam.
//!
//! The shell owns everything wire-critical (Rust, stable); a **backend** owns
//! ONLY the neural network. The trait is object-safe (no generic dataset type)
//! so the shell can hold a `&dyn NeuralBackend` and swap a Python backend now for
//! a fully-Rust one later WITHOUT a wire change — the wire is the [`crate::body`]
//! format either way.

use alloc::string::String;
use alloc::vec::Vec;
use semantic_abir::ContentId;
use semantic_abir_bcs::{ModelProvenance, PccpStatus};

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
    /// Immutable identity of the exact checkpoint and PCCP evidence this
    /// backend executes. The shell seals and verifies this value; callers
    /// cannot claim provenance independently of the inference implementation.
    fn model_provenance(&self) -> ModelProvenance;

    /// Encode a modality-blind signal (`[n_channels][n_samples]`, sampled at
    /// `sample_rate` Hz) into tokens.
    fn encode(&self, signal: &[Vec<i64>], sample_rate: f64) -> Result<NeuralTokens, BackendError>;

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
    fn model_provenance(&self) -> ModelProvenance {
        ModelProvenance {
            checkpoint_content_id: ContentId::from_bytes([0x51; 32]),
            checkpoint_sha256: [0x52; 32],
            pccp_change_id: String::from("LMQ-STUB-REFERENCE"),
            pccp_evidence_id: ContentId::from_bytes([0x53; 32]),
            pccp_status: PccpStatus::Candidate,
        }
    }

    fn encode(&self, signal: &[Vec<i64>], _sample_rate: f64) -> Result<NeuralTokens, BackendError> {
        if signal.is_empty() {
            return Err(BackendError(String::from("stub: empty signal")));
        }
        if self.alphabet < 2 {
            return Err(BackendError(String::from("stub: alphabet must be >= 2")));
        }
        let n_channels = signal.len() as u16;
        let n_samples = signal[0].len() as u32;
        if signal.iter().any(|c| c.len() as u32 != n_samples) {
            return Err(BackendError(String::from("stub: ragged channels")));
        }
        let l = self.alphabet as i64;
        let tokens: Vec<i32> = signal
            .iter()
            .flat_map(|ch| ch.iter().map(move |&s| s.rem_euclid(l) as i32))
            .collect();
        // One schedule entry per timestep (stand-in; the shell just carries it).
        let schedule = alloc::vec![self.alphabet as u8; n_samples as usize];
        Ok(NeuralTokens {
            tokens,
            schedule,
            alphabet: self.alphabet,
            n_channels,
            n_samples,
            backend_meta: Vec::new(), // the stub is self-contained; no metadata
        })
    }

    fn decode(&self, t: &NeuralTokens) -> Result<Vec<Vec<i64>>, BackendError> {
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
        let t = b.encode(&signal, 250.0).unwrap();
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
        assert!(b.encode(&[], 250.0).is_err());
        assert!(b
            .encode(&alloc::vec![alloc::vec![0i64, 1], alloc::vec![0i64]], 250.0)
            .is_err());
    }
}
