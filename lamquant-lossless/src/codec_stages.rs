//! Codec pipeline stages — concrete [`Stage`] impls for the lossless
//! compress/decompress operations.
//!
//! Each stage embodies the "one datatype in, one distinct datatype
//! out" discipline from Bible R6 / R8. The newtypes [`EncodedContainer`]
//! and [`DecodedSignal`] keep raw `Vec<u8>` and `Vec<Vec<i64>>` from
//! sliding past each other at chain boundaries — the type system
//! refuses to feed encoded bytes into a decoder that expects signal,
//! and vice versa.
//!
//! Topology:
//!
//! ```text
//! SemanticRead ── CompressStage ──► EncodedContainer
//! EncodedContainer ── DecompressStage ──► DecodedSignal
//! ```
//!
//! `EncodedContainer ≠ Vec<u8>` even though they share representation —
//! the strong type forces callers to be explicit about which kind of
//! bytes they hold. Same for `DecodedSignal ≠ Vec<Vec<i64>>` and validated
//! semantic ABIR.

use crate::container;
use crate::error::{LmlError, LmlResult};
use crate::lpc::LpcMode;
use crate::pipeline::Stage;
use crate::source::SemanticRead;

/// Encoded LML container bytes. Distinct from raw `Vec<u8>` at the
/// type level so a pipeline can't confuse "encoded container" with
/// "raw signal samples that happen to be in a byte buffer".
///
/// Construct via [`CompressStage::process`]; deconstruct with `.0`
/// when handing off to a `Write` sink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedContainer(pub Vec<u8>);

impl EncodedContainer {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<EncodedContainer> for Vec<u8> {
    fn from(c: EncodedContainer) -> Self {
        c.0
    }
}

/// Decoded signal plus raw metadata JSON for the quarantined legacy
/// `Stage`/`Pass` pipeline boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedSignal {
    pub signal: Vec<Vec<i64>>,
    pub metadata_json: String,
}

impl DecodedSignal {
    pub fn n_channels(&self) -> usize {
        self.signal.len()
    }
    pub fn n_samples(&self) -> usize {
        self.signal.first().map(|c| c.len()).unwrap_or(0)
    }
}

// ─── Compress ────────────────────────────────────────────────────────

/// Config-bearing compress stage over one validated semantic source result.
///
/// Sample rate and source metadata come from ABIR. Packet size and kernel mode
/// remain physical execution controls.
#[derive(Debug, Clone)]
pub struct CompressStage {
    pub window_size: usize,
    pub noise_bits: u8,
    pub mode: LpcMode,
}

impl CompressStage {
    /// Adaptive LPC, exact coding, 2500-sample packets.
    pub fn new() -> Self {
        Self {
            window_size: 2500,
            noise_bits: 0,
            mode: LpcMode::default(),
        }
    }

    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    pub fn with_noise_bits(mut self, noise_bits: u8) -> Self {
        self.noise_bits = noise_bits;
        self
    }

    pub fn with_mode(mut self, mode: LpcMode) -> Self {
        self.mode = mode;
        self
    }
}

impl Default for CompressStage {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage for CompressStage {
    type Input = SemanticRead;
    type Output = EncodedContainer;

    fn process(&mut self, read: SemanticRead) -> LmlResult<EncodedContainer> {
        if self.noise_bits != 0 {
            return Err(LmlError::InvalidHeader(
                "the BCS2 LML profile is exact; use a registered lossy profile for noise_bits > 0"
                    .into(),
            ));
        }
        let encoded = container::encode_semantic_read_with_options(
            &read,
            container::LmlEncodeOptions::new(self.window_size).with_lpc_mode(self.mode),
        )?;
        Ok(EncodedContainer(encoded.into_bytes()))
    }
}

// ─── Decompress ──────────────────────────────────────────────────────

/// Stateless decompress stage. `EncodedContainer` → `DecodedSignal`.
///
/// No config — the container header carries everything the decoder
/// needs. Lives as a unit struct so the pipeline still has a name to
/// attach in `.then(DecompressStage)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecompressStage;

impl Stage for DecompressStage {
    type Input = EncodedContainer;
    type Output = DecodedSignal;

    fn process(&mut self, input: EncodedContainer) -> LmlResult<DecodedSignal> {
        let mut cursor = std::io::Cursor::new(input.as_bytes());
        let (signal, metadata) = container::read_from(&mut cursor)?;
        Ok(DecodedSignal {
            signal,
            metadata_json: metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::StageExt;
    use crate::source::{from_owned_uniform_signal, SourceMetadata};

    fn synth_read(n_ch: usize, n_samples: usize) -> LmlResult<SemanticRead> {
        let mut state: u64 = 0x00C0_FFEE_BABE_DEAD;
        let mut signal: Vec<Vec<i64>> = (0..n_ch).map(|_| Vec::with_capacity(n_samples)).collect();
        for ch in &mut signal {
            for _ in 0..n_samples {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ch.push(((state >> 33) as i32) as i64 % 8000);
            }
        }
        from_owned_uniform_signal(
            signal,
            250.0,
            (0..n_ch).map(|i| format!("ch{i}")).collect(),
            vec![-200.0; n_ch],
            vec![200.0; n_ch],
            n_samples as f64 / 250.0,
            SourceMetadata {
                source_file: "synth.edf".into(),
                format: "EDF".into(),
                patient_id: "anon".into(),
                recording_info: String::new(),
                startdate: "2026-05-17".into(),
                phys_dim: "uV".into(),
            },
            vec![],
            semantic_abir::ValidationLimits::default(),
        )
    }

    #[test]
    fn compress_yields_current_abir_bcs2_container() {
        // ADR 0139/0143: the current archive is an authenticated ABIR/BCS2
        // codec bundle. The deterministic LML1 packet is sealed inside it.
        let read = synth_read(2, 256).unwrap();
        let mut stage = CompressStage::new();
        let encoded = stage.process(read).unwrap();
        assert_eq!(&encoded.as_bytes()[0..4], b"ABIR");
    }

    #[test]
    fn decompress_yields_decoded_signal_byte_exact() {
        let read = synth_read(3, 512).unwrap();
        let original = read.native_i64().unwrap().to_vec();
        let encoded = CompressStage::new().process(read).unwrap();
        let mut decompress = DecompressStage;
        let decoded = decompress.process(encoded).unwrap();
        assert_eq!(decoded.signal, original);
    }

    #[test]
    fn compress_then_decompress_chain_round_trips() {
        // The whole point: compose stages, type-checked by the compiler.
        let read = synth_read(4, 384).unwrap();
        let original = read.native_i64().unwrap().to_vec();
        let mut chain = CompressStage::new().then(DecompressStage);
        let decoded = chain.process(read).unwrap();
        assert_eq!(decoded.signal, original);
        let metadata: serde_json::Value =
            serde_json::from_str(&decoded.metadata_json).expect("ABIR metadata projection");
        assert_eq!(metadata["format"], "EDF");
        assert_eq!(metadata["source_file"], "synth.edf");
        assert_eq!(metadata["channels"].as_array().map(Vec::len), Some(4));
    }

    #[test]
    fn compress_refuses_unregistered_lossy_profile() {
        let read = synth_read(1, 128).unwrap();
        let mut stage = CompressStage::new().with_noise_bits(4);
        let error = stage.process(read).expect_err("unregistered lossy profile");
        assert!(error.to_string().contains("registered lossy profile"));
    }

    #[test]
    fn compress_rejects_empty_signal_without_panicking() {
        let error = synth_read(0, 0).expect_err("empty signal must fail closed");
        assert!(error.to_string().contains("at least one channel"));
    }

    #[test]
    fn config_with_builder_pattern() {
        let stage = CompressStage::new()
            .with_window_size(1024)
            .with_noise_bits(2);
        assert_eq!(stage.window_size, 1024);
        assert_eq!(stage.noise_bits, 2);
    }

    #[test]
    fn encoded_container_distinct_from_vec_u8_at_type_level() {
        // Compile-time check: this would fail to build if the newtype
        // wrapper degenerated to a transparent alias.
        let bytes: Vec<u8> = vec![1, 2, 3];
        let _wrapped = EncodedContainer(bytes.clone());
        // Going Vec<u8> → EncodedContainer requires explicit
        // construction; the From<EncodedContainer> for Vec<u8> impl
        // exists but not the reverse — that's the type discipline.
        let _bytes_back: Vec<u8> = _wrapped.into();
    }

    #[test]
    fn decoded_signal_helpers() {
        let s = DecodedSignal {
            signal: vec![vec![1, 2, 3]; 4],
            metadata_json: "{}".into(),
        };
        assert_eq!(s.n_channels(), 4);
        assert_eq!(s.n_samples(), 3);
    }

    #[test]
    fn empty_signal_handled() {
        let s = DecodedSignal {
            signal: vec![],
            metadata_json: "{}".into(),
        };
        assert_eq!(s.n_channels(), 0);
        assert_eq!(s.n_samples(), 0);
    }
}
