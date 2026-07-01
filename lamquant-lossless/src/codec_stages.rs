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
//! SignalBundle ── CompressStage ──► EncodedContainer
//! EncodedContainer ── DecompressStage ──► DecodedSignal
//! ```
//!
//! `EncodedContainer ≠ Vec<u8>` even though they share representation —
//! the strong type forces callers to be explicit about which kind of
//! bytes they hold. Same for `DecodedSignal ≠ Vec<Vec<i64>>` and the
//! richer `SignalBundle`.

use crate::container;
use crate::error::LmlResult;
use crate::lpc::LpcMode;
use crate::pipeline::Stage;
use crate::source::SignalBundle;

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

/// Decoded signal + the raw metadata JSON the container carried.
/// Distinct from `SignalBundle` because container-decode never has
/// the source-format sidecar (EDF raw_header etc.) — reconstructing
/// that lives in format-specific stages (e.g. an `EdfReconstruct`
/// stage downstream of `DecompressStage`).
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

/// Config-bearing compress stage. Config lives in the struct;
/// `process` takes one `SignalBundle`, returns one `EncodedContainer`.
///
/// Bible R6: per-call argument carries data only; config (window
/// size, noise bits, LPC mode, metadata) travels with the stage.
#[derive(Debug, Clone)]
pub struct CompressStage {
    pub sample_rate: f64,
    pub window_size: usize,
    pub noise_bits: u8,
    pub mode: LpcMode,
    pub metadata_json: String,
}

impl CompressStage {
    /// Convenience constructor with sensible defaults — adaptive LPC,
    /// lossless (noise_bits=0), 2500-sample windows.
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            window_size: 2500,
            noise_bits: 0,
            mode: LpcMode::default(),
            metadata_json: "{}".into(),
        }
    }

    pub fn with_metadata(mut self, metadata_json: impl Into<String>) -> Self {
        self.metadata_json = metadata_json.into();
        self
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

impl Stage for CompressStage {
    type Input = SignalBundle;
    type Output = EncodedContainer;

    fn process(&mut self, bundle: SignalBundle) -> LmlResult<EncodedContainer> {
        // Validate at the trust boundary — Bible R23.
        bundle.validate()?;
        let mut sink: Vec<u8> = Vec::new();
        let stats = container::write_into(
            &mut sink,
            &bundle.signal,
            self.sample_rate,
            self.window_size,
            self.noise_bits,
            &self.metadata_json,
            self.mode,
        )?;
        debug_assert_eq!(stats.compressed_size, sink.len());
        Ok(EncodedContainer(sink))
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
    use crate::source::SourceMetadata;

    fn synth_bundle(n_ch: usize, n_samples: usize) -> SignalBundle {
        let mut state: u64 = 0xC0FFEE_BABE_DEAD;
        let mut signal: Vec<Vec<i64>> = (0..n_ch).map(|_| Vec::with_capacity(n_samples)).collect();
        for ch in &mut signal {
            for _ in 0..n_samples {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ch.push(((state >> 33) as i32) as i64 % 8000);
            }
        }
        SignalBundle {
            signal,
            sample_rate: 250.0,
            channels: (0..n_ch).map(|i| format!("ch{i}")).collect(),
            phys_min: vec![-200.0; n_ch],
            phys_max: vec![200.0; n_ch],
            duration_s: n_samples as f64 / 250.0,
            metadata: SourceMetadata {
                source_file: "synth.edf".into(),
                format: "EDF".into(),
                patient_id: "anon".into(),
                recording_info: String::new(),
                startdate: "2026-05-17".into(),
                phys_dim: "uV".into(),
            },
            sidecar: vec![],
        }
    }

    #[test]
    fn compress_yields_encoded_container_with_bcs1_magic() {
        // ADR 0069/0071 L9: `CompressStage` routes through `write_abir`,
        // which now emits the `BCS1` typed header (wrapping the
        // byte-unchanged `LML1` per-window payloads inside) instead of a
        // bare `LML1`-prefixed container — see `lamquant_abir::bcs1`.
        let bundle = synth_bundle(2, 256);
        let mut stage = CompressStage::new(250.0);
        let encoded = stage.process(bundle).unwrap();
        assert_eq!(&encoded.as_bytes()[0..4], b"BCS1");
    }

    #[test]
    fn decompress_yields_decoded_signal_byte_exact() {
        let bundle = synth_bundle(3, 512);
        let original = bundle.signal.clone();
        let encoded = CompressStage::new(250.0).process(bundle).unwrap();
        let mut decompress = DecompressStage;
        let decoded = decompress.process(encoded).unwrap();
        assert_eq!(decoded.signal, original);
    }

    #[test]
    fn compress_then_decompress_chain_round_trips() {
        // The whole point: compose stages, type-checked by the compiler.
        let bundle = synth_bundle(4, 384);
        let original = bundle.signal.clone();
        let mut chain = CompressStage::new(250.0)
            .with_metadata("{\"src\":\"unit-test\"}")
            .then(DecompressStage);
        let decoded = chain.process(bundle).unwrap();
        assert_eq!(decoded.signal, original);
        // Metadata is augmented with the self-describing codec_mode stamp
        // (deployment tier); caller fields are preserved.
        assert!(
            decoded.metadata_json.contains("\"src\":\"unit-test\"")
                && decoded.metadata_json.contains("\"codec_mode\""),
            "metadata must preserve caller fields + carry codec_mode: {}",
            decoded.metadata_json
        );
    }

    #[test]
    fn compress_with_noise_bits_lossy() {
        let bundle = synth_bundle(1, 128);
        let mut stage = CompressStage::new(250.0).with_noise_bits(4);
        let encoded = stage.process(bundle).unwrap();
        // Confirm the encoder ran end-to-end through the noise_bits
        // path. Lossy round-trip correctness is pinned in container
        // tests; here we only verify the config threaded through.
        assert!(
            !encoded.is_empty(),
            "noise_bits=4 encoder must still produce non-empty output"
        );
        // ADR 0069/0071 L9: BCS1, not LML1 — see
        // `compress_yields_encoded_container_with_bcs1_magic` above.
        assert_eq!(&encoded.as_bytes()[0..4], b"BCS1");
    }

    #[test]
    fn config_with_builder_pattern() {
        let stage = CompressStage::new(500.0)
            .with_window_size(1024)
            .with_noise_bits(2)
            .with_metadata("{\"k\":1}");
        assert_eq!(stage.sample_rate, 500.0);
        assert_eq!(stage.window_size, 1024);
        assert_eq!(stage.noise_bits, 2);
        assert_eq!(stage.metadata_json, "{\"k\":1}");
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
