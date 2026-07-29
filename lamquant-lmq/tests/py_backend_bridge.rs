//! ADR 0074 Track N — PyBackend subprocess-bridge gate.
//!
//! Two tests:
//!   * `py_backend_selftest_...` — spawns the REAL `python3` helper in its weightless
//!     "selftest" mode and drives a full `AbirDataset → BCS2 LMQ → reconstruction`
//!     round-trip through the subprocess. Proves the bridge + JSON protocol +
//!     backend_meta round-trip WITHOUT any model/weights. Skips only if `python3` is
//!     absent.
//!   * `py_backend_model_...` — the real `SubbandCodec` end-to-end. Optional for
//!     developer runs; `LAMQUANT_LMQ_REQUIRE_MODEL_TEST=1` turns every missing
//!     dependency, checkpoint, or inference failure into a gate failure.

#![cfg(feature = "python")]

use std::path::PathBuf;

use lamquant_lmq::backend::{ModelInputContract, SignalDomain, TrainedModelArtifact};
use lamquant_lmq::py_backend::PyBackend;
use lamquant_lmq::shell;
use semantic_abir::{
    payload_content_id, Atom, AtomTag, ByteOrder, Calibration, Channel, ChannelBasis,
    ChannelBasisTag, ChannelBasisTerm, ChannelBasisVector, ChannelSpec, ChannelTag, ConceptId,
    ContentId, DatasetDraft, DatasetTag, Derivation, DerivationTag, ElementType,
    InMemoryPayloadAccess, Layout, ObjectId, OpenedDataset, PayloadDescriptor, Presence, Proof,
    ProofTag, Rational, Recording, RecordingTag, ReferenceKind, SemanticRef, SignalBlock, Stream,
    StreamTag, TimeAxis, TimeSegment, ValidationLimits,
};
use semantic_abir_bcs::{ModelProvenance, PccpStatus, ResourceBounds, BCS2_MAGIC};

fn helper() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/lmq_infer.py")
}

fn python_available(python: &str) -> bool {
    std::process::Command::new(python)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn model_dependencies_available(python: &str) -> bool {
    std::process::Command::new(python)
        .args(["-c", "import numpy, torch"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn process_containment_available() -> bool {
    cfg!(any(target_os = "linux", windows))
}

fn required_model_test() -> bool {
    std::env::var("LAMQUANT_LMQ_REQUIRE_MODEL_TEST").as_deref() == Ok("1")
}

fn required_model_provenance() -> Result<ModelProvenance, String> {
    let encoded = std::env::var("LAMQUANT_LMQ_CHECKPOINT_SHA256")
        .map_err(|_| "LAMQUANT_LMQ_CHECKPOINT_SHA256 is required".to_owned())?;
    let checkpoint_sha256 = parse_checkpoint_sha256(&encoded)?;
    Ok(ModelProvenance {
        checkpoint_content_id: ContentId::from_bytes([7; 32]),
        checkpoint_sha256,
        pccp_change_id: "LMQ-PY-REAL-MODEL-TEST".to_owned(),
        pccp_evidence_id: ContentId::from_bytes([9; 32]),
        pccp_status: PccpStatus::Candidate,
    })
}

fn parse_checkpoint_sha256(encoded: &str) -> Result<[u8; 32], String> {
    if encoded.len() != 64 || !encoded.is_ascii() {
        return Err("LAMQUANT_LMQ_CHECKPOINT_SHA256 must contain 64 hex digits".to_owned());
    }
    let mut checkpoint_sha256 = [0_u8; 32];
    for (byte, pair) in checkpoint_sha256
        .iter_mut()
        .zip(encoded.as_bytes().chunks_exact(2))
    {
        let pair = std::str::from_utf8(pair)
            .map_err(|_| "LAMQUANT_LMQ_CHECKPOINT_SHA256 is not ASCII".to_owned())?;
        *byte = u8::from_str_radix(pair, 16)
            .map_err(|_| "LAMQUANT_LMQ_CHECKPOINT_SHA256 is not hexadecimal".to_owned())?;
    }
    Ok(checkpoint_sha256)
}

#[test]
fn checkpoint_sha256_parser_rejects_non_ascii_without_panicking() {
    let mut malformed = "0".repeat(61);
    malformed.push('é');
    malformed.push('0');
    assert!(parse_checkpoint_sha256(&malformed).is_err());
}

fn eeg(signal: Vec<Vec<i64>>) -> OpenedDataset<InMemoryPayloadAccess> {
    let mut draft = DatasetDraft::new(ObjectId::<DatasetTag>::from_bytes([1; 16]));
    let recording_id = ObjectId::<RecordingTag>::from_bytes([2; 16]);
    let stream_id = ObjectId::<StreamTag>::from_bytes([3; 16]);
    let mut access = InMemoryPayloadAccess::new();
    let mut atom_ids = Vec::new();
    for (index, channel) in signal.iter().enumerate() {
        let bytes = channel
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let content_id = payload_content_id(ElementType::I64, &bytes);
        access.insert(content_id, bytes);
        let mut id = [0_u8; 16];
        id[15] = (index + 1) as u8;
        let atom_id = ObjectId::<AtomTag>::from_bytes(id);
        atom_ids.push(atom_id);
        draft.add_atom(Atom::SignalBlock(SignalBlock::new(
            atom_id,
            Presence::Present,
            Some(PayloadDescriptor::new(
                content_id,
                (channel.len() * 8) as u64,
                ElementType::I64,
                ByteOrder::Little,
                vec![1, channel.len() as u64],
                Layout::DenseRowMajor,
                None,
                None,
            )),
            TimeAxis::Regular(
                TimeSegment::new(
                    Rational::new(0, 1).unwrap(),
                    Rational::new(250, 1).unwrap(),
                    channel.len() as u64,
                )
                .unwrap(),
            ),
            Some(
                Calibration::new(
                    Rational::new(1, 1).unwrap(),
                    Rational::new(0, 1).unwrap(),
                    ConceptId::new("ucum:uV").unwrap(),
                )
                .unwrap(),
            ),
        )));
    }
    draft.add_recording(Recording::new(recording_id, vec![stream_id]));
    let basis_id = ObjectId::<ChannelBasisTag>::from_bytes([4; 16]);
    let reference_id = ObjectId::<ChannelTag>::from_bytes([5; 16]);
    draft.add_channel(Channel::new(
        reference_id,
        ConceptId::new("lamquant:test-source/reference").unwrap(),
    ));
    let channel_concepts = (0..signal.len())
        .map(|index| ConceptId::new(format!("lamquant:test-channel/{index}")).unwrap())
        .collect::<Vec<_>>();
    let mut vectors = Vec::with_capacity(signal.len());
    for index in 0..signal.len() {
        let mut source_bytes = [6; 16];
        source_bytes[15] = u8::try_from(index + 1).unwrap();
        let source_id = ObjectId::<ChannelTag>::from_bytes(source_bytes);
        draft.add_channel(Channel::new(
            source_id,
            ConceptId::new(format!("lamquant:test-source/{index}")).unwrap(),
        ));
        vectors.push(
            ChannelBasisVector::new(vec![
                ChannelBasisTerm::new(source_id, Rational::new(1, 1).unwrap()).unwrap(),
                ChannelBasisTerm::new(reference_id, Rational::new(-1, 1).unwrap()).unwrap(),
            ])
            .unwrap(),
        );
    }
    draft.add_channel_basis(
        ChannelBasis::new(
            basis_id,
            channel_concepts
                .iter()
                .cloned()
                .map(ChannelSpec::new)
                .collect(),
            ReferenceKind::Common,
        )
        .with_construction(vectors)
        .unwrap(),
    );
    draft.add_stream(Stream::new(
        stream_id,
        recording_id,
        ConceptId::new("abir:modality/eeg").unwrap(),
        atom_ids.clone(),
        None,
        Some(basis_id),
        None,
    ));
    let derivation_id = ObjectId::<DerivationTag>::from_bytes([10; 16]);
    draft.add_derivation(Derivation::new(
        derivation_id,
        ConceptId::new("lamquant:operation/model-input-v1").unwrap(),
        atom_ids.into_iter().map(SemanticRef::of).collect(),
        vec![SemanticRef::of(stream_id)],
    ));
    draft.add_proof(Proof::new(
        ObjectId::<ProofTag>::from_bytes([11; 16]),
        ConceptId::new("lamquant:proof/model-input-v1").unwrap(),
        SemanticRef::of(derivation_id),
        ContentId::from_bytes([12; 32]),
    ));
    OpenedDataset::new(draft.validate(ValidationLimits::default()).unwrap(), access)
}

fn model_input_contract(dataset: &semantic_abir::AbirDataset) -> ModelInputContract {
    let stream = &dataset.streams()[0];
    let samples = dataset
        .atoms()
        .iter()
        .find(|atom| atom.id() == stream.atoms()[0])
        .and_then(Atom::payload)
        .and_then(|payload| payload.shape().last().copied())
        .and_then(|samples| u32::try_from(samples).ok())
        .unwrap();
    let basis = dataset
        .channel_bases()
        .iter()
        .find(|basis| Some(basis.id()) == stream.channel_basis_id())
        .unwrap();
    ModelInputContract::new(
        stream.modality().clone(),
        basis
            .channels()
            .iter()
            .map(|channel| channel.concept().clone())
            .collect(),
        shell::model_channel_basis_content_id(dataset).unwrap(),
        Rational::new(250, 1).unwrap(),
        samples,
        SignalDomain::PhysicalMicrovoltQ16,
        ConceptId::new("lamquant:operation/model-input-v1").unwrap(),
        ConceptId::new("lamquant:proof/model-input-v1").unwrap(),
        ConceptId::new("lamquant:backend-pipeline/subband-v1").unwrap(),
    )
    .unwrap()
}

fn model() -> ModelProvenance {
    ModelProvenance {
        checkpoint_content_id: ContentId::from_bytes([7; 32]),
        checkpoint_sha256: [8; 32],
        pccp_change_id: "LMQ-PY-TEST".to_owned(),
        pccp_evidence_id: ContentId::from_bytes([9; 32]),
        pccp_status: PccpStatus::Candidate,
    }
}

fn reconstructed_signal(opened: &OpenedDataset<InMemoryPayloadAccess>) -> Vec<Vec<i64>> {
    opened.dataset().streams()[0]
        .atoms()
        .iter()
        .map(|atom_id| {
            opened
                .block_view(*atom_id)
                .unwrap()
                .bytes()
                .chunks_exact(8)
                .map(|sample| i64::from_le_bytes(sample.try_into().unwrap()))
                .collect()
        })
        .collect()
}

#[test]
fn py_backend_selftest_round_trips_through_the_subprocess_and_wire() {
    if !process_containment_available() {
        eprintln!("SKIP py_backend_selftest: bounded process containment unavailable");
        return;
    }
    if !python_available("python3") {
        eprintln!("SKIP py_backend_selftest: python3 not available");
        return;
    }
    let sig: Vec<Vec<i64>> = (0..4)
        .map(|c| {
            (0..64)
                .map(|i| ((i * 3 + c * 7) % 40) as i64 - 20)
                .collect()
        })
        .collect();
    let abir = eeg(sig.clone());
    let backend = PyBackend::selftest("python3", helper(), model());

    let bytes = shell::encode_bundle(
        abir.dataset(),
        abir.access(),
        &backend,
        shell::transformed_fidelity("selftest-residue"),
        shell::implementation_identity("python-selftest"),
        ResourceBounds::default(),
    )
    .expect("py selftest encode");
    assert!(bytes.starts_with(&BCS2_MAGIC));

    // decode (spawns python again, selftest dequantize) → the mod-5 residues.
    let decoded = shell::open_bundle(&bytes, &backend, ResourceBounds::default())
        .expect("py selftest decode");
    let got = reconstructed_signal(decoded.reconstructed());
    let expect: Vec<Vec<i64>> = sig
        .iter()
        .map(|ch| ch.iter().map(|&s| s.rem_euclid(5)).collect())
        .collect();
    assert_eq!(got, expect, "selftest wire round-trip == signal mod 5");
    assert_eq!(
        decoded.reconstructed().dataset().streams()[0]
            .modality()
            .as_str(),
        "abir:modality/eeg"
    );
}

#[test]
fn py_backend_model_rejects_invalid_numeric_results() {
    let required = required_model_test();
    let python = std::env::var("LAMQUANT_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    if !python_available(&python) || !model_dependencies_available(&python) {
        assert!(
            !required,
            "required model test needs Python with NumPy and PyTorch"
        );
        eprintln!("SKIP py_backend_nonfinite: model dependencies unavailable");
        return;
    }
    let script = r#"
import importlib.util
import json
import sys
import torch

spec = importlib.util.spec_from_file_location("lmq_infer", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class NonFiniteCodec:
    def encode(self, _signal):
        return torch.full((1, 32, 79), float("nan")), []

module._load_bound_model = lambda _request: (NonFiniteCodec(), "00" * 32)
request = {
    "signal_domain": module.MODEL_DOMAIN,
    "signal": [[0] * 8],
    "expected_checkpoint_sha256": "00" * 32,
}
try:
    module.model_encode(request)
except ValueError as error:
    assert "non-finite" in str(error)
else:
    raise AssertionError("non-finite latent was accepted")

class DecodeMustNotRun:
    def decode(self, _latent, _metadata):
        raise AssertionError("decode ran with non-finite metadata")

module._load_bound_model = lambda _request: (DecodeMustNotRun(), "00" * 32)
metadata = {
    "vmin": float("nan"),
    "vmax": 1.0,
    "shape": [1, 1],
    "metadata": [],
}
request = {
    "signal_domain": module.MODEL_DOMAIN,
    "tokens": [0],
    "alphabet": 32,
    "backend_meta": list(json.dumps(metadata).encode("utf-8")),
}
try:
    module.model_decode(request)
except ValueError as error:
    assert "non-finite" in str(error)
else:
    raise AssertionError("non-finite metadata reached model inference")

request["backend_meta"] = list(
    json.dumps({"__ndarray__": [0], "dtype": "V1048576"}).encode("utf-8")
)
try:
    module.model_decode(request)
except ValueError as error:
    assert "dtype" in str(error)
else:
    raise AssertionError("wire-controlled NumPy dtype was accepted")

class OverflowCodec:
    def decode(self, _latent, _metadata):
        return torch.tensor([[[float(2**47)]]])

module._load_bound_model = lambda _request: (OverflowCodec(), "00" * 32)
metadata = {
    "vmin": 0.0,
    "vmax": 1.0,
    "shape": [1, 1],
    "metadata": [],
}
request["backend_meta"] = list(json.dumps(metadata).encode("utf-8"))
try:
    module.model_decode(request)
except OverflowError:
    pass
else:
    raise AssertionError("Q47.16 value at the i64 upper bound wrapped")
"#;
    let output = std::process::Command::new(&python)
        .args(["-c", script])
        .arg(helper())
        .output()
        .expect("run non-finite model regression");
    assert!(
        output.status.success(),
        "non-finite regression failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn py_backend_model_end_to_end_is_optional_or_required() {
    let required = required_model_test();
    let python = std::env::var("LAMQUANT_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    if !process_containment_available() {
        assert!(
            !required,
            "required model test needs bounded process containment"
        );
        eprintln!("SKIP py_backend_model: bounded process containment unavailable");
        return;
    }
    if !python_available(&python) {
        assert!(!required, "required model test needs Python");
        eprintln!("SKIP py_backend_model: configured Python is not available");
        return;
    }
    let provenance = match required_model_provenance() {
        Ok(provenance) => provenance,
        Err(error) if required => panic!("required model test configuration invalid: {error}"),
        Err(error) => {
            eprintln!("SKIP py_backend_model: {error}");
            return;
        }
    };
    // The real SubbandCodec expects a 21-channel window; a short synthetic window
    // is enough to prove the wire path when the env is present.
    let sig: Vec<Vec<i64>> = (0..21)
        .map(|c| (0..2500).map(|i| ((i + c) % 200) as i64 - 100).collect())
        .collect();
    let abir = eeg(sig);
    let backend = PyBackend::model(
        python,
        helper(),
        TrainedModelArtifact::new(provenance, model_input_contract(abir.dataset())),
    );

    match shell::encode_bundle(
        abir.dataset(),
        abir.access(),
        &backend,
        shell::transformed_fidelity("model-reconstruction"),
        shell::implementation_identity("python-model"),
        ResourceBounds::default(),
    ) {
        Ok(bytes) => {
            assert!(bytes.starts_with(&BCS2_MAGIC));
            let decoded = shell::open_bundle(&bytes, &backend, ResourceBounds::default())
                .expect("model decode with weights present");
            // Honest end-to-end: it produced a valid lossy .lmq and reconstructed a
            // same-shape signal. The R number is reported by the R harness, not here.
            let reconstructed = decoded.reconstructed();
            let signal_atoms = reconstructed.dataset().streams()[0].atoms();
            assert_eq!(signal_atoms.len(), 21);
            assert_eq!(
                reconstructed
                    .block_view(signal_atoms[0])
                    .unwrap()
                    .bytes()
                    .len(),
                2500 * 8
            );
            eprintln!("py_backend_model: end-to-end OK (weights present)");
        }
        Err(e) => {
            assert!(
                !required,
                "required model test failed instead of completing: {e:?}"
            );
            // Optional developer run: environment absent means no model evidence.
            eprintln!("SKIP py_backend_model: environment/weights absent ({e:?})");
        }
    }
}
