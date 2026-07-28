//! ADR 0074 Track N — PyBackend subprocess-bridge gate.
//!
//! Two tests:
//!   * `py_backend_selftest_...` — spawns the REAL `python3` helper in its weightless
//!     "selftest" mode and drives a full `AbirDataset → BCS2 LMQ → reconstruction`
//!     round-trip through the subprocess. Proves the bridge + JSON protocol +
//!     backend_meta round-trip WITHOUT any model/weights. Skips only if `python3` is
//!     absent.
//!   * `py_backend_model_...` — the real `SubbandCodec` end-to-end. ENV-GATED: it
//!     skips when codec-neural / weights are absent, exactly like the SNN PCCP
//!     gates. When weights are present it produces a real lossy round-trip.
//!
//! "Env-gated" is not the same as "cannot fail", and the second test used to
//! conflate them: it caught every `Err` from `encode_bundle` and called it a
//! missing environment. Only `LmqError::Backend` can mean that — the weights
//! live behind the subprocess — so every other variant now fails the test. Read
//! `SKIP` in this file as "the model was not exercised", never as "the model
//! was exercised and was fine".

#![cfg(feature = "python")]

use std::path::PathBuf;

use lamquant_lmq::py_backend::PyBackend;
use lamquant_lmq::shell;
use semantic_abir::{
    payload_content_id, Atom, AtomTag, ByteOrder, ConceptId, ContentId, DatasetDraft, DatasetTag,
    ElementType, InMemoryPayloadAccess, Layout, ObjectId, OpenedDataset, PayloadDescriptor,
    Presence, Rational, Recording, RecordingTag, SignalBlock, Stream, StreamTag, TimeAxis,
    TimeSegment, ValidationLimits,
};
use semantic_abir_bcs::{ModelProvenance, PccpStatus, ResourceBounds, BCS2_MAGIC};

fn helper() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/lmq_infer.py")
}

/// Whether an encode failure is the environment's absence or our own defect.
///
/// Torch, the `codec-neural` tree and the weight files all sit behind the
/// helper subprocess, so their absence reaches this crate only as
/// `LmqError::Backend`. Every other variant is the shell mis-assembling a
/// bundle, which no missing file can cause.
///
/// This is a function rather than a `match` arm inside the test so that the
/// classification can be checked directly. The forgiving branch is the one that
/// runs in CI (the weights are not there), so the strict branch would otherwise
/// be code that never executes and never gets verified.
fn is_absent_environment(error: &shell::LmqError) -> bool {
    matches!(error, shell::LmqError::Backend(_))
}

fn python3_available() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
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
            None,
        )));
    }
    draft.add_recording(Recording::new(recording_id, vec![stream_id]));
    draft.add_stream(Stream::new(
        stream_id,
        recording_id,
        ConceptId::new("abir:modality/eeg").unwrap(),
        atom_ids,
        None,
        None,
        None,
    ));
    OpenedDataset::new(draft.validate(ValidationLimits::default()).unwrap(), access)
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
    if !python3_available() {
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
fn py_backend_model_end_to_end_is_env_gated() {
    if !python3_available() {
        eprintln!("SKIP py_backend_model: python3 not available");
        return;
    }
    // The real SubbandCodec expects a 21-channel window; a short synthetic window
    // is enough to prove the wire path when the env is present.
    let sig: Vec<Vec<i64>> = (0..21)
        .map(|c| (0..2500).map(|i| ((i + c) % 200) as i64 - 100).collect())
        .collect();
    let abir = eeg(sig);
    let backend = PyBackend::model("python3", helper(), model());

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
            assert_eq!(reconstructed.dataset().atoms().len(), 21);
            assert_eq!(
                reconstructed
                    .block_view(reconstructed.dataset().atoms()[0].id())
                    .unwrap()
                    .bytes()
                    .len(),
                2500 * 8
            );
            eprintln!("py_backend_model: end-to-end OK (weights present)");
        }
        Err(error) if is_absent_environment(&error) => {
            eprintln!("SKIP py_backend_model: environment/weights absent ({error})");
        }
        Err(other) => {
            // This arm used to be folded into the one above, under a blanket
            // `Err(e) => SKIP`. That made the test incapable of failing: a wire
            // regression, a malformed BCS2 header, a broken semantic closure --
            // each returned `Err`, printed the word SKIP, and passed. It matters
            // most in exactly the environment where the weights are missing,
            // because there the assertions never run and that line was the only
            // thing anyone saw.
            panic!(
                "py_backend_model: encode failed for a reason unrelated to the \
                 environment: {other}"
            );
        }
    }
}

#[test]
fn only_a_backend_failure_counts_as_a_missing_environment() {
    // The negative control for the skip above. Without this, the strict branch
    // is unreachable in every environment that lacks the weights -- which is
    // every environment that reaches the skip -- so a later edit could widen the
    // forgiveness back to everything and no test would notice.
    use lamquant_lmq::backend::BackendError;
    use shell::LmqError;

    assert!(is_absent_environment(&LmqError::Backend(BackendError(
        "ModuleNotFoundError: No module named 'lamquant_neural'".to_string()
    ))));

    for ours in [
        LmqError::CatalogContract,
        LmqError::Header,
        LmqError::BadTokens,
        LmqError::PayloadIdentityMismatch,
        LmqError::SemanticEncoding,
        LmqError::SemanticValidation,
        LmqError::SignalShapeMismatch,
        LmqError::UnsupportedSemantics("whatever the reason"),
    ] {
        assert!(
            !is_absent_environment(&ours),
            "{ours:?} describes this crate mis-assembling a bundle; no missing \
             weight file can produce it, so it must fail rather than skip"
        );
    }
}
