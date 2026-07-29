use std::collections::BTreeMap;

use crate::{
    lmq_descriptor, lmq_node_config, register_lmq_node, verify_pccp_gate_evidence,
    AbirDatasetValue, LamQuantKernelExecutor, LamQuantNodeValue, LmqAttestedBackend,
    LmqBackendDeploymentManifest, LmqBackendSession, LmqNodeProfile, LmqPccpAuthorizationEntry,
    LmqPccpAuthorizationEpochStore, LmqPccpAuthorizationLedger, NoopTransactionalSink,
    SignedLmqPccpAuthorizationSnapshot, LMQ_CURRENT_PCCP_POLICY, LMQ_MODEL_INPUT_PROOF,
    LMQ_NODE_TYPE,
};
use blut_graph_core::{
    Compiler, ExecutionRealm, Graph, KernelRegistry, NodeId, NodeInstance, PlanExecutor, PortRef,
    Target,
};
use ed25519_dalek::{Signer, SigningKey};
use lamquant_lmq::backend::NeuralBackend as _;
use lamquant_lmq::{backend, shell};
use semantic_abir::ElementType;
use semantic_abir::{
    payload_content_id, Atom, AtomTag, ByteOrder, Calibration, Channel, ChannelBasis,
    ChannelBasisTag, ChannelBasisTerm, ChannelBasisVector, ChannelSpec, ChannelTag, ConceptId,
    ContentId, DatasetDraft, DatasetTag, Derivation, DerivationTag, Layout, ObjectId,
    PayloadDescriptor, Presence, Proof, ProofTag, Rational, Recording, RecordingTag, ReferenceKind,
    SemanticRef, SignalBlock, Stream, StreamTag, TimeAxis, TimeSegment, ValidationLimits,
};
use semantic_abir_bcs::{CodecFidelityKind, ModelProvenance, PccpStatus};

fn fixture_dataset() -> (AbirDatasetValue, Vec<Vec<i64>>) {
    let mut draft = DatasetDraft::new(ObjectId::<DatasetTag>::from_bytes([1; 16]));
    let recording_id = ObjectId::<RecordingTag>::from_bytes([2; 16]);
    let stream_id = ObjectId::<StreamTag>::from_bytes([3; 16]);
    let mut payloads = Vec::new();
    let mut atom_ids = Vec::new();

    let channels = (0..4)
        .map(|channel| {
            (0..500)
                .map(|sample| {
                    let value = i64::from(sample) * 3 + i64::from(channel) * 7;
                    (value * 65_536 / 2) - 2_000_000
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    for (index, channel) in channels.iter().enumerate() {
        let bytes = channel
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let content_id = payload_content_id(ElementType::I64, &bytes);
        payloads.push((content_id, bytes));
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
                    Rational::new(65_536, 1).unwrap(),
                    Rational::new(0, 1).unwrap(),
                    ConceptId::new("ucum:uV").unwrap(),
                )
                .unwrap(),
            ),
        )));
    }

    let basis_id = ObjectId::<ChannelBasisTag>::from_bytes([20; 16]);
    let reference_id = ObjectId::<ChannelTag>::from_bytes([21; 16]);
    draft.add_channel(Channel::new(
        reference_id,
        ConceptId::new("lamquant:test-source/reference").unwrap(),
    ));

    let channel_specs = (0..4)
        .map(|index| {
            let mut source_id = [0_u8; 16];
            source_id[15] = (index + 1) as u8;
            let source_id = ObjectId::<ChannelTag>::from_bytes(source_id);
            draft.add_channel(Channel::new(
                source_id,
                ConceptId::new(format!("lamquant:test-source/{index}")).unwrap(),
            ));
            ChannelSpec::new(ConceptId::new(format!("lamquant:test-channel/{index}")).unwrap())
        })
        .collect::<Vec<_>>();

    let basis_vectors = (0..4)
        .map(|index| {
            let mut source_id = [0_u8; 16];
            source_id[15] = (index + 1) as u8;
            let source_id = ObjectId::<ChannelTag>::from_bytes(source_id);
            ChannelBasisVector::new(vec![
                ChannelBasisTerm::new(source_id, Rational::new(1, 1).unwrap()).unwrap(),
                ChannelBasisTerm::new(reference_id, Rational::new(-1, 1).unwrap()).unwrap(),
            ])
            .unwrap()
        })
        .collect::<Vec<_>>();

    draft.add_channel_basis(
        ChannelBasis::new(basis_id, channel_specs, ReferenceKind::Common)
            .with_construction(basis_vectors)
            .unwrap(),
    );

    draft.add_recording(Recording::new(recording_id, vec![stream_id]));
    draft.add_stream(Stream::new(
        stream_id,
        recording_id,
        ConceptId::new("abir:modality/eeg").unwrap(),
        atom_ids,
        None,
        Some(basis_id),
        None,
    ));

    let derivation_id = ObjectId::<DerivationTag>::from_bytes([22; 16]);
    draft.add_derivation(Derivation::new(
        derivation_id,
        ConceptId::new("lamquant:operation/model-input-v1").unwrap(),
        vec![],
        vec![SemanticRef::of(stream_id)],
    ));
    let proof_bytes = b"validated model-input proof".to_vec();
    let proof_content_id = payload_content_id(ElementType::Bytes, &proof_bytes);
    payloads.push((proof_content_id, proof_bytes));
    draft.add_proof(Proof::new(
        ObjectId::<ProofTag>::from_bytes([23; 16]),
        ConceptId::new("lamquant:proof/model-input-v1").unwrap(),
        SemanticRef::of(derivation_id),
        proof_content_id,
    ));

    let dataset = draft.validate(ValidationLimits::default()).unwrap();
    (
        AbirDatasetValue::try_new(dataset, payloads, 256 * 1024 * 1024).unwrap(),
        channels,
    )
}

fn test_contract(dataset: &semantic_abir::AbirDataset) -> backend::ModelInputContract {
    backend::ModelInputContract::new(
        ConceptId::new("abir:modality/eeg").unwrap(),
        vec![
            ConceptId::new("lamquant:test-channel/0").unwrap(),
            ConceptId::new("lamquant:test-channel/1").unwrap(),
            ConceptId::new("lamquant:test-channel/2").unwrap(),
            ConceptId::new("lamquant:test-channel/3").unwrap(),
        ],
        shell::model_channel_basis_content_id(dataset).unwrap(),
        Rational::new(250, 1).unwrap(),
        500,
        backend::SignalDomain::PhysicalMicrovoltQ16,
        ConceptId::new("lamquant:operation/model-input-v1").unwrap(),
        ConceptId::new("lamquant:proof/model-input-v1").unwrap(),
        ConceptId::new("lamquant:backend-pipeline/subband-v1").unwrap(),
    )
    .unwrap()
}

struct TestBackend {
    artifact: backend::TrainedModelArtifact,
    encoded: std::cell::Cell<usize>,
    fail_encode: std::cell::Cell<bool>,
    model_reads: std::cell::Cell<usize>,
    capabilities: backend::NeuralBackendCapabilities,
    executable_content_id: std::cell::Cell<ContentId>,
}

impl TestBackend {
    fn new(artifact: backend::TrainedModelArtifact) -> Self {
        let mut capabilities = backend::StubBackend::default().capabilities();
        capabilities.target = backend::BackendTarget::HostNative;
        capabilities.signal_domain = backend::SignalDomain::PhysicalMicrovoltQ16;
        capabilities.minimum_channels = 4;
        capabilities.maximum_channels = 4;
        capabilities.minimum_samples = 500;
        capabilities.maximum_samples = 500;
        capabilities.minimum_sample_rate = Rational::new(250, 1).unwrap();
        capabilities.maximum_sample_rate = Rational::new(250, 1).unwrap();
        capabilities.maximum_tokens = 2_000;
        capabilities.maximum_schedule_bytes = 500;
        capabilities.maximum_backend_metadata_bytes = 0;
        capabilities.minimum_alphabet = 32;
        capabilities.maximum_alphabet = 32;
        Self {
            artifact,
            encoded: std::cell::Cell::new(0),
            fail_encode: std::cell::Cell::new(false),
            model_reads: std::cell::Cell::new(0),
            capabilities,
            executable_content_id: std::cell::Cell::new(payload_content_id(
                ElementType::Bytes,
                TEST_BACKEND_EXECUTABLE,
            )),
        }
    }

    fn with_capabilities(
        artifact: backend::TrainedModelArtifact,
        capabilities: backend::NeuralBackendCapabilities,
    ) -> Self {
        Self {
            artifact,
            encoded: std::cell::Cell::new(0),
            fail_encode: std::cell::Cell::new(false),
            model_reads: std::cell::Cell::new(0),
            capabilities,
            executable_content_id: std::cell::Cell::new(payload_content_id(
                ElementType::Bytes,
                TEST_BACKEND_EXECUTABLE,
            )),
        }
    }

    fn with_executable_content_id(self, executable_content_id: ContentId) -> Self {
        self.executable_content_id.set(executable_content_id);
        self
    }

    fn set_executable_content_id(&self, executable_content_id: ContentId) {
        self.executable_content_id.set(executable_content_id);
    }

    fn encoded_calls(&self) -> usize {
        self.encoded.get()
    }

    fn model_reads(&self) -> usize {
        self.model_reads.get()
    }

    fn set_fail_encode(&self, fail: bool) {
        self.fail_encode.set(fail);
    }
}

impl backend::NeuralBackend for TestBackend {
    fn capabilities(&self) -> backend::NeuralBackendCapabilities {
        self.capabilities
    }

    fn model(&self) -> backend::BackendModel<'_> {
        self.model_reads.set(self.model_reads.get() + 1);
        backend::BackendModel::trained(&self.artifact)
    }

    fn encode(
        &self,
        signal: &backend::NeuralSignal,
        sample_rate: Rational,
    ) -> Result<backend::NeuralTokens, backend::BackendError> {
        self.encoded.set(self.encoded.get() + 1);
        if self.fail_encode.get() {
            return Err(backend::BackendError::new(
                backend::BackendErrorKind::Model,
                "injected backend failure",
            ));
        }
        self.capabilities()
            .validate_signal(signal, sample_rate)
            .map_err(|error| {
                backend::BackendError::new(
                    backend::BackendErrorKind::Capability,
                    format!("backend capability mismatch: {error:?}"),
                )
            })?;
        let tokens = signal
            .channels
            .iter()
            .flat_map(|channel| channel.iter().map(|sample| (sample.rem_euclid(32)) as i32))
            .collect::<Vec<_>>();
        let n_channels = u16::try_from(signal.channels.len()).map_err(|_| {
            backend::BackendError::new(
                backend::BackendErrorKind::ResourceLimit,
                "channel count overflow",
            )
        })?;
        let n_samples = u32::try_from(signal.channels[0].len()).map_err(|_| {
            backend::BackendError::new(
                backend::BackendErrorKind::ResourceLimit,
                "sample count overflow",
            )
        })?;
        Ok(backend::NeuralTokens {
            tokens,
            schedule: vec![3; n_samples as usize],
            alphabet: 32,
            n_channels,
            n_samples,
            backend_meta: vec![],
        })
    }

    fn decode(
        &self,
        tokens: &backend::NeuralTokens,
    ) -> Result<backend::NeuralSignal, backend::BackendError> {
        let n_channels = usize::from(tokens.n_channels);
        let n_samples = usize::try_from(tokens.n_samples).map_err(|_| {
            backend::BackendError::new(
                backend::BackendErrorKind::ResourceLimit,
                "sample count overflow",
            )
        })?;
        self.capabilities()
            .validate_output(tokens)
            .map_err(|error| {
                backend::BackendError::new(
                    backend::BackendErrorKind::Capability,
                    format!("backend capability mismatch: {error:?}"),
                )
            })?;
        if tokens.tokens.len() != n_channels * n_samples {
            return Err(backend::BackendError::new(
                backend::BackendErrorKind::Capability,
                "token shape mismatch",
            ));
        }
        let channels = (0..n_channels)
            .map(|index| {
                tokens.tokens[index * n_samples..(index + 1) * n_samples]
                    .iter()
                    .map(|token| i64::from(*token))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Ok(backend::NeuralSignal::physical_microvolt_q16(channels))
    }
}

impl LmqAttestedBackend for TestBackend {
    fn executable_content_id(&self) -> ContentId {
        self.executable_content_id.get()
    }
}

fn gate_evidence(
    checkpoint_sha256: [u8; 32],
    change_id: &str,
    floor: Rational,
    measured: Rational,
    passed: bool,
    skipped: bool,
) -> Vec<u8> {
    let decimal = |value: Rational| {
        let (numerator, denominator) = value.parts();
        serde_json::Number::from_f64(numerator as f64 / denominator as f64).unwrap()
    };
    let criteria = [
        "activation_memory_kb",
        "cr_avg",
        "cr_worst",
        "latency_rp2350_ms",
        "param_count",
        "pearson_r",
        "weight_memory_kb",
    ]
    .into_iter()
    .map(|name| {
        let value = if name == "pearson_r" {
            decimal(measured)
        } else {
            serde_json::Number::from(1)
        };
        serde_json::json!({
            "kind": "acceptance",
            "name": name,
            "floor": if name == "pearson_r" {
                serde_json::Value::Number(decimal(floor))
            } else {
                serde_json::Value::Null
            },
            "measured": value,
            "passed": passed,
            "skipped": skipped,
        })
    })
    .collect::<Vec<_>>();
    serde_json::to_vec(&serde_json::json!({
        "model": "encoder",
        "change_id": change_id,
        "candidate_sha256": checkpoint_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        "registry_sha256": "a".repeat(64),
        "passed": passed,
        "measurements": {
            "pearson_r": decimal(measured),
        },
        "criteria": criteria,
    }))
    .unwrap()
}

fn build_profile(floor: Rational, model_contract: &backend::ModelInputContract) -> LmqNodeProfile {
    let checkpoint_sha256 = [0x56; 32];
    let change_id = "LMQ-TEST-CHANGE";
    let evidence = gate_evidence(
        checkpoint_sha256,
        change_id,
        floor,
        Rational::new(99, 100).unwrap(),
        true,
        false,
    );
    let provenance = ModelProvenance {
        checkpoint_content_id: ContentId::from_bytes([0x55; 32]),
        checkpoint_sha256,
        pccp_change_id: change_id.into(),
        pccp_evidence_id: payload_content_id(ElementType::Bytes, &evidence),
        pccp_status: PccpStatus::GatePass,
    };
    let artifact = backend::TrainedModelArtifact::new(provenance, model_contract.clone());
    let verified = verify_pccp_gate_evidence(&evidence, [0xaa; 32]).unwrap();
    let backend = TestBackend::new(artifact);
    let session = test_session(&backend, vec![Target::Host, Target::BlutDurable]);
    LmqNodeProfile::from_session(&session, &verified, shell::LmqResourceBounds::default()).unwrap()
}

const TEST_BACKEND_EXECUTABLE: &[u8] = b"lamquant-test-backend-executable-v1";

fn test_manifest(targets: Vec<Target>) -> LmqBackendDeploymentManifest {
    LmqBackendDeploymentManifest::new(
        payload_content_id(ElementType::Bytes, TEST_BACKEND_EXECUTABLE),
        "lmq-test-build",
        16 * 1024 * 1024,
        3,
        Some("cpu".into()),
        targets,
    )
}

fn test_session<'a>(
    backend: &'a dyn LmqAttestedBackend,
    targets: Vec<Target>,
) -> LmqBackendSession<'a> {
    LmqBackendSession::verify_test(backend, test_manifest(targets), TEST_BACKEND_EXECUTABLE)
        .unwrap()
}

struct TestAuthorizationLedger {
    ledger: LmqPccpAuthorizationLedger,
    root: std::path::PathBuf,
}

impl std::ops::Deref for TestAuthorizationLedger {
    type Target = LmqPccpAuthorizationLedger;

    fn deref(&self) -> &Self::Target {
        &self.ledger
    }
}

impl Drop for TestAuthorizationLedger {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn authorization_ledger(profile: &LmqNodeProfile) -> TestAuthorizationLedger {
    let signing_key = authorization_signing_key();
    let snapshot = signed_authorization_snapshot(
        &signing_key,
        1,
        vec![LmqPccpAuthorizationEntry::from_profile(profile)],
    );
    let (epoch_store, root) = test_epoch_store(&signing_key);
    TestAuthorizationLedger {
        ledger: LmqPccpAuthorizationLedger::open(epoch_store, snapshot).unwrap(),
        root,
    }
}

fn authorization_signing_key() -> SigningKey {
    SigningKey::from_bytes(&[0x91; 32])
}

fn test_epoch_store(
    signing_key: &SigningKey,
) -> (LmqPccpAuthorizationEpochStore, std::path::PathBuf) {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("lmq-pccp-epochs-{}-{unique}", std::process::id()));
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
    }
    #[cfg(not(unix))]
    std::fs::create_dir(&root).unwrap();
    let store = LmqPccpAuthorizationEpochStore::open(&root, signing_key.verifying_key().to_bytes())
        .unwrap();
    (store, root)
}

fn signed_authorization_snapshot(
    signing_key: &SigningKey,
    epoch: u64,
    entries: Vec<LmqPccpAuthorizationEntry>,
) -> SignedLmqPccpAuthorizationSnapshot {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let not_before = now.saturating_sub(1);
    let not_after = now + 60;
    let message = SignedLmqPccpAuthorizationSnapshot::signing_message(
        epoch,
        not_before,
        not_after,
        entries.clone(),
    )
    .unwrap();
    SignedLmqPccpAuthorizationSnapshot::new(
        epoch,
        not_before,
        not_after,
        entries,
        signing_key.sign(&message).to_bytes(),
    )
    .unwrap()
}

fn build_plan(
    profile: &LmqNodeProfile,
) -> (
    Graph,
    PortRef,
    BTreeMap<String, blut_graph_core::ConfigValue>,
) {
    let config = lmq_node_config(profile);
    let descriptor = lmq_descriptor(profile);
    let input = PortRef {
        node: NodeId(0),
        port: "dataset".into(),
    };
    (
        Graph {
            version: 3,
            nodes: vec![NodeInstance {
                id: NodeId(0),
                descriptor: LMQ_NODE_TYPE.to_string(),
                descriptor_version: 1,
                config: config.clone(),
            }],
            edges: vec![],
            feedback: vec![],
            invocation_inputs: vec![input.clone()],
            required_capabilities: descriptor.capabilities,
            required_proofs: vec![LMQ_MODEL_INPUT_PROOF.into()],
            policy: vec![LMQ_CURRENT_PCCP_POLICY.into()],
            minimum_fidelity: u16::MAX - profile.maximum_loss(),
            session: None,
        },
        input,
        config,
    )
}

#[test]
fn lmq_profile_rejects_invalid_pccp_status_evidence_and_floor() {
    let (opened, _channels) = fixture_dataset();
    let model_contract = test_contract(opened.dataset());

    let floor = Rational::new(93, 100).unwrap();
    let checkpoint_sha256 = [0x56; 32];
    let change_id = "LMQ-TEST-CHANGE";
    let evidence = gate_evidence(
        checkpoint_sha256,
        change_id,
        floor,
        Rational::new(99, 100).unwrap(),
        true,
        false,
    );
    let verified = verify_pccp_gate_evidence(&evidence, [0xaa; 32]).unwrap();
    for status in [PccpStatus::Candidate, PccpStatus::Rejected] {
        let backend = TestBackend::new(backend::TrainedModelArtifact::new(
            ModelProvenance {
                checkpoint_content_id: ContentId::from_bytes([0x55; 32]),
                checkpoint_sha256,
                pccp_change_id: change_id.into(),
                pccp_evidence_id: payload_content_id(ElementType::Bytes, &evidence),
                pccp_status: status,
            },
            model_contract.clone(),
        ));
        let session = test_session(&backend, vec![Target::Host, Target::BlutDurable]);
        assert!(LmqNodeProfile::from_session(
            &session,
            &verified,
            shell::LmqResourceBounds::default(),
        )
        .is_err());
    }
    let zero_backend = TestBackend::new(backend::TrainedModelArtifact::new(
        ModelProvenance {
            checkpoint_content_id: ContentId::from_bytes([0; 32]),
            checkpoint_sha256: [0; 32],
            pccp_change_id: String::new(),
            pccp_evidence_id: payload_content_id(ElementType::Bytes, &evidence),
            pccp_status: PccpStatus::GatePass,
        },
        model_contract.clone(),
    ));
    let zero_session = test_session(&zero_backend, vec![Target::Host, Target::BlutDurable]);
    assert!(LmqNodeProfile::from_session(
        &zero_session,
        &verified,
        shell::LmqResourceBounds::default(),
    )
    .is_err());

    let status_profile = build_profile(floor, &model_contract);
    let other_evidence = gate_evidence(
        checkpoint_sha256,
        change_id,
        Rational::new(9, 10).unwrap(),
        Rational::new(99, 100).unwrap(),
        true,
        false,
    );
    let other_verified = verify_pccp_gate_evidence(&other_evidence, [0xaa; 32]).unwrap();
    let mismatch_backend = TestBackend::new(backend::TrainedModelArtifact::new(
        ModelProvenance {
            checkpoint_content_id: status_profile.checkpoint_content_id(),
            checkpoint_sha256: status_profile.checkpoint_sha256(),
            pccp_change_id: status_profile.pccp_change_id().to_string(),
            pccp_evidence_id: status_profile.pccp_evidence_id(),
            pccp_status: PccpStatus::GatePass,
        },
        model_contract.clone(),
    ));
    let mismatch_session = test_session(&mismatch_backend, vec![Target::Host, Target::BlutDurable]);
    assert!(LmqNodeProfile::from_session(
        &mismatch_session,
        &other_verified,
        shell::LmqResourceBounds::default(),
    )
    .is_err());
    assert_eq!(
        status_profile.pccp_evidence_id(),
        payload_content_id(ElementType::Bytes, &evidence)
    );

    for invalid_floor in [
        Rational::new(0, 1).unwrap(),
        Rational::new(1001, 1000).unwrap(),
    ] {
        let invalid_evidence = gate_evidence(
            checkpoint_sha256,
            change_id,
            invalid_floor,
            invalid_floor,
            true,
            false,
        );
        assert!(verify_pccp_gate_evidence(&invalid_evidence, [0xaa; 32]).is_err());
    }

    let skipped_evidence = gate_evidence(
        checkpoint_sha256,
        change_id,
        floor,
        Rational::new(99, 100).unwrap(),
        true,
        true,
    );
    assert!(verify_pccp_gate_evidence(&skipped_evidence, [0xaa; 32]).is_err());

    let below_floor = gate_evidence(
        checkpoint_sha256,
        change_id,
        floor,
        Rational::new(929, 1000).unwrap(),
        true,
        false,
    );
    assert!(verify_pccp_gate_evidence(&below_floor, [0xaa; 32]).is_err());
    assert!(verify_pccp_gate_evidence(&evidence, [0xbb; 32]).is_err());
}

#[test]
fn lmq_pccp_evidence_compares_decimal_lexemes_exactly() {
    let evidence = gate_evidence(
        [0x56; 32],
        "LMQ-EXACT-DECIMAL",
        Rational::new(93, 100).unwrap(),
        Rational::new(93, 100).unwrap(),
        true,
        false,
    );
    let evidence = String::from_utf8(evidence).unwrap();
    let adversarial = evidence.replacen("\"floor\":0.93", "\"floor\":0.93000000000000000001", 1);
    assert_ne!(adversarial, evidence, "fixture must replace Pearson floor");
    assert!(verify_pccp_gate_evidence(adversarial.as_bytes(), [0xaa; 32]).is_err());
}

#[test]
fn lmq_pccp_evidence_rejects_duplicate_object_members() {
    let evidence = gate_evidence(
        [0x56; 32],
        "LMQ-DUPLICATE-KEY",
        Rational::new(93, 100).unwrap(),
        Rational::new(99, 100).unwrap(),
        true,
        false,
    );
    let evidence = String::from_utf8(evidence).unwrap();
    let duplicate_root =
        evidence.replacen("\"passed\":true", "\"passed\":false,\"passed\":true", 1);
    assert_ne!(duplicate_root, evidence, "fixture must duplicate root key");
    assert!(verify_pccp_gate_evidence(duplicate_root.as_bytes(), [0xaa; 32]).is_err());

    let duplicate_nested = evidence.replacen(
        "\"kind\":\"acceptance\"",
        "\"kind\":\"diagnostic\",\"kind\":\"acceptance\"",
        1,
    );
    assert_ne!(
        duplicate_nested, evidence,
        "fixture must duplicate nested key"
    );
    assert!(verify_pccp_gate_evidence(duplicate_nested.as_bytes(), [0xaa; 32]).is_err());
}

#[test]
fn lmq_authorization_ledger_holds_revocation_until_lease_release() {
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    let (dataset, _) = fixture_dataset();
    let contract = test_contract(dataset.dataset());
    let profile = build_profile(Rational::new(93, 100).unwrap(), &contract);
    let signing_key = SigningKey::from_bytes(&[0x92; 32]);
    let initial = signed_authorization_snapshot(
        &signing_key,
        1,
        vec![LmqPccpAuthorizationEntry::from_profile(&profile)],
    );
    let (epoch_store, epoch_root) = test_epoch_store(&signing_key);
    let ledger = Arc::new(LmqPccpAuthorizationLedger::open(epoch_store, initial).unwrap());
    let request = profile.authorization_request();
    let lease = ledger.acquire(&request).expect("matching grant");
    let (sender, receiver) = mpsc::sync_channel(1);
    let revoker = Arc::clone(&ledger);
    let revoked = signed_authorization_snapshot(&signing_key, 2, Vec::new());
    let thread = std::thread::spawn(move || {
        revoker.apply_signed_snapshot(revoked).unwrap();
        sender.send(()).unwrap();
    });
    assert!(
        receiver.recv_timeout(Duration::from_millis(50)).is_err(),
        "revocation must wait while inference lease is held"
    );
    drop(lease);
    receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("revocation completes after inference");
    thread.join().unwrap();
    assert!(ledger.acquire(&request).is_none());
    drop(ledger);
    std::fs::remove_dir_all(epoch_root).unwrap();
}

#[test]
fn lmq_authorization_store_serializes_cross_ledger_revocation() {
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    let (dataset, _) = fixture_dataset();
    let contract = test_contract(dataset.dataset());
    let profile = build_profile(Rational::new(93, 100).unwrap(), &contract);
    let signing_key = SigningKey::from_bytes(&[0x95; 32]);
    let initial = signed_authorization_snapshot(
        &signing_key,
        1,
        vec![LmqPccpAuthorizationEntry::from_profile(&profile)],
    );
    let (first_store, epoch_root) = test_epoch_store(&signing_key);
    let first =
        LmqPccpAuthorizationLedger::open(first_store, initial.clone()).expect("first ledger");
    let second_store =
        LmqPccpAuthorizationEpochStore::open(&epoch_root, signing_key.verifying_key().to_bytes())
            .unwrap();
    let second =
        Arc::new(LmqPccpAuthorizationLedger::open(second_store, initial).expect("second ledger"));
    let lease = first
        .acquire(&profile.authorization_request())
        .expect("matching grant");
    let (sender, receiver) = mpsc::sync_channel(1);
    let revoker = Arc::clone(&second);
    let revoked = signed_authorization_snapshot(&signing_key, 2, Vec::new());
    let thread = std::thread::spawn(move || {
        revoker.apply_signed_snapshot(revoked).unwrap();
        sender.send(()).unwrap();
    });
    assert!(
        receiver.recv_timeout(Duration::from_millis(50)).is_err(),
        "store-wide exclusive revocation must wait for another ledger's shared lease"
    );
    drop(lease);
    receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("cross-ledger revocation completes after inference");
    thread.join().unwrap();
    assert!(first.acquire(&profile.authorization_request()).is_none());
    drop(first);
    drop(second);
    std::fs::remove_dir_all(epoch_root).unwrap();
}

#[test]
fn lmq_authorization_store_recovers_staged_and_missing_audit_records() {
    let (dataset, _) = fixture_dataset();
    let contract = test_contract(dataset.dataset());
    let profile = build_profile(Rational::new(93, 100).unwrap(), &contract);
    let signing_key = SigningKey::from_bytes(&[0x96; 32]);
    let initial = signed_authorization_snapshot(
        &signing_key,
        1,
        vec![LmqPccpAuthorizationEntry::from_profile(&profile)],
    );
    let (store, epoch_root) = test_epoch_store(&signing_key);
    let ledger = LmqPccpAuthorizationLedger::open(store, initial.clone()).expect("initial ledger");
    drop(ledger);

    let audit = epoch_root.join("epoch-00000000000000000001");
    std::fs::remove_file(&audit).unwrap();
    let orphan = epoch_root.join(".staging-interrupted-record");
    std::fs::write(&orphan, b"partial").unwrap();

    let recovered_store =
        LmqPccpAuthorizationEpochStore::open(&epoch_root, signing_key.verifying_key().to_bytes())
            .expect("store recovery");
    assert!(!orphan.exists(), "orphan staging file must be removed");
    assert!(
        audit.is_file(),
        "head record must restore missing audit link"
    );
    let recovered =
        LmqPccpAuthorizationLedger::open(recovered_store, initial).expect("recovered ledger");
    assert!(recovered
        .acquire(&profile.authorization_request())
        .is_some());
    drop(recovered);
    std::fs::remove_dir_all(epoch_root).unwrap();
}

#[cfg(unix)]
#[test]
fn lmq_authorization_store_files_ignore_permissive_umask() {
    const CHILD_ENV: &str = "LAMQUANT_AUTHORIZATION_UMASK_CHILD";
    if std::env::var_os(CHILD_ENV).is_some() {
        use std::os::unix::fs::PermissionsExt;

        let (dataset, _) = fixture_dataset();
        let contract = test_contract(dataset.dataset());
        let profile = build_profile(Rational::new(93, 100).unwrap(), &contract);
        let signing_key = SigningKey::from_bytes(&[0x97; 32]);
        let initial = signed_authorization_snapshot(
            &signing_key,
            1,
            vec![LmqPccpAuthorizationEntry::from_profile(&profile)],
        );
        let (store, epoch_root) = test_epoch_store(&signing_key);
        let ledger = LmqPccpAuthorizationLedger::open(store, initial).unwrap();
        for entry in std::fs::read_dir(&epoch_root).unwrap() {
            let entry = entry.unwrap();
            let metadata = entry.metadata().unwrap();
            assert_eq!(
                metadata.permissions().mode() & 0o777,
                0o600,
                "{} must be owner-only",
                entry.path().display()
            );
        }
        drop(ledger);
        let head = epoch_root.join("current");
        std::fs::set_permissions(&head, std::fs::Permissions::from_mode(0o660)).unwrap();
        assert!(
            LmqPccpAuthorizationEpochStore::open(
                &epoch_root,
                signing_key.verifying_key().to_bytes()
            )
            .is_err(),
            "group-writable control records must fail closed"
        );
        std::fs::remove_dir_all(epoch_root).unwrap();
        return;
    }

    let output = std::process::Command::new("/bin/sh")
        .arg("-c")
        .arg("umask 000; exec \"$@\"")
        .arg("sh")
        .arg(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("lmq_node::tests::lmq_authorization_store_files_ignore_permissive_umask")
        .arg("--test-threads=1")
        .env(CHILD_ENV, "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "isolated umask test failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[test]
fn lmq_authorization_store_rejects_special_files_without_blocking() {
    use std::os::unix::fs::DirBuilderExt;
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    const CHILD_ENV: &str = "LAMQUANT_AUTHORIZATION_SPECIAL_FILE_CHILD";
    if let Some(case) = std::env::var_os(CHILD_ENV) {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("lmq-pccp-special-{}-{unique}", std::process::id()));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
        let case = case.to_str().unwrap();
        match case {
            "authority-fifo" => {
                assert!(std::process::Command::new("mkfifo")
                    .arg(root.join("authority"))
                    .status()
                    .unwrap()
                    .success());
            }
            "authority-symlink-fifo" => {
                let target = root.join("target-fifo");
                assert!(std::process::Command::new("mkfifo")
                    .arg(&target)
                    .status()
                    .unwrap()
                    .success());
                std::os::unix::fs::symlink(&target, root.join("authority")).unwrap();
            }
            "lock-fifo" => {
                assert!(std::process::Command::new("mkfifo")
                    .arg(root.join(".lock"))
                    .status()
                    .unwrap()
                    .success());
            }
            _ => panic!("unknown special-file case"),
        }
        let signing_key = SigningKey::from_bytes(&[0x98; 32]);
        assert!(
            LmqPccpAuthorizationEpochStore::open(&root, signing_key.verifying_key().to_bytes())
                .is_err(),
            "{case} must fail closed"
        );
        std::fs::remove_dir_all(root).unwrap();
        return;
    }

    for case in ["authority-fifo", "authority-symlink-fifo", "lock-fifo"] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("lmq_node::tests::lmq_authorization_store_rejects_special_files_without_blocking")
            .arg("--test-threads=1")
            .env(CHILD_ENV, case)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "{case} child failed");
                break;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                let _ = child.wait();
                panic!("{case} child blocked during special-file rejection");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[cfg(unix)]
#[test]
fn lmq_authorization_store_pins_root_identity_and_rejects_unprotected_parent() {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    let signing_key = SigningKey::from_bytes(&[0x99; 32]);
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let unsafe_parent = std::env::temp_dir().join(format!(
        "lmq-pccp-unsafe-parent-{}-{unique}",
        std::process::id()
    ));
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&unsafe_parent)
        .unwrap();
    std::fs::set_permissions(&unsafe_parent, std::fs::Permissions::from_mode(0o777)).unwrap();
    let owned_parent = unsafe_parent.join("owned");
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&owned_parent)
        .unwrap();
    let unsafe_root = owned_parent.join("epochs");
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&unsafe_root)
        .unwrap();
    assert!(
        LmqPccpAuthorizationEpochStore::open(&unsafe_root, signing_key.verifying_key().to_bytes())
            .is_err(),
        "non-sticky writable ancestor must be rejected"
    );
    std::fs::remove_dir_all(&unsafe_parent).unwrap();

    let (dataset, _) = fixture_dataset();
    let contract = test_contract(dataset.dataset());
    let profile = build_profile(Rational::new(93, 100).unwrap(), &contract);
    let initial = signed_authorization_snapshot(
        &signing_key,
        1,
        vec![LmqPccpAuthorizationEntry::from_profile(&profile)],
    );
    let (store, epoch_root) = test_epoch_store(&signing_key);
    let ledger = LmqPccpAuthorizationLedger::open(store, initial).unwrap();
    let displaced = epoch_root.with_extension("displaced");
    std::fs::rename(&epoch_root, &displaced).unwrap();
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&epoch_root)
        .unwrap();
    assert!(
        ledger.acquire(&profile.authorization_request()).is_none(),
        "replacement directory must not enter the original lock domain"
    );
    drop(ledger);
    std::fs::remove_dir_all(epoch_root).unwrap();
    std::fs::remove_dir_all(displaced).unwrap();
}

#[test]
fn lmq_authorization_ledger_rejects_forgery_expiry_and_rollback() {
    let (dataset, _) = fixture_dataset();
    let contract = test_contract(dataset.dataset());
    let profile = build_profile(Rational::new(93, 100).unwrap(), &contract);
    let trusted = SigningKey::from_bytes(&[0x93; 32]);
    let attacker = SigningKey::from_bytes(&[0x94; 32]);
    let entries = vec![LmqPccpAuthorizationEntry::from_profile(&profile)];

    let forged = signed_authorization_snapshot(&attacker, 1, entries.clone());
    let (forged_store, forged_root) = test_epoch_store(&trusted);
    assert!(LmqPccpAuthorizationLedger::open(forged_store, forged).is_err());
    std::fs::remove_dir_all(forged_root).unwrap();

    let initial = signed_authorization_snapshot(&trusted, 2, entries.clone());
    let stale_authorization = initial.clone();
    let (epoch_store, epoch_root) = test_epoch_store(&trusted);
    let ledger = LmqPccpAuthorizationLedger::open(epoch_store, initial).unwrap();
    let observer_store =
        LmqPccpAuthorizationEpochStore::open(&epoch_root, trusted.verifying_key().to_bytes())
            .unwrap();
    let observer =
        LmqPccpAuthorizationLedger::open(observer_store, stale_authorization.clone()).unwrap();
    let rollback = signed_authorization_snapshot(&trusted, 1, entries);
    assert_eq!(
        ledger.apply_signed_snapshot(rollback),
        Err(super::LmqPccpAuthorizationLedgerError::SnapshotRollback)
    );
    ledger
        .apply_signed_snapshot(signed_authorization_snapshot(&trusted, 3, Vec::new()))
        .unwrap();
    assert!(
        observer.acquire(&profile.authorization_request()).is_none(),
        "another live ledger must fail closed after durable epoch advances"
    );

    let expired_entries = vec![LmqPccpAuthorizationEntry::from_profile(&profile)];
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let not_before = now.saturating_sub(60);
    let not_after = now.saturating_sub(1);
    let message = SignedLmqPccpAuthorizationSnapshot::signing_message(
        3,
        not_before,
        not_after,
        expired_entries.clone(),
    )
    .unwrap();
    let expired = SignedLmqPccpAuthorizationSnapshot::new(
        3,
        not_before,
        not_after,
        expired_entries,
        trusted.sign(&message).to_bytes(),
    )
    .unwrap();
    let (expired_store, expired_root) = test_epoch_store(&trusted);
    assert!(matches!(
        LmqPccpAuthorizationLedger::open(expired_store, expired),
        Err(super::LmqPccpAuthorizationLedgerError::SnapshotNotCurrent)
    ));
    std::fs::remove_dir_all(expired_root).unwrap();

    drop(observer);
    drop(ledger);
    let reopened_store =
        LmqPccpAuthorizationEpochStore::open(&epoch_root, trusted.verifying_key().to_bytes())
            .unwrap();
    assert!(matches!(
        LmqPccpAuthorizationLedger::open(reopened_store, stale_authorization),
        Err(super::LmqPccpAuthorizationLedgerError::SnapshotRollback)
    ));
    std::fs::remove_dir_all(epoch_root).unwrap();
}

#[test]
fn lmq_deployment_verification_binds_bytes_resources_and_realms() {
    let (opened, _) = fixture_dataset();
    let contract = test_contract(opened.dataset());
    let floor = Rational::new(93, 100).unwrap();
    let checkpoint_sha256 = [0x56; 32];
    let change_id = "LMQ-DEPLOYMENT-TEST";
    let evidence = gate_evidence(
        checkpoint_sha256,
        change_id,
        floor,
        Rational::new(99, 100).unwrap(),
        true,
        false,
    );
    let artifact = backend::TrainedModelArtifact::new(
        ModelProvenance {
            checkpoint_content_id: ContentId::from_bytes([0x55; 32]),
            checkpoint_sha256,
            pccp_change_id: change_id.into(),
            pccp_evidence_id: payload_content_id(ElementType::Bytes, &evidence),
            pccp_status: PccpStatus::GatePass,
        },
        contract,
    );
    let verified = verify_pccp_gate_evidence(&evidence, [0xaa; 32]).unwrap();
    let backend = TestBackend::new(artifact.clone());

    let forged = LmqBackendDeploymentManifest::new(
        ContentId::from_bytes([0x99; 32]),
        "lmq-test-build",
        16 * 1024 * 1024,
        3,
        Some("cpu".into()),
        vec![Target::Host],
    );
    assert!(LmqBackendSession::verify_test(&backend, forged, TEST_BACKEND_EXECUTABLE).is_err());
    let lying_backend = TestBackend::new(artifact.clone())
        .with_executable_content_id(ContentId::from_bytes([0x98; 32]));
    assert!(LmqBackendSession::verify_test(
        &lying_backend,
        test_manifest(vec![Target::Host]),
        TEST_BACKEND_EXECUTABLE,
    )
    .is_err());

    let session = test_session(&backend, vec![Target::Host]);
    let profile =
        LmqNodeProfile::from_session(&session, &verified, shell::LmqResourceBounds::default())
            .unwrap();
    let descriptor = lmq_descriptor(&profile);
    assert_eq!(descriptor.targets, vec![Target::Host]);
    assert_eq!(descriptor.resources.threads, 3);
    assert_eq!(descriptor.resources.device.as_deref(), Some("cpu"));
    assert_eq!(
        descriptor.resources.peak_bytes,
        descriptor.inputs[0].max_bytes
    );

    let (graph, _, _) = build_plan(&profile);
    let mut registry = KernelRegistry::default();
    register_lmq_node(&mut registry, &profile).unwrap();
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();
    let expected_peak = descriptor
        .resources
        .peak_bytes
        .checked_add(descriptor.resources.scratch_bytes)
        .unwrap();
    assert_eq!(plan.as_plan().peak_bytes, expected_peak);
    assert!(Compiler::new(&registry, ExecutionRealm::BlutDurable)
        .compile(&graph)
        .is_err());

    let impossible_frame = shell::LmqResourceBounds {
        bundle: semantic_abir_bcs::ResourceBounds {
            max_frame_bytes: 1,
            ..semantic_abir_bcs::ResourceBounds::default()
        },
        ..shell::LmqResourceBounds::default()
    };
    assert!(LmqNodeProfile::from_session(&session, &verified, impossible_frame).is_err());
    assert_eq!(backend.encoded_calls(), 0);

    let mut alphabet_capabilities = backend.capabilities();
    alphabet_capabilities.minimum_alphabet = 5_000;
    alphabet_capabilities.maximum_alphabet = 5_000;
    let alphabet_backend = TestBackend::with_capabilities(artifact, alphabet_capabilities);
    let alphabet_session = test_session(&alphabet_backend, vec![Target::Host]);
    let alphabet_bounds = shell::LmqResourceBounds {
        max_alphabet: 5_000,
        ..shell::LmqResourceBounds::default()
    };
    assert!(LmqNodeProfile::from_session(&alphabet_session, &verified, alphabet_bounds).is_err());
    assert_eq!(alphabet_backend.encoded_calls(), 0);

    let mut impossible_capabilities = backend.capabilities();
    impossible_capabilities.maximum_tokens = 1;
    impossible_capabilities.maximum_schedule_bytes = 1_024;
    impossible_capabilities.maximum_backend_metadata_bytes = 0;
    let impossible_backend =
        TestBackend::with_capabilities(backend.artifact.clone(), impossible_capabilities);
    let impossible_session = test_session(&impossible_backend, vec![Target::Host]);
    let impossible_packet = shell::LmqResourceBounds {
        bundle: semantic_abir_bcs::ResourceBounds {
            max_frame_bytes: 1_024,
            ..semantic_abir_bcs::ResourceBounds::default()
        },
        max_schedule_bytes: 1_024,
        ..shell::LmqResourceBounds::default()
    };
    assert!(
        LmqNodeProfile::from_session(&impossible_session, &verified, impossible_packet).is_err()
    );
    assert_eq!(impossible_backend.encoded_calls(), 0);
}

#[test]
fn lmq_plan_kernels_differ_between_host_and_blutdurable() {
    let (opened, _channels) = fixture_dataset();
    let contract = test_contract(opened.dataset());
    let profile = build_profile(Rational::new(93, 100).unwrap(), &contract);
    let (graph, _input, _config) = build_plan(&profile);
    let mut registry = KernelRegistry::default();
    register_lmq_node(&mut registry, &profile).unwrap();

    let host_plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();
    let durable_plan = Compiler::new(&registry, ExecutionRealm::BlutDurable)
        .compile(&graph)
        .unwrap();

    let host_step = &host_plan.as_plan().nodes[0];
    let durable_step = &durable_plan.as_plan().nodes[0];
    assert_eq!(host_step.semantic_types, durable_step.semantic_types);
    assert_eq!(
        host_step.determinism,
        blut_graph_core::Determinism::NumericallyEquivalent
    );
    assert_ne!(host_step.kernel, durable_step.kernel);
    assert_ne!(host_step.implementation_id, durable_step.implementation_id);
    assert!(Compiler::new(&registry, ExecutionRealm::McuAot)
        .compile(&graph)
        .is_err());
}

#[test]
fn lmq_execution_rechecks_current_pccp_authorization_before_inference() {
    let (dataset_value, _) = fixture_dataset();
    let contract = test_contract(dataset_value.dataset());
    let profile = build_profile(Rational::new(93, 100).unwrap(), &contract);
    let backend = TestBackend::new(backend::TrainedModelArtifact::new(
        ModelProvenance {
            checkpoint_content_id: profile.checkpoint_content_id(),
            checkpoint_sha256: profile.checkpoint_sha256(),
            pccp_change_id: profile.pccp_change_id().to_string(),
            pccp_evidence_id: profile.pccp_evidence_id(),
            pccp_status: PccpStatus::GatePass,
        },
        contract,
    ));
    let session = test_session(&backend, vec![Target::Host, Target::BlutDurable]);
    let authorizer = authorization_ledger(&profile);
    let mut kernels =
        LamQuantKernelExecutor::with_lmq_session(&session, &authorizer, &profile).unwrap();
    let (graph, input, _) = build_plan(&profile);
    let mut registry = KernelRegistry::default();
    register_lmq_node(&mut registry, &profile).unwrap();
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);

    backend.set_executable_content_id(ContentId::from_bytes([0x97; 32]));
    let (mutated_backend_dataset, _) = fixture_dataset();
    let error = executor
        .execute(
            &plan,
            [0x30; 32],
            BTreeMap::from([(
                input.clone(),
                LamQuantNodeValue::AbirDataset(Box::new(mutated_backend_dataset)),
            )]),
        )
        .unwrap_err();
    match error.error {
        blut_graph_core::ExecutionError::KernelFailed { failure, .. } => {
            assert_eq!(failure.code, "model-binding-mismatch");
        }
        other => panic!("unexpected execution error: {other:?}"),
    }
    assert_eq!(backend.encoded_calls(), 0);
    backend.set_executable_content_id(payload_content_id(
        ElementType::Bytes,
        TEST_BACKEND_EXECUTABLE,
    ));

    executor
        .execute(
            &plan,
            [0x31; 32],
            BTreeMap::from([(
                input.clone(),
                LamQuantNodeValue::AbirDataset(Box::new(dataset_value)),
            )]),
        )
        .unwrap();
    assert_eq!(backend.encoded_calls(), 1);

    backend.set_fail_encode(true);
    let (failed_backend_dataset, _) = fixture_dataset();
    let error = executor
        .execute(
            &plan,
            [0x32; 32],
            BTreeMap::from([(
                input.clone(),
                LamQuantNodeValue::AbirDataset(Box::new(failed_backend_dataset)),
            )]),
        )
        .unwrap_err();
    match error.error {
        blut_graph_core::ExecutionError::KernelFailed { failure, .. } => {
            assert_eq!(failure.code, "backend-model");
        }
        other => panic!("unexpected execution error: {other:?}"),
    }
    backend.set_fail_encode(false);
    assert_eq!(backend.encoded_calls(), 2);

    authorizer
        .apply_signed_snapshot(signed_authorization_snapshot(
            &authorization_signing_key(),
            2,
            Vec::new(),
        ))
        .unwrap();
    let (revoked_dataset, _) = fixture_dataset();
    let error = executor
        .execute(
            &plan,
            [0x33; 32],
            BTreeMap::from([(
                input,
                LamQuantNodeValue::AbirDataset(Box::new(revoked_dataset)),
            )]),
        )
        .unwrap_err();
    match error.error {
        blut_graph_core::ExecutionError::KernelFailed { failure, .. } => {
            assert_eq!(failure.code, "authorization-denied");
        }
        other => panic!("unexpected execution error: {other:?}"),
    }
    let denied = core::cell::Cell::new(false);
    let guarded = super::AuthorizingBackend {
        backend: &backend,
        authorizer: &authorizer,
        request: profile.authorization_request(),
        denied: &denied,
    };
    assert!(guarded
        .decode(&backend::NeuralTokens {
            tokens: vec![],
            schedule: vec![],
            alphabet: 2,
            n_channels: 1,
            n_samples: 1,
            backend_meta: vec![],
        })
        .is_err());
    assert!(denied.get());
    assert_eq!(backend.encoded_calls(), 2);
    assert_eq!(
        backend.model_reads(),
        1,
        "session must snapshot model identity instead of re-querying stateful accessors"
    );
}

#[test]
fn lmq_execution_matches_direct_shell_encode_bundle_bounded() {
    let (dataset_value, _channels) = fixture_dataset();
    let contract = test_contract(dataset_value.dataset());
    let profile = build_profile(Rational::new(93, 100).unwrap(), &contract);
    let (graph, input, _config) = build_plan(&profile);
    let mut registry = KernelRegistry::default();
    register_lmq_node(&mut registry, &profile).unwrap();
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();

    let backend = TestBackend::new(backend::TrainedModelArtifact::new(
        ModelProvenance {
            checkpoint_content_id: profile.checkpoint_content_id(),
            checkpoint_sha256: profile.checkpoint_sha256(),
            pccp_change_id: profile.pccp_change_id().to_string(),
            pccp_evidence_id: profile.pccp_evidence_id(),
            pccp_status: PccpStatus::GatePass,
        },
        contract,
    ));

    let shell_fidelity = profile.fidelity();
    let shell_implementation = profile.implementation();
    let shell_bounds = profile.bounds();
    let expected = shell::encode_bundle_bounded(
        dataset_value.dataset(),
        dataset_value.payloads(),
        &backend,
        shell_fidelity.clone(),
        shell_implementation.clone(),
        shell_bounds,
    )
    .unwrap();

    let session = test_session(&backend, vec![Target::Host, Target::BlutDurable]);
    let authorizer = authorization_ledger(&profile);
    let mut kernels =
        LamQuantKernelExecutor::with_lmq_session(&session, &authorizer, &profile).unwrap();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    let result = executor
        .execute(
            &plan,
            [0_u8; 32],
            BTreeMap::from([(
                input,
                LamQuantNodeValue::AbirDataset(Box::new(dataset_value)),
            )]),
        )
        .unwrap();

    let output = result
        .terminal_values
        .values()
        .next()
        .and_then(|values| values.first())
        .unwrap();
    match output {
        LamQuantNodeValue::Bcs2(bytes) => assert_eq!(bytes, &expected),
        other => panic!("unexpected output: {other:?}"),
    }
}

#[test]
fn lmq_execution_rejects_backend_and_input_ordering_failures() {
    let (dataset_value, _channels) = fixture_dataset();
    let contract = test_contract(dataset_value.dataset());
    let floor = Rational::new(93, 100).unwrap();
    let evidence_a = gate_evidence(
        [2; 32],
        "CHANGE-A",
        floor,
        Rational::new(99, 100).unwrap(),
        true,
        false,
    );
    let evidence_b = gate_evidence(
        [4; 32],
        "CHANGE-B",
        floor,
        Rational::new(99, 100).unwrap(),
        true,
        false,
    );

    let artifact_a = backend::TrainedModelArtifact::new(
        ModelProvenance {
            checkpoint_content_id: ContentId::from_bytes([1; 32]),
            checkpoint_sha256: [2; 32],
            pccp_change_id: String::from("CHANGE-A"),
            pccp_evidence_id: payload_content_id(ElementType::Bytes, &evidence_a),
            pccp_status: PccpStatus::GatePass,
        },
        contract.clone(),
    );
    let artifact_b = backend::TrainedModelArtifact::new(
        ModelProvenance {
            checkpoint_content_id: ContentId::from_bytes([3; 32]),
            checkpoint_sha256: [4; 32],
            pccp_change_id: String::from("CHANGE-B"),
            pccp_evidence_id: payload_content_id(ElementType::Bytes, &evidence_b),
            pccp_status: PccpStatus::GatePass,
        },
        contract,
    );

    let verified_a = verify_pccp_gate_evidence(&evidence_a, [0xaa; 32]).unwrap();
    let verified_b = verify_pccp_gate_evidence(&evidence_b, [0xaa; 32]).unwrap();
    let profile_backend_a = TestBackend::new(artifact_a.clone());
    let profile_session_a =
        test_session(&profile_backend_a, vec![Target::Host, Target::BlutDurable]);
    let profile_a = LmqNodeProfile::from_session(
        &profile_session_a,
        &verified_a,
        shell::LmqResourceBounds::default(),
    )
    .unwrap();
    let profile_backend_b = TestBackend::new(artifact_b);
    let profile_session_b =
        test_session(&profile_backend_b, vec![Target::Host, Target::BlutDurable]);
    let profile_b = LmqNodeProfile::from_session(
        &profile_session_b,
        &verified_b,
        shell::LmqResourceBounds::default(),
    )
    .unwrap();

    let (graph, input, _config) = build_plan(&profile_a);
    let mut registry = KernelRegistry::default();
    register_lmq_node(&mut registry, &profile_a).unwrap();
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();

    let mut no_runtime = LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut exec = PlanExecutor::new(&mut no_runtime, &mut sink);
    assert!(exec
        .execute(
            &plan,
            [1_u8; 32],
            BTreeMap::from([(
                input.clone(),
                LamQuantNodeValue::AbirDataset(Box::new(dataset_value)),
            )]),
        )
        .is_err());

    let backend = TestBackend::new(artifact_a);
    let runtime_session = test_session(&backend, vec![Target::Host, Target::BlutDurable]);
    let authorizer = authorization_ledger(&profile_a);
    assert_eq!(backend.encoded_calls(), 0);
    assert!(
        LamQuantKernelExecutor::with_lmq_session(&runtime_session, &authorizer, &profile_b)
            .is_err()
    );
    assert_eq!(backend.encoded_calls(), 0);

    let alternate_manifest = LmqBackendDeploymentManifest::new(
        payload_content_id(ElementType::Bytes, TEST_BACKEND_EXECUTABLE),
        "alternate-build",
        16 * 1024 * 1024,
        3,
        Some("cpu".into()),
        vec![Target::Host, Target::BlutDurable],
    );
    let alternate_session =
        LmqBackendSession::verify_test(&backend, alternate_manifest, TEST_BACKEND_EXECUTABLE)
            .unwrap();
    let alternate_profile = LmqNodeProfile::from_session(
        &alternate_session,
        &verified_a,
        shell::LmqResourceBounds::default(),
    )
    .unwrap();
    let (alternate_dataset, _) = fixture_dataset();
    let mut alternate_kernels = LamQuantKernelExecutor::with_lmq_session(
        &alternate_session,
        &authorizer,
        &alternate_profile,
    )
    .unwrap();
    let mut alternate_exec = PlanExecutor::new(&mut alternate_kernels, &mut sink);
    assert!(alternate_exec
        .execute(
            &plan,
            [2_u8; 32],
            BTreeMap::from([(
                input.clone(),
                LamQuantNodeValue::AbirDataset(Box::new(alternate_dataset)),
            )]),
        )
        .is_err());
    assert_eq!(backend.encoded_calls(), 0);

    let mut kernels =
        LamQuantKernelExecutor::with_lmq_session(&runtime_session, &authorizer, &profile_a)
            .unwrap();
    let mut exec = PlanExecutor::new(&mut kernels, &mut sink);
    assert_eq!(backend.encoded_calls(), 0);
    assert!(exec
        .execute(
            &plan,
            [1_u8; 32],
            BTreeMap::from([(input, LamQuantNodeValue::Bcs2(vec![]))]),
        )
        .is_err());
    assert_eq!(backend.encoded_calls(), 0);
}

#[test]
fn lmq_descriptor_keeps_corpus_floor_out_of_per_artifact_fidelity() {
    let (dataset_value, _) = fixture_dataset();
    let contract = test_contract(dataset_value.dataset());
    let floor = Rational::new(93, 100).unwrap();
    let profile = build_profile(floor, &contract);
    assert_eq!(profile.pearson_floor(), floor);
    assert_eq!(profile.fidelity().kind, CodecFidelityKind::Transformed);
    assert!(profile.fidelity().bound.is_none());
    assert_eq!(profile.maximum_loss(), u16::MAX);
}
