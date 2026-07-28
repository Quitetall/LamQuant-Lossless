#![cfg(feature = "standard-adapters")]

use std::collections::BTreeMap;

use abir_adapter::Adapter;
use abir_adapter::{ForeignEntry, ForeignObject, ProfileId};
use blut_graph_core::{
    Compiler, Determinism, Edge, Effect, ExecutionRealm, Graph, KernelRegistry, NodeId,
    NodeInstance, Partiality, PlanExecutor, PortRef, StateScope, Target,
};
use lamquant_nodes::{
    register_standard_nodes, standard_import_descriptor, standard_node_config,
    standard_restore_descriptor, standard_sink_descriptor, LamQuantKernelExecutor,
    LamQuantNodeValue, NoopTransactionalSink, BIDS_IMPORT_NODE_TYPE, BIDS_RESTORE_NODE_TYPE,
    BIDS_SINK_NODE_TYPE, DICOM_IMPORT_NODE_TYPE, DICOM_RESTORE_NODE_TYPE, DICOM_SINK_NODE_TYPE,
    EDFPLUS_IMPORT_NODE_TYPE, EDFPLUS_RESTORE_NODE_TYPE, EDFPLUS_SINK_NODE_TYPE,
    XDF_IMPORT_NODE_TYPE, XDF_RESTORE_NODE_TYPE, XDF_SINK_NODE_TYPE,
};
#[cfg(feature = "standard-nwb")]
use lamquant_nodes::{NWB_IMPORT_NODE_TYPE, NWB_RESTORE_NODE_TYPE, NWB_SINK_NODE_TYPE};
#[cfg(feature = "standard-nwb")]
use lamquant_standard_adapters::NwbAdapter;
use lamquant_standard_adapters::XdfAdapter;
#[cfg(feature = "standard-nwb")]
use lamquant_standard_adapters::{BidsSemanticAdapter, DicomSemanticAdapter, EdfAdapter};
#[cfg(feature = "standard-nwb")]
use semantic_abir::ValidationLimits;
use semantic_abir::{DatasetDraft, DatasetTag, ObjectId, Recording, RecordingTag};
use std::io::Write;

const SOURCE_CAPSULE_PROOF: &str = "org.quitetall.abir.proof.identity-bound-source-capsule-v1";
const TEST_LIMIT: u64 = 64 * 1024 * 1024;
const BOUNDARY_UUID: [u8; 16] = [
    0x43, 0xA5, 0x46, 0xDC, 0xCB, 0xF5, 0x41, 0x0F, 0xB3, 0x0E, 0xD5, 0x46, 0x73, 0x83, 0xCB, 0xE4,
];

struct Case {
    profile: &'static str,
    import_type: &'static str,
    restore_type: &'static str,
    sink_type: &'static str,
    source: ForeignObject,
}

fn entry(path: &str, media_type: &str, bytes: &[u8]) -> ForeignEntry {
    ForeignEntry {
        path: path.to_owned(),
        media_type: Some(media_type.to_owned()),
        bytes: bytes.to_vec(),
    }
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            profile: "edfplus.1",
            import_type: EDFPLUS_IMPORT_NODE_TYPE,
            restore_type: EDFPLUS_RESTORE_NODE_TYPE,
            sink_type: EDFPLUS_SINK_NODE_TYPE,
            source: ForeignObject {
                profile: ProfileId("edfplus.1".to_owned()),
                entries: vec![entry(
                    "recording.edf",
                    "application/edf",
                    include_bytes!(
                        "../../lamquant-standard-adapters/tests/fixtures/bids-full/sub-01/eeg/sub-01_task-rest_eeg.edf"
                    ),
                )],
            },
        },
        Case {
            profile: "bids.1.11.1",
            import_type: BIDS_IMPORT_NODE_TYPE,
            restore_type: BIDS_RESTORE_NODE_TYPE,
            sink_type: BIDS_SINK_NODE_TYPE,
            source: bids_source(),
        },
        Case {
            profile: "dicom.ps3.2026c",
            import_type: DICOM_IMPORT_NODE_TYPE,
            restore_type: DICOM_RESTORE_NODE_TYPE,
            sink_type: DICOM_SINK_NODE_TYPE,
            source: ForeignObject {
                profile: ProfileId("dicom.ps3.2026c".to_owned()),
                entries: vec![entry(
                    "waveform.dcm",
                    "application/dicom",
                    include_bytes!(
                        "../../lamquant-standard-adapters/tests/fixtures/ecg_with_references.dcm"
                    ),
                )],
            },
        },
        #[cfg(feature = "standard-nwb")]
        Case {
            profile: "nwb.2.10.0",
            import_type: NWB_IMPORT_NODE_TYPE,
            restore_type: NWB_RESTORE_NODE_TYPE,
            sink_type: NWB_SINK_NODE_TYPE,
            source: ForeignObject {
                profile: ProfileId("nwb.2.10.0".to_owned()),
                entries: vec![entry(
                    "session.nwb",
                    "application/x-nwb",
                    include_bytes!(
                        "../../lamquant-standard-adapters/tests/fixtures/multi_container_session.nwb"
                    ),
                )],
            },
        },
        Case {
            profile: "xdf.1.0",
            import_type: XDF_IMPORT_NODE_TYPE,
            restore_type: XDF_RESTORE_NODE_TYPE,
            sink_type: XDF_SINK_NODE_TYPE,
            source: ForeignObject {
                profile: ProfileId("xdf.1.0".to_owned()),
                entries: vec![entry(
                    "session.xdf",
                    "application/x-xdf",
                    &xdf_fixture(),
                )],
            },
        },
    ]
}

fn bids_source() -> ForeignObject {
    macro_rules! member {
        ($path:literal, $media:literal) => {
            entry(
                $path,
                $media,
                include_bytes!(concat!(
                    "../../lamquant-standard-adapters/tests/fixtures/bids-full/",
                    $path
                )),
            )
        };
    }
    let mut entries = vec![
        member!("dataset_description.json", "application/json"),
        member!("participants.tsv", "text/tab-separated-values"),
        member!("README", "text/plain"),
        member!("sub-01/eeg/sub-01_task-rest_eeg.edf", "application/edf"),
        member!("sub-01/eeg/sub-01_task-rest_eeg.json", "application/json"),
        member!(
            "sub-01/eeg/sub-01_task-rest_channels.tsv",
            "text/tab-separated-values"
        ),
        member!(
            "sub-01/eeg/sub-01_task-rest_events.tsv",
            "text/tab-separated-values"
        ),
        member!(
            "sub-01/eeg/sub-01_electrodes.tsv",
            "text/tab-separated-values"
        ),
        member!("sub-01/eeg/sub-01_coordsystem.json", "application/json"),
        member!(
            "sub-01/eeg/sub-01_task-rest_physio.tsv.gz",
            "application/gzip"
        ),
        member!(
            "sub-01/eeg/sub-01_task-rest_physio.json",
            "application/json"
        ),
        member!("sub-01/ieeg/sub-01_task-rest_ieeg.edf", "application/edf"),
        member!(
            "derivatives/cleaned/sub-01/eeg/sub-01_task-rest_desc-clean_eeg.edf",
            "application/edf"
        ),
    ];
    let physio = entries
        .iter_mut()
        .find(|entry| entry.path.ends_with("_physio.tsv.gz"))
        .unwrap();
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder
        .write_all(b"0.5\t1.5\n0.7\t1.4\n0.6\t1.6\n")
        .unwrap();
    physio.bytes = encoder.finish().unwrap();
    ForeignObject {
        profile: ProfileId("bids.1.11.1".to_owned()),
        entries,
    }
}

fn xdf_chunk(tag: u16, content: &[u8]) -> Vec<u8> {
    let length = (content.len() + 2) as u64;
    let mut bytes = Vec::new();
    if length < 256 {
        bytes.push(1);
        bytes.push(length as u8);
    } else {
        bytes.push(4);
        bytes.extend_from_slice(&(length as u32).to_le_bytes());
    }
    bytes.extend_from_slice(&tag.to_le_bytes());
    bytes.extend_from_slice(content);
    bytes
}

fn xdf_stream_header(id: u32, xml: &str) -> Vec<u8> {
    let mut content = id.to_le_bytes().to_vec();
    content.extend_from_slice(xml.as_bytes());
    xdf_chunk(2, &content)
}

fn xdf_fixture() -> Vec<u8> {
    let eeg_header = concat!(
        "<?xml version=\"1.0\"?><info><name>BioSemi</name><type>EEG</type>",
        "<channel_count>2</channel_count><nominal_srate>500</nominal_srate>",
        "<channel_format>int16</channel_format><desc><channels>",
        "<channel><label>Fp1</label></channel><channel><label>Fp2</label></channel>",
        "</channels></desc></info>"
    );
    let marker_header = concat!(
        "<?xml version=\"1.0\"?><info><name>Markers</name><type>Markers</type>",
        "<channel_count>1</channel_count><nominal_srate>0</nominal_srate>",
        "<channel_format>string</channel_format></info>"
    );
    let mut file = b"XDF:".to_vec();
    file.extend_from_slice(&xdf_chunk(
        1,
        b"<?xml version=\"1.0\"?><info><version>1.0</version></info>",
    ));
    file.extend_from_slice(&xdf_stream_header(1, eeg_header));
    file.extend_from_slice(&xdf_stream_header(2, marker_header));

    let mut samples = 1_u32.to_le_bytes().to_vec();
    samples.push(1);
    samples.push(4);
    for index in 0..4_i16 {
        samples.push(0);
        samples.extend_from_slice(&(index * 10).to_le_bytes());
        samples.extend_from_slice(&(-index * 10).to_le_bytes());
    }
    file.extend_from_slice(&xdf_chunk(3, &samples));

    let mut markers = 2_u32.to_le_bytes().to_vec();
    markers.push(1);
    markers.push(2);
    for (stamp, text) in [(0.5_f64, "start"), (1.25_f64, "stop")] {
        markers.push(8);
        markers.extend_from_slice(&stamp.to_le_bytes());
        markers.push(1);
        markers.push(text.len() as u8);
        markers.extend_from_slice(text.as_bytes());
    }
    file.extend_from_slice(&xdf_chunk(3, &markers));
    file.extend_from_slice(&xdf_chunk(5, &BOUNDARY_UUID));
    for (id, collection, offset) in [
        (1_u32, 10.0_f64, -0.001_f64),
        (1, 20.0, -0.002),
        (2, 10.0, 0.003),
    ] {
        let mut content = id.to_le_bytes().to_vec();
        content.extend_from_slice(&collection.to_le_bytes());
        content.extend_from_slice(&offset.to_le_bytes());
        file.extend_from_slice(&xdf_chunk(4, &content));
    }
    for id in [1_u32, 2] {
        let mut content = id.to_le_bytes().to_vec();
        content.extend_from_slice(
            b"<?xml version=\"1.0\"?><info><first_timestamp>0</first_timestamp></info>",
        );
        file.extend_from_slice(&xdf_chunk(6, &content));
    }
    file
}

fn graph(type_name: &str, profile: &str, restore: bool) -> Graph {
    let descriptor = if restore {
        standard_restore_descriptor(profile).unwrap()
    } else {
        standard_import_descriptor(profile).unwrap()
    };
    Graph {
        version: 3,
        nodes: vec![NodeInstance {
            id: NodeId(0),
            descriptor: type_name.into(),
            descriptor_version: 1,
            config: standard_node_config(TEST_LIMIT).unwrap(),
        }],
        edges: vec![],
        feedback: vec![],
        invocation_inputs: vec![PortRef {
            node: NodeId(0),
            port: if restore {
                "dataset".into()
            } else {
                "source".into()
            },
        }],
        required_capabilities: descriptor.capabilities,
        required_proofs: if restore {
            vec![format!("{SOURCE_CAPSULE_PROOF}.{profile}")]
        } else {
            vec![]
        },
        policy: vec![],
        minimum_fidelity: u16::MAX,
        session: None,
    }
}

fn sink_graph(sink_type: &str, profile: &str, destination_resource: &str) -> Graph {
    let descriptor = standard_sink_descriptor(profile).unwrap();
    Graph {
        version: 3,
        nodes: vec![NodeInstance {
            id: NodeId(0),
            descriptor: sink_type.into(),
            descriptor_version: 1,
            config: lamquant_nodes::standard_sink_node_config(destination_resource, TEST_LIMIT)
                .unwrap(),
        }],
        edges: vec![],
        feedback: vec![],
        invocation_inputs: vec![PortRef {
            node: NodeId(0),
            port: "source".into(),
        }],
        required_capabilities: descriptor.capabilities,
        required_proofs: vec![],
        policy: vec![],
        minimum_fidelity: u16::MAX,
        session: None,
    }
}

fn run_import(
    registry: &KernelRegistry,
    case: &Case,
) -> (
    lamquant_nodes::AbirDatasetValue,
    abir_adapter::MappingReport,
) {
    let plan = Compiler::new(registry, ExecutionRealm::HostStream)
        .compile(&graph(case.import_type, case.profile, false))
        .unwrap();
    let mut kernels = LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    let mut result = executor
        .execute(
            &plan,
            [0x17; 32],
            BTreeMap::from([(
                PortRef {
                    node: NodeId(0),
                    port: "source".into(),
                },
                LamQuantNodeValue::ForeignObject(case.source.clone()),
            )]),
        )
        .unwrap();
    let mut values = result.terminal_values.remove(&NodeId(0)).unwrap();
    assert_eq!(values.len(), 2);
    let report = match values.pop().unwrap() {
        LamQuantNodeValue::MappingReport(report) => report,
        other => panic!("unexpected import report: {other:?}"),
    };
    let dataset = match values.pop().unwrap() {
        LamQuantNodeValue::AbirDataset(dataset) => *dataset,
        other => panic!("unexpected import dataset: {other:?}"),
    };
    (dataset, report)
}

#[cfg(feature = "standard-nwb")]
fn run_restore(
    registry: &KernelRegistry,
    case: &Case,
    dataset: lamquant_nodes::AbirDatasetValue,
) -> (ForeignObject, abir_adapter::FidelityReceipt) {
    let plan = Compiler::new(registry, ExecutionRealm::HostStream)
        .compile(&graph(case.restore_type, case.profile, true))
        .unwrap();
    let mut kernels = LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    let mut result = executor
        .execute(
            &plan,
            [0x18; 32],
            BTreeMap::from([(
                PortRef {
                    node: NodeId(0),
                    port: "dataset".into(),
                },
                LamQuantNodeValue::AbirDataset(Box::new(dataset)),
            )]),
        )
        .unwrap();
    let mut values = result.terminal_values.remove(&NodeId(0)).unwrap();
    assert_eq!(values.len(), 2);
    let receipt = match values.pop().unwrap() {
        LamQuantNodeValue::FidelityReceipt(receipt) => receipt,
        other => panic!("unexpected restore receipt: {other:?}"),
    };
    let foreign = match values.pop().unwrap() {
        LamQuantNodeValue::ForeignObject(foreign) => foreign,
        other => panic!("unexpected restored source: {other:?}"),
    };
    (foreign, receipt)
}

#[cfg(feature = "standard-nwb")]
fn direct_report(case: &Case) -> abir_adapter::MappingReport {
    let adapter: Box<dyn Adapter> = match case.profile {
        "edfplus.1" => Box::new(EdfAdapter::new(TEST_LIMIT)),
        "bids.1.11.1" => Box::new(BidsSemanticAdapter::new(TEST_LIMIT)),
        "dicom.ps3.2026c" => Box::new(DicomSemanticAdapter::new(TEST_LIMIT)),
        #[cfg(feature = "standard-nwb")]
        "nwb.2.10.0" => Box::new(NwbAdapter::new(TEST_LIMIT)),
        "xdf.1.0" => Box::new(XdfAdapter::new(TEST_LIMIT)),
        _ => unreachable!(),
    };
    adapter
        .import(&case.source, ValidationLimits::default())
        .unwrap()
        .report
}

#[test]
fn import_and_restore_compose_in_one_compiled_graph() {
    let mut registry = KernelRegistry::default();
    register_standard_nodes(&mut registry).unwrap();
    let case = cases().remove(1);
    let descriptor = standard_import_descriptor(case.profile).unwrap();
    let graph = Graph {
        version: 3,
        nodes: vec![
            NodeInstance {
                id: NodeId(0),
                descriptor: case.import_type.into(),
                descriptor_version: 1,
                config: standard_node_config(TEST_LIMIT).unwrap(),
            },
            NodeInstance {
                id: NodeId(1),
                descriptor: case.restore_type.into(),
                descriptor_version: 1,
                config: standard_node_config(TEST_LIMIT).unwrap(),
            },
        ],
        edges: vec![Edge {
            from: PortRef {
                node: NodeId(0),
                port: "dataset".into(),
            },
            to: PortRef {
                node: NodeId(1),
                port: "dataset".into(),
            },
        }],
        feedback: vec![],
        invocation_inputs: vec![PortRef {
            node: NodeId(0),
            port: "source".into(),
        }],
        required_capabilities: descriptor.capabilities,
        required_proofs: vec![],
        policy: vec![],
        minimum_fidelity: u16::MAX,
        session: None,
    };
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();
    assert_eq!(plan.as_plan().nodes.len(), 2);

    let mut cross_profile = graph.clone();
    cross_profile.nodes[1].descriptor = DICOM_RESTORE_NODE_TYPE.into();
    cross_profile.required_capabilities.extend(
        standard_restore_descriptor("dicom.ps3.2026c")
            .unwrap()
            .capabilities,
    );
    let error = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&cross_profile)
        .unwrap_err();
    assert!(
        format!("{error:?}").contains("PortContractMismatch"),
        "cross-profile restore must fail at port proof-contract checking: {error:?}"
    );

    let bids_restore = standard_restore_descriptor("bids.1.11.1").unwrap();
    let dicom_import = standard_import_descriptor("dicom.ps3.2026c").unwrap();
    let restored_foreign_cross_profile = Graph {
        version: 3,
        nodes: vec![
            NodeInstance {
                id: NodeId(0),
                descriptor: BIDS_RESTORE_NODE_TYPE.into(),
                descriptor_version: 1,
                config: standard_node_config(TEST_LIMIT).unwrap(),
            },
            NodeInstance {
                id: NodeId(1),
                descriptor: DICOM_IMPORT_NODE_TYPE.into(),
                descriptor_version: 1,
                config: standard_node_config(TEST_LIMIT).unwrap(),
            },
        ],
        edges: vec![Edge {
            from: PortRef {
                node: NodeId(0),
                port: "source".into(),
            },
            to: PortRef {
                node: NodeId(1),
                port: "source".into(),
            },
        }],
        feedback: vec![],
        invocation_inputs: vec![PortRef {
            node: NodeId(0),
            port: "dataset".into(),
        }],
        required_capabilities: bids_restore
            .capabilities
            .into_iter()
            .chain(dicom_import.capabilities)
            .collect(),
        required_proofs: vec![format!("{SOURCE_CAPSULE_PROOF}.bids.1.11.1")],
        policy: vec![],
        minimum_fidelity: u16::MAX,
        session: None,
    };
    let error = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&restored_foreign_cross_profile)
        .unwrap_err();
    assert!(
        format!("{error:?}").contains("TypeMismatch"),
        "cross-profile foreign-object edge must fail at port checking: {error:?}"
    );

    let mut kernels = LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    let result = executor
        .execute(
            &plan,
            [0x16; 32],
            BTreeMap::from([(
                PortRef {
                    node: NodeId(0),
                    port: "source".into(),
                },
                LamQuantNodeValue::ForeignObject(case.source.clone()),
            )]),
        )
        .unwrap();
    let values = result.terminal_values.get(&NodeId(1)).unwrap();
    let restored = match &values[0] {
        LamQuantNodeValue::ForeignObject(source) => source,
        other => panic!("unexpected composed restore output: {other:?}"),
    };
    let mut actual = restored.entries.clone();
    let mut expected = case.source.entries;
    actual.sort_by(|left, right| left.path.cmp(&right.path));
    expected.sort_by(|left, right| left.path.cmp(&right.path));
    assert_eq!(actual, expected);
}

#[test]
fn dataset_value_rejects_metadata_that_alone_exceeds_retained_limit() {
    let mut draft = DatasetDraft::new(ObjectId::<DatasetTag>::from_bytes([0x42; 16]));
    draft.add_recording(Recording::new(
        ObjectId::<RecordingTag>::from_bytes([0x43; 16]),
        vec![],
    ));
    let dataset = draft
        .validate(semantic_abir::ValidationLimits::default())
        .unwrap();
    assert!(dataset.semantic_metadata_budget_bytes() > 0);
    assert!(matches!(
        lamquant_nodes::AbirDatasetValue::try_new(dataset, core::iter::empty(), 0),
        Err(lamquant_nodes::AbirDatasetValueError::RetainedExtentExceeded)
    ));
}

#[test]
fn dataset_value_rejects_missing_clock_relation_provenance() {
    let outcome = XdfAdapter::new(TEST_LIMIT)
        .import(
            &ForeignObject {
                profile: ProfileId("xdf.1.0".to_owned()),
                entries: vec![entry("session.xdf", "application/x-xdf", &xdf_fixture())],
            },
            semantic_abir::ValidationLimits::default(),
        )
        .unwrap();
    let missing = outcome.dataset.clock_relations()[0].provenance();
    let payloads = outcome
        .payloads
        .into_iter()
        .filter(|payload| payload.content_id != missing)
        .map(|payload| (payload.content_id, payload.bytes));
    assert!(matches!(
        lamquant_nodes::AbirDatasetValue::try_new(outcome.dataset, payloads, TEST_LIMIT),
        Err(lamquant_nodes::AbirDatasetValueError::MissingPayload(content_id))
            if content_id == missing
    ));
}

#[test]
fn registers_all_enabled_profile_specific_host_nodes() {
    let mut registry = KernelRegistry::default();
    register_standard_nodes(&mut registry).unwrap();
    let mut prior_profile_sink_plan = None;
    for case in cases() {
        let import = standard_import_descriptor(case.profile).unwrap();
        let restore = standard_restore_descriptor(case.profile).unwrap();
        let sink = standard_sink_descriptor(case.profile).unwrap();
        assert_eq!(import.type_name, case.import_type);
        assert_eq!(restore.type_name, case.restore_type);
        assert_eq!(sink.type_name, case.sink_type);
        assert_eq!(import.targets, vec![blut_graph_core::Target::Host]);
        assert_eq!(restore.targets, vec![blut_graph_core::Target::Host]);
        assert_eq!(sink.targets, vec![Target::Host]);
        assert_ne!(import.type_name, restore.type_name);
        assert_ne!(sink.type_name, import.type_name);
        assert_eq!(sink.outputs, vec![]);
        assert_eq!(sink.inputs.len(), 1);
        assert_eq!(sink.inputs[0].name, "source");
        assert_eq!(
            sink.inputs[0].semantic_type,
            format!("abir.foreign-object.{}", case.profile)
        );
        assert_eq!(sink.determinism, Determinism::BitExact);
        assert_eq!(sink.retry_limit, 0);
        assert_eq!(sink.effect, Effect::Transactional);
        assert_eq!(sink.partiality, Partiality::Atomic);
        assert_eq!(sink.state.scope, StateScope::Stateless);
        assert_eq!(sink.fidelity.minimum_input, u16::MAX);
        assert_eq!(sink.fidelity.maximum_loss, 0);
        assert!(sink
            .capabilities
            .iter()
            .any(|capability| capability.0 == format!("abir.foreign-tree.{}", case.profile)));
        assert!(sink
            .capabilities
            .iter()
            .any(|capability| capability.0 == "org.quitetall.lamquant.sink.durable-file-v1"));
        let expected_binding = lamquant_nodes::standard_sink_kernel_binding(case.profile).unwrap();
        assert_ne!(
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .compile(&graph(case.import_type, case.profile, false))
                .unwrap()
                .as_plan()
                .plan_id,
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .compile(&graph(case.restore_type, case.profile, true))
                .unwrap()
                .as_plan()
                .plan_id
        );
        assert_ne!(
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .compile(&sink_graph(case.sink_type, case.profile, "archive:profile"))
                .unwrap()
                .as_plan()
                .plan_id,
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .compile(&sink_graph(
                    case.sink_type,
                    case.profile,
                    "archive:profile2"
                ))
                .unwrap()
                .as_plan()
                .plan_id
        );
        let sink_plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&sink_graph(case.sink_type, case.profile, "archive:profile"))
            .unwrap();
        let sink_step = &sink_plan.as_plan().nodes[0];
        assert_eq!(
            (sink_step.kernel, sink_step.implementation_id),
            expected_binding
        );
        let contract = lamquant_nodes::parse_standard_sink_contract(sink_step).unwrap();
        assert_eq!(contract.profile, case.profile);
        assert_eq!(contract.destination_resource, "archive:profile");
        assert_eq!(contract.max_source_bytes, TEST_LIMIT);
        let sink_plan_id = sink_plan.as_plan().plan_id;
        if let Some(prior) = prior_profile_sink_plan {
            assert_ne!(prior, sink_plan_id);
        }
        prior_profile_sink_plan = Some(sink_plan_id);
        assert!(Compiler::new(&registry, ExecutionRealm::McuAot)
            .compile(&graph(case.import_type, case.profile, false))
            .is_err());
        assert!(Compiler::new(&registry, ExecutionRealm::BlutDurable)
            .compile(&graph(case.import_type, case.profile, false))
            .is_err());
        assert!(Compiler::new(&registry, ExecutionRealm::McuAot)
            .compile(&sink_graph(case.sink_type, case.profile, "archive:profile"))
            .is_err());
        assert!(Compiler::new(&registry, ExecutionRealm::BlutDurable)
            .compile(&sink_graph(case.sink_type, case.profile, "archive:profile"))
            .is_err());
    }
}

#[test]
#[cfg(feature = "standard-nwb")]
fn five_node_imports_match_adapters_and_restore_exact_sources() {
    let mut registry = KernelRegistry::default();
    register_standard_nodes(&mut registry).unwrap();
    for case in cases() {
        let direct = direct_report(&case);
        let (dataset, report) = run_import(&registry, &case);
        assert_eq!(report, direct, "profile {}", case.profile);
        assert!(!dataset.dataset().source_capsules().is_empty());
        if let Some(atom) = dataset
            .dataset()
            .atoms()
            .iter()
            .find(|atom| atom.payload().is_some())
        {
            let view = dataset.opened().block_view(atom.id()).unwrap();
            assert!(!view.bytes().is_empty());
        }
        let (restored, receipt) = run_restore(&registry, &case, dataset);
        assert_eq!(
            restored.profile, case.source.profile,
            "profile {}",
            case.profile
        );
        let mut restored_entries = restored.entries;
        let mut source_entries = case.source.entries;
        restored_entries.sort_by(|left, right| left.path.cmp(&right.path));
        source_entries.sort_by(|left, right| left.path.cmp(&right.path));
        assert_eq!(restored_entries, source_entries, "profile {}", case.profile);
        assert!(receipt.exact_source_restoration);
        assert!(!receipt.output_content_ids.is_empty());
    }
}

#[test]
fn resource_limits_and_cross_profile_restore_fail_closed() {
    let mut registry = KernelRegistry::default();
    register_standard_nodes(&mut registry).unwrap();
    let cases = cases();
    let edf = &cases[0];

    let mut limited = graph(edf.import_type, edf.profile, false);
    limited.nodes[0].config = standard_node_config(1).unwrap();
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&limited)
        .unwrap();
    let mut kernels = LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    assert!(executor
        .execute(
            &plan,
            [0x19; 32],
            BTreeMap::from([(
                PortRef {
                    node: NodeId(0),
                    port: "source".into(),
                },
                LamQuantNodeValue::ForeignObject(edf.source.clone()),
            )]),
        )
        .is_err());

    let hostile_profile = ForeignObject {
        profile: ProfileId("x".repeat(1024 * 1024)),
        entries: vec![ForeignEntry {
            path: "x".into(),
            media_type: None,
            bytes: vec![],
        }],
    };
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&limited)
        .unwrap();
    let mut kernels = LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    let error = executor
        .execute(
            &plan,
            [0x1b; 32],
            BTreeMap::from([(
                PortRef {
                    node: NodeId(0),
                    port: "source".into(),
                },
                LamQuantNodeValue::ForeignObject(hostile_profile),
            )]),
        )
        .unwrap_err();
    assert!(format!("{error:?}").contains("resource-limit"));

    let (edf_dataset, _) = run_import(&registry, edf);
    let mut limited_restore = graph(edf.restore_type, edf.profile, true);
    limited_restore.nodes[0].config = standard_node_config(1).unwrap();
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&limited_restore)
        .unwrap();
    let mut kernels = LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    assert!(executor
        .execute(
            &plan,
            [0x1a; 32],
            BTreeMap::from([(
                PortRef {
                    node: NodeId(0),
                    port: "dataset".into(),
                },
                LamQuantNodeValue::AbirDataset(Box::new(edf_dataset)),
            )]),
        )
        .is_err());

    let (edf_dataset, _) = run_import(&registry, edf);
    let dicom = &cases[2];
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph(dicom.restore_type, dicom.profile, true))
        .unwrap();
    let mut kernels = LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    assert!(executor
        .execute(
            &plan,
            [0x20; 32],
            BTreeMap::from([(
                PortRef {
                    node: NodeId(0),
                    port: "dataset".into(),
                },
                LamQuantNodeValue::AbirDataset(Box::new(edf_dataset)),
            )]),
        )
        .is_err());
}
