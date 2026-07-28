use std::collections::BTreeMap;

use blut_graph_core::{
    AbirRootType, AbirViewType, Capability, Compiler, ExecutionRealm, Graph, KernelRegistry,
    NodeId, NodeInstance, PlanExecutor, PortRef, Target,
};
use lamquant_abir_codec::encode_lml_bundle_from_views_explicit;
use lamquant_lml_mcu::{lml::EncodeFeatures, lpc::LpcMode};
use lamquant_nodes::{
    arithmetic_lml_descriptor, baseline_lml_descriptor, lml_node_config, register_lml_nodes,
    LamQuantNodeValue, LmlNodeConfigError, LmlSignalView, NoopTransactionalSink,
    CAP_LML_ARITHMETIC_NODE, LML_ARITHMETIC_NODE_TYPE, LML_BASELINE_NODE_TYPE,
};
use semantic_abir::{
    payload_content_id, AbirDataset, Atom, AtomTag, ByteOrder, ConceptId, DatasetDraft, DatasetTag,
    ElementType, Layout, ObjectId, PayloadDescriptor, Presence, Rational, Recording, RecordingTag,
    SignalBlock, Stream, StreamTag, TimeAxis, TimeSegment, ValidationLimits,
};
use semantic_abir_bcs::ResourceBounds;

fn fixture_dataset(signal: &[Vec<i64>]) -> AbirDataset {
    let dataset_id = ObjectId::<DatasetTag>::from_bytes([1; 16]);
    let recording_id = ObjectId::<RecordingTag>::from_bytes([2; 16]);
    let stream_id = ObjectId::<StreamTag>::from_bytes([3; 16]);
    let mut draft = DatasetDraft::new(dataset_id);
    let mut atom_ids = Vec::new();
    for (index, channel) in signal.iter().enumerate() {
        let bytes = channel
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let content_id = payload_content_id(ElementType::I64, &bytes);
        let descriptor = PayloadDescriptor::new(
            content_id,
            bytes.len() as u64,
            ElementType::I64,
            ByteOrder::Little,
            vec![1, channel.len() as u64],
            Layout::DenseRowMajor,
            None,
            None,
        );
        let mut atom_id = [0_u8; 16];
        atom_id[15] = (index + 1) as u8;
        let atom_id = ObjectId::<AtomTag>::from_bytes(atom_id);
        atom_ids.push(atom_id);
        draft.add_atom(Atom::SignalBlock(SignalBlock::new(
            atom_id,
            Presence::Present,
            Some(descriptor),
            TimeAxis::Regular(
                TimeSegment::new(
                    Rational::new(0, 1).unwrap(),
                    Rational::new(256, 1).unwrap(),
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
    draft.validate(ValidationLimits::default()).unwrap()
}

fn fixture_signal() -> Vec<Vec<i64>> {
    (0..4)
        .map(|channel| {
            (0..313)
                .map(|sample| {
                    let base = ((sample * 3 + channel * 7) % 512) as i64 - 256;
                    let wobble = ((sample * sample + channel) % 97) as i64 - 48;
                    base * 40 + wobble
                })
                .collect()
        })
        .collect()
}

fn single_lml_graph(
    descriptor: &str,
    capabilities: Vec<Capability>,
    config: BTreeMap<String, blut_graph_core::ConfigValue>,
) -> Graph {
    Graph {
        version: 3,
        nodes: vec![NodeInstance {
            id: NodeId(0),
            descriptor: descriptor.into(),
            descriptor_version: 1,
            config,
        }],
        edges: vec![],
        feedback: vec![],
        invocation_inputs: vec![PortRef {
            node: NodeId(0),
            port: "signal".into(),
        }],
        required_capabilities: capabilities,
        required_proofs: vec![],
        policy: vec![],
        minimum_fidelity: u16::MAX,
        session: None,
    }
}

#[test]
fn descriptor_identity_is_feature_specific() {
    let baseline = baseline_lml_descriptor();
    let arithmetic = arithmetic_lml_descriptor();

    assert_eq!(baseline.type_name, LML_BASELINE_NODE_TYPE);
    assert_eq!(arithmetic.type_name, LML_ARITHMETIC_NODE_TYPE);
    assert_ne!(baseline.type_name, arithmetic.type_name);
    assert!(!baseline
        .capabilities
        .contains(&Capability(CAP_LML_ARITHMETIC_NODE.into())));
    assert!(arithmetic
        .capabilities
        .contains(&Capability(CAP_LML_ARITHMETIC_NODE.into())));
    assert_eq!(baseline.resources.threads, 1024);
    assert_eq!(baseline.inputs[0].abir.root, AbirRootType::Dataset);
    assert_eq!(baseline.inputs[0].abir.view, AbirViewType::Root);
    assert_eq!(baseline.outputs[0].abir.root, AbirRootType::Dataset);
    assert_eq!(baseline.outputs[0].abir.view, AbirViewType::Root);
    assert_eq!(baseline.targets, vec![Target::Host, Target::BlutDurable]);
}

#[test]
fn node_config_rejects_unserializable_or_unbounded_schedules() {
    assert_eq!(
        lml_node_config(LpcMode::Fixed, 0),
        Err(LmlNodeConfigError::WindowSizeOutOfRange)
    );
    assert_eq!(
        lml_node_config(LpcMode::Adaptive { max_order: 65 }, 1024),
        Err(LmlNodeConfigError::MaxOrderOutOfRange)
    );
    assert_eq!(
        lml_node_config(
            LpcMode::Anytime {
                max_order: 16,
                deadline: Some(std::time::Instant::now()),
            },
            1024,
        ),
        Err(LmlNodeConfigError::LiveDeadlineUnsupported)
    );

    let fixed = lml_node_config(LpcMode::Fixed, 1024).unwrap();
    assert_eq!(fixed.len(), 2);
    assert_eq!(
        fixed.get("lpc_schedule"),
        Some(&blut_graph_core::ConfigValue::Text("fixed".into()))
    );
    assert!(!fixed.contains_key("max_order"));
    assert_ne!(
        lml_node_config(LpcMode::Adaptive { max_order: 8 }, 1024).unwrap(),
        lml_node_config(LpcMode::Adaptive { max_order: 9 }, 1024).unwrap()
    );
}

#[test]
fn signal_view_rejects_shapes_outside_runtime_contract() {
    let signal = fixture_signal();
    let dataset = fixture_dataset(&signal);
    let ragged = [&signal[0][..], &signal[1][..100]];
    assert!(LmlSignalView::new(&dataset, &[], ResourceBounds::default()).is_err());
    assert!(LmlSignalView::new(&dataset, &ragged, ResourceBounds::default()).is_err());
    let empty = [&[][..]];
    assert!(LmlSignalView::new(&dataset, &empty, ResourceBounds::default()).is_err());
    let maximum = vec![0_i64; 131_072];
    assert!(LmlSignalView::new(&dataset, &[&maximum], ResourceBounds::default()).is_ok());
    let too_long = vec![0_i64; 131_073];
    assert!(LmlSignalView::new(&dataset, &[&too_long], ResourceBounds::default()).is_err());
}

#[test]
fn compiled_baseline_node_matches_direct_fused_bundle() {
    let signal = fixture_signal();
    let dataset = fixture_dataset(&signal);
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bounds = ResourceBounds::default();

    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).unwrap();
    let graph = single_lml_graph(
        LML_BASELINE_NODE_TYPE,
        baseline_lml_descriptor().capabilities,
        lml_node_config(LpcMode::Fixed, u16::MAX as usize).unwrap(),
    );
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();
    let direct = encode_lml_bundle_from_views_explicit(
        &dataset,
        &views,
        u16::MAX as usize,
        LpcMode::Fixed,
        EncodeFeatures::default(),
        bounds,
    )
    .unwrap();

    let mut kernels = lamquant_nodes::LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    let result = executor
        .execute(
            &plan,
            [9; 32],
            BTreeMap::from([(
                PortRef {
                    node: NodeId(0),
                    port: "signal".into(),
                },
                LamQuantNodeValue::LmlSignal(LmlSignalView::new(&dataset, &views, bounds).unwrap()),
            )]),
        )
        .unwrap();
    let output = result.terminal_values.get(&NodeId(0)).unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        LamQuantNodeValue::Bcs2(bytes) => assert_eq!(bytes, &direct),
        other => panic!("unexpected node output: {other:?}"),
    }
}

#[test]
fn node_rejects_pathological_entropy_output_without_allocating_it() {
    let signal = vec![vec![i64::MAX]];
    let dataset = fixture_dataset(&signal);
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bounds = ResourceBounds {
        max_frame_bytes: 1024,
        ..ResourceBounds::default()
    };

    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).unwrap();
    let graph = single_lml_graph(
        LML_BASELINE_NODE_TYPE,
        baseline_lml_descriptor().capabilities,
        lml_node_config(LpcMode::Fixed, 1).unwrap(),
    );
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();
    let mut kernels = lamquant_nodes::LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    let result = executor.execute(
        &plan,
        [7; 32],
        BTreeMap::from([(
            PortRef {
                node: NodeId(0),
                port: "signal".into(),
            },
            LamQuantNodeValue::LmlSignal(LmlSignalView::new(&dataset, &views, bounds).unwrap()),
        )]),
    );

    assert!(result.is_err());
}

#[cfg(feature = "experimental-arithmetic")]
#[test]
fn compiled_plan_identity_changes_for_arithmetic_capability() {
    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).unwrap();
    let config = lml_node_config(LpcMode::Fixed, u16::MAX as usize).unwrap();
    let baseline = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&single_lml_graph(
            LML_BASELINE_NODE_TYPE,
            baseline_lml_descriptor().capabilities,
            config.clone(),
        ))
        .unwrap();
    let arithmetic = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&single_lml_graph(
            LML_ARITHMETIC_NODE_TYPE,
            arithmetic_lml_descriptor().capabilities,
            config,
        ))
        .unwrap();

    assert_ne!(baseline.as_plan().graph_id, arithmetic.as_plan().graph_id);
    assert_ne!(baseline.as_plan().plan_id, arithmetic.as_plan().plan_id);
    assert_ne!(
        baseline.as_plan().nodes[0].semantic_types,
        arithmetic.as_plan().nodes[0].semantic_types
    );
}
