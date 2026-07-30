use std::collections::BTreeMap;

use blut_graph_core::{
    AbirRootType, AbirViewType, Capability, Compiler, ExecutionRealm, Graph, KernelRegistry,
    NodeId, NodeInstance, PlanExecutor, PortRef, Target,
};
use lamquant_abir_codec::encode_lml_bundle_from_views_explicit;
use lamquant_lml_mcu::{
    lml::{compress_with_mode_views_explicit, EncodeFeatures},
    lpc::LpcMode,
};
use lamquant_nodes::{
    arithmetic_lml_descriptor, baseline_lml_descriptor, baseline_lml_packet_descriptor,
    lml_node_config, lml_packet_node_config, register_lml_nodes, LamQuantNodeValue,
    LmlNodeConfigError, LmlSignalView, NoopTransactionalSink, CAP_LML_ARITHMETIC_NODE,
    LML_ARITHMETIC_NODE_TYPE, LML_ASSEMBLE_NODE_TYPE, LML_BASELINE_NODE_TYPE,
    LML_ENTROPY_NODE_TYPE, LML_PACKET_BASELINE_NODE_TYPE, LML_PREDICT_NODE_TYPE,
    LML_QUANTIZE_NODE_TYPE, LML_TRANSFORM_NODE_TYPE,
};
#[cfg(feature = "optimum-v2")]
use lamquant_nodes::{
    optimum_v2_node_config, optimum_v2_peer_descriptor, CAP_LML_OPTIMUM_V2_PEER_NODE,
    OPTIMUM_V2_PEER_NODE_TYPE,
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

fn execute_packet_plan(
    plan: &blut_graph_core::AuthorizedPlan,
    input_port: PortRef,
    dataset: &AbirDataset,
    views: &[&[i64]],
    bounds: ResourceBounds,
) -> Vec<u8> {
    let mut kernels = lamquant_nodes::LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    let result = executor
        .execute(
            plan,
            [0x42; 32],
            BTreeMap::from([(
                input_port,
                LamQuantNodeValue::LmlSignal(LmlSignalView::new(dataset, views, bounds).unwrap()),
            )]),
        )
        .unwrap();
    match result
        .terminal_values
        .values()
        .next()
        .and_then(|values| values.first())
        .expect("one terminal packet")
    {
        LamQuantNodeValue::LmlPackets(packets) => {
            assert_eq!(packets.packets().len(), 1);
            packets.packets()[0].clone()
        }
        other => panic!("unexpected reference output: {other:?}"),
    }
}

#[test]
fn descriptor_identity_is_feature_specific() {
    let baseline = baseline_lml_descriptor();
    let arithmetic = arithmetic_lml_descriptor();
    let packet = baseline_lml_packet_descriptor();

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
    // McuAot joined deliberately: the fused body is `lamquant-lml-mcu`, the
    // no_std crate firmware links, so the MCU realm was the one target this node
    // could most honestly claim and the only one it did not.
    assert_eq!(
        baseline.targets,
        vec![Target::Host, Target::McuAot, Target::BlutDurable]
    );
    assert_eq!(packet.type_name, LML_PACKET_BASELINE_NODE_TYPE);
    assert_eq!(packet.targets, vec![Target::Host]);
    assert!(packet.subgraph.is_some());
    assert_eq!(packet.outputs[0].abir.root, AbirRootType::EncodedBlock);
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

#[cfg(feature = "optimum-v2")]
#[test]
fn optimum_v2_peer_node_matches_direct_codec_and_bcs2_closure() {
    use lamquant_abir_codec::{
        open_optimum_v2_bundle, optimum_v2_implementation_identity, seal_optimum_v2_packets,
    };
    use lamquant_lml_optimum_v2::{PeerCodec, PeerEncodeContext};

    let signal = fixture_signal();
    let dataset = fixture_dataset(&signal);
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bounds = ResourceBounds::default();
    let descriptor = optimum_v2_peer_descriptor();
    assert_eq!(descriptor.type_name, OPTIMUM_V2_PEER_NODE_TYPE);
    assert_eq!(descriptor.targets, vec![Target::Host, Target::BlutDurable]);
    assert!(descriptor
        .capabilities
        .contains(&Capability(CAP_LML_OPTIMUM_V2_PEER_NODE.into())));

    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).unwrap();
    let graph = single_lml_graph(
        OPTIMUM_V2_PEER_NODE_TYPE,
        descriptor.capabilities,
        optimum_v2_node_config(256_000, 16).unwrap(),
    );
    let direct_packet = PeerCodec
        .encode_window(
            &signal,
            PeerEncodeContext {
                sample_rate_mhz: 256_000,
                bit_depth: 16,
            },
        )
        .unwrap();
    let direct_bundle = seal_optimum_v2_packets(
        &dataset,
        &[direct_packet.as_slice()],
        optimum_v2_implementation_identity(),
        bounds,
    )
    .unwrap();

    let execute = |realm| {
        let plan = Compiler::new(&registry, realm).compile(&graph).unwrap();
        let selected = plan
            .as_plan()
            .nodes
            .iter()
            .map(|node| (node.kernel, node.implementation_id))
            .collect::<Vec<_>>();
        let mut kernels = lamquant_nodes::LamQuantKernelExecutor::default();
        let mut sink = NoopTransactionalSink;
        let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
        let result = executor
            .execute(
                &plan,
                [0x81; 32],
                BTreeMap::from([(
                    PortRef {
                        node: NodeId(0),
                        port: "signal".into(),
                    },
                    LamQuantNodeValue::LmlSignal(
                        LmlSignalView::new(&dataset, &views, bounds).unwrap(),
                    ),
                )]),
            )
            .unwrap();
        let bytes = match &result.terminal_values.get(&NodeId(0)).unwrap()[0] {
            LamQuantNodeValue::Bcs2(bytes) => bytes.clone(),
            other => panic!("unexpected peer node output: {other:?}"),
        };
        (selected, bytes)
    };

    let (host_kernels, host) = execute(ExecutionRealm::HostStream);
    let (blut_kernels, blut) = execute(ExecutionRealm::BlutDurable);
    assert_ne!(host_kernels, blut_kernels);
    assert_eq!(host, direct_bundle);
    assert_eq!(blut, host);
    let opened = open_optimum_v2_bundle(&host, bounds).unwrap();
    assert_eq!(opened.packet(), direct_packet);
    assert_eq!(opened.signal(), signal);
    assert!(Compiler::new(&registry, ExecutionRealm::McuAot)
        .compile(&graph)
        .is_err());
}

/// P8.3 #22 — one graph, two realms, identical bytes.
///
/// The MCU and host kernels are NOT the same kernel. They carry different
/// `KernelId`s and different `ImplementationId`s (target is hashed into it), and
/// they are permitted different resources: the host kernel may use up to 1024
/// parallel channels, the MCU kernel exactly one. That difference is the point.
/// If channel parallelism could perturb the encoder, this is where it would
/// show, and it would show as a wire fork between the device that records and
/// the host that reads.
///
/// Byte equality across the two is therefore not a restatement of the existing
/// `byte_equal_backends` gate, which compares two backend modes inside one
/// crate. This compares two COMPILED PLANS in two execution realms.
#[test]
fn one_graph_encodes_identically_in_the_mcu_and_host_realms() {
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

    let encode = |realm| {
        let plan = Compiler::new(&registry, realm)
            .compile(&graph)
            .expect("the graph must compile in this realm");
        let selected: Vec<_> = plan
            .as_plan()
            .nodes
            .iter()
            .map(|node| (node.kernel, node.implementation_id))
            .collect();
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
                    LamQuantNodeValue::LmlSignal(
                        LmlSignalView::new(&dataset, &views, bounds).unwrap(),
                    ),
                )]),
            )
            .expect("execute");
        let bytes = match &result.terminal_values.get(&NodeId(0)).unwrap()[0] {
            LamQuantNodeValue::Bcs2(bytes) => bytes.clone(),
            other => panic!("unexpected node output: {other:?}"),
        };
        (selected, bytes)
    };

    let (host_kernels, host) = encode(ExecutionRealm::HostStream);
    let (mcu_kernels, mcu) = encode(ExecutionRealm::McuAot);

    // Without this the test is worthless. If the MCU realm quietly selected the
    // host kernel, the byte comparison below would compare a thing to itself and
    // pass no matter what. Equality of OUTPUT is only evidence when the
    // IMPLEMENTATIONS differ.
    assert_ne!(
        host_kernels, mcu_kernels,
        "the two realms selected the same kernel, so equal bytes prove nothing"
    );
    assert!(
        !host.is_empty(),
        "an empty encode would make equality vacuous"
    );
    assert_eq!(
        host, mcu,
        "the same graph produced different bytes in the host and MCU realms: a \
         recording made on the device would not match one made on the host"
    );
}

#[test]
fn compiler_produced_reference_dag_matches_fused_lml_node() {
    let signal = fixture_signal();
    let dataset = fixture_dataset(&signal);
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bounds = ResourceBounds::default();
    let mode = LpcMode::Adaptive { max_order: 8 };
    let config = lml_packet_node_config(mode).unwrap();

    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).unwrap();
    let reference_graph = registry
        .materialize_subgraph(&NodeInstance {
            id: NodeId(99),
            descriptor: LML_PACKET_BASELINE_NODE_TYPE.into(),
            descriptor_version: 1,
            config,
        })
        .unwrap();
    assert_eq!(reference_graph.graph.nodes.len(), 5);
    assert_eq!(
        reference_graph
            .graph
            .nodes
            .iter()
            .map(|node| node.descriptor.as_str())
            .collect::<Vec<_>>(),
        vec![
            LML_TRANSFORM_NODE_TYPE,
            LML_QUANTIZE_NODE_TYPE,
            LML_PREDICT_NODE_TYPE,
            LML_ENTROPY_NODE_TYPE,
            LML_ASSEMBLE_NODE_TYPE,
        ]
    );
    let reference = Compiler::new(&registry, ExecutionRealm::HostStream)
        .with_fusion(false)
        .compile(&reference_graph.graph)
        .unwrap();
    let fused = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&reference_graph.graph)
        .unwrap();
    assert!(Compiler::new(&registry, ExecutionRealm::BlutDurable)
        .compile(&reference_graph.graph)
        .is_err());
    assert_eq!(reference.as_plan().nodes.len(), 5);
    assert_eq!(fused.as_plan().nodes.len(), 1);

    let input_port = reference_graph.inputs[0].inner.clone();
    let reference_output =
        execute_packet_plan(&reference, input_port.clone(), &dataset, &views, bounds);
    let fused_output = execute_packet_plan(&fused, input_port, &dataset, &views, bounds);
    assert_eq!(reference_output, fused_output);
    let direct_output = compress_with_mode_views_explicit(
        &views,
        0,
        mode,
        EncodeFeatures {
            max_packet_bytes: Some(bounds.max_frame_bytes as usize),
            ..EncodeFeatures::default()
        },
    )
    .unwrap();
    assert_eq!(reference_output, direct_output);
}

#[test]
fn canonical_compiler_dag_reference_equals_fused_matrix() {
    let shapes = [
        (1, 4),
        (1, 8),
        (1, 20),
        (1, 100),
        (4, 313),
        (8, 2500),
        (32, 2500),
    ];
    let modes = [
        LpcMode::Fixed,
        LpcMode::Adaptive { max_order: 16 },
        LpcMode::Anytime {
            max_order: 16,
            deadline: None,
        },
    ];
    let bounds = ResourceBounds::default();
    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).unwrap();

    for (channels, samples) in shapes {
        let signal = (0..channels)
            .map(|channel| {
                (0..samples)
                    .map(|sample| {
                        let base = ((sample * 3 + channel * 7) % 512) as i64 - 256;
                        let wobble = ((sample * sample + channel) % 97) as i64 - 48;
                        base * 40 + wobble
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let dataset = fixture_dataset(&signal);
        let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
        for mode in modes {
            let materialized = registry
                .materialize_subgraph(&NodeInstance {
                    id: NodeId(77),
                    descriptor: LML_PACKET_BASELINE_NODE_TYPE.into(),
                    descriptor_version: 1,
                    config: lml_packet_node_config(mode).unwrap(),
                })
                .unwrap();
            let reference = Compiler::new(&registry, ExecutionRealm::HostStream)
                .with_fusion(false)
                .compile(&materialized.graph)
                .unwrap();
            let fused = Compiler::new(&registry, ExecutionRealm::HostStream)
                .compile(&materialized.graph)
                .unwrap();
            let input = materialized.inputs[0].inner.clone();
            let reference_bytes =
                execute_packet_plan(&reference, input.clone(), &dataset, &views, bounds);
            let fused_bytes = execute_packet_plan(&fused, input, &dataset, &views, bounds);
            let direct = compress_with_mode_views_explicit(
                &views,
                0,
                mode,
                EncodeFeatures {
                    max_packet_bytes: Some(bounds.max_frame_bytes as usize),
                    ..EncodeFeatures::default()
                },
            )
            .unwrap();
            assert_eq!(reference_bytes, fused_bytes, "{channels}ch x {samples}");
            assert_eq!(reference_bytes, direct, "{channels}ch x {samples}");
        }
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

#[test]
fn reference_packet_dag_rejects_multi_packet_input_before_transform() {
    let signal = vec![vec![0_i64; u16::MAX as usize + 1]];
    let dataset = fixture_dataset(&signal);
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bounds = ResourceBounds::default();
    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).unwrap();
    let materialized = registry
        .materialize_subgraph(&NodeInstance {
            id: NodeId(88),
            descriptor: LML_PACKET_BASELINE_NODE_TYPE.into(),
            descriptor_version: 1,
            config: lml_packet_node_config(LpcMode::Fixed).unwrap(),
        })
        .unwrap();
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .with_fusion(false)
        .compile(&materialized.graph)
        .unwrap();
    let mut kernels = lamquant_nodes::LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    let result = executor.execute(
        &plan,
        [0x55; 32],
        BTreeMap::from([(
            materialized.inputs[0].inner.clone(),
            LamQuantNodeValue::LmlSignal(LmlSignalView::new(&dataset, &views, bounds).unwrap()),
        )]),
    );
    assert!(result.is_err());
}

#[test]
fn reference_packet_dag_rejects_dataset_payload_substitution() {
    let declared = vec![vec![1_i64, 2, 3, 4]];
    let substituted = [vec![9_i64, 9, 9, 9]];
    let dataset = fixture_dataset(&declared);
    let views = substituted.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bounds = ResourceBounds::default();
    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).unwrap();
    let materialized = registry
        .materialize_subgraph(&NodeInstance {
            id: NodeId(89),
            descriptor: LML_PACKET_BASELINE_NODE_TYPE.into(),
            descriptor_version: 1,
            config: lml_packet_node_config(LpcMode::Fixed).unwrap(),
        })
        .unwrap();

    for fusion in [false, true] {
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .with_fusion(fusion)
            .compile(&materialized.graph)
            .unwrap();
        let mut kernels = lamquant_nodes::LamQuantKernelExecutor::default();
        let mut sink = NoopTransactionalSink;
        let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
        let result = executor.execute(
            &plan,
            [0x66; 32],
            BTreeMap::from([(
                materialized.inputs[0].inner.clone(),
                LamQuantNodeValue::LmlSignal(LmlSignalView::new(&dataset, &views, bounds).unwrap()),
            )]),
        );
        assert!(result.is_err(), "fusion={fusion}");
    }
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
