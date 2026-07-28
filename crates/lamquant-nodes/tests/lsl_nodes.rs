#![cfg(feature = "standard-adapters")]

use blut_graph_core::{
    CheckpointMode, Compiler, ConfigType, ConfigValue, Determinism, Edge, Effect, ExecutionRealm,
    Graph, KernelRegistry, NodeId, NodeInstance, Partiality, PortRef, SessionContract, StateScope,
    Target,
};
use lamquant_nodes::{
    lsl_accept_export_descriptor, lsl_accept_export_kernel_binding, lsl_export_descriptor,
    lsl_export_kernel_binding, lsl_inlet_descriptor, lsl_inlet_kernel_binding,
    lsl_outlet_descriptor, lsl_outlet_kernel_binding, register_lsl_nodes,
    LslImplementationIdentity, LSL_ACCEPT_EXPORT_NODE_TYPE, LSL_EXPORT_NODE_TYPE,
    LSL_OUTLET_NODE_TYPE,
};
use std::collections::BTreeMap;

const TEST_IDENTITY: LslImplementationIdentity<'static> = LslImplementationIdentity {
    host_source_id: "test-host-source",
    host_feature_set: "live-lsl",
    liblsl_revision: "test-liblsl-revision",
};

#[test]
fn live_lsl_descriptors_are_host_only_and_effect_honest() {
    let inlet = lsl_inlet_descriptor();
    let export = lsl_export_descriptor();
    let accept = lsl_accept_export_descriptor();
    let outlet = lsl_outlet_descriptor();

    assert_eq!(inlet.targets, vec![Target::Host]);
    assert_eq!(inlet.effect, Effect::AtMostOnce);
    assert_eq!(inlet.determinism, Determinism::Nondeterministic);
    assert!(inlet.inputs.is_empty());
    assert_eq!(inlet.outputs.len(), 3);
    assert_eq!(inlet.state.scope, StateScope::Session);
    assert_eq!(inlet.state.max_bytes, 1);
    assert_eq!(inlet.state.checkpoint.mode, CheckpointMode::Disabled);
    assert_eq!(inlet.partiality, Partiality::ExplicitGaps);
    assert_eq!(inlet.fidelity.maximum_loss, u16::MAX);
    assert_eq!(inlet.resources.threads, 3);
    assert!(inlet.capabilities.iter().any(
        |capability| capability.0 == "org.quitetall.lamquant.lsl.experimental-numeric-inlet-v1"
    ));
    assert!(!inlet
        .capabilities
        .iter()
        .any(|capability| capability.0 == "abir.adapter.lsl.1.16"));
    assert!(inlet
        .policy
        .requires
        .iter()
        .any(|policy| policy == "abir.policy.lsl-protocol-110-peer-attested"));

    assert_eq!(export.effect, Effect::Pure);
    assert_eq!(export.determinism, Determinism::BitExact);
    assert_eq!(export.inputs.len(), 1);
    assert_eq!(export.outputs.len(), 1);
    assert_eq!(
        export.outputs[0].semantic_type,
        "abir.lsl.live-export-plan.v1"
    );
    assert!(export
        .config
        .fields
        .iter()
        .all(|field| field.name != "accept_projection"));
    assert_eq!(export.resources.threads, 1);

    assert_eq!(accept.effect, Effect::Pure);
    assert_eq!(accept.determinism, Determinism::BitExact);
    assert_eq!(accept.inputs.len(), 2);
    assert_eq!(accept.outputs.len(), 2);
    assert!(accept
        .config
        .fields
        .iter()
        .any(|field| field.name == "accept_projection"));
    assert_eq!(accept.fidelity.maximum_loss, u16::MAX);
    assert_eq!(accept.resources.threads, 1);

    assert_eq!(outlet.effect, Effect::AtMostOnce);
    assert_eq!(outlet.determinism, Determinism::Nondeterministic);
    assert_eq!(outlet.inputs.len(), 1);
    assert_eq!(outlet.outputs.len(), 1);
    assert_eq!(outlet.state.scope, StateScope::Session);
    assert_eq!(outlet.state.max_bytes, 1);
    assert_eq!(outlet.state.checkpoint.mode, CheckpointMode::Disabled);
    assert_eq!(outlet.resources.threads, 3);
    assert_eq!(outlet.fidelity.maximum_loss, u16::MAX);
}

#[test]
fn bounded_inlet_capability_does_not_advertise_string_transport() {
    let inlet = lsl_inlet_descriptor();
    let sample_type = inlet
        .config
        .fields
        .iter()
        .find(|field| field.name == "sample_type")
        .unwrap();
    let ConfigType::Choice { values } = &sample_type.value_type else {
        panic!("sample_type must be a closed choice");
    };
    assert!(!values.iter().any(|value| value == "string"));
    #[cfg(windows)]
    assert!(!values.iter().any(|value| value == "int64"));
}

#[test]
fn live_lsl_kernel_registration_matches_host_bindings() {
    let mut registry = KernelRegistry::default();
    register_lsl_nodes(&mut registry, &TEST_IDENTITY).unwrap();

    assert_ne!(
        lsl_inlet_kernel_binding(&TEST_IDENTITY),
        lsl_export_kernel_binding(&TEST_IDENTITY)
    );
    assert_ne!(
        lsl_export_kernel_binding(&TEST_IDENTITY),
        lsl_accept_export_kernel_binding(&TEST_IDENTITY)
    );
    assert_ne!(
        lsl_accept_export_kernel_binding(&TEST_IDENTITY),
        lsl_outlet_kernel_binding(&TEST_IDENTITY)
    );
}

#[test]
fn projected_export_acceptance_compiles_into_outlet() {
    let export_dataset = PortRef {
        node: NodeId(0),
        port: "dataset".to_owned(),
    };
    let accept_dataset = PortRef {
        node: NodeId(1),
        port: "dataset".to_owned(),
    };
    let descriptors = [
        lsl_export_descriptor(),
        lsl_accept_export_descriptor(),
        lsl_outlet_descriptor(),
    ];
    let graph = Graph {
        version: 3,
        nodes: vec![
            NodeInstance {
                id: NodeId(0),
                descriptor: LSL_EXPORT_NODE_TYPE.to_owned(),
                descriptor_version: 1,
                config: BTreeMap::from([
                    ("name".to_owned(), ConfigValue::Text("out".to_owned())),
                    ("rebase_at_first_push".to_owned(), ConfigValue::Bool(true)),
                    (
                        "source_id".to_owned(),
                        ConfigValue::Text("source".to_owned()),
                    ),
                    (
                        "stream_id".to_owned(),
                        ConfigValue::Text("stream".to_owned()),
                    ),
                    (
                        "stream_type".to_owned(),
                        ConfigValue::Text("EEG".to_owned()),
                    ),
                    (
                        "timestamp_policy".to_owned(),
                        ConfigValue::Text("quantize_to_nanoseconds".to_owned()),
                    ),
                ]),
            },
            NodeInstance {
                id: NodeId(1),
                descriptor: LSL_ACCEPT_EXPORT_NODE_TYPE.to_owned(),
                descriptor_version: 1,
                config: BTreeMap::from([("accept_projection".to_owned(), ConfigValue::Bool(true))]),
            },
            NodeInstance {
                id: NodeId(2),
                descriptor: LSL_OUTLET_NODE_TYPE.to_owned(),
                descriptor_version: 1,
                config: BTreeMap::from([(
                    "resource_id".to_owned(),
                    ConfigValue::Text("outlet".to_owned()),
                )]),
            },
        ],
        edges: vec![
            Edge {
                from: PortRef {
                    node: NodeId(0),
                    port: "export_plan".to_owned(),
                },
                to: PortRef {
                    node: NodeId(1),
                    port: "export_plan".to_owned(),
                },
            },
            Edge {
                from: PortRef {
                    node: NodeId(1),
                    port: "accepted_export".to_owned(),
                },
                to: PortRef {
                    node: NodeId(2),
                    port: "accepted_export".to_owned(),
                },
            },
        ],
        feedback: vec![],
        invocation_inputs: vec![export_dataset, accept_dataset],
        required_capabilities: descriptors
            .into_iter()
            .flat_map(|descriptor| descriptor.capabilities)
            .collect(),
        required_proofs: vec![],
        policy: vec!["abir.policy.network-export-authorized".to_owned()],
        minimum_fidelity: 0,
        session: Some(SessionContract {
            namespace: "lsl-outlet-test".to_owned(),
            max_concurrent_sessions: 1,
            max_idle_millis: 60_000,
            reset_on_plan_change: true,
        }),
    };
    let mut registry = KernelRegistry::default();
    register_lsl_nodes(&mut registry, &TEST_IDENTITY).unwrap();
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();
    assert_eq!(plan.persistent_state_bytes, 1);
}
