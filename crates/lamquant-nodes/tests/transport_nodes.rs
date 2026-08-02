use std::collections::BTreeMap;

use blut_graph_core::{
    Capability, Compiler, Effect, ExecutionRealm, Graph, ImplementationId, KernelRegistry, Layout,
    NodeId, NodeInstance, Partiality, PortRef, StateScope, Target,
};
use lamquant_nodes::{
    ble_transmit_descriptor, register_mcu_transport_nodes, usb_transmit_descriptor,
    BLE_TRANSMIT_KERNEL, BLE_TRANSMIT_NODE_TYPE, CAP_BCS2_STREAM_BOUNDED, CAP_BLE_TRANSPORT,
    CAP_LQF2_TRANSPORT, CAP_USB_TRANSPORT, POLICY_DEVICE_EXPORT, POLICY_NETWORK_EXPORT,
    TRANSPORT_ATTEMPT_RECEIPT_BYTES, TRANSPORT_ATTEMPT_RECEIPT_SEMANTIC_TYPE,
    TRANSPORT_MAX_PAYLOAD_BYTES, TRANSPORT_PAYLOAD_SEMANTIC_TYPE, USB_TRANSMIT_KERNEL,
    USB_TRANSMIT_NODE_TYPE,
};

const BLE_IMPLEMENTATION: ImplementationId = ImplementationId([0x42; 32]);
const USB_IMPLEMENTATION: ImplementationId = ImplementationId([0x55; 32]);

#[test]
fn ble_and_usb_descriptors_are_bounded_idempotent_mcu_effects() {
    let ble = ble_transmit_descriptor();
    let usb = usb_transmit_descriptor();

    for descriptor in [&ble, &usb] {
        assert_eq!(descriptor.targets, vec![Target::McuAot]);
        assert_eq!(descriptor.effect, Effect::Idempotent);
        assert_eq!(descriptor.retry_limit, 0);
        assert_eq!(descriptor.state.scope, StateScope::Stateless);
        assert_eq!(descriptor.partiality, Partiality::Atomic);
        assert_eq!(descriptor.inputs.len(), 1);
        assert_eq!(descriptor.outputs.len(), 1);
        assert_eq!(
            descriptor.inputs[0].semantic_type,
            TRANSPORT_PAYLOAD_SEMANTIC_TYPE
        );
        assert_eq!(descriptor.inputs[0].layouts, vec![Layout::Packed]);
        assert_eq!(descriptor.inputs[0].max_bytes, TRANSPORT_MAX_PAYLOAD_BYTES);
        assert_eq!(
            descriptor.outputs[0].semantic_type,
            TRANSPORT_ATTEMPT_RECEIPT_SEMANTIC_TYPE
        );
        assert_eq!(descriptor.outputs[0].layouts, vec![Layout::Opaque]);
        assert_eq!(
            descriptor.outputs[0].max_bytes,
            TRANSPORT_ATTEMPT_RECEIPT_BYTES
        );
        assert!(descriptor
            .capabilities
            .contains(&Capability(CAP_BCS2_STREAM_BOUNDED.to_owned())));
        assert!(descriptor
            .capabilities
            .contains(&Capability(CAP_LQF2_TRANSPORT.to_owned())));
    }

    assert!(ble
        .policy
        .requires
        .iter()
        .any(|policy| policy == POLICY_NETWORK_EXPORT));
    assert!(usb
        .policy
        .requires
        .iter()
        .any(|policy| policy == POLICY_DEVICE_EXPORT));
    assert!(ble
        .capabilities
        .contains(&Capability(CAP_BLE_TRANSPORT.to_owned())));
    assert!(usb
        .capabilities
        .contains(&Capability(CAP_USB_TRANSPORT.to_owned())));
    assert_ne!(ble.type_name, usb.type_name);
}

#[test]
fn transport_kernel_bindings_are_distinct_and_compile_for_mcu() {
    let mut registry = KernelRegistry::default();
    register_mcu_transport_nodes(&mut registry, BLE_IMPLEMENTATION, USB_IMPLEMENTATION).unwrap();

    assert_ne!(BLE_TRANSMIT_KERNEL, USB_TRANSMIT_KERNEL);
    assert_ne!(BLE_IMPLEMENTATION, USB_IMPLEMENTATION);

    for (descriptor, node_type, policy) in [
        (
            ble_transmit_descriptor(),
            BLE_TRANSMIT_NODE_TYPE,
            POLICY_NETWORK_EXPORT,
        ),
        (
            usb_transmit_descriptor(),
            USB_TRANSMIT_NODE_TYPE,
            POLICY_DEVICE_EXPORT,
        ),
    ] {
        let graph = Graph {
            version: 3,
            nodes: vec![NodeInstance {
                id: NodeId(1),
                descriptor: node_type.to_owned(),
                descriptor_version: 1,
                config: BTreeMap::new(),
            }],
            edges: vec![],
            feedback: vec![],
            invocation_inputs: vec![PortRef {
                node: NodeId(1),
                port: "payload".to_owned(),
            }],
            required_capabilities: descriptor.capabilities.clone(),
            required_proofs: descriptor.inputs[0].proof.requires.clone(),
            policy: vec![policy.to_owned()],
            minimum_fidelity: u16::MAX,
            session: None,
        };
        let plan = Compiler::new(&registry, ExecutionRealm::McuAot)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert_eq!(plan.nodes[0].effect, Effect::Idempotent);
        assert_eq!(plan.nodes[0].input_contracts[0].max_bytes, 8 * 1024);
    }
}
