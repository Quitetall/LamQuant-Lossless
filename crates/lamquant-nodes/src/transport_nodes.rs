//! Bounded MCU transport Node contracts.
//!
//! BLE and USB share one semantic payload and receipt contract while retaining
//! distinct Node and kernel identities. Physical peripheral ownership stays in
//! firmware; this catalog declares what firmware must implement.

use alloc::string::ToString;
use alloc::vec;

use blut_graph_core::{
    AbirRootType, AbirSemanticType, AbirViewType, Capability, CompileError, ConfigSchema,
    Determinism, Effect, ExtentContract, FailureContract, FidelityContract, ImplementationId,
    KernelDescriptor, KernelId, KernelRegistry, Layout, LeaseAccess, LeaseContract, LeaseLifetime,
    NodeDescriptor, NodeTypeRef, Partiality, PolicyContract, PortDescriptor, ProofContract,
    ResourceEnvelope, StateContract, Target,
};

pub const BLE_TRANSMIT_NODE_TYPE: &str = "org.quitetall.lamquant.transport.ble.transmit";
pub const USB_TRANSMIT_NODE_TYPE: &str = "org.quitetall.lamquant.transport.usb.transmit";

/// Shared `TX` allocator namespace established by ADR 0151.
pub const BLE_TRANSMIT_KERNEL: KernelId = KernelId(0x5458_0101);
pub const USB_TRANSMIT_KERNEL: KernelId = KernelId(0x5458_0102);

pub const CAP_BCS2_STREAM_BOUNDED: &str = "bcs2.profile.stream-bounded-v1";
pub const CAP_LQF2_TRANSPORT: &str = "org.quitetall.lamquant.transport.lqf2-v2";
pub const CAP_BLE_TRANSPORT: &str = "org.quitetall.lamquant.transport.ble-v1";
pub const CAP_USB_TRANSPORT: &str = "org.quitetall.lamquant.transport.usb-v1";
pub const POLICY_NETWORK_EXPORT: &str = "abir.policy.network-export-authorized";
pub const POLICY_DEVICE_EXPORT: &str = "abir.policy.device-export-authorized";
pub const TRANSPORT_PAYLOAD_PROOF: &str = "org.quitetall.abir.proof.bcs2-stream-bounded-closure-v1";
pub const TRANSPORT_ATTEMPT_RECEIPT_PROOF: &str =
    "org.quitetall.lamquant.proof.transport-attempt-receipt-v1";
pub const TRANSPORT_PAYLOAD_SEMANTIC_TYPE: &str = "bcs2.stream-bounded-v1";
pub const TRANSPORT_ATTEMPT_RECEIPT_SEMANTIC_TYPE: &str = "abir.transport.attempt-receipt-v1";
pub const TRANSPORT_MAX_PAYLOAD_BYTES: u64 = 8 * 1024;
/// Fixed `LQTR` v1 receipt length emitted by firmware transport executors.
pub const TRANSPORT_ATTEMPT_RECEIPT_BYTES: u64 = 92;
/// One LQF2 v2 header plus one maximum fragment, reused for every fragment.
pub const TRANSPORT_LQF2_SCRATCH_BYTES: u64 = 116 + 256;

const FAILURE_DOMAIN: &str = "org.quitetall.lamquant.transport";

#[derive(Clone, Copy)]
enum TransportKind {
    Ble,
    Usb,
}

pub fn ble_transmit_descriptor() -> NodeDescriptor {
    descriptor(TransportKind::Ble)
}

pub fn usb_transmit_descriptor() -> NodeDescriptor {
    descriptor(TransportKind::Usb)
}

/// Register semantic transport contracts against exact linked firmware builds.
pub fn register_mcu_transport_nodes(
    registry: &mut KernelRegistry,
    ble_implementation: ImplementationId,
    usb_implementation: ImplementationId,
) -> Result<(), CompileError> {
    for (kind, implementation) in [
        (TransportKind::Ble, ble_implementation),
        (TransportKind::Usb, usb_implementation),
    ] {
        let descriptor = descriptor(kind);
        registry.register_descriptor(descriptor)?;
        registry.register_kernel(kernel(kind, implementation))?;
    }
    Ok(())
}

fn descriptor(kind: TransportKind) -> NodeDescriptor {
    NodeDescriptor {
        type_name: node_type(kind).to_string(),
        version: 1,
        inputs: vec![payload_port()],
        outputs: vec![receipt_port()],
        capabilities: vec![
            Capability("abir.semantic-v1".to_string()),
            Capability(CAP_BCS2_STREAM_BOUNDED.to_string()),
            Capability(CAP_LQF2_TRANSPORT.to_string()),
            Capability(transport_capability(kind).to_string()),
        ],
        targets: vec![Target::McuAot],
        resources: resources(),
        determinism: Determinism::Nondeterministic,
        config: ConfigSchema::default(),
        state: StateContract::stateless(),
        subgraph: None,
        proof: ProofContract {
            requires: vec![TRANSPORT_PAYLOAD_PROOF.to_string()],
            provides: vec![TRANSPORT_ATTEMPT_RECEIPT_PROOF.to_string()],
            invalidates: vec![],
        },
        policy: PolicyContract {
            requires: vec![policy(kind).to_string()],
            adds: vec![],
        },
        fidelity: exact_fidelity(),
        partiality: Partiality::Atomic,
        failure: FailureContract {
            domains: vec![FAILURE_DOMAIN.to_string()],
        },
        effect: Effect::Idempotent,
        // Transport returns an explicit attempt state. Framework retries would
        // obscure whether a peripheral accepted an earlier fragment.
        retry_limit: 0,
    }
}

fn kernel(kind: TransportKind, implementation_id: ImplementationId) -> KernelDescriptor {
    KernelDescriptor {
        id: kernel_id(kind),
        implements: vec![NodeTypeRef {
            type_name: node_type(kind).to_string(),
            version: 1,
        }],
        implementation_id,
        conversion: None,
        target: Target::McuAot,
        input_layouts: vec![Layout::Packed],
        output_layouts: vec![Layout::Opaque],
        resources: resources(),
        determinism: Determinism::Nondeterministic,
        lowering: lowering(kind).to_string(),
    }
}

const fn kernel_id(kind: TransportKind) -> KernelId {
    match kind {
        TransportKind::Ble => BLE_TRANSMIT_KERNEL,
        TransportKind::Usb => USB_TRANSMIT_KERNEL,
    }
}

const fn node_type(kind: TransportKind) -> &'static str {
    match kind {
        TransportKind::Ble => BLE_TRANSMIT_NODE_TYPE,
        TransportKind::Usb => USB_TRANSMIT_NODE_TYPE,
    }
}

const fn transport_capability(kind: TransportKind) -> &'static str {
    match kind {
        TransportKind::Ble => CAP_BLE_TRANSPORT,
        TransportKind::Usb => CAP_USB_TRANSPORT,
    }
}

const fn policy(kind: TransportKind) -> &'static str {
    match kind {
        TransportKind::Ble => POLICY_NETWORK_EXPORT,
        TransportKind::Usb => POLICY_DEVICE_EXPORT,
    }
}

const fn lowering(kind: TransportKind) -> &'static str {
    match kind {
        TransportKind::Ble => "mcu-aot.transport.ble.lqf2-idempotent-v1",
        TransportKind::Usb => "mcu-aot.transport.usb.lqf2-idempotent-v1",
    }
}

const fn resources() -> ResourceEnvelope {
    ResourceEnvelope::bounded(0, TRANSPORT_LQF2_SCRATCH_BYTES, 1)
}

fn payload_port() -> PortDescriptor {
    PortDescriptor {
        name: "payload".to_string(),
        semantic_type: TRANSPORT_PAYLOAD_SEMANTIC_TYPE.to_string(),
        optional: false,
        layouts: vec![Layout::Packed],
        max_bytes: TRANSPORT_MAX_PAYLOAD_BYTES,
        abir: AbirSemanticType {
            root: AbirRootType::EncodedBlock,
            view: AbirViewType::Atom,
        },
        proof: ProofContract {
            requires: vec![TRANSPORT_PAYLOAD_PROOF.to_string()],
            provides: vec![],
            invalidates: vec![],
        },
        policy: PolicyContract {
            requires: vec![],
            adds: vec![],
        },
        fidelity: exact_fidelity(),
        extent: ExtentContract {
            rank: 1,
            maximum_shape: vec![TRANSPORT_MAX_PAYLOAD_BYTES],
            max_elements: TRANSPORT_MAX_PAYLOAD_BYTES,
            ragged: false,
            sparse: false,
        },
        lease: LeaseContract {
            access: LeaseAccess::ReadOnly,
            lifetime: LeaseLifetime::Invocation,
            zero_copy_permitted: true,
            contiguous_required: true,
        },
    }
}

fn receipt_port() -> PortDescriptor {
    PortDescriptor {
        name: "attempt_receipt".to_string(),
        semantic_type: TRANSPORT_ATTEMPT_RECEIPT_SEMANTIC_TYPE.to_string(),
        optional: false,
        layouts: vec![Layout::Opaque],
        max_bytes: TRANSPORT_ATTEMPT_RECEIPT_BYTES,
        abir: AbirSemanticType {
            root: AbirRootType::Unknown("transport-attempt-receipt".to_string()),
            view: AbirViewType::Unknown("transport-attempt-receipt".to_string()),
        },
        proof: ProofContract {
            requires: vec![],
            provides: vec![TRANSPORT_ATTEMPT_RECEIPT_PROOF.to_string()],
            invalidates: vec![],
        },
        policy: PolicyContract {
            requires: vec![],
            adds: vec![],
        },
        fidelity: exact_fidelity(),
        extent: ExtentContract {
            rank: 1,
            maximum_shape: vec![TRANSPORT_ATTEMPT_RECEIPT_BYTES],
            max_elements: TRANSPORT_ATTEMPT_RECEIPT_BYTES,
            ragged: false,
            sparse: false,
        },
        lease: LeaseContract {
            access: LeaseAccess::ReadOnly,
            lifetime: LeaseLifetime::Invocation,
            zero_copy_permitted: false,
            contiguous_required: true,
        },
    }
}

const fn exact_fidelity() -> FidelityContract {
    FidelityContract {
        minimum_input: u16::MAX,
        maximum_loss: 0,
    }
}
