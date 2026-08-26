//! Host-only live LSL Node contracts.
//!
//! Descriptors declare semantic ports and bounded resources. Host transport
//! ownership remains in `lamquant-io-nodes`; this crate has no liblsl link.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use blut_graph_core::{
    Capability, CheckpointContract, CheckpointMode, CompileError, ConfigField, ConfigSchema,
    ConfigType, Determinism, DomainToken, DomainType, Effect, ExtentContract, FailureContract,
    FidelityContract, ImplementationId, KernelDescriptor, KernelId, KernelRegistry, Layout,
    LeaseAccess, LeaseContract, LeaseLifetime, NodeDescriptor, NodeTypeRef, Partiality,
    PolicyContract, PortDescriptor, ProofContract, ResourceEnvelope, StateContract, StateScope,
    Target,
};

pub const LSL_INLET_NODE_TYPE: &str = "org.quitetall.lamquant.lsl.inlet";
pub const LSL_EXPORT_NODE_TYPE: &str = "org.quitetall.lamquant.lsl.export";
pub const LSL_ACCEPT_EXPORT_NODE_TYPE: &str = "org.quitetall.lamquant.lsl.accept-export";
pub const LSL_OUTLET_NODE_TYPE: &str = "org.quitetall.lamquant.lsl.outlet";

pub const LSL_INLET_KERNEL: KernelId = KernelId(0x4c53_0101);
pub const LSL_EXPORT_KERNEL: KernelId = KernelId(0x4c53_0102);
pub const LSL_ACCEPT_EXPORT_KERNEL: KernelId = KernelId(0x4c53_0103);
pub const LSL_OUTLET_KERNEL: KernelId = KernelId(0x4c53_0104);

const CAP_ABIR: &str = "abir.semantic-v1";
const CAP_LSL_ADAPTER: &str = "abir.adapter.lsl.1.16";
const CAP_LSL_INLET: &str = "org.quitetall.lamquant.lsl.experimental-isolated-inlet-v1";
const CAP_LSL_OUTLET: &str = "org.quitetall.lamquant.lsl.outlet-v1";
const FAILURE_DOMAIN: &str = "org.quitetall.lamquant.lsl";
const MAX_DATASET_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_LIVE_PLAN_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RECEIPT_BYTES: u64 = 64 * 1024;
const MAX_REPORT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PEAK_BYTES: u64 = MAX_DATASET_BYTES + 3 * MAX_LIVE_PLAN_BYTES;
// Worst-case admission sums one materialized dataset, the host worker's hard
// resident bound, and the isolated helper's RLIMIT_AS. These are simultaneous
// safety ceilings, not measured steady-state RSS.
const MAX_INLET_PARENT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_INLET_HELPER_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_INLET_PEAK_BYTES: u64 =
    MAX_DATASET_BYTES + MAX_INLET_PARENT_BYTES + MAX_INLET_HELPER_BYTES;
const INLET_AVAILABLE: bool = cfg!(unix);
const POLICY_NETWORK_IMPORT: &str = "abir.policy.network-import-authorized";
const POLICY_NETWORK_EXPORT: &str = "abir.policy.network-export-authorized";
const POLICY_PROTOCOL_110_PEER: &str = "abir.policy.lsl-protocol-110-peer-attested";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LslImplementationIdentity<'a> {
    pub host_source_id: &'a str,
    pub host_feature_set: &'a str,
    pub liblsl_revision: &'a str,
}

pub fn register_lsl_nodes(
    registry: &mut KernelRegistry,
    identity: &LslImplementationIdentity<'_>,
) -> Result<(), CompileError> {
    for operation in [
        Operation::Inlet,
        Operation::Export,
        Operation::AcceptExport,
        Operation::Outlet,
    ] {
        if !INLET_AVAILABLE && matches!(operation, Operation::Inlet) {
            continue;
        }
        let descriptor = descriptor(operation);
        let type_name = descriptor.type_name.clone();
        registry.register_descriptor(descriptor)?;
        registry.register_kernel(kernel(operation, &type_name, identity))?;
    }
    Ok(())
}

pub fn lsl_inlet_descriptor() -> NodeDescriptor {
    descriptor(Operation::Inlet)
}

pub fn lsl_export_descriptor() -> NodeDescriptor {
    descriptor(Operation::Export)
}

pub fn lsl_accept_export_descriptor() -> NodeDescriptor {
    descriptor(Operation::AcceptExport)
}

pub fn lsl_outlet_descriptor() -> NodeDescriptor {
    descriptor(Operation::Outlet)
}

pub fn lsl_inlet_kernel_binding(
    identity: &LslImplementationIdentity<'_>,
) -> (KernelId, ImplementationId) {
    (
        LSL_INLET_KERNEL,
        lsl_implementation_id(LSL_INLET_NODE_TYPE, identity),
    )
}

pub fn lsl_export_kernel_binding(
    identity: &LslImplementationIdentity<'_>,
) -> (KernelId, ImplementationId) {
    (
        LSL_EXPORT_KERNEL,
        lsl_implementation_id(LSL_EXPORT_NODE_TYPE, identity),
    )
}

pub fn lsl_accept_export_kernel_binding(
    identity: &LslImplementationIdentity<'_>,
) -> (KernelId, ImplementationId) {
    (
        LSL_ACCEPT_EXPORT_KERNEL,
        lsl_implementation_id(LSL_ACCEPT_EXPORT_NODE_TYPE, identity),
    )
}

pub fn lsl_outlet_kernel_binding(
    identity: &LslImplementationIdentity<'_>,
) -> (KernelId, ImplementationId) {
    (
        LSL_OUTLET_KERNEL,
        lsl_implementation_id(LSL_OUTLET_NODE_TYPE, identity),
    )
}

#[derive(Clone, Copy)]
enum Operation {
    Inlet,
    Export,
    AcceptExport,
    Outlet,
}

fn descriptor(operation: Operation) -> NodeDescriptor {
    let (type_name, inputs, outputs, capabilities, determinism, config, effect) = match operation {
        Operation::Inlet => (
            LSL_INLET_NODE_TYPE,
            vec![],
            vec![
                dataset_port("dataset"),
                mapping_report_port(),
                receipt_port("inlet_receipt"),
            ],
            vec![
                Capability(CAP_ABIR.into()),
                Capability(CAP_LSL_INLET.into()),
            ],
            Determinism::Nondeterministic,
            inlet_config(),
            Effect::AtMostOnce,
        ),
        Operation::Export => (
            LSL_EXPORT_NODE_TYPE,
            vec![dataset_port("dataset")],
            vec![export_plan_port()],
            vec![
                Capability(CAP_ABIR.into()),
                Capability(CAP_LSL_ADAPTER.into()),
            ],
            Determinism::BitExact,
            export_config(),
            Effect::Pure,
        ),
        Operation::AcceptExport => (
            LSL_ACCEPT_EXPORT_NODE_TYPE,
            vec![dataset_port("dataset"), export_plan_port()],
            vec![accepted_export_port(), fidelity_receipt_port()],
            vec![
                Capability(CAP_ABIR.into()),
                Capability(CAP_LSL_ADAPTER.into()),
            ],
            Determinism::BitExact,
            accept_export_config(),
            Effect::Pure,
        ),
        Operation::Outlet => (
            LSL_OUTLET_NODE_TYPE,
            vec![accepted_export_port()],
            vec![receipt_port("outlet_receipt")],
            vec![
                Capability(CAP_LSL_ADAPTER.into()),
                Capability(CAP_LSL_OUTLET.into()),
            ],
            Determinism::Nondeterministic,
            outlet_config(),
            Effect::AtMostOnce,
        ),
    };
    NodeDescriptor {
        type_name: type_name.into(),
        version: 1,
        inputs,
        outputs,
        capabilities,
        targets: vec![Target::Host],
        resources: operation_resources(operation),
        determinism,
        config,
        state: state_contract(operation),
        subgraph: None,
        proof: empty_proof(),
        policy: operation_policy(operation),
        fidelity: operation_fidelity(operation),
        partiality: if matches!(operation, Operation::Inlet) {
            Partiality::ExplicitGaps
        } else {
            Partiality::Atomic
        },
        failure: FailureContract {
            domains: vec![FAILURE_DOMAIN.into()],
        },
        effect,
        retry_limit: 0,
    }
}

fn kernel(
    operation: Operation,
    type_name: &str,
    identity: &LslImplementationIdentity<'_>,
) -> KernelDescriptor {
    let (id, input_layouts, output_layouts, lowering, determinism) = match operation {
        Operation::Inlet => (
            LSL_INLET_KERNEL,
            vec![],
            vec![Layout::Opaque, Layout::Opaque, Layout::Opaque],
            "lsl:bounded-inlet+abir-import:v1",
            Determinism::Nondeterministic,
        ),
        Operation::Export => (
            LSL_EXPORT_KERNEL,
            vec![Layout::Opaque],
            vec![Layout::Opaque],
            "lsl:abir-live-export-plan:v1",
            Determinism::BitExact,
        ),
        Operation::AcceptExport => (
            LSL_ACCEPT_EXPORT_KERNEL,
            vec![Layout::Opaque, Layout::Opaque],
            vec![Layout::Opaque, Layout::Opaque],
            "lsl:accept-live-export-plan:v1",
            Determinism::BitExact,
        ),
        Operation::Outlet => (
            LSL_OUTLET_KERNEL,
            vec![Layout::Opaque],
            vec![Layout::Opaque],
            "lsl:bounded-outlet-at-most-once:v1",
            Determinism::Nondeterministic,
        ),
    };
    KernelDescriptor {
        id,
        implements: vec![NodeTypeRef {
            type_name: type_name.into(),
            version: 1,
        }],
        implementation_id: lsl_implementation_id(type_name, identity),
        conversion: None,
        target: Target::Host,
        input_layouts,
        output_layouts,
        resources: operation_resources(operation),
        determinism,
        lowering: lowering.into(),
    }
}

fn lsl_implementation_id(
    type_name: &str,
    identity: &LslImplementationIdentity<'_>,
) -> ImplementationId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.lsl-host-implementation-v1\0");
    for field in [
        identity.host_source_id,
        identity.host_feature_set,
        identity.liblsl_revision,
        type_name,
    ] {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    ImplementationId(*hasher.finalize().as_bytes())
}

fn inlet_config() -> ConfigSchema {
    let mut fields = vec![
        text_field("resource_id", 256, true),
        text_field("source_id", 1_024, true),
        text_field("name", 1_024, true),
        text_field("stream_type", 1_024, true),
        text_field("expected_uid", 1_024, false),
        ConfigField {
            name: "channel_count".into(),
            value_type: ConfigType::U64 {
                minimum: 1,
                maximum: 4_096,
            },
            required: true,
            default: None,
        },
        ConfigField {
            name: "sample_type".into(),
            value_type: ConfigType::Choice {
                values: supported_inlet_sample_types(),
            },
            required: true,
            default: None,
        },
        text_field("modality", 1_024, true),
        text_field("receiver_clock_id", 1_024, true),
    ];
    fields.sort_by(|left, right| left.name.cmp(&right.name));
    ConfigSchema { fields }
}

fn export_config() -> ConfigSchema {
    let mut fields = vec![
        text_field("stream_id", 1_024, true),
        text_field("name", 1_024, true),
        text_field("stream_type", 1_024, true),
        text_field("source_id", 1_024, true),
        text_field("description_xml", 65_536, false),
        ConfigField {
            name: "timestamp_policy".into(),
            value_type: ConfigType::Choice {
                values: vec!["exact_nanoseconds".into(), "quantize_to_nanoseconds".into()],
            },
            required: true,
            default: None,
        },
        ConfigField {
            name: "rebase_at_first_push".into(),
            value_type: ConfigType::Bool,
            required: true,
            default: None,
        },
    ];
    fields.sort_by(|left, right| left.name.cmp(&right.name));
    ConfigSchema { fields }
}

fn accept_export_config() -> ConfigSchema {
    ConfigSchema {
        fields: vec![ConfigField {
            name: "accept_projection".into(),
            value_type: ConfigType::Bool,
            required: true,
            default: None,
        }],
    }
}

fn outlet_config() -> ConfigSchema {
    ConfigSchema {
        fields: vec![text_field("resource_id", 256, true)],
    }
}

/// Pinned liblsl binding exposes every LSL 1.16 sample format across targets;
/// Node registration separately gates platform availability.
fn supported_inlet_sample_types() -> Vec<String> {
    vec![
        "float32".into(),
        "double64".into(),
        "string".into(),
        "int32".into(),
        "int16".into(),
        "int8".into(),
        "int64".into(),
    ]
}

fn text_field(name: &str, max_bytes: u32, required: bool) -> ConfigField {
    ConfigField {
        name: name.into(),
        value_type: ConfigType::Text { max_bytes },
        required,
        default: None,
    }
}

fn dataset_port(name: &str) -> PortDescriptor {
    PortDescriptor {
        name: name.into(),
        semantic_type: "abir.dataset".into(),
        optional: false,
        layouts: vec![Layout::Opaque],
        max_bytes: MAX_DATASET_BYTES,
        domain: DomainType {
            root: DomainToken::new("dataset"),
            view: DomainToken::new("root"),
        },
        proof: empty_proof(),
        policy: empty_policy(),
        fidelity: exact_fidelity(),
        extent: opaque_extent(),
        lease: read_lease(),
    }
}

fn accepted_export_port() -> PortDescriptor {
    opaque_port(
        "accepted_export",
        "abir.lsl.accepted-live-export.v1",
        MAX_LIVE_PLAN_BYTES,
    )
}

fn export_plan_port() -> PortDescriptor {
    opaque_port(
        "export_plan",
        "abir.lsl.live-export-plan.v1",
        MAX_LIVE_PLAN_BYTES,
    )
}

fn fidelity_receipt_port() -> PortDescriptor {
    opaque_port(
        "fidelity_receipt",
        "abir.lsl.live-fidelity-receipt.v1",
        MAX_RECEIPT_BYTES,
    )
}

fn mapping_report_port() -> PortDescriptor {
    PortDescriptor {
        name: "mapping_report".into(),
        semantic_type: "abir.mapping_report".into(),
        optional: false,
        layouts: vec![Layout::Opaque],
        max_bytes: MAX_REPORT_BYTES,
        domain: DomainType {
            root: DomainToken::new("mapping_report"),
            view: DomainToken::new("mapping_report"),
        },
        proof: empty_proof(),
        policy: empty_policy(),
        fidelity: exact_fidelity(),
        extent: opaque_extent(),
        lease: read_lease(),
    }
}

fn receipt_port(name: &str) -> PortDescriptor {
    opaque_port(name, &format!("abir.lsl.{name}.v1"), MAX_RECEIPT_BYTES)
}

fn opaque_port(name: &str, semantic_type: &str, max_bytes: u64) -> PortDescriptor {
    PortDescriptor {
        name: name.into(),
        semantic_type: semantic_type.into(),
        optional: false,
        layouts: vec![Layout::Opaque],
        max_bytes,
        domain: DomainType {
            root: DomainToken::new(semantic_type),
            view: DomainToken::new(semantic_type),
        },
        proof: empty_proof(),
        policy: empty_policy(),
        fidelity: exact_fidelity(),
        extent: opaque_extent(),
        lease: read_lease(),
    }
}

fn empty_proof() -> ProofContract {
    ProofContract {
        requires: vec![],
        provides: vec![],
        invalidates: vec![],
    }
}

fn empty_policy() -> PolicyContract {
    PolicyContract {
        requires: vec![],
        adds: vec![],
    }
}

fn operation_policy(operation: Operation) -> PolicyContract {
    match operation {
        Operation::Inlet => PolicyContract {
            requires: vec![
                POLICY_NETWORK_IMPORT.into(),
                POLICY_PROTOCOL_110_PEER.into(),
            ],
            adds: vec![],
        },
        Operation::AcceptExport | Operation::Outlet => PolicyContract {
            requires: vec![POLICY_NETWORK_EXPORT.into()],
            adds: vec![],
        },
        Operation::Export => empty_policy(),
    }
}

fn operation_fidelity(operation: Operation) -> FidelityContract {
    match operation {
        Operation::Inlet | Operation::AcceptExport => FidelityContract {
            minimum_input: u16::MAX,
            maximum_loss: u16::MAX,
        },
        Operation::Outlet => FidelityContract {
            minimum_input: 0,
            maximum_loss: u16::MAX,
        },
        Operation::Export => exact_fidelity(),
    }
}

fn state_contract(operation: Operation) -> StateContract {
    let (scope, max_bytes) = match operation {
        Operation::Inlet | Operation::Outlet => (StateScope::Session, 1),
        Operation::Export | Operation::AcceptExport => (StateScope::Stateless, 0),
    };
    StateContract {
        scope,
        max_bytes,
        checkpoint: CheckpointContract {
            mode: CheckpointMode::Disabled,
            max_snapshot_bytes: 0,
            max_interval_invocations: 0,
        },
    }
}

fn operation_resources(operation: Operation) -> ResourceEnvelope {
    let (peak_bytes, threads) = match operation {
        Operation::Inlet => (MAX_INLET_PEAK_BYTES, 4),
        Operation::Outlet => (MAX_PEAK_BYTES, 3),
        Operation::Export | Operation::AcceptExport => (MAX_PEAK_BYTES, 1),
    };
    ResourceEnvelope::bounded(peak_bytes, MAX_LIVE_PLAN_BYTES, threads)
}

fn exact_fidelity() -> FidelityContract {
    FidelityContract {
        minimum_input: u16::MAX,
        maximum_loss: 0,
    }
}

fn opaque_extent() -> ExtentContract {
    ExtentContract {
        rank: 0,
        maximum_shape: vec![],
        max_elements: 1,
        ragged: false,
        sparse: false,
    }
}

fn read_lease() -> LeaseContract {
    LeaseContract {
        access: LeaseAccess::ReadOnly,
        lifetime: LeaseLifetime::Invocation,
        zero_copy_permitted: false,
        contiguous_required: false,
    }
}
