//! Host-only ABIR standard imports and exact source restoration.
//!
//! Imports retain identity-bound source capsules. Restore nodes emit those
//! exact source bytes; they are not general-purpose foreign serializers.

use std::collections::BTreeMap;

use abir_adapter::{Adapter, AdapterError, PayloadResolver};
use blut_graph_core::{
    AbirRootType, AbirSemanticType, AbirViewType, Capability, CompileError, CompiledNode,
    ConfigField, ConfigSchema, ConfigType, ConfigValue, Determinism, Effect, ExecutionError,
    ExtentContract, FailureContract, FidelityContract, ImplementationId, KernelDescriptor,
    KernelId, KernelRegistry, Layout, LeaseAccess, LeaseContract, LeaseLifetime, NodeDescriptor,
    NodeTypeRef, Partiality, PolicyContract, PortDescriptor, ProofContract, ResourceEnvelope,
    StateContract, StateScope, StructuredFailure, Target,
};
#[cfg(feature = "standard-nwb")]
use lamquant_standard_adapters::NwbAdapter;
use lamquant_standard_adapters::{
    BidsSemanticAdapter, DicomSemanticAdapter, EdfAdapter, XdfAdapter,
};
use semantic_abir::{ContentId, ValidationLimits};

use crate::{implementation_id, AbirDatasetValue, LamQuantNodeValue, NodePayloadStore};

pub const EDFPLUS_IMPORT_NODE_TYPE: &str = "org.quitetall.lamquant.standard.edfplus.import";
pub const BIDS_IMPORT_NODE_TYPE: &str = "org.quitetall.lamquant.standard.bids.import";
pub const DICOM_IMPORT_NODE_TYPE: &str = "org.quitetall.lamquant.standard.dicom.import";
pub const NWB_IMPORT_NODE_TYPE: &str = "org.quitetall.lamquant.standard.nwb.import";
pub const XDF_IMPORT_NODE_TYPE: &str = "org.quitetall.lamquant.standard.xdf.import";

pub const EDFPLUS_RESTORE_NODE_TYPE: &str = "org.quitetall.lamquant.standard.edfplus.restore";
pub const BIDS_RESTORE_NODE_TYPE: &str = "org.quitetall.lamquant.standard.bids.restore";
pub const DICOM_RESTORE_NODE_TYPE: &str = "org.quitetall.lamquant.standard.dicom.restore";
pub const NWB_RESTORE_NODE_TYPE: &str = "org.quitetall.lamquant.standard.nwb.restore";
pub const XDF_RESTORE_NODE_TYPE: &str = "org.quitetall.lamquant.standard.xdf.restore";

pub const EDFPLUS_SINK_NODE_TYPE: &str = "org.quitetall.lamquant.standard.edfplus.sink";
pub const BIDS_SINK_NODE_TYPE: &str = "org.quitetall.lamquant.standard.bids.sink";
pub const DICOM_SINK_NODE_TYPE: &str = "org.quitetall.lamquant.standard.dicom.sink";
pub const NWB_SINK_NODE_TYPE: &str = "org.quitetall.lamquant.standard.nwb.sink";
pub const XDF_SINK_NODE_TYPE: &str = "org.quitetall.lamquant.standard.xdf.sink";

const CAP_ABIR: &str = "abir.semantic-v1";
const CAP_SOURCE_CAPSULE: &str = "abir.source-capsule.identity-bound-v1";
const SOURCE_CAPSULE_PROOF: &str = "org.quitetall.abir.proof.identity-bound-source-capsule-v1";
pub const EXACT_SOURCE_RESTORATION_PROOF: &str =
    "org.quitetall.abir.proof.exact-source-restoration-v1";
const CAP_DURABLE_FILE_SINK: &str = "org.quitetall.lamquant.sink.durable-file-v1";
const FAILURE_DOMAIN: &str = "org.quitetall.lamquant.standard";
// Current adapters materialize decoded host values before ABIR payloads.
// Keep source cap conservative until streaming decoders replace those copies.
const MAX_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DATASET_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_SEMANTIC_PAYLOAD_BYTES: u64 = 512 * 1024 * 1024;
const MAX_REPORT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PEAK_BYTES: u64 =
    2 * MAX_SOURCE_BYTES + 2 * MAX_SEMANTIC_PAYLOAD_BYTES + MAX_REPORT_BYTES;
const MAX_FOREIGN_ENTRIES: usize = 1_000_000;

const STANDARD_SPECS: &[StandardSpec] = &[
    StandardSpec {
        import_type: EDFPLUS_IMPORT_NODE_TYPE,
        restore_type: EDFPLUS_RESTORE_NODE_TYPE,
        sink_type: EDFPLUS_SINK_NODE_TYPE,
        profile: "edfplus.1",
        kernel_base: 0x5354_0100,
        adapter: AdapterKind::EdfPlus,
    },
    StandardSpec {
        import_type: BIDS_IMPORT_NODE_TYPE,
        restore_type: BIDS_RESTORE_NODE_TYPE,
        sink_type: BIDS_SINK_NODE_TYPE,
        profile: "bids.1.11.1",
        kernel_base: 0x5354_0200,
        adapter: AdapterKind::Bids,
    },
    StandardSpec {
        import_type: DICOM_IMPORT_NODE_TYPE,
        restore_type: DICOM_RESTORE_NODE_TYPE,
        sink_type: DICOM_SINK_NODE_TYPE,
        profile: "dicom.ps3.2026c",
        kernel_base: 0x5354_0300,
        adapter: AdapterKind::Dicom,
    },
    #[cfg(feature = "standard-nwb")]
    StandardSpec {
        import_type: NWB_IMPORT_NODE_TYPE,
        restore_type: NWB_RESTORE_NODE_TYPE,
        sink_type: NWB_SINK_NODE_TYPE,
        profile: "nwb.2.10.0",
        kernel_base: 0x5354_0400,
        adapter: AdapterKind::Nwb,
    },
    StandardSpec {
        import_type: XDF_IMPORT_NODE_TYPE,
        restore_type: XDF_RESTORE_NODE_TYPE,
        sink_type: XDF_SINK_NODE_TYPE,
        profile: "xdf.1.0",
        kernel_base: 0x5354_0500,
        adapter: AdapterKind::Xdf,
    },
];

#[derive(Clone, Copy)]
struct StandardSpec {
    import_type: &'static str,
    restore_type: &'static str,
    sink_type: &'static str,
    profile: &'static str,
    kernel_base: u32,
    adapter: AdapterKind,
}

#[derive(Clone, Copy)]
enum AdapterKind {
    EdfPlus,
    Bids,
    Dicom,
    #[cfg(feature = "standard-nwb")]
    Nwb,
    Xdf,
}

impl StandardSpec {
    fn for_type(type_name: &str) -> Option<(Self, Operation)> {
        STANDARD_SPECS.iter().copied().find_map(|spec| {
            if type_name == spec.import_type {
                Some((spec, Operation::Import))
            } else if type_name == spec.restore_type {
                Some((spec, Operation::Restore))
            } else if type_name == spec.sink_type {
                Some((spec, Operation::Sink))
            } else {
                None
            }
        })
    }

    fn adapter(self, max_source_bytes: u64) -> Box<dyn Adapter> {
        match self.adapter {
            AdapterKind::EdfPlus => Box::new(EdfAdapter::new(max_source_bytes)),
            AdapterKind::Bids => Box::new(BidsSemanticAdapter::new(max_source_bytes)),
            AdapterKind::Dicom => Box::new(DicomSemanticAdapter::new(max_source_bytes)),
            #[cfg(feature = "standard-nwb")]
            AdapterKind::Nwb => Box::new(NwbAdapter::with_decoded_limit(
                max_source_bytes,
                MAX_SEMANTIC_PAYLOAD_BYTES,
            )),
            AdapterKind::Xdf => Box::new(XdfAdapter::new(max_source_bytes)),
        }
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Import,
    Restore,
    Sink,
}

impl PayloadResolver for NodePayloadStore {
    fn resolve(&self, content_id: ContentId) -> Result<Vec<u8>, AdapterError> {
        self.get(content_id)
            .map(<[u8]>::to_vec)
            .ok_or(AdapterError::MissingPayload(content_id))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StandardNodeConfigError {
    DestinationResourceInvalid,
    MaxSourceBytesOutOfRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StandardSinkContract {
    pub profile: &'static str,
    pub destination_resource: String,
    pub max_source_bytes: u64,
}

impl core::fmt::Display for StandardNodeConfigError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DestinationResourceInvalid => {
                formatter.write_str("destination resource identifier is invalid")
            }
            Self::MaxSourceBytesOutOfRange => {
                formatter.write_str("maximum source bytes is outside the supported range")
            }
        }
    }
}

impl std::error::Error for StandardNodeConfigError {}

pub fn standard_node_config(
    max_source_bytes: u64,
) -> Result<BTreeMap<String, ConfigValue>, StandardNodeConfigError> {
    if !(1..=MAX_SOURCE_BYTES).contains(&max_source_bytes) {
        return Err(StandardNodeConfigError::MaxSourceBytesOutOfRange);
    }
    Ok(BTreeMap::from([(
        "max_source_bytes".into(),
        ConfigValue::U64(max_source_bytes),
    )]))
}

pub fn standard_sink_node_config(
    destination_resource: &str,
    max_source_bytes: u64,
) -> Result<BTreeMap<String, ConfigValue>, StandardNodeConfigError> {
    if !valid_resource_id(destination_resource) {
        return Err(StandardNodeConfigError::DestinationResourceInvalid);
    }
    if !(1..=MAX_SOURCE_BYTES).contains(&max_source_bytes) {
        return Err(StandardNodeConfigError::MaxSourceBytesOutOfRange);
    }
    Ok(BTreeMap::from([
        (
            "destination_resource".into(),
            ConfigValue::Text(destination_resource.to_owned()),
        ),
        (
            "max_source_bytes".into(),
            ConfigValue::U64(max_source_bytes),
        ),
    ]))
}

pub fn is_standard_node(type_name: &str) -> bool {
    StandardSpec::for_type(type_name).is_some()
}

pub fn execute_standard<'a>(
    node: &CompiledNode,
    type_name: &str,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    let (spec, operation) = StandardSpec::for_type(type_name)
        .ok_or_else(|| kernel_failure(node, "unsupported-node", "unknown standard node"))?;
    let max_source_bytes = parse_max_source_bytes(node)?;
    let adapter = spec.adapter(max_source_bytes);
    match operation {
        Operation::Import => execute_import(node, adapter.as_ref(), max_source_bytes, inputs),
        Operation::Restore => execute_restore(node, adapter.as_ref(), max_source_bytes, inputs),
        Operation::Sink => {
            let _ = parse_standard_sink_contract(node)?;
            Err(kernel_failure(
                node,
                "not-implemented",
                "durable sink execution is not implemented in this build",
            ))
        }
    }
}

fn execute_import<'a>(
    node: &CompiledNode,
    adapter: &dyn Adapter,
    max_source_bytes: u64,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    let source = match inputs {
        [Some(LamQuantNodeValue::ForeignObject(source))] => source,
        _ => {
            return Err(kernel_failure(
                node,
                "invalid-input",
                "standard import requires one ForeignObject input",
            ))
        }
    };
    if source.entries.len() > MAX_FOREIGN_ENTRIES
        || foreign_object_bytes(source).is_none_or(|bytes| bytes > max_source_bytes)
    {
        return Err(kernel_failure(
            node,
            "resource-limit",
            "foreign source exceeds configured byte limit",
        ));
    }
    let outcome = adapter
        .import(source, node_validation_limits())
        .map_err(|error| adapter_failure(node, error))?;
    if !mapping_report_fits(&outcome.report) {
        return Err(kernel_failure(
            node,
            "resource-limit",
            "imported mapping report exceeds node output limits",
        ));
    }
    let exact_source_plan = adapter
        .plan_export(&outcome.dataset)
        .is_ok_and(|plan| plan.accepts_without_loss());
    if outcome.dataset.source_capsules().is_empty() || !exact_source_plan {
        return Err(kernel_failure(
            node,
            "source-capsule-proof",
            "imported dataset lacks an exact source capsule for this adapter profile",
        ));
    }
    let value = AbirDatasetValue::try_new(
        outcome.dataset,
        outcome
            .payloads
            .into_iter()
            .map(|payload| (payload.content_id, payload.bytes)),
        MAX_DATASET_BYTES,
    )
    .map_err(|error| {
        kernel_failure(
            node,
            "payload-closure",
            &format!("imported ABIR payload closure is invalid: {error}"),
        )
    })?;
    Ok(vec![
        LamQuantNodeValue::AbirDataset(Box::new(value)),
        LamQuantNodeValue::MappingReport(outcome.report),
    ])
}

fn execute_restore<'a>(
    node: &CompiledNode,
    adapter: &dyn Adapter,
    max_source_bytes: u64,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    let source = match inputs {
        [Some(LamQuantNodeValue::AbirDataset(source))] => source,
        _ => {
            return Err(kernel_failure(
                node,
                "invalid-input",
                "standard restore requires one AbirDataset input",
            ))
        }
    };
    let plan = adapter
        .plan_export(source.dataset())
        .map_err(|error| adapter_failure(node, error))?;
    if !plan.accepts_without_loss() {
        return Err(kernel_failure(
            node,
            "semantic-loss",
            "exact source restoration is unavailable for this dataset and profile",
        ));
    }
    let (foreign, receipt) = adapter
        .export(source.dataset(), &plan, source.payloads())
        .map_err(|error| adapter_failure(node, error))?;
    if !receipt.exact_source_restoration
        || foreign.entries.len() > MAX_FOREIGN_ENTRIES
        || foreign_object_bytes(&foreign).is_none_or(|bytes| bytes > max_source_bytes)
        || !fidelity_receipt_fits(&receipt)
    {
        return Err(kernel_failure(
            node,
            "resource-or-fidelity",
            "restored source exceeds limits or lacks exact-source attestation",
        ));
    }
    Ok(vec![
        LamQuantNodeValue::ForeignObject(foreign),
        LamQuantNodeValue::FidelityReceipt(receipt),
    ])
}

fn parse_max_source_bytes(node: &CompiledNode) -> Result<u64, ExecutionError> {
    let config = node
        .semantic_configs
        .first()
        .ok_or_else(|| kernel_failure(node, "invalid-plan", "missing standard node config"))?;
    match config.get("max_source_bytes") {
        Some(ConfigValue::U64(bytes)) if (1..=MAX_SOURCE_BYTES).contains(bytes) => Ok(*bytes),
        _ => Err(kernel_failure(
            node,
            "invalid-config",
            "max_source_bytes is missing or outside the supported range",
        )),
    }
}

pub fn parse_standard_sink_contract(
    node: &CompiledNode,
) -> Result<StandardSinkContract, ExecutionError> {
    let type_name = node
        .semantic_types
        .first()
        .map(|node_type| node_type.type_name.as_str())
        .ok_or_else(|| kernel_failure(node, "invalid-plan", "missing standard sink node type"))?;
    let (spec, operation) = StandardSpec::for_type(type_name)
        .ok_or_else(|| kernel_failure(node, "invalid-plan", "unknown standard sink node type"))?;
    if !matches!(operation, Operation::Sink) {
        return Err(kernel_failure(
            node,
            "invalid-plan",
            "compiled node is not a standard sink",
        ));
    }
    let config = node
        .semantic_configs
        .first()
        .ok_or_else(|| kernel_failure(node, "invalid-plan", "missing standard node config"))?;
    let destination_resource = match config.get("destination_resource") {
        Some(ConfigValue::Text(destination_resource))
            if valid_resource_id(destination_resource) =>
        {
            destination_resource.clone()
        }
        _ => {
            return Err(kernel_failure(
                node,
                "invalid-config",
                "destination_resource is missing or invalid",
            ))
        }
    };
    let max_source_bytes = parse_max_source_bytes(node)?;
    Ok(StandardSinkContract {
        profile: spec.profile,
        destination_resource,
        max_source_bytes,
    })
}

fn valid_resource_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value != "."
        && !value.contains("..")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':'))
}

fn adapter_failure(node: &CompiledNode, error: AdapterError) -> ExecutionError {
    kernel_failure(node, adapter_error_code(&error), &error.to_string())
}

fn adapter_error_code(error: &AdapterError) -> &'static str {
    match error {
        AdapterError::SourceTooLarge => "resource-limit",
        AdapterError::ProfileMismatch { .. }
        | AdapterError::EmptySource
        | AdapterError::DuplicatePath(_)
        | AdapterError::InvalidPath(_)
        | AdapterError::InvalidSource(_) => "invalid-input",
        AdapterError::AbirValidation | AdapterError::MissingPayload(_) => "payload-closure",
        AdapterError::ExportPlanMismatch => "invalid-plan",
        AdapterError::ExportRequiresAcceptance | AdapterError::UnsupportedMeaning(_) => {
            "semantic-loss"
        }
        AdapterError::AdapterUnavailable { .. } => "adapter-unavailable",
    }
}

fn kernel_failure(node: &CompiledNode, code: &str, message: &str) -> ExecutionError {
    ExecutionError::KernelFailed {
        kernel: node.kernel,
        failure: StructuredFailure {
            domain: FAILURE_DOMAIN.into(),
            code: code.into(),
            message: message.into(),
            retryable: false,
        },
    }
}

fn node_validation_limits() -> ValidationLimits {
    ValidationLimits {
        max_logical_payload_bytes: MAX_SEMANTIC_PAYLOAD_BYTES,
        ..ValidationLimits::default()
    }
}

fn foreign_object_bytes(source: &abir_adapter::ForeignObject) -> Option<u64> {
    source
        .entries
        .iter()
        .try_fold(source.profile.0.len() as u64 + 64, |total, entry| {
            total
                .checked_add(entry.bytes.len() as u64)?
                .checked_add(entry.path.len() as u64)?
                .checked_add(
                    entry
                        .media_type
                        .as_ref()
                        .map_or(0_u64, |value| value.len() as u64),
                )?
                .checked_add(64)
        })
}

fn mapping_report_fits(report: &abir_adapter::MappingReport) -> bool {
    let initial = (report.source_profile.0.len() + report.target_profile.0.len()) as u64;
    report
        .entries
        .iter()
        .try_fold(initial, |total, entry| {
            total
                .checked_add(entry.source_path.len() as u64)?
                .checked_add(entry.target.len() as u64)?
                .checked_add(
                    entry
                        .reason
                        .as_ref()
                        .map_or(0_u64, |value| value.len() as u64),
                )?
                .checked_add(64)
        })
        .is_some_and(|bytes| bytes <= MAX_REPORT_BYTES)
}

fn fidelity_receipt_fits(receipt: &abir_adapter::FidelityReceipt) -> bool {
    receipt
        .output_content_ids
        .iter()
        .try_fold(receipt.plan_id.len() as u64, |total, content_id| {
            total.checked_add(content_id.len() as u64)?.checked_add(32)
        })
        .is_some_and(|bytes| bytes <= MAX_REPORT_BYTES)
}

pub fn register_standard_nodes(registry: &mut KernelRegistry) -> Result<(), CompileError> {
    for spec in STANDARD_SPECS {
        for operation in [Operation::Import, Operation::Restore, Operation::Sink] {
            let descriptor = descriptor(*spec, operation);
            let type_name = descriptor.type_name.clone();
            registry.register_descriptor(descriptor)?;
            registry.register_kernel(standard_kernel(
                KernelId(
                    spec.kernel_base
                        + match operation {
                            Operation::Import => 1,
                            Operation::Restore => 2,
                            Operation::Sink => 3,
                        },
                ),
                &type_name,
                operation,
            ))?;
        }
    }
    Ok(())
}

pub fn standard_import_descriptor(profile: &str) -> Option<NodeDescriptor> {
    STANDARD_SPECS
        .iter()
        .copied()
        .find(|spec| spec.profile == profile)
        .map(|spec| descriptor(spec, Operation::Import))
}

pub fn standard_restore_descriptor(profile: &str) -> Option<NodeDescriptor> {
    STANDARD_SPECS
        .iter()
        .copied()
        .find(|spec| spec.profile == profile)
        .map(|spec| descriptor(spec, Operation::Restore))
}

pub fn standard_sink_descriptor(profile: &str) -> Option<NodeDescriptor> {
    STANDARD_SPECS
        .iter()
        .copied()
        .find(|spec| spec.profile == profile)
        .map(|spec| descriptor(spec, Operation::Sink))
}

pub fn standard_sink_kernel_binding(profile: &str) -> Option<(KernelId, ImplementationId)> {
    STANDARD_SPECS
        .iter()
        .copied()
        .find(|spec| spec.profile == profile)
        .map(|spec| {
            (
                KernelId(spec.kernel_base + 3),
                implementation_id(spec.sink_type, Target::Host),
            )
        })
}

fn descriptor(spec: StandardSpec, operation: Operation) -> NodeDescriptor {
    let source_proof = source_capsule_proof(spec.profile);
    let (type_name, inputs, outputs, proof, config, capabilities, effect) = match operation {
        Operation::Import => (
            spec.import_type,
            vec![foreign_port("source", spec.profile)],
            vec![
                dataset_port("dataset", &source_proof, true),
                report_port("mapping_report"),
            ],
            ProofContract {
                requires: vec![],
                provides: vec![source_proof.clone()],
                invalidates: vec![],
            },
            config_schema(),
            vec![
                Capability(CAP_ABIR.into()),
                Capability(CAP_SOURCE_CAPSULE.into()),
                Capability(format!("abir.adapter.{}", spec.profile)),
            ],
            Effect::Pure,
        ),
        Operation::Restore => (
            spec.restore_type,
            vec![dataset_port("dataset", &source_proof, false)],
            vec![
                foreign_port("source", spec.profile),
                fidelity_receipt_port("fidelity_receipt", true),
            ],
            ProofContract {
                requires: vec![source_proof],
                provides: vec![EXACT_SOURCE_RESTORATION_PROOF.into()],
                invalidates: vec![],
            },
            config_schema(),
            vec![
                Capability(CAP_ABIR.into()),
                Capability(CAP_SOURCE_CAPSULE.into()),
                Capability(format!("abir.adapter.{}", spec.profile)),
            ],
            Effect::Pure,
        ),
        Operation::Sink => (
            spec.sink_type,
            vec![
                foreign_port("source", spec.profile),
                fidelity_receipt_port("fidelity_receipt", false),
            ],
            vec![],
            ProofContract {
                requires: vec![EXACT_SOURCE_RESTORATION_PROOF.into()],
                provides: vec![],
                invalidates: vec![],
            },
            sink_config_schema(),
            vec![
                Capability(format!("abir.foreign-tree.{}", spec.profile)),
                Capability(CAP_DURABLE_FILE_SINK.into()),
            ],
            Effect::Transactional,
        ),
    };
    NodeDescriptor {
        type_name: type_name.into(),
        version: 1,
        inputs,
        outputs,
        capabilities,
        targets: vec![Target::Host],
        resources: ResourceEnvelope::bounded(MAX_PEAK_BYTES, MAX_SOURCE_BYTES, 1),
        determinism: Determinism::BitExact,
        config,
        state: StateContract {
            scope: StateScope::Stateless,
            max_bytes: 0,
            checkpoint: blut_graph_core::CheckpointContract {
                mode: blut_graph_core::CheckpointMode::Disabled,
                max_snapshot_bytes: 0,
                max_interval_invocations: 0,
            },
        },
        subgraph: None,
        proof,
        policy: PolicyContract {
            requires: vec![],
            adds: vec![],
        },
        fidelity: FidelityContract {
            minimum_input: u16::MAX,
            maximum_loss: 0,
        },
        partiality: Partiality::Atomic,
        failure: FailureContract {
            domains: vec![FAILURE_DOMAIN.into()],
        },
        effect,
        retry_limit: 0,
    }
}

fn config_schema() -> ConfigSchema {
    ConfigSchema {
        fields: vec![ConfigField {
            name: "max_source_bytes".into(),
            value_type: ConfigType::U64 {
                minimum: 1,
                maximum: MAX_SOURCE_BYTES,
            },
            required: true,
            default: None,
        }],
    }
}

fn sink_config_schema() -> ConfigSchema {
    ConfigSchema {
        fields: vec![
            ConfigField {
                name: "destination_resource".into(),
                value_type: ConfigType::Text { max_bytes: 256 },
                required: true,
                default: None,
            },
            ConfigField {
                name: "max_source_bytes".into(),
                value_type: ConfigType::U64 {
                    minimum: 1,
                    maximum: MAX_SOURCE_BYTES,
                },
                required: true,
                default: None,
            },
        ],
    }
}

fn foreign_port(name: &str, profile: &str) -> PortDescriptor {
    let semantic_type = format!("abir.foreign-object.{profile}");
    PortDescriptor {
        name: name.into(),
        semantic_type: semantic_type.clone(),
        optional: false,
        layouts: vec![Layout::Packed],
        max_bytes: MAX_SOURCE_BYTES,
        abir: AbirSemanticType {
            root: AbirRootType::Unknown(semantic_type.clone()),
            view: AbirViewType::Unknown(semantic_type),
        },
        proof: empty_proof(),
        policy: empty_policy(),
        fidelity: exact_fidelity(),
        extent: byte_extent(MAX_SOURCE_BYTES),
        lease: read_lease(false),
    }
}

fn source_capsule_proof(profile: &str) -> String {
    format!("{SOURCE_CAPSULE_PROOF}.{profile}")
}

fn dataset_port(
    name: &str,
    source_capsule_proof: &str,
    provides_source_capsule: bool,
) -> PortDescriptor {
    PortDescriptor {
        name: name.into(),
        semantic_type: "abir.dataset".into(),
        optional: false,
        layouts: vec![Layout::Opaque],
        max_bytes: MAX_DATASET_BYTES,
        abir: AbirSemanticType {
            root: AbirRootType::Dataset,
            view: AbirViewType::Root,
        },
        proof: if provides_source_capsule {
            ProofContract {
                requires: vec![],
                provides: vec![source_capsule_proof.into()],
                invalidates: vec![],
            }
        } else {
            ProofContract {
                requires: vec![source_capsule_proof.into()],
                provides: vec![],
                invalidates: vec![],
            }
        },
        policy: empty_policy(),
        fidelity: exact_fidelity(),
        extent: opaque_extent(),
        lease: read_lease(false),
    }
}

fn report_port(name: &str) -> PortDescriptor {
    PortDescriptor {
        name: name.into(),
        semantic_type: format!("abir.{name}"),
        optional: false,
        layouts: vec![Layout::Opaque],
        max_bytes: MAX_REPORT_BYTES,
        abir: AbirSemanticType {
            root: AbirRootType::Unknown(name.into()),
            view: AbirViewType::Unknown(name.into()),
        },
        proof: empty_proof(),
        policy: empty_policy(),
        fidelity: exact_fidelity(),
        extent: opaque_extent(),
        lease: read_lease(false),
    }
}

fn fidelity_receipt_port(name: &str, provides_proof: bool) -> PortDescriptor {
    let mut port = report_port(name);
    port.proof = if provides_proof {
        ProofContract {
            requires: vec![],
            provides: vec![EXACT_SOURCE_RESTORATION_PROOF.into()],
            invalidates: vec![],
        }
    } else {
        ProofContract {
            requires: vec![EXACT_SOURCE_RESTORATION_PROOF.into()],
            provides: vec![],
            invalidates: vec![],
        }
    };
    port
}

fn standard_kernel(id: KernelId, type_name: &str, operation: Operation) -> KernelDescriptor {
    let (input_layouts, output_layouts, lowering) = match operation {
        Operation::Import => (
            vec![Layout::Packed],
            vec![Layout::Opaque, Layout::Opaque],
            "adapter:import+payload-closure:v1",
        ),
        Operation::Restore => (
            vec![Layout::Opaque],
            vec![Layout::Packed, Layout::Opaque],
            "adapter:plan-export+exact-source-export:v1",
        ),
        Operation::Sink => (
            vec![Layout::Packed, Layout::Opaque],
            vec![],
            "adapter:accepted-plan-durable-tree-sink:v1",
        ),
    };
    KernelDescriptor {
        id,
        implements: vec![NodeTypeRef {
            type_name: type_name.into(),
            version: 1,
        }],
        implementation_id: implementation_id(type_name, Target::Host),
        conversion: None,
        target: Target::Host,
        input_layouts,
        output_layouts,
        resources: ResourceEnvelope::bounded(MAX_PEAK_BYTES, MAX_SOURCE_BYTES, 1),
        determinism: Determinism::BitExact,
        lowering: lowering.into(),
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

fn exact_fidelity() -> FidelityContract {
    FidelityContract {
        minimum_input: u16::MAX,
        maximum_loss: 0,
    }
}

fn byte_extent(maximum: u64) -> ExtentContract {
    ExtentContract {
        rank: 1,
        maximum_shape: vec![maximum],
        max_elements: maximum,
        ragged: false,
        sparse: false,
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

fn read_lease(zero_copy_permitted: bool) -> LeaseContract {
    LeaseContract {
        access: LeaseAccess::ReadOnly,
        lifetime: LeaseLifetime::Invocation,
        zero_copy_permitted,
        contiguous_required: false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        adapter_error_code, standard_sink_descriptor, standard_sink_node_config,
        StandardNodeConfigError, EXACT_SOURCE_RESTORATION_PROOF,
    };
    use abir_adapter::{AdapterError, ProfileId};
    use semantic_abir::ContentId;

    #[test]
    fn adapter_errors_have_stable_structured_failure_classes() {
        let profile = ProfileId("test".to_owned());
        let cases = [
            (AdapterError::SourceTooLarge, "resource-limit"),
            (
                AdapterError::ProfileMismatch {
                    expected: profile.clone(),
                    actual: profile,
                },
                "invalid-input",
            ),
            (AdapterError::EmptySource, "invalid-input"),
            (AdapterError::DuplicatePath("x".to_owned()), "invalid-input"),
            (AdapterError::InvalidPath("x".to_owned()), "invalid-input"),
            (AdapterError::InvalidSource("x".to_owned()), "invalid-input"),
            (AdapterError::AbirValidation, "payload-closure"),
            (
                AdapterError::MissingPayload(ContentId::from_bytes([0; 32])),
                "payload-closure",
            ),
            (AdapterError::ExportPlanMismatch, "invalid-plan"),
            (AdapterError::ExportRequiresAcceptance, "semantic-loss"),
            (
                AdapterError::UnsupportedMeaning("x".to_owned()),
                "semantic-loss",
            ),
            (
                AdapterError::AdapterUnavailable {
                    package: "x".to_owned(),
                    capability: "y".to_owned(),
                },
                "adapter-unavailable",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(adapter_error_code(&error), expected);
        }
    }

    #[test]
    fn sink_resource_ids_are_bounded_opaque_names() {
        assert!(standard_sink_node_config("archive:clinical-01", 1).is_ok());
        for invalid in [
            "",
            ".",
            "..",
            "archive..escape",
            "../escape",
            "path/escape",
            "white space",
            "nul\0byte",
        ] {
            assert_eq!(
                standard_sink_node_config(invalid, 1),
                Err(StandardNodeConfigError::DestinationResourceInvalid)
            );
        }
        assert_eq!(
            standard_sink_node_config(&"x".repeat(257), 1),
            Err(StandardNodeConfigError::DestinationResourceInvalid)
        );
    }

    #[test]
    fn sink_requires_exact_export_receipt_and_proof() {
        let descriptor = standard_sink_descriptor("bids.1.11.1").unwrap();
        assert_eq!(descriptor.inputs.len(), 2);
        assert_eq!(descriptor.inputs[0].name, "source");
        assert_eq!(descriptor.inputs[1].name, "fidelity_receipt");
        assert_eq!(
            descriptor.inputs[1].proof.requires,
            [EXACT_SOURCE_RESTORATION_PROOF]
        );
        assert_eq!(descriptor.proof.requires, [EXACT_SOURCE_RESTORATION_PROOF]);
    }
}
