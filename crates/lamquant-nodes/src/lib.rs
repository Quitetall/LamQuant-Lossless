#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
//! Production LamQuant ABIR Node descriptors and fused kernel adapters.

extern crate alloc;

mod abir_value;
mod lml_reference;
#[cfg(feature = "standard-adapters")]
mod standard_nodes;

pub use abir_value::{AbirDatasetValue, AbirDatasetValueError, NodePayloadStore};
pub use lml_reference::{
    LmlPackets, ReferenceEntropy, ReferencePredicted, ReferenceQuantized, ReferenceSubbands,
};
#[cfg(feature = "standard-adapters")]
pub use standard_nodes::{
    parse_standard_sink_contract, register_standard_nodes, standard_import_descriptor,
    standard_node_config, standard_restore_descriptor, standard_sink_descriptor,
    standard_sink_kernel_binding, standard_sink_node_config, StandardNodeConfigError,
    StandardSinkContract, BIDS_IMPORT_NODE_TYPE, BIDS_RESTORE_NODE_TYPE, BIDS_SINK_NODE_TYPE,
    DICOM_IMPORT_NODE_TYPE, DICOM_RESTORE_NODE_TYPE, DICOM_SINK_NODE_TYPE,
    EDFPLUS_IMPORT_NODE_TYPE, EDFPLUS_RESTORE_NODE_TYPE, EDFPLUS_SINK_NODE_TYPE,
    NWB_IMPORT_NODE_TYPE, NWB_RESTORE_NODE_TYPE, NWB_SINK_NODE_TYPE, XDF_IMPORT_NODE_TYPE,
    XDF_RESTORE_NODE_TYPE, XDF_SINK_NODE_TYPE,
};

pub const LML_TRANSFORM_NODE_TYPE: &str = lml_reference::LML_TRANSFORM_NODE_TYPE;
pub const LML_QUANTIZE_NODE_TYPE: &str = lml_reference::LML_QUANTIZE_NODE_TYPE;
pub const LML_PREDICT_NODE_TYPE: &str = lml_reference::LML_PREDICT_NODE_TYPE;
pub const LML_ENTROPY_NODE_TYPE: &str = lml_reference::LML_ENTROPY_NODE_TYPE;
pub const LML_ASSEMBLE_NODE_TYPE: &str = lml_reference::LML_ASSEMBLE_NODE_TYPE;
pub const LML_PACKET_BASELINE_NODE_TYPE: &str = lml_reference::LML_PACKET_BASELINE_NODE_TYPE;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use blut_graph_core::{
    AbirRootType, AbirSemanticType, AbirViewType, Capability, CheckpointMode, CompileError,
    CompiledNode, ConfigField, ConfigSchema, ConfigType, ConfigValue, Determinism, Effect,
    ExecutionError, ExtentContract, FailureContract, FidelityContract, ImplementationId,
    KernelDescriptor, KernelExecutor, KernelId, KernelRegistry, Layout, LeaseAccess, LeaseContract,
    LeaseLifetime, NodeDescriptor, NodeTypeRef, Partiality, PolicyContract, PortDescriptor,
    ProofContract, ResourceEnvelope, StateContract, StateScope, Target, TransactionalSink,
};
use lamquant_abir_codec::encode_lml_bundle_from_views_explicit;
use lamquant_lml_mcu::lml::EncodeFeatures;
use lamquant_lml_mcu::lpc::LpcMode;
use semantic_abir::AbirDataset;
use semantic_abir_bcs::ResourceBounds;

pub const LML_BASELINE_NODE_TYPE: &str = "org.quitetall.lamquant.lml.encode.baseline";
pub const LML_ARITHMETIC_NODE_TYPE: &str = "org.quitetall.lamquant.lml.encode.arithmetic";
pub const CAP_LML_ARITHMETIC_NODE: &str = "bcs2.cap.lml-arithmetic-v1";
const CAP_ABIR: &str = "abir.semantic-v1";
const CAP_LML: &str = "bcs.lml.lossless-v1";
const FAILURE_DOMAIN: &str = "org.quitetall.lamquant.lml.encode";
const SIGNAL_SEMANTIC_TYPE: &str = "abir.dataset.uniform-signed-integer-recording";
const BUNDLE_SEMANTIC_TYPE: &str = "bcs2.bundle.lml-lossless";
const MAX_SIGNAL_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_PACKET_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_PARALLEL_CHANNELS: u16 = 1024;
const SOURCE_ID: &str = env!("LAMQUANT_NODES_SOURCE_ID");
const FEATURE_SET: &str = env!("LAMQUANT_NODES_FEATURE_SET");

const BASELINE_HOST_KERNEL: KernelId = KernelId(0x4c4d_0101);
/// The slot between host and BLUT was already reserved for it.
const BASELINE_MCU_KERNEL: KernelId = KernelId(0x4c4d_0102);
const BASELINE_BLUT_KERNEL: KernelId = KernelId(0x4c4d_0103);
#[cfg(feature = "experimental-arithmetic")]
const ARITHMETIC_HOST_KERNEL: KernelId = KernelId(0x4c4d_0201);
#[cfg(feature = "experimental-arithmetic")]
const ARITHMETIC_BLUT_KERNEL: KernelId = KernelId(0x4c4d_0202);

/// Borrowed input shape accepted by LML encoder nodes.
///
/// Kernels verify dataset payload closure before transforming samples.
#[derive(Clone, Copy, Debug)]
pub struct LmlSignalView<'a> {
    dataset: &'a AbirDataset,
    channels: &'a [&'a [i64]],
    bounds: ResourceBounds,
}

impl<'a> LmlSignalView<'a> {
    pub fn new(
        dataset: &'a AbirDataset,
        channels: &'a [&'a [i64]],
        bounds: ResourceBounds,
    ) -> Result<Self, LmlSignalViewError> {
        let channel_count = channels.len();
        if !(1..=1024).contains(&channel_count) {
            return Err(LmlSignalViewError::ChannelCountOutOfRange);
        }
        let sample_count = channels[0].len();
        if !(1..=131_072).contains(&sample_count) {
            return Err(LmlSignalViewError::SampleCountOutOfRange);
        }
        if channels.iter().any(|channel| channel.len() != sample_count) {
            return Err(LmlSignalViewError::RaggedChannels);
        }
        let bytes = channel_count
            .checked_mul(sample_count)
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<i64>()))
            .ok_or(LmlSignalViewError::ByteExtentOverflow)?;
        if bytes as u64 > MAX_SIGNAL_BYTES {
            return Err(LmlSignalViewError::ByteExtentExceeded);
        }
        Ok(Self {
            dataset,
            channels,
            bounds,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmlSignalViewError {
    ChannelCountOutOfRange,
    SampleCountOutOfRange,
    RaggedChannels,
    ByteExtentOverflow,
    ByteExtentExceeded,
}

impl fmt::Display for LmlSignalViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LmlSignalViewError {}

#[derive(Debug)]
pub enum LamQuantNodeValue<'a> {
    LmlSignal(LmlSignalView<'a>),
    #[doc(hidden)]
    ReferenceSubbands(ReferenceSubbands),
    #[doc(hidden)]
    ReferenceQuantized(ReferenceQuantized),
    #[doc(hidden)]
    ReferencePredicted(ReferencePredicted),
    #[doc(hidden)]
    ReferenceEntropy(ReferenceEntropy),
    #[doc(hidden)]
    LmlPackets(LmlPackets),
    Bcs2(Vec<u8>),
    #[cfg(feature = "standard-adapters")]
    ForeignObject(abir_adapter::ForeignObject),
    AbirDataset(Box<AbirDatasetValue>),
    #[cfg(feature = "standard-adapters")]
    MappingReport(abir_adapter::MappingReport),
    #[cfg(feature = "standard-adapters")]
    FidelityReceipt(abir_adapter::FidelityReceipt),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmlNodeConfigError {
    LiveDeadlineUnsupported,
    MaxOrderOutOfRange,
    WindowSizeOutOfRange,
}

impl fmt::Display for LmlNodeConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LmlNodeConfigError {}

pub struct LamQuantKernelExecutor<'a> {
    marker: PhantomData<&'a ()>,
}

impl Default for LamQuantKernelExecutor<'_> {
    fn default() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<'a> KernelExecutor for LamQuantKernelExecutor<'a> {
    type Value = LamQuantNodeValue<'a>;

    fn execute(
        &mut self,
        node: &CompiledNode,
        inputs: &[Option<&Self::Value>],
    ) -> Result<Vec<Self::Value>, ExecutionError> {
        let semantic_types = node
            .semantic_types
            .iter()
            .map(|node_type| node_type.type_name.as_str())
            .collect::<Vec<_>>();
        if semantic_types.as_slice() == reference_stage_types() {
            return execute_fused_reference(node, inputs);
        }
        let type_name = semantic_types.first().copied().ok_or_else(|| {
            lml_reference::kernel_failure(node, "invalid-plan", "missing semantic node type")
        })?;
        #[cfg(feature = "standard-adapters")]
        if standard_nodes::is_standard_node(type_name) {
            return standard_nodes::execute_standard(node, type_name, inputs);
        }
        match type_name {
            LML_TRANSFORM_NODE_TYPE => execute_transform(node, inputs),
            LML_QUANTIZE_NODE_TYPE => execute_quantize(node, inputs),
            LML_PREDICT_NODE_TYPE => execute_predict(node, inputs),
            LML_ENTROPY_NODE_TYPE => execute_entropy(node, inputs),
            LML_ASSEMBLE_NODE_TYPE => execute_assemble(node, inputs),
            LML_BASELINE_NODE_TYPE | LML_ARITHMETIC_NODE_TYPE => {
                execute_fused_outer(node, type_name, inputs)
            }
            _ => Err(lml_reference::kernel_failure(
                node,
                "unsupported-node",
                "kernel does not implement requested semantic node",
            )),
        }
    }
}

fn execute_fused_outer<'a>(
    node: &CompiledNode,
    type_name: &str,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    #[cfg(not(feature = "experimental-arithmetic"))]
    if type_name == LML_ARITHMETIC_NODE_TYPE {
        return Err(lml_reference::kernel_failure(
            node,
            "missing-capability",
            "arithmetic LML kernel not compiled into executor",
        ));
    }
    let signal = match inputs {
        [Some(LamQuantNodeValue::LmlSignal(signal))] => signal,
        _ => {
            return Err(lml_reference::kernel_failure(
                node,
                "invalid-input",
                "LML encoder requires one signal input",
            ))
        }
    };
    let config = node.semantic_configs.first().ok_or_else(|| {
        lml_reference::kernel_failure(node, "invalid-plan", "missing semantic config")
    })?;
    let (mode, window_size) = lml_reference::parse_lml_config(config)
        .map_err(|message| lml_reference::kernel_failure(node, "invalid-config", message))?;
    let features = if type_name == LML_ARITHMETIC_NODE_TYPE {
        EncodeFeatures {
            arithmetic: true,
            max_packet_bytes: Some(lml_reference::max_frame_bytes(node, signal.bounds)?),
            ..EncodeFeatures::default()
        }
    } else {
        EncodeFeatures {
            max_packet_bytes: Some(lml_reference::max_frame_bytes(node, signal.bounds)?),
            ..EncodeFeatures::default()
        }
    };
    let bytes = encode_lml_bundle_from_views_explicit(
        signal.dataset,
        signal.channels,
        window_size,
        mode,
        features,
        signal.bounds,
    )
    .map_err(|error| lml_reference::codec_failure(node, error))?;
    Ok(vec![LamQuantNodeValue::Bcs2(bytes)])
}

fn execute_fused_reference<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    lml_reference::execute_fused_reference(node, inputs)
}

fn execute_transform<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    lml_reference::execute_transform(node, inputs)
}

fn execute_quantize<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    lml_reference::execute_quantize(node, inputs)
}

fn execute_predict<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    lml_reference::execute_predict(node, inputs)
}

fn execute_entropy<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    lml_reference::execute_entropy(node, inputs)
}

fn execute_assemble<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    lml_reference::execute_assemble(node, inputs)
}

/// Sink for pure plans. Transaction callbacks fail closed if compiler emits one.
pub struct NoopTransactionalSink;

impl TransactionalSink for NoopTransactionalSink {
    fn prepare(&mut self, _idempotency_key: &str) -> Result<(), ExecutionError> {
        Err(ExecutionError::TransactionPrepare(
            "pure LamQuant node plan requested transaction".into(),
        ))
    }

    fn commit(&mut self, _idempotency_key: &str) -> Result<String, ExecutionError> {
        Err(ExecutionError::TransactionCommit(
            "pure LamQuant node plan requested transaction".into(),
        ))
    }

    fn abort(&mut self, _idempotency_key: &str) {}
}

pub fn baseline_lml_descriptor() -> NodeDescriptor {
    lml_descriptor(LML_BASELINE_NODE_TYPE, false)
}

pub fn baseline_lml_packet_descriptor() -> NodeDescriptor {
    lml_reference::baseline_lml_packet_descriptor()
}

pub fn arithmetic_lml_descriptor() -> NodeDescriptor {
    lml_descriptor(LML_ARITHMETIC_NODE_TYPE, true)
}

pub fn lml_node_config(
    mode: LpcMode,
    window_size: usize,
) -> Result<BTreeMap<String, ConfigValue>, LmlNodeConfigError> {
    if !(1..=u16::MAX as usize).contains(&window_size) {
        return Err(LmlNodeConfigError::WindowSizeOutOfRange);
    }
    let schedule = serialize_lpc_schedule(mode)?;
    Ok(BTreeMap::from([
        ("lpc_schedule".into(), ConfigValue::Text(schedule)),
        ("window_size".into(), ConfigValue::U64(window_size as u64)),
    ]))
}

pub fn lml_packet_node_config(
    mode: LpcMode,
) -> Result<BTreeMap<String, ConfigValue>, LmlNodeConfigError> {
    Ok(BTreeMap::from([(
        "lpc_schedule".into(),
        ConfigValue::Text(serialize_lpc_schedule(mode)?),
    )]))
}

fn serialize_lpc_schedule(mode: LpcMode) -> Result<String, LmlNodeConfigError> {
    let schedule = match mode {
        LpcMode::Fixed => "fixed".into(),
        LpcMode::Adaptive { max_order } => {
            validate_max_order(max_order)?;
            format!("adaptive-{max_order}")
        }
        #[cfg(feature = "std")]
        LpcMode::Anytime {
            deadline: Some(_), ..
        } => return Err(LmlNodeConfigError::LiveDeadlineUnsupported),
        #[cfg(feature = "std")]
        LpcMode::Anytime {
            max_order,
            deadline: None,
        } => {
            validate_max_order(max_order)?;
            format!("anytime-none-{max_order}")
        }
        #[cfg(not(feature = "std"))]
        LpcMode::Anytime { max_order } => {
            validate_max_order(max_order)?;
            format!("anytime-none-{max_order}")
        }
    };
    Ok(schedule)
}

fn validate_max_order(max_order: usize) -> Result<(), LmlNodeConfigError> {
    if (1..=64).contains(&max_order) {
        Ok(())
    } else {
        Err(LmlNodeConfigError::MaxOrderOutOfRange)
    }
}

/// Register production LML semantic nodes and target-specific fused kernels.
pub fn register_lml_nodes(registry: &mut KernelRegistry) -> Result<(), CompileError> {
    for descriptor in reference_stage_descriptors() {
        registry.register_descriptor(descriptor)?;
    }
    let schema = baseline_reference_subgraph();
    registry.register_subgraph(schema)?;
    registry.register_descriptor(baseline_lml_packet_descriptor())?;
    registry.register_descriptor(baseline_lml_descriptor())?;
    registry.register_descriptor(arithmetic_lml_descriptor())?;

    for (offset, type_name) in reference_stage_types().iter().enumerate() {
        registry.register_kernel(reference_kernel(
            KernelId(lml_reference::REFERENCE_HOST_KERNEL_BASE + offset as u32),
            &[*type_name],
            Target::Host,
        ))?;
    }
    registry.register_kernel(reference_kernel(
        lml_reference::REFERENCE_FUSED_HOST_KERNEL,
        reference_stage_types(),
        Target::Host,
    ))?;

    for (id, target) in [
        (BASELINE_HOST_KERNEL, Target::Host),
        // The fused kernel body IS `lamquant-lml-mcu`, the no_std crate firmware
        // links and the byte-equality gate pins. Without this registration the
        // MCU realm had no LML kernel at all, so a cross-realm equivalence claim
        // had no MCU side to compare against.
        (BASELINE_MCU_KERNEL, Target::McuAot),
        (BASELINE_BLUT_KERNEL, Target::BlutDurable),
    ] {
        registry.register_kernel(lml_kernel(id, LML_BASELINE_NODE_TYPE, target))?;
    }
    #[cfg(feature = "experimental-arithmetic")]
    for (id, target) in [
        (ARITHMETIC_HOST_KERNEL, Target::Host),
        (ARITHMETIC_BLUT_KERNEL, Target::BlutDurable),
    ] {
        registry.register_kernel(lml_kernel(id, LML_ARITHMETIC_NODE_TYPE, target))?;
    }
    Ok(())
}

fn lml_descriptor(type_name: &str, arithmetic: bool) -> NodeDescriptor {
    let mut capabilities = vec![Capability(CAP_ABIR.into()), Capability(CAP_LML.into())];
    if arithmetic {
        capabilities.push(Capability(CAP_LML_ARITHMETIC_NODE.into()));
    }
    NodeDescriptor {
        type_name: type_name.into(),
        version: 1,
        inputs: vec![signal_port()],
        outputs: vec![bundle_port()],
        capabilities,
        targets: vec![Target::Host, Target::McuAot, Target::BlutDurable],
        resources: ResourceEnvelope::bounded(
            MAX_SIGNAL_BYTES,
            MAX_PACKET_BYTES,
            MAX_PARALLEL_CHANNELS,
        ),
        determinism: Determinism::BitExact,
        config: lml_config_schema(),
        state: StateContract {
            scope: StateScope::Stateless,
            max_bytes: 0,
            checkpoint: blut_graph_core::CheckpointContract {
                mode: CheckpointMode::Disabled,
                max_snapshot_bytes: 0,
                max_interval_invocations: 0,
            },
        },
        subgraph: None,
        proof: ProofContract {
            requires: vec![],
            provides: vec!["org.quitetall.lamquant.proof.exact-lml-closure-v1".into()],
            invalidates: vec![],
        },
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
        effect: Effect::Pure,
        retry_limit: 0,
    }
}

fn reference_stage_types() -> &'static [&'static str] {
    lml_reference::reference_stage_types()
}

fn baseline_reference_subgraph() -> blut_graph_core::SubgraphSchema {
    lml_reference::baseline_reference_subgraph()
}

fn reference_stage_descriptors() -> Vec<NodeDescriptor> {
    lml_reference::reference_stage_descriptors()
}

fn signal_port() -> PortDescriptor {
    PortDescriptor {
        name: "signal".into(),
        semantic_type: SIGNAL_SEMANTIC_TYPE.into(),
        optional: false,
        layouts: vec![Layout::ChannelMajor],
        max_bytes: MAX_SIGNAL_BYTES,
        abir: AbirSemanticType {
            root: AbirRootType::Dataset,
            view: AbirViewType::Root,
        },
        proof: ProofContract {
            requires: vec![],
            provides: vec![],
            invalidates: vec![],
        },
        policy: PolicyContract {
            requires: vec![],
            adds: vec![],
        },
        fidelity: FidelityContract {
            minimum_input: u16::MAX,
            maximum_loss: 0,
        },
        extent: ExtentContract {
            rank: 2,
            maximum_shape: vec![1024, 131_072],
            max_elements: 134_217_728,
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

fn bundle_port() -> PortDescriptor {
    PortDescriptor {
        name: "bundle".into(),
        semantic_type: BUNDLE_SEMANTIC_TYPE.into(),
        optional: false,
        layouts: vec![Layout::Packed],
        max_bytes: MAX_PACKET_BYTES,
        abir: AbirSemanticType {
            root: AbirRootType::Dataset,
            view: AbirViewType::Root,
        },
        proof: ProofContract {
            requires: vec![],
            provides: vec!["org.quitetall.lamquant.proof.exact-lml-closure-v1".into()],
            invalidates: vec![],
        },
        policy: PolicyContract {
            requires: vec![],
            adds: vec![],
        },
        fidelity: FidelityContract {
            minimum_input: u16::MAX,
            maximum_loss: 0,
        },
        extent: ExtentContract {
            rank: 1,
            maximum_shape: vec![MAX_PACKET_BYTES],
            max_elements: MAX_PACKET_BYTES,
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

fn lml_config_schema() -> ConfigSchema {
    ConfigSchema {
        fields: vec![
            ConfigField {
                name: "lpc_schedule".into(),
                value_type: ConfigType::Choice {
                    values: lml_reference::lpc_schedule_values(),
                },
                required: true,
                default: None,
            },
            ConfigField {
                name: "window_size".into(),
                value_type: ConfigType::U64 {
                    minimum: 1,
                    maximum: u16::MAX as u64,
                },
                required: true,
                default: None,
            },
        ],
    }
}

fn lml_kernel(id: KernelId, type_name: &str, target: Target) -> KernelDescriptor {
    let threads = match target {
        Target::Host | Target::BlutDurable => MAX_PARALLEL_CHANNELS,
        Target::McuAot => 1,
    };
    KernelDescriptor {
        id,
        implements: vec![NodeTypeRef {
            type_name: type_name.into(),
            version: 1,
        }],
        implementation_id: implementation_id(type_name, target),
        conversion: None,
        target,
        input_layouts: vec![Layout::ChannelMajor],
        output_layouts: vec![Layout::Packed],
        resources: ResourceEnvelope::bounded(MAX_SIGNAL_BYTES, MAX_PACKET_BYTES, threads),
        determinism: Determinism::BitExact,
        lowering: format!("fused:{type_name}:encode_lml_bundle_from_views_explicit:v1"),
    }
}

fn reference_kernel(id: KernelId, type_names: &[&str], target: Target) -> KernelDescriptor {
    lml_reference::reference_kernel(id, type_names, target)
}

fn implementation_id(type_name: &str, target: Target) -> ImplementationId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.nodes.implementation-v1\0");
    hasher.update(SOURCE_ID.as_bytes());
    hasher.update(&[0]);
    hasher.update(FEATURE_SET.as_bytes());
    hasher.update(&[0]);
    hasher.update(type_name.as_bytes());
    hasher.update(&[0]);
    hasher.update(&[match target {
        Target::McuAot => 0,
        Target::Host => 1,
        Target::BlutDurable => 2,
    }]);
    ImplementationId(*hasher.finalize().as_bytes())
}
