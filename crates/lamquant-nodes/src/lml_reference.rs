use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use blut_graph_core::{
    subgraph_identity, AbirRootType, AbirSemanticType, AbirViewType, Capability, CheckpointMode,
    CompiledNode, ConfigField, ConfigSchema, ConfigType, ConfigValue, Determinism, Edge, Effect,
    ExecutionError, ExtentContract, FailureContract, FidelityContract, ImplementationId,
    KernelDescriptor, KernelId, Layout, LeaseAccess, LeaseContract, LeaseLifetime, NodeDescriptor,
    NodeId, NodeTypeRef, Partiality, PolicyContract, PortDescriptor, PortMap, PortRef,
    ProofContract, ResourceEnvelope, StateContract, StateScope, StructuredFailure,
    SubgraphConfigMap, SubgraphInterfacePort, SubgraphLowering, SubgraphNode, SubgraphSchema,
    Target,
};
use lamquant_abir_codec::{verify_lml_signal_views_closure, LmlBundleError};
use lamquant_lml_mcu::golomb;
use lamquant_lml_mcu::lml::{
    assemble_lml_packet, channel_payload_limit, compress_with_mode_views_explicit,
    compute_n_levels, forward_subbands, lpc_max_order, scope_lpc_mode, EncodeFeatures, BIAS_CTX,
};
use lamquant_lml_mcu::lpc;
use lamquant_lml_mcu::lpc::LpcMode;
use semantic_abir_bcs::ResourceBounds;

use crate::{
    LamQuantNodeValue, LmlSignalView, CAP_ABIR, CAP_LML, FAILURE_DOMAIN, MAX_PARALLEL_CHANNELS,
};

pub const LML_TRANSFORM_NODE_TYPE: &str = "org.quitetall.lamquant.lml.stage.transform";
pub const LML_QUANTIZE_NODE_TYPE: &str = "org.quitetall.lamquant.lml.stage.quantize-lossless";
pub const LML_PREDICT_NODE_TYPE: &str = "org.quitetall.lamquant.lml.stage.predict";
pub const LML_ENTROPY_NODE_TYPE: &str = "org.quitetall.lamquant.lml.stage.entropy-golomb";
pub const LML_ASSEMBLE_NODE_TYPE: &str = "org.quitetall.lamquant.lml.stage.assemble-lml1";
pub const LML_PACKET_BASELINE_NODE_TYPE: &str = "org.quitetall.lamquant.lml.packet.encode.baseline";

const SIGNAL_SEMANTIC_TYPE: &str = "abir.dataset.uniform-signed-integer-recording";
const SUBBANDS_SEMANTIC_TYPE: &str = "lamquant.lml.reference.subbands";
const QUANTIZED_SEMANTIC_TYPE: &str = "lamquant.lml.reference.quantized-subbands";
const PREDICTED_SEMANTIC_TYPE: &str = "lamquant.lml.reference.predicted-subbands";
const ENTROPY_SEMANTIC_TYPE: &str = "lamquant.lml.reference.entropy-segments";
const PACKETS_SEMANTIC_TYPE: &str = "bcs.lml.packet-sequence.lossless-v1";
pub const REFERENCE_MAX_SIGNAL_BYTES: u64 = 128 * 1024 * 1024;
pub const REFERENCE_MAX_PACKET_BYTES: u64 = 512 * 1024 * 1024;
const REFERENCE_PEAK_BYTES: u64 = 1024 * 1024 * 1024;
pub const REFERENCE_MCU_SCRATCH_BYTES: u64 = 256 * 1024;
const REFERENCE_MAX_ELEMENTS: u64 = REFERENCE_MAX_SIGNAL_BYTES / core::mem::size_of::<i64>() as u64;
const REFERENCE_MAX_CHANNELS: usize = 256;

pub(crate) const REFERENCE_HOST_KERNEL_BASE: u32 = 0x4c4d_1000;
pub const REFERENCE_MCU_KERNEL_BASE: u32 = 0x4c4d_1100;
pub const REFERENCE_FUSED_MCU_KERNEL: KernelId = KernelId(0x4c4d_1200);
pub(crate) const REFERENCE_FUSED_HOST_KERNEL: KernelId = KernelId(0x4c4d_1201);
include!(concat!(env!("OUT_DIR"), "/mcu_implementation_ids.rs"));
pub const REFERENCE_FUSED_MCU_IMPLEMENTATION_ID: ImplementationId =
    ImplementationId(REFERENCE_FUSED_MCU_IMPLEMENTATION_BYTES);

#[doc(hidden)]
#[derive(Debug)]
pub struct ReferenceSubbands {
    bounds: ResourceBounds,
    packet: Arc<TransformedPacket>,
}

#[derive(Debug)]
struct TransformedPacket {
    n_ch: usize,
    t: usize,
    n_levels: u8,
    channels: Vec<Vec<Vec<i64>>>,
}

#[doc(hidden)]
#[derive(Debug)]
pub struct ReferenceQuantized {
    bounds: ResourceBounds,
    packet: Arc<TransformedPacket>,
}

#[doc(hidden)]
#[derive(Debug)]
pub struct ReferencePredicted {
    bounds: ResourceBounds,
    packet: PredictedPacket,
}

#[derive(Debug)]
struct PredictedPacket {
    n_ch: usize,
    t: usize,
    n_levels: u8,
    channels: Vec<Vec<PredictedSubband>>,
}

#[derive(Debug)]
struct PredictedSubband {
    coeffs: Vec<i32>,
    order: usize,
    residual: Vec<i64>,
}

#[doc(hidden)]
#[derive(Debug)]
pub struct ReferenceEntropy {
    bounds: ResourceBounds,
    packet: EntropyPacket,
}

#[derive(Debug)]
struct EntropyPacket {
    n_ch: usize,
    t: usize,
    n_levels: u8,
    lpc_meta: Vec<u8>,
    payload: Vec<u8>,
}

#[doc(hidden)]
#[derive(Debug)]
pub struct LmlPackets {
    packets: Vec<Vec<u8>>,
}

impl LmlPackets {
    pub fn packets(&self) -> &[Vec<u8>] {
        &self.packets
    }
}

pub(crate) fn execute_fused_reference<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    let signal = require_signal(node, inputs)?;
    let predict_config = node
        .semantic_configs
        .get(2)
        .ok_or_else(|| kernel_failure(node, "invalid-plan", "missing predictor config"))?;
    let mode = parse_lpc_schedule(predict_config)
        .map_err(|message| kernel_failure(node, "invalid-config", message))?;
    validate_reference_signal(node, signal)?;
    let packet = compress_with_mode_views_explicit(
        signal.channels,
        0,
        mode,
        EncodeFeatures {
            max_packet_bytes: Some(max_frame_bytes(node, signal.bounds)?),
            ..EncodeFeatures::default()
        },
    )
    .map_err(|error| lml_failure(node, error))?;
    Ok(vec![LamQuantNodeValue::LmlPackets(LmlPackets {
        packets: vec![packet],
    })])
}

pub(crate) fn execute_transform<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    let signal = require_signal(node, inputs)?;
    validate_reference_signal(node, signal)?;
    let t = signal.channels[0].len();
    let n_levels = compute_n_levels(t);
    Ok(vec![LamQuantNodeValue::ReferenceSubbands(
        ReferenceSubbands {
            bounds: signal.bounds,
            packet: Arc::new(TransformedPacket {
                n_ch: signal.channels.len(),
                t,
                n_levels,
                channels: signal
                    .channels
                    .iter()
                    .map(|channel| forward_subbands(channel, n_levels))
                    .collect(),
            }),
        },
    )])
}

pub(crate) fn execute_quantize<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    let subbands = match inputs {
        [Some(LamQuantNodeValue::ReferenceSubbands(value))] => value,
        _ => {
            return Err(kernel_failure(
                node,
                "invalid-input",
                "lossless quantizer requires transformed subbands",
            ));
        }
    };
    Ok(vec![LamQuantNodeValue::ReferenceQuantized(
        ReferenceQuantized {
            bounds: subbands.bounds,
            packet: Arc::clone(&subbands.packet),
        },
    )])
}

pub(crate) fn execute_predict<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    let quantized = match inputs {
        [Some(LamQuantNodeValue::ReferenceQuantized(value))] => value,
        _ => {
            return Err(kernel_failure(
                node,
                "invalid-input",
                "predictor requires lossless-quantized subbands",
            ));
        }
    };
    let mode = parse_lpc_schedule(require_config(node)?)
        .map_err(|message| kernel_failure(node, "invalid-config", message))?;
    let packet = &quantized.packet;
    Ok(vec![LamQuantNodeValue::ReferencePredicted(
        ReferencePredicted {
            bounds: quantized.bounds,
            packet: PredictedPacket {
                n_ch: packet.n_ch,
                t: packet.t,
                n_levels: packet.n_levels,
                channels: packet
                    .channels
                    .iter()
                    .map(|subbands| {
                        subbands
                            .iter()
                            .enumerate()
                            .map(|(index, subband)| {
                                let scoped = scope_lpc_mode(mode, lpc_max_order(subband.len()));
                                let (coeffs, residual, order) =
                                    lpc::analyze_with_mode(subband, index, scoped, BIAS_CTX, None);
                                PredictedSubband {
                                    coeffs,
                                    order,
                                    residual,
                                }
                            })
                            .collect()
                    })
                    .collect(),
            },
        },
    )])
}

pub(crate) fn execute_entropy<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    let predicted = match inputs {
        [Some(LamQuantNodeValue::ReferencePredicted(value))] => value,
        _ => {
            return Err(kernel_failure(
                node,
                "invalid-input",
                "entropy stage requires predicted subbands",
            ));
        }
    };
    let max_packet_bytes = max_frame_bytes(node, predicted.bounds)?;
    let packet = &predicted.packet;
    let per_channel_limit = channel_payload_limit(
        Some(max_packet_bytes),
        packet.n_ch,
        packet.t,
        packet.n_levels,
    )
    .map_err(|error| lml_failure(node, error))?;
    let mut lpc_meta = Vec::new();
    let mut payload = Vec::new();
    for channel in &packet.channels {
        let mut remaining = per_channel_limit;
        for subband in channel {
            lpc_meta.push(u8::try_from(subband.order).map_err(|_| {
                kernel_failure(node, "invalid-predictor", "LPC order exceeds wire field")
            })?);
            for coefficient in &subband.coeffs {
                lpc_meta.extend_from_slice(&coefficient.to_le_bytes());
            }
            let encoded = golomb::encode_dense_bounded(&subband.residual, remaining)
                .map_err(|error| primitive_failure(node, "golomb", error))?;
            remaining = remaining.checked_sub(encoded.len()).ok_or_else(|| {
                kernel_failure(node, "resource-limit", "channel payload budget exceeded")
            })?;
            payload.extend_from_slice(&encoded);
        }
    }
    Ok(vec![LamQuantNodeValue::ReferenceEntropy(
        ReferenceEntropy {
            bounds: predicted.bounds,
            packet: EntropyPacket {
                n_ch: packet.n_ch,
                t: packet.t,
                n_levels: packet.n_levels,
                lpc_meta,
                payload,
            },
        },
    )])
}

pub(crate) fn execute_assemble<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    let entropy = match inputs {
        [Some(LamQuantNodeValue::ReferenceEntropy(value))] => value,
        _ => {
            return Err(kernel_failure(
                node,
                "invalid-input",
                "assembler requires entropy segments",
            ));
        }
    };
    let max_packet_bytes = max_frame_bytes(node, entropy.bounds)?;
    let encoded = &entropy.packet;
    let packet = assemble_lml_packet(
        encoded.n_ch,
        encoded.t,
        encoded.n_levels,
        0,
        false,
        &encoded.lpc_meta,
        &encoded.payload,
    );
    if packet.len() > max_packet_bytes {
        return Err(kernel_failure(
            node,
            "resource-limit",
            "assembled LML packet exceeds frame bound",
        ));
    }
    Ok(vec![LamQuantNodeValue::LmlPackets(LmlPackets {
        packets: vec![packet],
    })])
}

pub(crate) fn reference_stage_types() -> &'static [&'static str] {
    &[
        LML_TRANSFORM_NODE_TYPE,
        LML_QUANTIZE_NODE_TYPE,
        LML_PREDICT_NODE_TYPE,
        LML_ENTROPY_NODE_TYPE,
        LML_ASSEMBLE_NODE_TYPE,
    ]
}

pub(crate) fn baseline_reference_subgraph() -> SubgraphSchema {
    let types = reference_stage_types();
    let configs = [
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::from([("lpc_schedule".into(), ConfigValue::Text("fixed".into()))]),
        BTreeMap::new(),
        BTreeMap::new(),
    ];
    let mut schema = SubgraphSchema {
        id: blut_graph_core::SubgraphId([0; 32]),
        version: 1,
        nodes: types
            .iter()
            .enumerate()
            .map(|(index, type_name)| SubgraphNode {
                id: NodeId(index as u32),
                node_type: NodeTypeRef {
                    type_name: (*type_name).into(),
                    version: 1,
                },
                config: configs[index].clone(),
                child: None,
            })
            .collect(),
        edges: (0..types.len() - 1)
            .map(|index| Edge {
                from: PortRef {
                    node: NodeId(index as u32),
                    port: reference_output_name(index).into(),
                },
                to: PortRef {
                    node: NodeId(index as u32 + 1),
                    port: reference_input_name(index + 1).into(),
                },
            })
            .collect(),
        inputs: vec![SubgraphInterfacePort {
            name: "signal".into(),
            inner: PortRef {
                node: NodeId(0),
                port: "signal".into(),
            },
        }],
        outputs: vec![SubgraphInterfacePort {
            name: "packet".into(),
            inner: PortRef {
                node: NodeId(4),
                port: "packets".into(),
            },
        }],
    };
    schema.id = subgraph_identity(&schema);
    schema
}

pub(crate) fn baseline_lml_packet_descriptor() -> NodeDescriptor {
    let schema = baseline_reference_subgraph();
    let mut output = packet_port();
    output.name = "packet".into();
    NodeDescriptor {
        type_name: LML_PACKET_BASELINE_NODE_TYPE.into(),
        version: 1,
        inputs: vec![signal_port()],
        outputs: vec![output],
        capabilities: vec![Capability(CAP_ABIR.into()), Capability(CAP_LML.into())],
        targets: vec![Target::McuAot, Target::Host],
        resources: ResourceEnvelope::bounded(
            REFERENCE_PEAK_BYTES,
            REFERENCE_MAX_PACKET_BYTES,
            MAX_PARALLEL_CHANNELS,
        ),
        determinism: Determinism::BitExact,
        config: ConfigSchema {
            fields: vec![ConfigField {
                name: "lpc_schedule".into(),
                value_type: ConfigType::Choice {
                    values: lpc_schedule_values(),
                },
                required: true,
                default: None,
            }],
        },
        state: stateless_contract(),
        subgraph: Some(SubgraphLowering {
            subgraph: schema.id,
            input_map: vec![PortMap {
                outer: "signal".into(),
                inner: "signal".into(),
            }],
            output_map: vec![PortMap {
                outer: "packet".into(),
                inner: "packet".into(),
            }],
            config_map: vec![SubgraphConfigMap {
                outer: "lpc_schedule".into(),
                node: NodeId(2),
                inner: "lpc_schedule".into(),
            }],
        }),
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
        partiality: Partiality::Atomic,
        failure: FailureContract {
            domains: vec![FAILURE_DOMAIN.into()],
        },
        effect: Effect::Pure,
        retry_limit: 0,
    }
}

fn reference_input_name(index: usize) -> &'static str {
    match index {
        0 => "signal",
        1 => "subbands",
        2 => "quantized",
        3 => "predicted",
        4 => "entropy",
        _ => unreachable!("reference stage index"),
    }
}

fn reference_output_name(index: usize) -> &'static str {
    match index {
        0 => "subbands",
        1 => "quantized",
        2 => "predicted",
        3 => "entropy",
        4 => "packets",
        _ => unreachable!("reference stage index"),
    }
}

pub(crate) fn reference_stage_descriptors() -> Vec<NodeDescriptor> {
    let semantic_types = [
        (SIGNAL_SEMANTIC_TYPE, SUBBANDS_SEMANTIC_TYPE),
        (SUBBANDS_SEMANTIC_TYPE, QUANTIZED_SEMANTIC_TYPE),
        (QUANTIZED_SEMANTIC_TYPE, PREDICTED_SEMANTIC_TYPE),
        (PREDICTED_SEMANTIC_TYPE, ENTROPY_SEMANTIC_TYPE),
        (ENTROPY_SEMANTIC_TYPE, PACKETS_SEMANTIC_TYPE),
    ];
    reference_stage_types()
        .iter()
        .enumerate()
        .map(|(index, type_name)| {
            let config = match index {
                2 => ConfigSchema {
                    fields: vec![ConfigField {
                        name: "lpc_schedule".into(),
                        value_type: ConfigType::Choice {
                            values: lpc_schedule_values(),
                        },
                        required: true,
                        default: None,
                    }],
                },
                _ => ConfigSchema { fields: vec![] },
            };
            reference_stage_descriptor(
                type_name,
                reference_input_name(index),
                semantic_types[index].0,
                reference_output_name(index),
                semantic_types[index].1,
                config,
                index == 4,
            )
        })
        .collect()
}

fn reference_stage_descriptor(
    type_name: &str,
    input_name: &str,
    input_semantic_type: &str,
    output_name: &str,
    output_semantic_type: &str,
    config: ConfigSchema,
    produces_packet: bool,
) -> NodeDescriptor {
    let mut input = intermediate_port(input_name, input_semantic_type);
    if type_name == LML_TRANSFORM_NODE_TYPE {
        input = signal_port();
    }
    let mut output = intermediate_port(output_name, output_semantic_type);
    if produces_packet {
        output = packet_port();
    }
    NodeDescriptor {
        type_name: type_name.into(),
        version: 1,
        inputs: vec![input],
        outputs: vec![output],
        capabilities: vec![
            Capability(crate::CAP_LML.into()),
            Capability(crate::CAP_ABIR.into()),
        ],
        targets: vec![Target::McuAot, Target::Host],
        resources: ResourceEnvelope::bounded(
            REFERENCE_PEAK_BYTES,
            REFERENCE_MAX_PACKET_BYTES,
            MAX_PARALLEL_CHANNELS,
        ),
        determinism: Determinism::BitExact,
        config,
        state: stateless_contract(),
        subgraph: None,
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
        partiality: Partiality::Atomic,
        failure: FailureContract {
            domains: vec![FAILURE_DOMAIN.into()],
        },
        effect: Effect::Pure,
        retry_limit: 0,
    }
}

fn stateless_contract() -> StateContract {
    StateContract {
        scope: StateScope::Stateless,
        max_bytes: 0,
        checkpoint: blut_graph_core::CheckpointContract {
            mode: CheckpointMode::Disabled,
            max_snapshot_bytes: 0,
            max_interval_invocations: 0,
        },
    }
}

fn intermediate_port(name: &str, semantic_type: &str) -> PortDescriptor {
    PortDescriptor {
        name: name.into(),
        semantic_type: semantic_type.into(),
        optional: false,
        layouts: vec![Layout::Opaque],
        max_bytes: REFERENCE_MAX_PACKET_BYTES,
        abir: AbirSemanticType {
            root: AbirRootType::Unknown("native-lml-reference-v1".into()),
            view: AbirViewType::Unknown("native".into()),
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
            rank: 0,
            maximum_shape: vec![],
            max_elements: 1,
            ragged: false,
            sparse: false,
        },
        lease: LeaseContract {
            access: LeaseAccess::ReadOnly,
            lifetime: LeaseLifetime::Invocation,
            zero_copy_permitted: true,
            contiguous_required: false,
        },
    }
}

fn signal_port() -> PortDescriptor {
    PortDescriptor {
        name: "signal".into(),
        semantic_type: SIGNAL_SEMANTIC_TYPE.into(),
        optional: false,
        layouts: vec![Layout::ChannelMajor],
        max_bytes: REFERENCE_MAX_SIGNAL_BYTES,
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
            maximum_shape: vec![REFERENCE_MAX_CHANNELS as u64, u16::MAX as u64],
            max_elements: REFERENCE_MAX_ELEMENTS,
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

fn packet_port() -> PortDescriptor {
    PortDescriptor {
        name: "packets".into(),
        semantic_type: PACKETS_SEMANTIC_TYPE.into(),
        optional: false,
        layouts: vec![Layout::Packed],
        max_bytes: REFERENCE_MAX_PACKET_BYTES,
        abir: AbirSemanticType {
            root: AbirRootType::EncodedBlock,
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
            rank: 1,
            maximum_shape: vec![REFERENCE_MAX_PACKET_BYTES],
            max_elements: REFERENCE_MAX_PACKET_BYTES,
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

pub(crate) fn lpc_schedule_values() -> Vec<String> {
    let mut values = Vec::with_capacity(129);
    values.push("fixed".into());
    for order in 1..=64 {
        values.push(format!("adaptive-{order}"));
        values.push(format!("anytime-none-{order}"));
    }
    values
}

pub(crate) fn reference_kernel(
    id: KernelId,
    type_names: &[&str],
    target: Target,
) -> KernelDescriptor {
    debug_assert!(matches!(target, Target::McuAot | Target::Host));
    let threads = if target == Target::McuAot {
        1
    } else {
        MAX_PARALLEL_CHANNELS
    };
    let lowering = if type_names.len() == 1 {
        format!("reference:{}", type_names[0])
    } else {
        "fused:org.quitetall.lamquant.lml.reference-v1".into()
    };
    let (peak_bytes, scratch_bytes) = if target == Target::McuAot {
        (0, REFERENCE_MCU_SCRATCH_BYTES)
    } else {
        (REFERENCE_PEAK_BYTES, REFERENCE_MAX_PACKET_BYTES)
    };
    KernelDescriptor {
        id,
        implements: type_names
            .iter()
            .map(|type_name| NodeTypeRef {
                type_name: (*type_name).into(),
                version: 1,
            })
            .collect(),
        implementation_id: implementation_id(&lowering, target),
        conversion: None,
        target,
        input_layouts: vec![if type_names.first() == Some(&LML_TRANSFORM_NODE_TYPE) {
            Layout::ChannelMajor
        } else {
            Layout::Opaque
        }],
        output_layouts: vec![if type_names.last() == Some(&LML_ASSEMBLE_NODE_TYPE) {
            Layout::Packed
        } else {
            Layout::Opaque
        }],
        resources: ResourceEnvelope::bounded(peak_bytes, scratch_bytes, threads),
        determinism: Determinism::BitExact,
        lowering,
    }
}

fn implementation_id(type_name: &str, target: Target) -> ImplementationId {
    const MCU_SOURCE_ID: &str = env!("LAMQUANT_NODES_MCU_SOURCE_ID");
    const MCU_FEATURE_SET: &str = "mcu-aot-baseline";
    let (source_id, feature_set) = if target == Target::McuAot {
        (MCU_SOURCE_ID, MCU_FEATURE_SET)
    } else {
        (crate::SOURCE_ID, crate::FEATURE_SET)
    };
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.nodes.implementation-v1\0");
    hasher.update(source_id.as_bytes());
    hasher.update(&[0]);
    hasher.update(feature_set.as_bytes());
    hasher.update(&[0]);
    hasher.update(type_name.as_bytes());
    hasher.update(&[0]);
    hasher.update(&[match target {
        Target::McuAot => 0,
        Target::Host => 1,
        Target::BlutDurable => 2,
    }]);
    let id = ImplementationId(*hasher.finalize().as_bytes());
    if target == Target::McuAot && type_name == "fused:org.quitetall.lamquant.lml.reference-v1" {
        debug_assert_eq!(id, REFERENCE_FUSED_MCU_IMPLEMENTATION_ID);
    }
    id
}

pub(crate) fn parse_window_size(
    config: &BTreeMap<String, ConfigValue>,
) -> Result<usize, &'static str> {
    match config.get("window_size") {
        Some(ConfigValue::U64(value)) => {
            let window_size = usize::try_from(*value).map_err(|_| "window_size overflow")?;
            if !(1..=u16::MAX as usize).contains(&window_size) {
                return Err("window_size out of range");
            }
            Ok(window_size)
        }
        _ => Err("missing window_size"),
    }
}

pub(crate) fn parse_lpc_schedule(
    config: &BTreeMap<String, ConfigValue>,
) -> Result<LpcMode, &'static str> {
    let schedule = match config.get("lpc_schedule") {
        Some(ConfigValue::Text(value)) => value.as_str(),
        _ => return Err("missing lpc_schedule"),
    };
    let mode = if schedule == "fixed" {
        LpcMode::Fixed
    } else if let Some(order) = parse_schedule_order(schedule, "adaptive-") {
        LpcMode::Adaptive { max_order: order? }
    } else if let Some(order) = parse_schedule_order(schedule, "anytime-none-") {
        let max_order = order?;
        {
            #[cfg(feature = "std")]
            {
                LpcMode::Anytime {
                    max_order,
                    deadline: None,
                }
            }
            #[cfg(not(feature = "std"))]
            {
                LpcMode::Anytime { max_order }
            }
        }
    } else {
        return Err("invalid lpc_schedule");
    };
    Ok(mode)
}

pub(crate) fn parse_schedule_order(
    schedule: &str,
    prefix: &str,
) -> Option<Result<usize, &'static str>> {
    schedule.strip_prefix(prefix).map(|value| {
        let order = value.parse::<usize>().map_err(|_| "invalid max_order")?;
        if (1..=64).contains(&order) {
            Ok(order)
        } else {
            Err("max_order out of range")
        }
    })
}

pub(crate) fn parse_lml_config(
    config: &BTreeMap<String, ConfigValue>,
) -> Result<(LpcMode, usize), &'static str> {
    Ok((parse_lpc_schedule(config)?, parse_window_size(config)?))
}

fn require_signal<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
) -> Result<LmlSignalView<'a>, ExecutionError> {
    match inputs {
        [Some(LamQuantNodeValue::LmlSignal(signal))] => Ok(*signal),
        _ => Err(kernel_failure(
            node,
            "invalid-input",
            "LML transform requires one signal input",
        )),
    }
}

fn require_config(node: &CompiledNode) -> Result<&BTreeMap<String, ConfigValue>, ExecutionError> {
    node.semantic_configs
        .first()
        .ok_or_else(|| kernel_failure(node, "invalid-plan", "missing semantic config"))
}

fn validate_reference_signal(
    node: &CompiledNode,
    signal: LmlSignalView<'_>,
) -> Result<(), ExecutionError> {
    let samples = signal.channels[0].len();
    if signal.channels.len() > REFERENCE_MAX_CHANNELS {
        return Err(kernel_failure(
            node,
            "resource-limit",
            "reference input exceeds bounded channel count",
        ));
    }
    if samples > u16::MAX as usize {
        return Err(kernel_failure(
            node,
            "resource-limit",
            "reference packet input exceeds LML1 sample extent",
        ));
    }
    let bytes = signal
        .channels
        .len()
        .checked_mul(samples)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<i64>()))
        .ok_or_else(|| kernel_failure(node, "resource-limit", "reference input extent overflow"))?;
    if bytes as u64 > REFERENCE_MAX_SIGNAL_BYTES {
        return Err(kernel_failure(
            node,
            "resource-limit",
            "reference input exceeds bounded native-stage budget",
        ));
    }
    if u64::from(signal.bounds.max_frame_bytes) > REFERENCE_MAX_PACKET_BYTES {
        return Err(kernel_failure(
            node,
            "resource-limit",
            "reference frame bound exceeds bounded packet budget",
        ));
    }
    verify_lml_signal_views_closure(signal.dataset, signal.channels)
        .map_err(|error| codec_failure(node, error))?;
    Ok(())
}

pub(crate) fn max_frame_bytes(
    node: &CompiledNode,
    bounds: ResourceBounds,
) -> Result<usize, ExecutionError> {
    usize::try_from(bounds.max_frame_bytes).map_err(|_| {
        kernel_failure(
            node,
            "resource-limit",
            "frame byte bound exceeds target address space",
        )
    })
}

pub(crate) fn kernel_failure(node: &CompiledNode, code: &str, message: &str) -> ExecutionError {
    ExecutionError::KernelFailed {
        kernel: node.kernel,
        failure: StructuredFailure {
            domain: FAILURE_DOMAIN.into(),
            code: code.into(),
            message: message.into(),
            retryable: false,
            evidence: Vec::new(),
        },
    }
}

pub(crate) fn codec_failure(node: &CompiledNode, error: LmlBundleError) -> ExecutionError {
    ExecutionError::KernelFailed {
        kernel: node.kernel,
        failure: StructuredFailure {
            domain: FAILURE_DOMAIN.into(),
            code: "codec-failure".into(),
            message: format!("{error:?}"),
            retryable: false,
            evidence: Vec::new(),
        },
    }
}

pub(crate) fn lml_failure(
    node: &CompiledNode,
    error: lamquant_lml_mcu::error::LmlError,
) -> ExecutionError {
    ExecutionError::KernelFailed {
        kernel: node.kernel,
        failure: StructuredFailure {
            domain: FAILURE_DOMAIN.into(),
            code: "lml-failure".into(),
            message: format!("{error:?}"),
            retryable: false,
            evidence: Vec::new(),
        },
    }
}

pub(crate) fn primitive_failure(
    node: &CompiledNode,
    primitive: &str,
    error: impl fmt::Debug,
) -> ExecutionError {
    ExecutionError::KernelFailed {
        kernel: node.kernel,
        failure: StructuredFailure {
            domain: FAILURE_DOMAIN.into(),
            code: format!("{primitive}-failure"),
            message: format!("{error:?}"),
            retryable: false,
            evidence: Vec::new(),
        },
    }
}
