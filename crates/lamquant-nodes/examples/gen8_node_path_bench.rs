//! Paired Gen 8 benchmark: direct LML encoding versus compiled Node execution.
//!
//! This probe intentionally measures complete per-window calls for both the
//! BCS2 wrapper seam and the compiler-materialized five-stage packet DAG. The
//! Node arm constructs the invocation map, executes the precompiled plan, and
//! returns the execution receipt and terminal value. Dataset construction and
//! plan compilation are reported but excluded from steady-state timings.

use std::collections::BTreeMap;
use std::env;
use std::hint::black_box;
use std::mem::size_of;
use std::time::Instant;

use blut_graph_core::{
    Capability, Compiler, ExecutionRealm, Graph, KernelRegistry, NodeId, NodeInstance,
    PlanExecutor, PortRef,
};
use lamquant_abir_codec::{encode_lml_bundle_from_views_explicit, verify_lml_signal_views_closure};
use lamquant_lml_mcu::{
    lml::{compress_with_mode_views_explicit, EncodeFeatures},
    lpc::LpcMode,
};
use lamquant_nodes::{
    baseline_lml_descriptor, lml_node_config, lml_packet_node_config, register_lml_nodes,
    LamQuantKernelExecutor, LamQuantNodeValue, LmlSignalView, NoopTransactionalSink,
    LML_BASELINE_NODE_TYPE, LML_PACKET_BASELINE_NODE_TYPE,
};
use semantic_abir::{
    payload_content_id, AbirDataset, Atom, AtomTag, ByteOrder, ConceptId, DatasetDraft, DatasetTag,
    ElementType, Layout, ObjectId, PayloadDescriptor, Presence, Rational, Recording, RecordingTag,
    SignalBlock, Stream, StreamTag, TimeAxis, TimeSegment, ValidationLimits,
};
use semantic_abir_bcs::ResourceBounds;

const DEFAULT_ROUNDS: usize = 31;
const DEFAULT_TARGET_ROUND_MS: u64 = 50;
const MIN_ROUNDS: usize = 9;
const MAX_ROUNDS: usize = 101;
const MIN_TARGET_ROUND_MS: u64 = 10;
const MAX_TARGET_ROUND_MS: u64 = 500;
const MAX_BATCH_ITERATIONS: usize = 256;
const WARMUP_ITERATIONS: usize = 3;
const WINDOW_SIZE: usize = u16::MAX as usize;

#[derive(Clone, Copy)]
struct BenchCase {
    name: &'static str,
    channels: usize,
    samples: usize,
    path: BenchPath,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BenchPath {
    Bundle,
    PacketDag,
}

const CASES: [BenchCase; 4] = [
    BenchCase {
        name: "eeg-8x2500",
        channels: 8,
        samples: 2_500,
        path: BenchPath::Bundle,
    },
    BenchCase {
        name: "eeg-32x2500",
        channels: 32,
        samples: 2_500,
        path: BenchPath::Bundle,
    },
    BenchCase {
        name: "packet-dag-eeg-8x2500",
        channels: 8,
        samples: 2_500,
        path: BenchPath::PacketDag,
    },
    BenchCase {
        name: "packet-dag-eeg-32x2500",
        channels: 32,
        samples: 2_500,
        path: BenchPath::PacketDag,
    },
];

fn fixture_signal(channels: usize, samples: usize) -> Vec<Vec<i64>> {
    (0..channels)
        .map(|channel| {
            (0..samples)
                .map(|sample| {
                    let base = ((sample * 31 + channel * 17) % 4_096) as i64 - 2_048;
                    let wobble = ((sample * sample + channel * 13) % 257) as i64 - 128;
                    let slow = (((sample / 25) + channel * 5) % 89) as i64 - 44;
                    base * 32 + wobble * 3 + slow
                })
                .collect()
        })
        .collect()
}

fn fixture_dataset(signal: &[Vec<i64>]) -> AbirDataset {
    let dataset_id = ObjectId::<DatasetTag>::from_bytes([0x21; 16]);
    let recording_id = ObjectId::<RecordingTag>::from_bytes([0x22; 16]);
    let stream_id = ObjectId::<StreamTag>::from_bytes([0x23; 16]);
    let mut draft = DatasetDraft::new(dataset_id);
    let mut atom_ids = Vec::with_capacity(signal.len());

    for (index, channel) in signal.iter().enumerate() {
        let bytes = channel
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let descriptor = PayloadDescriptor::new(
            payload_content_id(ElementType::I64, &bytes),
            bytes.len() as u64,
            ElementType::I64,
            ByteOrder::Little,
            vec![1, channel.len() as u64],
            Layout::DenseRowMajor,
            None,
            None,
        );
        let mut atom_bytes = [0_u8; 16];
        atom_bytes[14..].copy_from_slice(&((index + 1) as u16).to_be_bytes());
        let atom_id = ObjectId::<AtomTag>::from_bytes(atom_bytes);
        atom_ids.push(atom_id);
        draft.add_atom(Atom::SignalBlock(SignalBlock::new(
            atom_id,
            Presence::Present,
            Some(descriptor),
            TimeAxis::Regular(
                TimeSegment::new(
                    Rational::new(0, 1).expect("valid origin"),
                    Rational::new(250, 1).expect("valid sample rate"),
                    channel.len() as u64,
                )
                .expect("valid regular time segment"),
            ),
            None,
        )));
    }

    draft.add_recording(Recording::new(recording_id, vec![stream_id]));
    draft.add_stream(Stream::new(
        stream_id,
        recording_id,
        ConceptId::new("abir:modality/eeg").expect("valid modality"),
        atom_ids,
        None,
        None,
        None,
    ));
    draft
        .validate(ValidationLimits::default())
        .expect("benchmark fixture is valid ABIR")
}

fn single_lml_graph(
    capabilities: Vec<Capability>,
    config: BTreeMap<String, blut_graph_core::ConfigValue>,
) -> Graph {
    Graph {
        version: 3,
        nodes: vec![NodeInstance {
            id: NodeId(0),
            descriptor: LML_BASELINE_NODE_TYPE.into(),
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

fn elapsed_ns<F>(iterations: usize, mut operation: F) -> u128
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    start.elapsed().as_nanos()
}

fn parse_positive<T>(flag: &str, value: Option<String>) -> T
where
    T: std::str::FromStr + PartialOrd + From<u8>,
    T::Err: std::fmt::Display,
{
    let text = value.unwrap_or_else(|| panic!("{flag} requires a value"));
    let parsed = text
        .parse::<T>()
        .unwrap_or_else(|error| panic!("invalid {flag} value {text:?}: {error}"));
    assert!(parsed > T::from(0), "{flag} must be positive");
    parsed
}

fn arguments() -> (usize, u64) {
    let mut rounds = DEFAULT_ROUNDS;
    let mut target_round_ms = DEFAULT_TARGET_ROUND_MS;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--rounds" => rounds = parse_positive("--rounds", arguments.next()),
            "--target-round-ms" => {
                target_round_ms = parse_positive("--target-round-ms", arguments.next());
            }
            "--help" | "-h" => {
                println!("usage: gen8_node_path_bench [--rounds N] [--target-round-ms N]");
                std::process::exit(0);
            }
            _ => panic!("unsupported argument: {argument}"),
        }
    }
    assert!(
        (MIN_ROUNDS..=MAX_ROUNDS).contains(&rounds) && rounds % 2 == 1,
        "--rounds must be an odd integer in {MIN_ROUNDS}..={MAX_ROUNDS}"
    );
    assert!(
        (MIN_TARGET_ROUND_MS..=MAX_TARGET_ROUND_MS).contains(&target_round_ms),
        "--target-round-ms must be in {MIN_TARGET_ROUND_MS}..={MAX_TARGET_ROUND_MS}"
    );
    (rounds, target_round_ms)
}

fn direct_encode(
    path: BenchPath,
    dataset: &AbirDataset,
    views: &[&[i64]],
    bounds: ResourceBounds,
) -> Vec<u8> {
    match path {
        BenchPath::Bundle => encode_lml_bundle_from_views_explicit(
            dataset,
            views,
            WINDOW_SIZE,
            LpcMode::Fixed,
            EncodeFeatures::default(),
            bounds,
        )
        .expect("direct BCS2 LML encode"),
        BenchPath::PacketDag => {
            let _validated_view =
                LmlSignalView::new(dataset, views, bounds).expect("valid direct packet signal");
            verify_lml_signal_views_closure(dataset, views)
                .expect("direct packet signal closes over ABIR");
            compress_with_mode_views_explicit(
                views,
                0,
                LpcMode::Fixed,
                EncodeFeatures {
                    max_packet_bytes: Some(bounds.max_frame_bytes as usize),
                    ..EncodeFeatures::default()
                },
            )
            .expect("direct LML packet encode")
        }
    }
}

fn run_case(case: BenchCase, rounds: usize, target_round_ms: u64) {
    let signal = fixture_signal(case.channels, case.samples);
    let dataset = fixture_dataset(&signal);
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bounds = ResourceBounds::default();
    let compile_start = Instant::now();
    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).expect("register LML Nodes");
    let (graph, input_port) = match case.path {
        BenchPath::Bundle => (
            single_lml_graph(
                baseline_lml_descriptor().capabilities,
                lml_node_config(LpcMode::Fixed, WINDOW_SIZE).expect("valid fixed config"),
            ),
            PortRef {
                node: NodeId(0),
                port: "signal".into(),
            },
        ),
        BenchPath::PacketDag => {
            let materialized = registry
                .materialize_subgraph(&NodeInstance {
                    id: NodeId(0),
                    descriptor: LML_PACKET_BASELINE_NODE_TYPE.into(),
                    descriptor_version: 1,
                    config: lml_packet_node_config(LpcMode::Fixed)
                        .expect("valid fixed packet config"),
                })
                .expect("materialize compiler packet DAG");
            (materialized.graph, materialized.inputs[0].inner.clone())
        }
    };
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .expect("compile LML Node path");
    let plan_compile_ns = compile_start.elapsed().as_nanos();

    let direct = direct_encode(case.path, &dataset, &views, bounds);

    let mut kernels = LamQuantKernelExecutor::default();
    let mut sink = NoopTransactionalSink;
    let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
    let node_result = executor
        .execute(
            &plan,
            [0x91; 32],
            BTreeMap::from([(
                input_port.clone(),
                LamQuantNodeValue::LmlSignal(
                    LmlSignalView::new(&dataset, &views, bounds).expect("valid Node signal"),
                ),
            )]),
        )
        .expect("execute compiled baseline LML Node");
    let terminal = match case.path {
        BenchPath::Bundle => node_result
            .terminal_values
            .get(&NodeId(0))
            .and_then(|outputs| outputs.first()),
        BenchPath::PacketDag => node_result
            .terminal_values
            .values()
            .next()
            .and_then(|outputs| outputs.first()),
    }
    .expect("one terminal Node value");
    let node_bytes = match (case.path, terminal) {
        (BenchPath::Bundle, LamQuantNodeValue::Bcs2(bytes)) => bytes.as_slice(),
        (BenchPath::PacketDag, LamQuantNodeValue::LmlPackets(packets))
            if packets.packets().len() == 1 =>
        {
            packets.packets()[0].as_slice()
        }
        (_, other) => panic!("unexpected Node output: {other:?}"),
    };
    assert_eq!(node_bytes, direct, "direct and Node bytes diverged");

    let output_hash = blake3::hash(&direct);
    let input_bytes = case
        .channels
        .checked_mul(case.samples)
        .and_then(|samples| samples.checked_mul(size_of::<i64>()))
        .expect("input byte count");

    println!(
        "CASE name={} channels={} samples={} input_bytes={} output_bytes={} \
         output_blake3={} plan_compile_ns={} byte_equal=true",
        case.name,
        case.channels,
        case.samples,
        input_bytes,
        direct.len(),
        output_hash.to_hex(),
        plan_compile_ns,
    );

    let measure_direct = |iterations| {
        elapsed_ns(iterations, || {
            let output = direct_encode(case.path, &dataset, &views, bounds);
            black_box(output);
        })
    };
    let mut measure_node = |iterations| {
        elapsed_ns(iterations, || {
            let result = executor
                .execute(
                    &plan,
                    [0x92; 32],
                    BTreeMap::from([(
                        input_port.clone(),
                        LamQuantNodeValue::LmlSignal(
                            LmlSignalView::new(&dataset, &views, bounds)
                                .expect("valid measured Node signal"),
                        ),
                    )]),
                )
                .expect("Node LML encode during measurement");
            black_box(result);
        })
    };

    for _ in 0..WARMUP_ITERATIONS {
        measure_direct(1);
        measure_node(1);
    }
    let calibration_ns = measure_direct(1).max(measure_node(1)).max(1);
    let target_ns = u128::from(target_round_ms) * 1_000_000;
    let iterations = usize::try_from(target_ns / calibration_ns)
        .unwrap_or(MAX_BATCH_ITERATIONS)
        .clamp(1, MAX_BATCH_ITERATIONS);
    println!(
        "CALIBRATION case={} iterations={} target_round_ms={}",
        case.name, iterations, target_round_ms
    );

    for round in 0..rounds {
        let direct_first = round % 2 == 0;
        let (direct_ns, node_ns) = if direct_first {
            (measure_direct(iterations), measure_node(iterations))
        } else {
            let node_ns = measure_node(iterations);
            (measure_direct(iterations), node_ns)
        };
        println!(
            "ROUND case={} index={} order={} iterations={} direct_ns={} node_ns={}",
            case.name,
            round,
            if direct_first {
                "direct-node"
            } else {
                "node-direct"
            },
            iterations,
            direct_ns,
            node_ns,
        );
    }
}

fn main() {
    let (rounds, target_round_ms) = arguments();
    println!(
        "LAMQUANT_GEN8_NODE_PATH_BENCH_V1 rounds={} target_round_ms={} cases={}",
        rounds,
        target_round_ms,
        CASES.len()
    );
    for case in CASES {
        run_case(case, rounds, target_round_ms);
    }
}
