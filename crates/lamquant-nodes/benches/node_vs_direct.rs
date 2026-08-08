//! P8.3 #21 — the node path measured against the path it would replace.
//!
//! Principle 1 says nothing is promoted to default until a superior replacement
//! is live. That makes this harness the evidence for #23's flip, so what it
//! measures has to be the thing the flip actually changes.
//!
//! Two arms:
//!
//!   * `direct` — `encode_lml_bundle_from_views_explicit`, the call the current
//!     path makes.
//!   * `node/execute` — executing an already-compiled plan over the same input.
//!
//! Compilation is deliberately OUTSIDE the measured region, and that choice is
//! the one most likely to flatter the node path, so it is worth defending: a
//! plan is compiled once and executed per window, so folding compilation into
//! per-window cost would measure a workload nobody runs. `node/compile` is
//! reported separately rather than hidden, so the amortisation can be checked
//! instead of assumed.
//!
//! The harness also asserts the two arms agree byte-for-byte before timing
//! anything. A throughput comparison between two functions that disagree about
//! their output is a comparison of unrelated quantities.

use std::collections::BTreeMap;
use std::hint::black_box;

use blut_graph_core::{
    Capability, Compiler, ExecutionRealm, Graph, KernelRegistry, NodeId, NodeInstance,
    PlanExecutor, PortRef,
};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use lamquant_abir_codec::encode_lml_bundle_from_views_explicit;
use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_nodes::{
    baseline_lml_descriptor, lml_node_config, register_lml_nodes, LamQuantNodeValue, LmlSignalView,
    NoopTransactionalSink, LML_BASELINE_NODE_TYPE,
};
use semantic_abir::{
    payload_content_id, AbirDataset, Atom, AtomTag, ByteOrder, ConceptId, DatasetDraft, DatasetTag,
    ElementType, Layout, ObjectId, PayloadDescriptor, Presence, Rational, Recording, RecordingTag,
    SignalBlock, Stream, StreamTag, TimeAxis, TimeSegment, ValidationLimits,
};
use semantic_abir_bcs::ResourceBounds;

/// 21 channels of 2560 samples — a montage-shaped window rather than a toy, so
/// per-window fixed costs are amortised the way production amortises them.
const CHANNELS: usize = 21;
const SAMPLES: usize = 2560;
const WINDOW_SIZE: usize = u16::MAX as usize;

fn signal() -> Vec<Vec<i64>> {
    (0..CHANNELS)
        .map(|channel| {
            (0..SAMPLES)
                .map(|sample| {
                    let base = ((sample * 3 + channel * 7) % 512) as i64 - 256;
                    let wobble = ((sample * sample + channel) % 97) as i64 - 48;
                    base * 40 + wobble
                })
                .collect()
        })
        .collect()
}

fn dataset(signal: &[Vec<i64>]) -> AbirDataset {
    let mut draft = DatasetDraft::new(ObjectId::<DatasetTag>::from_bytes([1; 16]));
    let recording_id = ObjectId::<RecordingTag>::from_bytes([2; 16]);
    let stream_id = ObjectId::<StreamTag>::from_bytes([3; 16]);
    let mut atom_ids = Vec::new();
    for (index, channel) in signal.iter().enumerate() {
        let bytes = channel
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let content_id = payload_content_id(ElementType::I64, &bytes);
        let mut atom_id = [0_u8; 16];
        atom_id[14] = (index / 256) as u8;
        atom_id[15] = (index % 256) as u8;
        let atom_id = ObjectId::<AtomTag>::from_bytes(atom_id);
        atom_ids.push(atom_id);
        draft.add_atom(Atom::SignalBlock(SignalBlock::new(
            atom_id,
            Presence::Present,
            Some(PayloadDescriptor::new(
                content_id,
                bytes.len() as u64,
                ElementType::I64,
                ByteOrder::Little,
                vec![1, channel.len() as u64],
                Layout::DenseRowMajor,
                None,
                None,
            )),
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

fn graph(capabilities: Vec<Capability>) -> Graph {
    Graph {
        version: 3,
        nodes: vec![NodeInstance {
            id: NodeId(0),
            descriptor: LML_BASELINE_NODE_TYPE.into(),
            descriptor_version: 1,
            config: lml_node_config(LpcMode::Fixed, WINDOW_SIZE).unwrap(),
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

fn head_to_head(criterion: &mut Criterion) {
    let signal = signal();
    let dataset = dataset(&signal);
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bounds = ResourceBounds::default();

    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).unwrap();
    let graph = graph(baseline_lml_descriptor().capabilities);

    let run_plan = |plan: &blut_graph_core::AuthorizedPlan| {
        let mut kernels = lamquant_nodes::LamQuantKernelExecutor::default();
        let mut sink = NoopTransactionalSink;
        let mut executor = PlanExecutor::new(&mut kernels, &mut sink);
        let result = executor
            .execute(
                plan,
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
            .unwrap();
        match &result.terminal_values.get(&NodeId(0)).unwrap()[0] {
            LamQuantNodeValue::Bcs2(bytes) => bytes.clone(),
            other => panic!("unexpected node output: {other:?}"),
        }
    };

    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();
    let direct_bytes = encode_lml_bundle_from_views_explicit(
        &dataset,
        &views,
        WINDOW_SIZE,
        LpcMode::Fixed,
        lamquant_lml_mcu::lml::EncodeFeatures::default(),
        bounds,
    )
    .unwrap();
    let node_bytes = run_plan(&plan);
    assert_eq!(
        direct_bytes, node_bytes,
        "the two arms disagree on output, so any throughput comparison between \
         them would be comparing unrelated quantities"
    );

    // Bytes of INPUT signal, so both arms report throughput over the same
    // denominator and the numbers can be read against each other directly.
    let input_bytes = (CHANNELS * SAMPLES * core::mem::size_of::<i64>()) as u64;
    let mut group = criterion.benchmark_group("lml_encode");
    group.throughput(Throughput::Bytes(input_bytes));

    group.bench_function("direct", |bencher| {
        bencher.iter(|| {
            black_box(
                encode_lml_bundle_from_views_explicit(
                    black_box(&dataset),
                    black_box(&views),
                    WINDOW_SIZE,
                    LpcMode::Fixed,
                    lamquant_lml_mcu::lml::EncodeFeatures::default(),
                    bounds,
                )
                .unwrap(),
            )
        })
    });

    group.bench_function("node/execute", |bencher| {
        bencher.iter(|| black_box(run_plan(black_box(&plan))))
    });
    group.finish();

    // Reported separately, not folded in: a plan is compiled once and executed
    // per window. Keeping it visible is what lets the amortisation be checked
    // rather than taken on trust.
    criterion.bench_function("lml_compile/plan", |bencher| {
        bencher.iter(|| {
            black_box(
                Compiler::new(&registry, ExecutionRealm::HostStream)
                    .compile(black_box(&graph))
                    .unwrap(),
            )
        })
    });
}

criterion_group!(benches, head_to_head);
criterion_main!(benches);
