//! P8.3 #21, memory axis — does routing through a compiled plan cost more heap?
//!
//! The head-to-head bench answers throughput. Ratio is settled by construction
//! (both arms emit identical bytes, so identical compression). Semantic coverage
//! is a property, not a measurement. Memory was the one axis of the four P8.3
//! names with no evidence at all, and "we did not look" is not the same as
//! "no difference".
//!
//! This lives in its own test binary because it installs a global allocator.
//! A `#[global_allocator]` is process-wide, so measuring peak heap only means
//! anything when nothing else is allocating concurrently — hence one test, one
//! binary, and a serialised structure inside it.

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use blut_graph_core::{
    Capability, Compiler, ExecutionRealm, Graph, KernelRegistry, NodeId, NodeInstance,
    PlanExecutor, PortRef,
};
use lamquant_abir_codec::encode_lml_bundle_from_views_explicit;
use lamquant_lml_mcu::{lml::EncodeFeatures, lpc::LpcMode};
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

struct PeakTracking;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for PeakTracking {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: AllocLayout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: PeakTracking = PeakTracking;

/// Peak live heap ABOVE the baseline, in bytes, reached while running `body`.
///
/// The baseline is read before `body` runs and subtracted from the peak. An
/// earlier version subtracted the live total *after* `body` instead, which
/// silently discounted everything the body left allocated — and since each arm
/// leaves its output buffer live, that made the arm with the larger retained
/// output look cheaper. It reported the node path using 17% LESS heap than the
/// direct call, which is not a result anyone should have believed.
fn peak_bytes(body: impl FnOnce()) -> usize {
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    body();
    PEAK.load(Ordering::Relaxed).saturating_sub(baseline)
}

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

/// Peak heap for one window through each path, on the same input.
///
/// The bound is a ratio rather than an absolute: the node path legitimately
/// carries a plan and an executor the direct call does not, so demanding exact
/// parity would fail on a cost that is real and accepted. What must not happen
/// is the node path allocating on a different ORDER — that would mean routing
/// through a plan changes the memory profile of the codec, which is the thing
/// an MCU realm could not absorb.
#[test]
fn the_node_path_does_not_cost_materially_more_heap_per_window() {
    let signal = signal();
    let dataset = dataset(&signal);
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bounds = ResourceBounds::default();

    let mut registry = KernelRegistry::default();
    register_lml_nodes(&mut registry).unwrap();
    let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph(baseline_lml_descriptor().capabilities))
        .unwrap();

    // Warm both paths once before measuring: first-call lazy initialisation
    // would otherwise be charged to whichever arm ran first.
    let mut direct_out = Vec::new();
    let mut node_out = Vec::new();

    let direct_peak = {
        let mut run = || {
            direct_out = encode_lml_bundle_from_views_explicit(
                &dataset,
                &views,
                WINDOW_SIZE,
                LpcMode::Fixed,
                EncodeFeatures::default(),
                bounds,
            )
            .unwrap();
        };
        run();
        peak_bytes(run)
    };

    let node_peak = {
        let mut run = || {
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
                .unwrap();
            node_out = match &result.terminal_values.get(&NodeId(0)).unwrap()[0] {
                LamQuantNodeValue::Bcs2(bytes) => bytes.clone(),
                other => panic!("unexpected node output: {other:?}"),
            };
        };
        run();
        peak_bytes(run)
    };

    assert_eq!(
        direct_out, node_out,
        "the arms disagree on output, so comparing their memory compares \
         unrelated computations"
    );
    println!(
        "peak heap per window — direct {direct_peak} B, node {node_peak} B, \
         ratio {:.3}",
        node_peak as f64 / direct_peak.max(1) as f64
    );
    assert!(
        direct_peak > 0,
        "a zero direct peak means the allocator saw nothing and the comparison \
         is vacuous"
    );
    assert!(
        node_peak <= direct_peak * 2,
        "node path peaked at {node_peak} B against {direct_peak} B direct: \
         routing through a plan changed the memory profile of the codec"
    );
}
