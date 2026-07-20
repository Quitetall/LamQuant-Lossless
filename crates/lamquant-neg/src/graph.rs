//! The graph (ADR 0114) — [`NegGraph`], a flat content-addressed set of nodes
//! and edges with a verify pass and a stable JSON serialization.
//!
//! The graph is the platform's currency: producers `add_node` typed nodes and
//! `add_edge` typed relations; consumers `view::<C>` a node at the verified
//! boundary. [`NegGraph::verify`] is the integrity gate — every edge endpoint
//! exists, every provenance parent exists, and every node's stored id equals its
//! recomputed content address (tamper-evidence). Serialization is deterministic
//! (nodes and edges sorted before emit) so the same graph always yields the same
//! bytes and the same content address.

use serde::{Deserialize, Serialize};

use crate::class::EpistemicClass;
use crate::edge::{Edge, EdgeClass};
use crate::node::{ClassMismatch, Node, NodeId, NodeRecord};

/// A Neural Evidence Graph: nodes (class-erased [`NodeRecord`]s) and the typed
/// [`Edge`]s between them.
///
/// **Scale (N0):** lookups (`get`/`view`/`add_node` idempotency/
/// `materialize_provenance_edges`) are O(n) linear scans — deliberately simple
/// for the schema slice, where graphs are small (one window's worth of nodes). A
/// producer that puts real volume through this (N3+) should add a
/// `HashMap<NodeId, usize>` index; the public API is shaped so that swap is
/// internal and non-breaking.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NegGraph {
    /// The nodes, by insertion; ids are unique (a re-added identical node is a
    /// no-op — content addressing makes that safe).
    pub nodes: Vec<NodeRecord>,
    /// The typed relations.
    pub edges: Vec<Edge>,
}

/// An integrity violation found by [`NegGraph::verify`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// A node's stored id does not equal its recomputed content address — the
    /// record was mutated after construction (tampering, or a hand-edit).
    IdMismatch {
        /// The stored (claimed) id.
        stored: NodeId,
        /// The id the content actually hashes to.
        recomputed: NodeId,
    },
    /// Two distinct records share an id — a hash collision or a forged id.
    DuplicateId(NodeId),
    /// A record cites a provenance parent that is not in the graph.
    DanglingParent {
        /// The node with the bad parent reference.
        node: NodeId,
        /// The missing parent.
        parent: NodeId,
    },
    /// An edge references an endpoint that is not in the graph.
    DanglingEdge {
        /// The missing endpoint.
        missing: NodeId,
    },
    /// A record's class tag is not a known [`EpistemicClass`] — a graph from a
    /// newer schema or a corrupt one. Fail-closed: unknown class is an error,
    /// never silently trusted.
    UnknownClass {
        /// The offending node.
        node: NodeId,
        /// The unrecognized tag.
        tag: u8,
    },
    /// A node carries a non-finite [`crate::Uncertainty`] value (NaN/±inf) —
    /// meaningless, and it would poison equality + the content address. Rejected
    /// fail-closed rather than silently hashed.
    NonFiniteUncertainty {
        /// The offending node.
        node: NodeId,
        /// The uncertainty metric name.
        metric: String,
    },
}

impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VerifyError::IdMismatch { stored, recomputed } => write!(
                f,
                "node id {stored} does not match recomputed content address {recomputed} (tampered)"
            ),
            VerifyError::DuplicateId(id) => write!(f, "duplicate node id {id}"),
            VerifyError::DanglingParent { node, parent } => {
                write!(f, "node {node} cites missing provenance parent {parent}")
            }
            VerifyError::DanglingEdge { missing } => {
                write!(f, "edge references missing node {missing}")
            }
            VerifyError::UnknownClass { node, tag } => {
                write!(f, "node {node} has unknown epistemic-class tag {tag}")
            }
            VerifyError::NonFiniteUncertainty { node, metric } => {
                write!(f, "node {node} has non-finite uncertainty '{metric}'")
            }
        }
    }
}

impl std::error::Error for VerifyError {}

impl NegGraph {
    /// An empty graph.
    pub fn new() -> Self {
        NegGraph::default()
    }

    /// Insert a typed node, returning its content address. Idempotent: adding a
    /// node whose id already exists does nothing (content addressing guarantees
    /// identical content ⇒ identical id).
    pub fn add_node<C: EpistemicClass>(&mut self, node: Node<C>) -> NodeId {
        let rec = node.into_record();
        let id = rec.id.clone();
        if !self.nodes.iter().any(|n| n.id == id) {
            self.nodes.push(rec);
        }
        id
    }

    /// Add a typed edge between two nodes.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, class: EdgeClass) {
        self.edges.push(Edge::new(from, to, class));
    }

    /// Look up a record by id.
    pub fn get(&self, id: &NodeId) -> Option<&NodeRecord> {
        self.nodes.iter().find(|n| &n.id == id)
    }

    /// The verified typed accessor over the graph: recover a [`Node<C>`] for
    /// `id` iff it exists AND its class matches `C`. This is how a consumer that
    /// requires `Node<Measured>` pulls one out of a deserialized graph — the
    /// class is checked once, here, and the type system enforces it thereafter.
    pub fn view<C: EpistemicClass>(&self, id: &NodeId) -> Option<Result<Node<C>, ClassMismatch>> {
        self.get(id).map(|rec| rec.view::<C>())
    }

    /// Materialize every node's stored `provenance.parents` as explicit
    /// [`EdgeClass::ProvenanceDependency`] edges (parent → child), so the
    /// provenance backbone is traversable, not just hashed into the id. Safe to
    /// call once after building the node set; skips edges already present.
    pub fn materialize_provenance_edges(&mut self) {
        let mut to_add: Vec<Edge> = Vec::new();
        for node in &self.nodes {
            for parent in &node.provenance.parents {
                let e = Edge::new(
                    parent.clone(),
                    node.id.clone(),
                    EdgeClass::ProvenanceDependency,
                );
                if !self.edges.contains(&e) {
                    to_add.push(e);
                }
            }
        }
        self.edges.extend(to_add);
    }

    /// Integrity check. Returns every violation found (empty ⇒ sound). Fail-CLOSED
    /// on unknown class. This is the runtime backbone of the ADR 0114 gate.
    pub fn verify(&self) -> Result<(), Vec<VerifyError>> {
        let mut errors = Vec::new();

        // Unique ids + tamper check + known class.
        let mut seen: Vec<&NodeId> = Vec::new();
        for node in &self.nodes {
            if seen.contains(&&node.id) {
                errors.push(VerifyError::DuplicateId(node.id.clone()));
            } else {
                seen.push(&node.id);
            }
            let recomputed = node.recompute_id();
            if recomputed != node.id {
                errors.push(VerifyError::IdMismatch {
                    stored: node.id.clone(),
                    recomputed,
                });
            }
            if node.class_name().is_none() {
                errors.push(VerifyError::UnknownClass {
                    node: node.id.clone(),
                    tag: node.class_tag,
                });
            }
            if let Some(u) = &node.uncertainty {
                if !u.value.is_finite() {
                    errors.push(VerifyError::NonFiniteUncertainty {
                        node: node.id.clone(),
                        metric: u.metric.clone(),
                    });
                }
            }
        }

        // Provenance parents must exist.
        for node in &self.nodes {
            for parent in &node.provenance.parents {
                if self.get(parent).is_none() {
                    errors.push(VerifyError::DanglingParent {
                        node: node.id.clone(),
                        parent: parent.clone(),
                    });
                }
            }
        }

        // Edge endpoints must exist.
        for edge in &self.edges {
            if self.get(&edge.from).is_none() {
                errors.push(VerifyError::DanglingEdge {
                    missing: edge.from.clone(),
                });
            }
            if self.get(&edge.to).is_none() {
                errors.push(VerifyError::DanglingEdge {
                    missing: edge.to.clone(),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// A deterministic, canonically-ordered clone (nodes sorted by id, edges by
    /// (from, to, class-name)). Serializing this yields stable bytes regardless
    /// of insertion order — the basis of the graph's own content address.
    pub fn canonicalized(&self) -> NegGraph {
        let mut nodes = self.nodes.clone();
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        let mut edges = self.edges.clone();
        edges.sort_by(|a, b| {
            (a.from.as_str(), a.to.as_str(), a.class.name()).cmp(&(
                b.from.as_str(),
                b.to.as_str(),
                b.class.name(),
            ))
        });
        NegGraph { nodes, edges }
    }

    /// Deterministic JSON of the canonicalized graph.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.canonicalized())
    }

    /// Parse a graph from JSON. Does NOT verify — call [`NegGraph::verify`] after.
    pub fn from_json(s: &str) -> Result<NegGraph, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// The graph's own content address: SHA-256 over its canonical JSON. Two
    /// graphs with the same nodes+edges (any insertion order) hash identically —
    /// the round-trip-stability property the ADR 0114 gate checks.
    pub fn content_address(&self) -> String {
        let json = self
            .to_json()
            .expect("canonicalized NegGraph always serializes");
        crate::node::hex_sha256(json.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class::{Derived, Estimated, Generated, Measured};
    use crate::node::{NodePayload, Provenance, Uncertainty};

    fn small_graph() -> (NegGraph, NodeId, NodeId) {
        let mut g = NegGraph::new();
        let m = Node::<Measured>::new(
            NodePayload {
                content_ref: Some("abir:w0".into()),
                summary: Some("21ch x 2500".into()),
            },
            Provenance::root("edf-reader@abir-s4"),
            None,
        );
        let m_id = g.add_node(m);
        let est = Node::<Estimated>::new(
            NodePayload {
                content_ref: Some("lmq:recon-w0".into()),
                summary: Some("R=0.63".into()),
            },
            Provenance::from_parents("lmq-decoder@0068-e1", vec![m_id.clone()]),
            Some(Uncertainty {
                metric: "prd".into(),
                value: 18.4,
            }),
        );
        let est_id = g.add_node(est);
        (g, m_id, est_id)
    }

    #[test]
    fn build_view_and_verify() {
        let (mut g, m_id, est_id) = small_graph();
        g.materialize_provenance_edges();
        assert!(g.verify().is_ok());

        // The measured node views as Measured, not as Generated.
        assert!(g.view::<Measured>(&m_id).unwrap().is_ok());
        assert!(g.view::<Generated>(&m_id).unwrap().is_err());
        // The estimate views as Estimated and carries its uncertainty.
        let est = g.view::<Estimated>(&est_id).unwrap().unwrap();
        assert_eq!(est.uncertainty().unwrap().value, 18.4);
        assert!(!est.is_evidence());
    }

    #[test]
    fn provenance_edges_are_materialized() {
        let (mut g, m_id, est_id) = small_graph();
        assert!(g.edges.is_empty());
        g.materialize_provenance_edges();
        assert!(g.edges.iter().any(|e| e.from == m_id
            && e.to == est_id
            && e.class == EdgeClass::ProvenanceDependency));
        // Idempotent.
        let n = g.edges.len();
        g.materialize_provenance_edges();
        assert_eq!(g.edges.len(), n);
    }

    #[test]
    fn add_node_is_idempotent_by_content() {
        let mut g = NegGraph::new();
        let mk = || Node::<Derived>::new(NodePayload::default(), Provenance::root("t"), None);
        let a = g.add_node(mk());
        let b = g.add_node(mk());
        assert_eq!(a, b);
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn json_round_trips_and_content_address_is_stable() {
        let (mut g, _, _) = small_graph();
        g.materialize_provenance_edges();
        let json = g.to_json().unwrap();
        let back = NegGraph::from_json(&json).unwrap();
        assert!(back.verify().is_ok());
        // Content address is insertion-order-independent.
        assert_eq!(g.content_address(), back.content_address());
        assert_eq!(g.content_address(), g.canonicalized().content_address());
    }

    #[test]
    fn verify_catches_tampering() {
        let (mut g, m_id, _) = small_graph();
        // Mutate a payload without updating the id -> IdMismatch.
        let idx = g.nodes.iter().position(|n| n.id == m_id).unwrap();
        g.nodes[idx].payload.summary = Some("FORGED".into());
        let errs = g.verify().unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, VerifyError::IdMismatch { .. })));
    }

    #[test]
    fn verify_catches_dangling_edge() {
        let (mut g, m_id, _) = small_graph();
        g.add_edge(
            m_id,
            NodeId("does-not-exist".into()),
            EdgeClass::TemporalDependence,
        );
        let errs = g.verify().unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, VerifyError::DanglingEdge { .. })));
    }

    #[test]
    fn fractional_uncertainty_survives_json_round_trip_and_verifies() {
        // Regression: a node's content address must be stable across a
        // to_json/from_json round-trip even for an f64 whose JSON serialize→parse
        // is not bit-exact (e.g. 100.54785919189453, an exact f32 promoted to
        // f64). Hashing raw IEEE-754 bytes broke this; hashing the decimal form
        // fixes it. `verify()` after reload is the real check (comparing only
        // graph content_address would NOT have caught it).
        for v in [100.54785919189453f64, 18.4, 1.0 / 3.0, 0.1 + 0.2] {
            let mut g = NegGraph::new();
            g.add_node(Node::<Estimated>::new(
                NodePayload {
                    content_ref: Some("sha256:bb".into()),
                    summary: Some("R=0.24".into()),
                    ..Default::default()
                },
                Provenance::root("lqs@openecs"),
                Some(Uncertainty {
                    metric: "prd_pct".into(),
                    value: v,
                }),
            ));
            let back = NegGraph::from_json(&g.to_json().unwrap()).unwrap();
            assert!(
                back.verify().is_ok(),
                "reloaded node failed verify for uncertainty value {v}"
            );
            assert_eq!(g.content_address(), back.content_address());
        }
    }

    #[test]
    fn verify_rejects_non_finite_uncertainty() {
        let mut g = NegGraph::new();
        g.add_node(Node::<Estimated>::new(
            NodePayload::default(),
            Provenance::root("lmq"),
            Some(Uncertainty {
                metric: "prd".into(),
                value: f64::NAN,
            }),
        ));
        let errs = g.verify().unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, VerifyError::NonFiniteUncertainty { .. })));
    }

    #[test]
    fn verify_fails_closed_on_unknown_class() {
        let (mut g, m_id, _) = small_graph();
        // Forge an unknown class tag on-wire, then re-stamp the id so ONLY the
        // unknown-class rule fires (not IdMismatch).
        let idx = g.nodes.iter().position(|n| n.id == m_id).unwrap();
        g.nodes[idx].class_tag = 200;
        g.nodes[idx].id = g.nodes[idx].recompute_id();
        let errs = g.verify().unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, VerifyError::UnknownClass { tag: 200, .. })));
    }
}
