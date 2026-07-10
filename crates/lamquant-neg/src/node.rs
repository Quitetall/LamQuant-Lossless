//! Nodes (ADR 0114) — the typed [`Node<C>`] over the erased [`NodeRecord`].
//!
//! Storage is **class-erased and serializable** ([`NodeRecord`], with a `u8`
//! `class_tag`); the **accessor is typed and verified** ([`Node<C>`], a
//! zero-cost wrapper carrying `PhantomData<C>`). This is exactly ABIR's split
//! (erased wire, typed accessor): a serialized graph is a flat list of records,
//! but the only way to *use* one as a given class is [`NodeRecord::view`], which
//! checks the tag and hands back a `Node<C>` the compiler then polices.
//!
//! Content addressing: a node's [`NodeId`] is the SHA-256 of a canonical,
//! order-independent serialization of its class, payload, provenance, and
//! uncertainty. The id is therefore a function of everything the node claims —
//! tampering with any field (including swapping the class, or rewriting a parent
//! hash) changes the id, so [`crate::NegGraph::verify`] can detect it.

use core::marker::PhantomData;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::class::{name_for_tag, EpistemicClass, Measured};

/// Lowercase-hex a SHA-256 digest of `bytes`. Shared by node and graph content
/// addressing so the encoding is defined once.
pub(crate) fn hex_sha256(bytes: &[u8]) -> String {
    use core::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for b in digest {
        // Infallible: writing to a String never errors.
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

/// A content address: the lowercase-hex SHA-256 of a node's canonical bytes.
/// Newtype (not a bare `String`) so a node id can't be confused with an
/// arbitrary label or a payload hash at a call site.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    /// The hex string, for logging / edge construction / the PyO3 boundary.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Display for NodeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What produced a node and what it was computed from — the provenance carried
/// *on the node itself* (so it is part of the content address and thus
/// tamper-evident), separate from the [`crate::Edge`]s a graph materializes for
/// traversal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The named thing that produced this node, ideally with a version — e.g.
    /// `"lmq-decoder@0068-e1"`, `"lqs@1.2"`, `"edf-reader@abir-s4"`. Free-form in
    /// N0; a producer registry can tighten this later.
    pub producer: String,
    /// Content addresses of the nodes this one was computed from. Sorted before
    /// hashing so provenance order does not affect the id. Empty for a root
    /// [`Measured`] node.
    pub parents: Vec<NodeId>,
    /// Optional human note — e.g. the transform name, or a promotion audit line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Provenance {
    /// A root producer with no parents (the shape a [`Measured`] node takes).
    pub fn root(producer: impl Into<String>) -> Self {
        Provenance {
            producer: producer.into(),
            parents: Vec::new(),
            note: None,
        }
    }

    /// A producer computed from the given parent nodes.
    pub fn from_parents(producer: impl Into<String>, parents: Vec<NodeId>) -> Self {
        Provenance {
            producer: producer.into(),
            parents,
            note: None,
        }
    }

    /// Attach a note (builder-style).
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

/// A calibrated-uncertainty tag on a node. Load-bearing for [`crate::class::Estimated`]
/// / [`crate::class::Generated`] and for the future risk-limiting codec (its
/// `uncertainty-propagation` edges read this). Minimal in N0: a named metric and
/// its scalar value (e.g. `{"prd", 18.4}`, `{"lqs-grade-margin", 0.06}`).
///
/// `value` MUST be finite. A NaN would make `PartialEq` (and thus node equality)
/// behave surprisingly (`NaN != NaN`) and would leak into the content address via
/// its bit pattern; a non-finite uncertainty is meaningless anyway. This is
/// enforced fail-closed by [`crate::NegGraph::verify`]
/// ([`crate::VerifyError::NonFiniteUncertainty`]), not just documented.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Uncertainty {
    /// What the number measures (a PRD, an entropy in bits, a posterior width, …).
    pub metric: String,
    /// The value. Must be finite (see the type note).
    pub value: f64,
}

/// The payload a node points at — the actual bytes/signal live elsewhere
/// (content-addressed in the artifact plane); the node references them by hash.
/// Keeps the graph a light provenance skeleton rather than a data blob.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodePayload {
    /// Content hash of the actual payload bytes (an ABIR atom, an `.lml`
    /// window, a decoder output tensor). `None` for a node that is purely a
    /// provenance/hypothesis marker with no byte payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_ref: Option<String>,
    /// A short human summary of the payload (channel×window shape, a grade, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// The class-erased, serializable node as it lives in a [`crate::NegGraph`] and
/// on the wire. `class_tag` is the epistemic class as a `u8`
/// ([`EpistemicClass::TAG`]); the typed [`Node<C>`] is recovered via
/// [`NodeRecord::view`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeRecord {
    /// Content address — SHA-256 of the canonical bytes of every other field.
    pub id: NodeId,
    /// Epistemic class as a wire tag ([`EpistemicClass::TAG`]).
    pub class_tag: u8,
    /// What this node points at.
    pub payload: NodePayload,
    /// How it was produced and from what.
    pub provenance: Provenance,
    /// Calibrated uncertainty, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<Uncertainty>,
}

impl NodeRecord {
    /// The canonical, order-independent byte serialization hashed into the id.
    /// Deterministic and independent of `serde_json` key ordering: fields are
    /// written in a fixed order with explicit length prefixes, and `parents`
    /// are sorted. Any change to any field (class included) changes these bytes.
    fn canonical_bytes(
        class_tag: u8,
        payload: &NodePayload,
        provenance: &Provenance,
        uncertainty: &Option<Uncertainty>,
    ) -> Vec<u8> {
        fn push_str(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        fn push_opt(buf: &mut Vec<u8>, s: &Option<String>) {
            match s {
                Some(v) => {
                    buf.push(1);
                    push_str(buf, v);
                }
                None => buf.push(0),
            }
        }

        let mut buf = Vec::new();
        buf.push(class_tag);
        push_opt(&mut buf, &payload.content_ref);
        push_opt(&mut buf, &payload.summary);
        push_str(&mut buf, &provenance.producer);
        push_opt(&mut buf, &provenance.note);

        // Parents sorted -> provenance order does not affect the id.
        let mut parents: Vec<&str> = provenance.parents.iter().map(|p| p.as_str()).collect();
        parents.sort_unstable();
        buf.extend_from_slice(&(parents.len() as u64).to_le_bytes());
        for p in parents {
            push_str(&mut buf, p);
        }

        match uncertainty {
            Some(u) => {
                buf.push(1);
                push_str(&mut buf, &u.metric);
                buf.extend_from_slice(&u.value.to_le_bytes());
            }
            None => buf.push(0),
        }
        buf
    }

    /// Compute the content address for the given fields.
    fn compute_id(
        class_tag: u8,
        payload: &NodePayload,
        provenance: &Provenance,
        uncertainty: &Option<Uncertainty>,
    ) -> NodeId {
        let bytes = Self::canonical_bytes(class_tag, payload, provenance, uncertainty);
        NodeId(hex_sha256(&bytes))
    }

    /// Recompute this record's id from its own fields — the tamper check. Returns
    /// the id the content *should* have; [`crate::NegGraph::verify`] compares it
    /// against the stored `id`.
    pub fn recompute_id(&self) -> NodeId {
        Self::compute_id(
            self.class_tag,
            &self.payload,
            &self.provenance,
            &self.uncertainty,
        )
    }

    /// Human-readable class name, or `None` if the tag is unknown (a record
    /// deserialized from a newer/corrupt graph).
    pub fn class_name(&self) -> Option<&'static str> {
        name_for_tag(self.class_tag)
    }

    /// The verified typed accessor: recover a [`Node<C>`] iff this record's
    /// stored `class_tag` matches `C`. This is THE boundary — the only way to go
    /// from an erased record to a typed node — so a consumer that holds a
    /// `Node<Measured>` knows the storage really was measured, and the compiler
    /// then forbids passing it anywhere a different class is required.
    pub fn view<C: EpistemicClass>(&self) -> Result<Node<C>, ClassMismatch> {
        if self.class_tag == C::TAG {
            Ok(Node {
                record: self.clone(),
                _class: PhantomData,
            })
        } else {
            Err(ClassMismatch {
                expected: C::TAG,
                actual: self.class_tag,
            })
        }
    }
}

/// A typed node — a zero-cost wrapper over a [`NodeRecord`] whose `class_tag` is
/// guaranteed to equal `C::TAG`. `Node<Measured>` and `Node<Generated>` are
/// distinct types; a `fn f(&Node<Measured>)` cannot be called with a
/// `Node<Generated>` (compile error — see the crate-level `compile_fail`
/// doctest). This is the mechanization of invariant #1 (measurement != inference).
#[derive(Clone, Debug)]
pub struct Node<C: EpistemicClass> {
    record: NodeRecord,
    _class: PhantomData<C>,
}

impl<C: EpistemicClass> Node<C> {
    /// Construct a typed node, computing its content address. The class is fixed
    /// by `C` — a node is *born* into its class (a producer writes
    /// `Node::<Generated>::new(...)`), never cast into it later.
    pub fn new(
        payload: NodePayload,
        provenance: Provenance,
        uncertainty: Option<Uncertainty>,
    ) -> Self {
        let id = NodeRecord::compute_id(C::TAG, &payload, &provenance, &uncertainty);
        Node {
            record: NodeRecord {
                id,
                class_tag: C::TAG,
                payload,
                provenance,
                uncertainty,
            },
            _class: PhantomData,
        }
    }

    /// This node's content address.
    pub fn id(&self) -> &NodeId {
        &self.record.id
    }

    /// The class wire tag (`== C::TAG`).
    pub fn class_tag(&self) -> u8 {
        self.record.class_tag
    }

    /// Whether this class may be treated as measured evidence
    /// ([`EpistemicClass::IS_EVIDENCE`]) — a compile-time constant, no lookup.
    pub fn is_evidence(&self) -> bool {
        C::IS_EVIDENCE
    }

    /// Borrow the underlying erased record (for serialization / graph insertion).
    pub fn record(&self) -> &NodeRecord {
        &self.record
    }

    /// Consume into the erased record.
    pub fn into_record(self) -> NodeRecord {
        self.record
    }

    /// Read the payload.
    pub fn payload(&self) -> &NodePayload {
        &self.record.payload
    }

    /// Read the provenance.
    pub fn provenance(&self) -> &Provenance {
        &self.record.provenance
    }

    /// Read the uncertainty, if any.
    pub fn uncertainty(&self) -> Option<&Uncertainty> {
        self.record.uncertainty.as_ref()
    }
}

impl Node<crate::class::Estimated> {
    /// The ONE deliberate class-narrowing bridge (ADR 0114): turn an estimate
    /// into a measurement-grade node. This should essentially never appear on
    /// the codec/eval path — an estimate is not a measurement — and every use is
    /// a reviewable event (the Validation clause watches for it). It is not a
    /// silent cast: it mints a NEW [`Measured`] node whose provenance records the
    /// estimate it came from and the authority that promoted it, so the promotion
    /// is permanently visible in the graph.
    pub fn promote(&self, authority: &str) -> Node<Measured> {
        let prov = Provenance::from_parents(
            format!("promotion:{authority}"),
            vec![self.record.id.clone()],
        )
        .with_note("PROMOTED estimated->measured (ADR 0114 audited boundary)");
        // Uncertainty is intentionally NOT carried onto the promoted node: a
        // promotion asserts "treat this as measured", and the original estimate
        // — with its uncertainty intact — remains reachable as the provenance
        // parent, so the signal is preserved one hop back, not lost.
        Node::<Measured>::new(self.record.payload.clone(), prov, None)
    }
}

/// Attempted to view an erased [`NodeRecord`] as the wrong epistemic class — the
/// runtime twin of the compile-time barrier, raised when the class is only known
/// as a `u8` (a deserialized graph, the PyO3 boundary).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassMismatch {
    /// The class the caller asked for ([`EpistemicClass::TAG`]).
    pub expected: u8,
    /// The class the record actually is.
    pub actual: u8,
}

impl core::fmt::Display for ClassMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let e = name_for_tag(self.expected).unwrap_or("<unknown>");
        let a = name_for_tag(self.actual).unwrap_or("<unknown>");
        write!(
            f,
            "epistemic-class mismatch: asked for {e} (tag {}), record is {a} (tag {})",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for ClassMismatch {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class::{Generated, Measured};

    fn measured(summary: &str) -> Node<Measured> {
        Node::<Measured>::new(
            NodePayload {
                content_ref: Some("abir:deadbeef".into()),
                summary: Some(summary.into()),
            },
            Provenance::root("edf-reader@abir-s4"),
            None,
        )
    }

    #[test]
    fn id_is_deterministic() {
        assert_eq!(measured("w0").id(), measured("w0").id());
    }

    #[test]
    fn id_changes_with_any_field() {
        assert_ne!(measured("w0").id(), measured("w1").id());
    }

    #[test]
    fn class_is_part_of_the_id() {
        // Same payload/provenance, different class -> different content address.
        // (This is why you cannot relabel a Generated node as Measured without
        // the id revealing it.)
        let payload = NodePayload {
            content_ref: Some("x".into()),
            summary: None,
        };
        let prov = Provenance::root("p");
        let m = Node::<Measured>::new(payload.clone(), prov.clone(), None);
        let g = Node::<Generated>::new(payload, prov, None);
        assert_ne!(m.id(), g.id());
    }

    #[test]
    fn parents_order_does_not_affect_id() {
        let a = NodeId("aaa".into());
        let b = NodeId("bbb".into());
        let p1 = Provenance::from_parents("t", vec![a.clone(), b.clone()]);
        let p2 = Provenance::from_parents("t", vec![b, a]);
        let n1 = Node::<crate::class::Derived>::new(NodePayload::default(), p1, None);
        let n2 = Node::<crate::class::Derived>::new(NodePayload::default(), p2, None);
        assert_eq!(n1.id(), n2.id());
    }

    #[test]
    fn view_recovers_matching_class_and_rejects_others() {
        let rec = measured("w0").into_record();
        assert!(rec.view::<Measured>().is_ok());
        let err = rec.view::<Generated>().unwrap_err();
        assert_eq!(err.expected, Generated::TAG);
        assert_eq!(err.actual, Measured::TAG);
    }

    #[test]
    fn is_evidence_reflects_class() {
        assert!(measured("w0").is_evidence());
        let g = Node::<Generated>::new(NodePayload::default(), Provenance::root("gan"), None);
        assert!(!g.is_evidence());
    }

    #[test]
    fn recompute_id_matches_stored_id() {
        let rec = measured("w0").into_record();
        assert_eq!(rec.recompute_id(), rec.id);
    }

    #[test]
    fn promotion_mints_measured_node_recording_its_source() {
        let est = Node::<crate::class::Estimated>::new(
            NodePayload {
                content_ref: Some("lmq:recon".into()),
                summary: Some("R=0.63".into()),
            },
            Provenance::root("lmq-decoder@0068-e1"),
            Some(Uncertainty {
                metric: "prd".into(),
                value: 18.4,
            }),
        );
        let promoted = est.promote("clinician-signoff-42");
        assert_eq!(promoted.class_tag(), Measured::TAG);
        assert!(promoted.is_evidence());
        // The promotion is visible: the new node's parent is the estimate.
        assert_eq!(promoted.provenance().parents, vec![est.id().clone()]);
        assert!(promoted.provenance().producer.starts_with("promotion:"));
    }
}
