//! # `lamquant-neg` — the Neural Evidence Graph (ADR 0114)
//!
//! A typed measurement/inference/generation substrate. Every value the platform
//! produces is a [`Node`] of one **epistemic class** — [`Measured`], [`Derived`],
//! [`Estimated`], [`Generated`], [`Hypothesis`], [`Action`], or [`Outcome`] —
//! connected by one of eight typed [`EdgeClass`] relations. The point is to turn
//! three platform invariants from *conventions* into *type errors*:
//!
//! 1. **Measurement and inference are different types.** `Node<Measured>` and
//!    `Node<Estimated>` are distinct types; a consumer requiring measured
//!    evidence cannot be handed an estimate (compile error).
//! 2. **Generative content is never evidence.** `Node<Generated>` carries
//!    `IS_EVIDENCE = false`; it cannot reach a `&Node<Measured>` consumer, and a
//!    type-erased runtime gate ([`class::tag_is_evidence`]) fails closed on it.
//! 3. **Every conclusion is replayable.** A node's content address
//!    ([`NodeId`]) is the SHA-256 of its class + payload + provenance, so a
//!    result hashes back through its parents to the measured nodes it stands on;
//!    [`NegGraph::verify`] detects any tampering.
//!
//! This crate is foundational — consumers depend *down* on it
//! (`lamquant-common ← abir ← lamquant-neg ← {LML, LMQ, training, eval, BLUT}`).
//! N0 (this slice) is the schema; the typed ABIR-atom `Measured` producer, the
//! PyO3 handle, and the LMQ/LQS producers land in later slices.
//!
//! ## The type barrier (this must NOT compile)
//!
//! A generated sample cannot be passed where measured evidence is required — the
//! mechanization of invariants #1 and #2:
//!
//! ```compile_fail,E0308
//! use lamquant_neg::{Node, NodePayload, Provenance};
//! use lamquant_neg::class::{Measured, Generated};
//!
//! // A consumer that only accepts measured evidence.
//! fn diagnose(_evidence: &Node<Measured>) {}
//!
//! let synthetic = Node::<Generated>::new(
//!     NodePayload::default(),
//!     Provenance::root("lmq-generative-decoder"),
//!     None,
//! );
//!
//! // ERROR[E0308]: expected `&Node<Measured>`, found `&Node<Generated>`.
//! diagnose(&synthetic);
//! ```
//!
//! ## Building a small evidence graph (this DOES compile)
//!
//! ```
//! use lamquant_neg::{NegGraph, Node, NodePayload, Provenance, Uncertainty, EdgeClass};
//! use lamquant_neg::class::{Measured, Estimated};
//!
//! let mut g = NegGraph::new();
//!
//! // A measured window (root evidence).
//! let m = g.add_node(Node::<Measured>::new(
//!     NodePayload { content_ref: Some("abir:w0".into()), summary: Some("21ch x 2500".into()) },
//!     Provenance::root("edf-reader@abir-s4"),
//!     None,
//! ));
//!
//! // An LMQ reconstruction OF that window — an estimate, born non-evidence,
//! // carrying its provenance back to the measured node.
//! let _est = g.add_node(Node::<Estimated>::new(
//!     NodePayload { content_ref: Some("lmq:recon-w0".into()), summary: Some("R=0.63".into()) },
//!     Provenance::from_parents("lmq-decoder@0068-e1", vec![m.clone()]),
//!     Some(Uncertainty { metric: "prd".into(), value: 18.4 }),
//! ));
//!
//! g.materialize_provenance_edges();
//! assert!(g.verify().is_ok());
//!
//! // Serialization is deterministic + content-addressed.
//! let json = g.to_json().unwrap();
//! let back = NegGraph::from_json(&json).unwrap();
//! assert_eq!(g.content_address(), back.content_address());
//! ```

pub mod class;
pub mod edge;
pub mod graph;
pub mod node;

pub use class::{
    name_for_tag, tag_is_evidence, Action, Derived, EpistemicClass, Estimated, Generated,
    Hypothesis, Measured, Outcome,
};
pub use edge::{Edge, EdgeClass};
pub use graph::{NegGraph, VerifyError};
pub use node::{ClassMismatch, Node, NodeId, NodePayload, NodeRecord, Provenance, Uncertainty};
