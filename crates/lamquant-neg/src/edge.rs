//! Edges (ADR 0114) — the eight typed relations between nodes.
//!
//! An edge names *how* one node relates to another: a deterministic transform, a
//! probabilistic inference, a temporal/spatial correspondence, a causal
//! intervention, a calibration or provenance dependency, or an uncertainty
//! propagation. Edges are the traversal surface — a scientific result node walks
//! `provenance-dependency` and `calibration-dependency` edges back to the
//! measured nodes it stands on; the future risk-limiting codec reads
//! `uncertainty-propagation` edges to define "sufficient evidence".

use serde::{Deserialize, Serialize};

use crate::node::NodeId;

/// The relation an [`Edge`] encodes. Closed set (ADR 0114): a new relation is a
/// deliberate, reviewed addition here, not an ad-hoc string a caller invents.
///
/// `rename_all = "kebab-case"` makes the serde wire form match [`EdgeClass::name`]
/// exactly, so a foreign (e.g. Python) consumer reading the JSON graph keys on
/// the SAME string the Rust `name()` returns — one wire vocabulary, not two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EdgeClass {
    /// `to` is a deterministic, replayable transform of `from`.
    DeterministicTransform,
    /// `to` is a model's inference conditioned on `from`.
    ProbabilisticInference,
    /// `from` precedes `to` in time (a temporal dependence).
    TemporalDependence,
    /// `from` and `to` correspond across space (channels / electrodes / regions).
    SpatialCorrespondence,
    /// `from` is an intervention whose causal effect is `to` (an action→outcome
    /// causal claim, distinct from mere temporal succession).
    CausalIntervention,
    /// `to` depends on the calibration state recorded in `from`.
    CalibrationDependency,
    /// `to` was computed *from* `from` — the provenance backbone. Materializes a
    /// node's stored `provenance.parents` as traversable edges.
    ProvenanceDependency,
    /// `to`'s uncertainty is propagated from `from`'s (the risk-limiting codec's
    /// sufficiency machinery reads these).
    UncertaintyPropagation,
}

impl EdgeClass {
    /// Stable serialized name (also what a foreign consumer keys on).
    pub const fn name(self) -> &'static str {
        match self {
            EdgeClass::DeterministicTransform => "deterministic-transform",
            EdgeClass::ProbabilisticInference => "probabilistic-inference",
            EdgeClass::TemporalDependence => "temporal-dependence",
            EdgeClass::SpatialCorrespondence => "spatial-correspondence",
            EdgeClass::CausalIntervention => "causal-intervention",
            EdgeClass::CalibrationDependency => "calibration-dependency",
            EdgeClass::ProvenanceDependency => "provenance-dependency",
            EdgeClass::UncertaintyPropagation => "uncertainty-propagation",
        }
    }
}

/// A directed, typed relation between two nodes, referenced by content address.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    /// Source node.
    pub from: NodeId,
    /// Destination node.
    pub to: NodeId,
    /// The relation.
    pub class: EdgeClass,
}

impl Edge {
    /// Construct an edge.
    pub fn new(from: NodeId, to: NodeId, class: EdgeClass) -> Self {
        Edge { from, to, class }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_edge_class_has_a_distinct_name() {
        let all = [
            EdgeClass::DeterministicTransform,
            EdgeClass::ProbabilisticInference,
            EdgeClass::TemporalDependence,
            EdgeClass::SpatialCorrespondence,
            EdgeClass::CausalIntervention,
            EdgeClass::CalibrationDependency,
            EdgeClass::ProvenanceDependency,
            EdgeClass::UncertaintyPropagation,
        ];
        let mut names: Vec<&str> = all.iter().map(|c| c.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len(), "edge-class names must be unique");
    }

    #[test]
    fn serde_wire_form_equals_name() {
        // The one-wire-vocabulary invariant: serde emits exactly what name()
        // returns, so a foreign consumer keys on a single string per relation.
        for c in [
            EdgeClass::DeterministicTransform,
            EdgeClass::ProbabilisticInference,
            EdgeClass::TemporalDependence,
            EdgeClass::SpatialCorrespondence,
            EdgeClass::CausalIntervention,
            EdgeClass::CalibrationDependency,
            EdgeClass::ProvenanceDependency,
            EdgeClass::UncertaintyPropagation,
        ] {
            let j = serde_json::to_string(&c).unwrap();
            assert_eq!(j, format!("\"{}\"", c.name()));
        }
    }

    #[test]
    fn edge_class_serde_round_trips() {
        for c in [
            EdgeClass::DeterministicTransform,
            EdgeClass::ProvenanceDependency,
            EdgeClass::UncertaintyPropagation,
        ] {
            let j = serde_json::to_string(&c).unwrap();
            let back: EdgeClass = serde_json::from_str(&j).unwrap();
            assert_eq!(c, back);
        }
    }
}
