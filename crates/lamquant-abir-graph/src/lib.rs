//! The conformance projection between ABIR's semantic atoms and
//! `blut-graph-core`'s node-graph type mirror.
//!
//! # Why this crate exists at all
//!
//! ADR 0139 contract 4 keeps BLUT domain-neutral: `blut-graph-core` takes **no**
//! ABIR dependency. It models ABIR structurally instead — `AbirRootType` and
//! `AbirViewType` are plain serde enums whose variants happen to be named after
//! ABIR's, with an `Unknown(String)` arm for anything they do not recognise.
//!
//! That independence is correct, and it has a cost that nothing was paying: the
//! mirror can drift. Add an atom kind to ABIR and every graph plan involving it
//! silently becomes `Unknown("...")` — which compiles, serialises, round-trips,
//! and produces a plan that has quietly stopped describing the data. No error is
//! raised anywhere, because from graph-core's point of view an unknown type is a
//! perfectly ordinary value.
//!
//! So the projection lives here, on the LamQuant side, depending on both. This
//! crate is deliberately **not** a type bridge — it introduces no new
//! representation and does not wrap graph-core's types. It is a *statement that
//! the mirror still agrees*, expressed as code that fails to compile or fails a
//! test when it stops being true.
//!
//! # What is actually pinned
//!
//! [`root_type_for_atom`] is total over `abir::Atom` by construction: it matches
//! exhaustively, so adding a variant to ABIR is a **compile error here**, not a
//! silent degradation downstream. The tests then check the other direction —
//! that no ABIR atom projects to `Unknown`, and that the wire names agree — so a
//! *rename* on either side is caught too.
//!
//! Compile-time for additions, test-time for renames. Both directions covered,
//! because they fail differently.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod montage;

use blut_graph_core::{AbirRootType, AbirViewType};
use semantic_abir::Atom;

/// Project an ABIR atom onto the graph compiler's root-type mirror.
///
/// Exhaustive on purpose. A wildcard arm would make this function keep
/// compiling when ABIR gains an atom kind, and the new kind would reach the
/// compiler as `Unknown` — the exact failure this crate exists to prevent.
pub fn root_type_for_atom(atom: &Atom) -> AbirRootType {
    match atom {
        Atom::SignalBlock(_) => AbirRootType::SignalBlock,
        Atom::TemporalTable(_) => AbirRootType::TemporalTable,
        Atom::Table(_) => AbirRootType::Table,
        Atom::Tensor(_) => AbirRootType::Tensor,
        Atom::EncodedBlock(_) => AbirRootType::EncodedBlock,
        Atom::BlobRef(_) => AbirRootType::BlobRef,
    }
}

/// Every root type an ABIR atom can project to.
///
/// The graph mirror additionally carries `Dataset`, `Recording` and `Stream` —
/// container levels rather than atom kinds — which is why this is a subset
/// rather than the whole enum. Keeping the distinction explicit stops a future
/// reader from "fixing" the apparent gap by mapping an atom onto a container.
pub const ATOM_ROOT_TYPES: [AbirRootType; 6] = [
    AbirRootType::SignalBlock,
    AbirRootType::TemporalTable,
    AbirRootType::Table,
    AbirRootType::Tensor,
    AbirRootType::EncodedBlock,
    AbirRootType::BlobRef,
];

/// The view types a LamQuant node may declare.
///
/// `Unknown` is deliberately absent: a LamQuant node that cannot name its view
/// type is a node whose contract nobody can check, and admitting one would make
/// the compiler's type agreement vacuous for exactly the plans that need it
/// most.
pub const DECLARABLE_VIEW_TYPES: [AbirViewType; 6] = [
    AbirViewType::Root,
    AbirViewType::Recording,
    AbirViewType::Stream,
    AbirViewType::Block,
    AbirViewType::Tensor,
    AbirViewType::Atom,
];

/// True when `root` is a type an ABIR atom can actually project to.
pub fn is_atom_root_type(root: &AbirRootType) -> bool {
    ATOM_ROOT_TYPES.iter().any(|known| known == root)
}

/// True when `root` names something the mirror does not recognise.
///
/// The check a producer should make before sealing a plan: an `Unknown` root
/// type means the plan is describing data the compiler cannot reason about.
pub fn is_unrecognised(root: &AbirRootType) -> bool {
    matches!(root, AbirRootType::Unknown(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn no_atom_root_type_is_unrecognised() {
        // The core claim. If any ABIR atom projected to `Unknown`, plans naming
        // it would compile and serialise while having stopped describing the
        // data -- silent, and invisible to every downstream check.
        for root in &ATOM_ROOT_TYPES {
            assert!(
                !is_unrecognised(root),
                "{root:?} projects to Unknown; the graph mirror has drifted from ABIR"
            );
            assert!(is_atom_root_type(root));
        }
    }

    #[test]
    fn the_projection_is_injective() {
        // Two atom kinds sharing a root type would make the compiler unable to
        // tell them apart, which is a type-agreement hole rather than a naming
        // nicety.
        for (index, root) in ATOM_ROOT_TYPES.iter().enumerate() {
            for other in &ATOM_ROOT_TYPES[index + 1..] {
                assert_ne!(root, other, "two atom kinds share a root type");
            }
        }
    }

    #[test]
    fn wire_names_agree_with_abir_variant_names() {
        // Catches a RENAME on either side, which the exhaustive match cannot:
        // renaming a variant on both sides still compiles, but changes the
        // kebab-case wire name and breaks every already-sealed plan.
        let expected = [
            ("signal-block", AbirRootType::SignalBlock),
            ("temporal-table", AbirRootType::TemporalTable),
            ("table", AbirRootType::Table),
            ("tensor", AbirRootType::Tensor),
            ("encoded-block", AbirRootType::EncodedBlock),
            ("blob-ref", AbirRootType::BlobRef),
        ];
        for (name, root) in expected {
            let encoded = serde_json::to_string(&root).expect("root type serialises");
            assert_eq!(
                encoded,
                alloc::format!("\"{name}\""),
                "the mirror's wire name for {root:?} changed; already-sealed plans name the old one"
            );
        }
    }

    #[test]
    fn unknown_is_recognised_as_unrecognised() {
        // Guards the guard: if `is_unrecognised` stopped detecting Unknown, the
        // producer-side check above it would pass everything.
        let stray = AbirRootType::Unknown("montage-block".to_string());
        assert!(is_unrecognised(&stray));
        assert!(!is_atom_root_type(&stray));
    }

    #[test]
    fn declarable_view_types_exclude_unknown() {
        for view in &DECLARABLE_VIEW_TYPES {
            assert!(
                !matches!(view, AbirViewType::Unknown(_)),
                "a LamQuant node must not declare an unnameable view type"
            );
        }
    }
}
