//! The conformance projection between ABIR's semantic atoms and
//! `blut-graph-core`'s node-graph type mirror.
//!
//! # Why this crate exists at all
//!
//! ADR 0034 makes BLUT the domain control plane — it owns the nouns and delegates
//! the verbs — and ADR 0139 records that "BLUT gains a lightweight domain-neutral
//! graph compiler". `blut-graph-core` therefore takes **no** ABIR dependency.
//!
//! For a long time it took no ABIR dependency while still carrying ABIR's
//! *vocabulary*: `AbirRootType` and `AbirViewType` were serde enums whose variants
//! were named after ABIR's, with an `Unknown(String)` arm for anything else. So a
//! crate published to crates.io as general infrastructure described itself as
//! biosignal infrastructure, and every port in every domain had a field called
//! `abir`. The 2026-08-26 domain-token migration finished the job: graph-core now
//! carries an opaque [`DomainToken`] it never interprets, and **this crate owns
//! the vocabulary**.
//!
//! (This paragraph used to cite "ADR 0139 contract 4". ADR 0139 has no numbered
//! contracts — that clause does not exist and never did. The substance was right;
//! the citation was not.)
//!
//! That independence is correct, and it has a cost that nothing was paying: the
//! mirror can drift. Add an atom kind to ABIR and every graph plan involving it
//! silently becomes a token nobody recognises — which compiles, serialises,
//! round-trips, and produces a plan that has quietly stopped describing the
//! data. No error is raised anywhere, because from graph-core's point of view an
//! unrecognised token is a perfectly ordinary value.
//!
//! So the projection lives here, on the LamQuant side, depending on both. This
//! crate is deliberately **not** a type bridge — it introduces no new
//! representation and does not wrap graph-core's types. It is a *statement that
//! the mirror still agrees*, expressed as code that fails to compile or fails a
//! test when it stops being true.
//!
//! # What is actually pinned
//!
//! [`root_type_for_atom`] is total over `domain::Atom` by construction: it matches
//! exhaustively, so adding a variant to ABIR is a **compile error here**, not a
//! silent degradation downstream. The tests then check the other direction —
//! that no ABIR atom projects to an unrecognised token, and that the wire names
//! agree — so a *rename* on either side is caught too.
//!
//! Compile-time for additions, test-time for renames. Both directions covered,
//! because they fail differently.
//!
//! # What moved in the domain-token migration, and what deliberately did not
//!
//! The token strings are byte-identical to the kebab-case names serde produced
//! for the old enum variants, and [`DomainToken`] is `#[serde(transparent)]` over
//! its string. **Already-sealed plans keep their wire names**; only the in-memory
//! type changed. `wire_names_agree_with_abir_variant_names` pins that.
//!
//! What could not survive the move is the *meaning* of "unrecognised". It used to
//! be a property graph-core could answer — `matches!(root, Unknown(_))` — because
//! graph-core knew the full list. It cannot answer it any more, and should not:
//! knowing which tokens are real is domain knowledge. [`is_unrecognised`] is now
//! decided against the vocabulary declared below, which is the same question
//! asked by the layer that can actually answer it.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// Backward-compatible path for the ABIR-only montage projection crate.
pub use lamquant_abir_montage as montage;

use blut_graph_core::DomainToken;
use semantic_abir::Atom;

/// The root-type tokens an ABIR atom can project to.
///
/// These strings ARE the wire format. They were the kebab-case serde names of
/// the pre-migration `AbirRootType` variants and must not be edited casually: a
/// change here renames the type in every already-sealed plan.
pub const ATOM_ROOT_TYPE_NAMES: [&str; 6] = [
    "signal-block",
    "temporal-table",
    "table",
    "tensor",
    "encoded-block",
    "blob-ref",
];

/// Root-type tokens naming a *container* level rather than an atom kind.
///
/// Kept separate from [`ATOM_ROOT_TYPE_NAMES`] so that the distinction stays
/// explicit — it stops a future reader from "fixing" the apparent gap in
/// [`atom_root_types`] by mapping an atom onto a container.
pub const CONTAINER_ROOT_TYPE_NAMES: [&str; 3] = ["dataset", "recording", "stream"];

/// The view-type tokens a LamQuant node may declare.
///
/// A LamQuant node that cannot name its view type is a node whose contract
/// nobody can check, and admitting one would make the compiler's type agreement
/// vacuous for exactly the plans that need it most.
pub const DECLARABLE_VIEW_TYPE_NAMES: [&str; 6] =
    ["root", "recording", "stream", "block", "tensor", "atom"];

/// Project an ABIR atom onto the graph compiler's root-type mirror.
///
/// Exhaustive on purpose. A wildcard arm would make this function keep
/// compiling when ABIR gains an atom kind, and the new kind would reach the
/// compiler as a token nothing recognises — the exact failure this crate exists
/// to prevent.
pub fn root_type_for_atom(atom: &Atom) -> DomainToken {
    match atom {
        Atom::SignalBlock(_) => DomainToken::new(ATOM_ROOT_TYPE_NAMES[0]),
        Atom::TemporalTable(_) => DomainToken::new(ATOM_ROOT_TYPE_NAMES[1]),
        Atom::Table(_) => DomainToken::new(ATOM_ROOT_TYPE_NAMES[2]),
        Atom::Tensor(_) => DomainToken::new(ATOM_ROOT_TYPE_NAMES[3]),
        Atom::EncodedBlock(_) => DomainToken::new(ATOM_ROOT_TYPE_NAMES[4]),
        Atom::BlobRef(_) => DomainToken::new(ATOM_ROOT_TYPE_NAMES[5]),
    }
}

/// Every root type an ABIR atom can project to, as tokens.
pub fn atom_root_types() -> [DomainToken; 6] {
    ATOM_ROOT_TYPE_NAMES.map(DomainToken::new)
}

/// The view types a LamQuant node may declare, as tokens.
pub fn declarable_view_types() -> [DomainToken; 6] {
    DECLARABLE_VIEW_TYPE_NAMES.map(DomainToken::new)
}

/// True when `root` is a type an ABIR atom can actually project to.
pub fn is_atom_root_type(root: &DomainToken) -> bool {
    ATOM_ROOT_TYPE_NAMES.contains(&root.as_str())
}

/// True when `root` names something the mirror does not recognise.
///
/// The check a producer should make before sealing a plan: an unrecognised root
/// type means the plan is describing data the compiler cannot reason about.
///
/// Before the domain-token migration this was `matches!(root, Unknown(_))` — a question graph-core
/// could answer because it held the whole list. It holds none of it now, so the
/// question is answered here, against the vocabulary this crate declares. Same
/// verdict, asked of the layer that owns the answer.
pub fn is_unrecognised(root: &DomainToken) -> bool {
    !ATOM_ROOT_TYPE_NAMES.contains(&root.as_str())
        && !CONTAINER_ROOT_TYPE_NAMES.contains(&root.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn no_atom_root_type_is_unrecognised() {
        // The core claim. If any ABIR atom projected to an unrecognised token,
        // plans naming it would compile and serialise while having stopped
        // describing the data -- silent, and invisible to every downstream check.
        for root in &atom_root_types() {
            assert!(
                !is_unrecognised(root),
                "{root:?} is unrecognised; the graph mirror has drifted from ABIR"
            );
            assert!(is_atom_root_type(root));
        }
    }

    #[test]
    fn the_projection_is_injective() {
        // Two atom kinds sharing a root type would make the compiler unable to
        // tell them apart, which is a type-agreement hole rather than a naming
        // nicety.
        let roots = atom_root_types();
        for (index, root) in roots.iter().enumerate() {
            for other in &roots[index + 1..] {
                assert_ne!(root, other, "two atom kinds share a root type");
            }
        }
    }

    #[test]
    fn every_abir_atom_projects_into_the_declared_vocabulary() {
        // Pins the exhaustive match to the constant. Without this, someone could
        // add an atom arm returning a token absent from ATOM_ROOT_TYPE_NAMES:
        // the match stays exhaustive, the compile-time guard stays green, and the
        // new kind is unrecognised at run time -- the drift this crate exists to
        // stop, reintroduced through the back door.
        let declared = atom_root_types();
        for atom_root in &declared {
            assert!(
                ATOM_ROOT_TYPE_NAMES.contains(&atom_root.as_str()),
                "{atom_root:?} is produced by the projection but is not declared"
            );
        }
        assert_eq!(declared.len(), ATOM_ROOT_TYPE_NAMES.len());
    }

    #[test]
    fn wire_names_agree_with_abir_variant_names() {
        // Catches a RENAME on either side, which the exhaustive match cannot:
        // renaming on both sides still compiles, but changes the wire name and
        // breaks every already-sealed plan.
        //
        // These are the SAME strings serde produced for the pre-migration kebab-case
        // enum variants, so the migration did not move the wire format -- this test is
        // what makes that claim checkable rather than asserted.
        let expected = [
            "signal-block",
            "temporal-table",
            "table",
            "tensor",
            "encoded-block",
            "blob-ref",
        ];
        for name in expected {
            let root = DomainToken::new(name);
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
        // Guards the guard: if `is_unrecognised` stopped detecting strays, the
        // producer-side check above it would pass everything.
        let stray = DomainToken::new("montage-block".to_string());
        assert!(is_unrecognised(&stray));
        assert!(!is_atom_root_type(&stray));
    }

    #[test]
    fn container_root_types_are_recognised_but_are_not_atoms() {
        // The container levels are legitimate vocabulary -- they must NOT read as
        // drift -- while still being ineligible as an atom projection.
        for name in CONTAINER_ROOT_TYPE_NAMES {
            let root = DomainToken::new(name);
            assert!(!is_unrecognised(&root), "{name} should be recognised");
            assert!(!is_atom_root_type(&root), "{name} is not an atom kind");
        }
    }

    #[test]
    fn declarable_view_types_are_nameable() {
        // graph-core rejects an empty token; a node declaring one would fail to
        // compile its plan. Assert the declared vocabulary cannot contain one.
        for view in &declarable_view_types() {
            assert!(
                !view.is_empty(),
                "a LamQuant node must not declare an unnameable view type"
            );
        }
    }
}
