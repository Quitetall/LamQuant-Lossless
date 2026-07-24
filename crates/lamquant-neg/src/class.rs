//! The epistemic-class typestate (ADR 0114) — the `Node<C>` trust boundary.
//!
//! Every value the platform produces belongs to exactly one **epistemic
//! class**: was it *measured* by a sensor, *derived* deterministically from
//! evidence, *estimated* by a model, *generated* synthetically, or is it a
//! *hypothesis*, an *action*, or a measured *outcome*? Today this distinction
//! lives in prose and reviewer vigilance (the LML/LMQ product split, the
//! "generative is never evidence" rule of ADR 0068). Here it becomes a **type**.
//!
//! This mirrors the retired codec IR's modality tags
//! ([`Modality`](../../lamquant-legacy-ir/src/modality.rs)) exactly, one
//! axis over: `Node<Measured>` and `Node<Generated>` are DIFFERENT types, so a
//! consumer that requires measured evidence (`fn diagnose(&Node<Measured>)`)
//! simply cannot be handed a generated sample — the mismatch is a compile error,
//! not a bug caught in review. The class markers are sealed ZSTs; no downstream
//! crate can invent a new one or spoof the set.
//!
//! The one deliberate bridge between classes is
//! [`Node::promote`](crate::Node::promote) — an explicit, provenance-recording
//! narrowing that should essentially never appear on the codec/eval path (its
//! use is itself the audit signal that a boundary is being leaned on; ADR 0114
//! Validation).

mod sealed {
    /// Sealed supertrait — only this crate can implement [`super::EpistemicClass`].
    pub trait Sealed {}
}

/// An epistemic-class marker. Sealed: the only implementors are the ZST markers
/// in this module ([`Measured`], [`Derived`], …). A downstream crate cannot add
/// a class, which is what makes `Node<C>` a closed, trustworthy set rather than
/// an open one any caller could extend to smuggle inference in as measurement.
pub trait EpistemicClass: sealed::Sealed + Copy + Clone + core::fmt::Debug + 'static {
    /// Human-readable class name (logs / errors / serialized `class` field).
    const NAME: &'static str;
    /// The stable wire tag stored in a serialized [`crate::NodeRecord`]. Pinned:
    /// a silent renumbering would let an old graph deserialize into a different
    /// class than it was written as. Never reuse a retired tag.
    const TAG: u8;
    /// Whether a node of this class may be **treated as measured evidence** by a
    /// runtime gate. This is the load-bearing property behind invariant #2
    /// ("generative content is never evidence"). The *type* prevents the cast at
    /// compile time; this flag lets a type-erased gate (e.g. a serialized graph
    /// crossing the PyO3 boundary) enforce the same rule at runtime.
    ///
    /// - `Measured`  — true (a sensor produced it).
    /// - `Derived`   — true (a deterministic, replayable function of evidence).
    /// - `Outcome`   — true (a measured response to an action).
    /// - `Estimated` — FALSE (a model's posterior, not an observation).
    /// - `Generated` — FALSE (synthetic; the red line of ADR 0068).
    /// - `Hypothesis`/`Action` — FALSE (a proposition / a command, not evidence).
    const IS_EVIDENCE: bool;
}

macro_rules! class_marker {
    ($(#[$meta:meta])* $name:ident, $tag:expr, $label:expr, $is_evidence:expr) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl EpistemicClass for $name {
            const NAME: &'static str = $label;
            const TAG: u8 = $tag;
            const IS_EVIDENCE: bool = $is_evidence;
        }
    };
}

class_marker!(
    /// A value a sensor directly produced — with timestamp, position, reference,
    /// calibration identity, hardware state, and integrity status. The ground
    /// truth every other class ultimately traces back to. (Wraps an ABIR atom
    /// once the N3 producer lands.)
    Measured, 0, "measured", true
);
class_marker!(
    /// A **deterministic** transform whose provenance replays exactly — an LML
    /// lossless encoding, a `Reversible` ABIR pass, a doc-composer transclusion.
    /// Evidence-grade because re-running the named transform on the same inputs
    /// reproduces it bit-for-bit.
    Derived, 1, "derived", true
);
class_marker!(
    /// A **posterior** produced by a named model over named evidence — an LMQ
    /// near-lossless reconstruction, an LQS grade, an SNN state estimate. Carries
    /// uncertainty. NOT evidence: it is the model's belief, not an observation.
    Estimated, 2, "estimated", false
);
class_marker!(
    /// A **synthetic** or enhanced sample — an LMQ generative-decoder output, a
    /// GAN sample. Permanently labeled; NEVER substitutable for [`Measured`]
    /// (ADR 0068's red line, now a type). Acceptable as monitoring output,
    /// forbidden as evidence.
    Generated, 3, "generated", false
);
class_marker!(
    /// A scientific proposition with explicit supporting / contradicting
    /// evidence — an ADR, a technique-mine backlog entry, a DON'T-retry finding.
    Hypothesis, 4, "hypothesis", false
);
class_marker!(
    /// A stimulus / device command / experiment choice, with constraints and
    /// authorization. No producer exists today — the closed-loop action plane is
    /// a deferred consumer ADR (ADR 0114 Non-goals). Reserved so the schema is
    /// stable when it lands.
    Action, 5, "action", false
);
class_marker!(
    /// The **measured** response after an [`Action`]. Evidence-grade (it is a
    /// measurement), but distinct from [`Measured`] so an outcome always points
    /// back to the action that provoked it. Reserved alongside [`Action`].
    Outcome, 6, "outcome", true
);

/// Map a class wire tag ([`EpistemicClass::TAG`]) to its name — the SINGLE
/// SOURCE OF TRUTH for tag→name, referencing the same associated constants the
/// `class_marker!` invocations generate (a marker whose tag/name changes tracks
/// automatically). A foreign crate (e.g. the PyO3 handle) must call this rather
/// than keep its own copy. `None` for an unknown tag — never a stale default.
pub fn name_for_tag(tag: u8) -> Option<&'static str> {
    const TABLE: &[(u8, &str)] = &[
        (Measured::TAG, Measured::NAME),
        (Derived::TAG, Derived::NAME),
        (Estimated::TAG, Estimated::NAME),
        (Generated::TAG, Generated::NAME),
        (Hypothesis::TAG, Hypothesis::NAME),
        (Action::TAG, Action::NAME),
        (Outcome::TAG, Outcome::NAME),
    ];
    TABLE.iter().find(|(t, _)| *t == tag).map(|(_, n)| *n)
}

/// Whether the given class wire tag may be treated as measured evidence — the
/// type-erased twin of [`EpistemicClass::IS_EVIDENCE`], for a runtime gate over
/// a deserialized [`crate::NodeRecord`] whose class is only known as a `u8`
/// (e.g. after crossing the PyO3/JSON boundary). `None` for an unknown tag:
/// **fail-closed** — an unrecognized class is never credited as evidence.
pub fn tag_is_evidence(tag: u8) -> Option<bool> {
    const TABLE: &[(u8, bool)] = &[
        (Measured::TAG, Measured::IS_EVIDENCE),
        (Derived::TAG, Derived::IS_EVIDENCE),
        (Estimated::TAG, Estimated::IS_EVIDENCE),
        (Generated::TAG, Generated::IS_EVIDENCE),
        (Hypothesis::TAG, Hypothesis::IS_EVIDENCE),
        (Action::TAG, Action::IS_EVIDENCE),
        (Outcome::TAG, Outcome::IS_EVIDENCE),
    ];
    TABLE.iter().find(|(t, _)| *t == tag).map(|(_, e)| *e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_tags_and_names_pinned() {
        assert_eq!((Measured::TAG, Measured::NAME), (0, "measured"));
        assert_eq!((Derived::TAG, Derived::NAME), (1, "derived"));
        assert_eq!((Estimated::TAG, Estimated::NAME), (2, "estimated"));
        assert_eq!((Generated::TAG, Generated::NAME), (3, "generated"));
        assert_eq!((Hypothesis::TAG, Hypothesis::NAME), (4, "hypothesis"));
        assert_eq!((Action::TAG, Action::NAME), (5, "action"));
        assert_eq!((Outcome::TAG, Outcome::NAME), (6, "outcome"));
    }

    #[test]
    fn evidence_flags_encode_the_invariant() {
        // Invariant #2: estimated and generated are NEVER evidence.
        const {
            assert!(Measured::IS_EVIDENCE);
            assert!(Derived::IS_EVIDENCE);
            assert!(Outcome::IS_EVIDENCE);
            assert!(!Estimated::IS_EVIDENCE);
            assert!(!Generated::IS_EVIDENCE);
            assert!(!Hypothesis::IS_EVIDENCE);
            assert!(!Action::IS_EVIDENCE);
        }
    }

    #[test]
    fn name_for_tag_round_trips_every_class() {
        for tag in 0..=6u8 {
            let name = name_for_tag(tag).expect("every 0..=6 tag has a name");
            assert!(!name.is_empty());
        }
        assert_eq!(name_for_tag(7), None);
        assert_eq!(name_for_tag(255), None);
    }

    #[test]
    fn tag_is_evidence_fails_closed_on_unknown() {
        assert_eq!(tag_is_evidence(Measured::TAG), Some(true));
        assert_eq!(tag_is_evidence(Generated::TAG), Some(false));
        // Unknown class -> None, and a caller must treat None as "not evidence".
        assert_eq!(tag_is_evidence(7), None);
        assert_eq!(tag_is_evidence(200), None);
    }

    #[test]
    fn markers_are_zero_sized() {
        assert_eq!(core::mem::size_of::<Measured>(), 0);
        assert_eq!(core::mem::size_of::<Generated>(), 0);
    }
}
