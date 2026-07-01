//! The modality trust model (ADR 0069 S3a) — the `Abir<M>` typestate.
//!
//! `Abir` is generic over a **sealed** [`Modality`] marker (default
//! [`Untyped`]). This is a compile-time trust boundary, not a runtime check:
//! `Abir<Eeg>` and `Abir<Ecg>` are DIFFERENT types, so a modality-typed
//! consumer (e.g. `fn train(&Abir<Eeg>)`) simply cannot be called with the
//! wrong modality — the mismatch is a type error, not a bug caught in CI.
//!
//! **Reconciling with Pillar 1 (no cross-modality mixing) and the encoder
//! egress.** The codec's hot path (`Abir::window_views`, consumed by
//! `write_abir`) is modality-**blind** by design — bytes in, bytes out,
//! generic over any `M: Modality`, `Abir<Untyped>` included. That is
//! intentional and stays that way: the wire format has never encoded
//! modality, and this step keeps it byte-frozen. The trust boundary is
//! enforced one layer up, at the SEMANTIC accessors: [`Abir::into_modality`]
//! is the one deliberate, explicit narrowing point (untyped → typed,
//! recorded in `prov`), and [`Abir::verify`] is the boundary invariant check
//! a typed consumer calls before trusting the data. Nothing downstream of
//! `into_modality` can silently widen back to a different modality or mix
//! two modalities under one type — the sealed trait means only the markers
//! in this module implement `Modality` at all.
//!
//! All markers are zero-sized types (ZSTs); `PhantomData<M>` inside `Abir<M>`
//! costs nothing at runtime — see the `size_of` proof in `atoms.rs` tests.

mod sealed {
    /// Sealed supertrait — only this crate can implement [`super::Modality`].
    pub trait Sealed {}
}

/// A biosignal modality marker. Sealed: the only implementors are the ZST
/// markers defined in this module (`Untyped`, `Eeg`, `Ieeg`, …) — downstream
/// crates cannot invent new ones, which is what makes the `Abir<M>` typestate
/// a closed, trustworthy set rather than an open one any caller could spoof.
pub trait Modality: sealed::Sealed + Copy + Clone + core::fmt::Debug + 'static {
    /// Human-readable modality name (for logs/errors/provenance).
    const NAME: &'static str;
    /// The wire-adjacent tag byte recorded in [`ModalityProvenance::tag`].
    /// NOT part of the LML/LMO wire format itself (S3a is encoder-egress
    /// invariant) — this is IR-side bookkeeping only.
    const TAG: u8;
}

macro_rules! modality_marker {
    ($(#[$meta:meta])* $name:ident, $tag:expr, $label:expr) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl Modality for $name {
            const NAME: &'static str = $label;
            const TAG: u8 = $tag;
        }
    };
}

modality_marker!(
    /// The default, modality-erased marker — today's `Abir` behavior
    /// (`Abir` == `Abir<Untyped>`). The encoder egress path
    /// (`window_views`/`write_abir`) runs on this marker unchanged.
    Untyped,
    255,
    "untyped"
);
modality_marker!(
    /// Scalp electroencephalography.
    Eeg, 0, "eeg"
);
modality_marker!(
    /// Intracranial electroencephalography.
    Ieeg, 1, "ieeg"
);
modality_marker!(
    /// Electrocorticography.
    Ecog, 2, "ecog"
);
modality_marker!(
    /// Stereo-EEG.
    Seeg, 3, "seeg"
);
modality_marker!(
    /// Electrocardiography.
    Ecg, 4, "ecg"
);
modality_marker!(
    /// Electromyography.
    Emg, 5, "emg"
);
modality_marker!(
    /// Electrooculography.
    Eog, 6, "eog"
);
modality_marker!(
    /// Respiration.
    Resp, 7, "resp"
);
modality_marker!(
    /// Accelerometry.
    Accel, 8, "accel"
);
modality_marker!(
    /// Anything not covered by a dedicated marker above.
    Other, 9, "other"
);

/// How a modality assignment was decided — carried alongside the tag in
/// [`ModalityProvenance`] so a `verify()` failure (or an audit) can explain
/// WHY a stream ended up typed the way it did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModalitySource {
    /// Inferred from a channel label (e.g. "Fp1" → EEG-like 10-20 montage).
    ChannelLabel,
    /// Declared by the source format itself (e.g. a format that carries an
    /// explicit modality field).
    FormatDeclared,
    /// Explicitly set by the caller (`into_modality`, manual construction).
    Manual,
}

/// Provenance of an `Abir`'s modality assignment: how it was decided
/// ([`ModalitySource`]) and which modality it resolved to (`tag`, matching
/// the assigned `Modality::TAG`).
#[derive(Clone, Debug)]
pub struct ModalityProvenance {
    /// How the modality was decided.
    pub source: ModalitySource,
    /// The assigned modality's [`Modality::TAG`].
    pub tag: u8,
}

/// A boundary-invariant violation caught by [`crate::Abir::verify`].
#[derive(Clone, Debug)]
pub enum VerifyError {
    /// Not every channel's column has the same length as `n_samples`.
    ChannelLengthMismatch {
        /// Index of the first offending channel.
        channel: usize,
        /// That channel's actual column length.
        actual: usize,
        /// The `Abir`'s declared `n_samples`.
        expected: usize,
    },
    /// `prov.tag` does not match the `Abir<M>`'s type-level `M::TAG` — the
    /// provenance record has drifted from the type it is attached to.
    ProvenanceTagMismatch {
        /// The tag recorded in `prov`.
        recorded: u8,
        /// The tag the type parameter `M` demands.
        expected: u8,
    },
}

impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VerifyError::ChannelLengthMismatch {
                channel,
                actual,
                expected,
            } => write!(
                f,
                "channel {channel} has {actual} samples, expected n_samples={expected}"
            ),
            VerifyError::ProvenanceTagMismatch { recorded, expected } => write!(
                f,
                "provenance tag {recorded} does not match modality tag {expected}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VerifyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_tags_and_names() {
        assert_eq!(Untyped::TAG, 255);
        assert_eq!(Untyped::NAME, "untyped");
        assert_eq!(Eeg::TAG, 0);
        assert_eq!(Eeg::NAME, "eeg");
        assert_eq!(Ieeg::TAG, 1);
        assert_eq!(Ecog::TAG, 2);
        assert_eq!(Seeg::TAG, 3);
        assert_eq!(Ecg::TAG, 4);
        assert_eq!(Emg::TAG, 5);
        assert_eq!(Eog::TAG, 6);
        assert_eq!(Resp::TAG, 7);
        assert_eq!(Accel::TAG, 8);
        assert_eq!(Other::TAG, 9);
    }

    #[test]
    fn markers_are_zero_sized() {
        assert_eq!(core::mem::size_of::<Eeg>(), 0);
        assert_eq!(core::mem::size_of::<Untyped>(), 0);
    }
}
