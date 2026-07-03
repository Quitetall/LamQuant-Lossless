//! ADR 0074 · Track M — the typed stage-DAG authoring layer (host-only, `archive`).
//!
//! The codec's stages become owned newtypes generic over modality `M`; a `lower_*`
//! step **DISPATCHES** to the fused kernel — it never reimplements the DSP (ADR
//! 0074's "dispatch, not codegen"), so its output is byte-identical to the shipped
//! kernel by construction. This module holds the DAG's ENDPOINTS:
//!
//!   * [`Raw<M>`]  — one window of the typed recording (the input), and
//!   * [`Coded<M>`] — the compressed LML1 packet (the output),
//!
//! with a single `Raw → Coded` lowering ([`lower_encode`]) that equals
//! `compress_with_mode_views`, and its inverse ([`lower_decode`]). The
//! intermediate stages (`Subbands`/`Residuals`) and the reproduced assembler land
//! in M3, proven byte-equal to the kernel by oracle arm H.
//!
//! Modality `M` threads through every stage: `Raw<Eeg>`, `Coded<Eeg>` and
//! `Raw<Ecg>` are distinct types, so a stage function bound to one modality can
//! never be handed another — see the `compile_fail` doctest on [`Raw`].

use core::marker::PhantomData;

use abir::{Abir, Modality, ModalitySource};

use crate::error::LmlResult;
use crate::lml::{compress_with_mode_views, decompress};
use crate::lpc::LpcMode;

/// The **Raw** stage — one window of the typed recording; the DAG's input. The
/// whole `Abir` is treated as one window (`n_samples` samples × `n_channels`),
/// matching the per-window kernel entry.
///
/// Stage + modality are both in the type, so mixing them is a compile error.
///
/// A `Coded` is not a `Raw` (distinct stages) — a single `compile_fail` block
/// stops at the first error, so the two properties get one block each:
///
/// ```compile_fail
/// use lamquant_core::stage::{Coded, Raw};
/// fn takes_eeg_raw(_: Raw<abir::Eeg>) {}
/// fn bad(c: Coded<abir::Eeg>) { takes_eeg_raw(c); }
/// ```
///
/// And `Raw<Ecg>` is not `Raw<Eeg>` (distinct modalities — Pillar 1):
///
/// ```compile_fail
/// use lamquant_core::stage::Raw;
/// fn takes_eeg_raw(_: Raw<abir::Eeg>) {}
/// fn bad(r: Raw<abir::Ecg>) { takes_eeg_raw(r); }
/// ```
#[derive(Debug, Clone)]
pub struct Raw<M: Modality> {
    abir: Abir<M>,
}

impl<M: Modality> Raw<M> {
    /// Wrap a typed window.
    pub fn new(abir: Abir<M>) -> Self {
        Self { abir }
    }
    /// Borrow the underlying typed currency.
    pub fn abir(&self) -> &Abir<M> {
        &self.abir
    }
    /// Consume back into the typed `Abir`.
    pub fn into_abir(self) -> Abir<M> {
        self.abir
    }
}

/// The **Coded** stage — the compressed LML1 packet; the DAG's output. Carries the
/// modality type so [`lower_decode`] restores a typed `Raw<M>`. (The bytes
/// themselves are the modality-blind codec payload; `M` is a compile-time-only
/// tag — `Coded<M>` is `size_of == size_of::<Vec<u8>>`.)
#[derive(Debug, Clone)]
pub struct Coded<M: Modality> {
    bytes: Vec<u8>,
    _m: PhantomData<M>,
}

impl<M: Modality> Coded<M> {
    /// The compressed packet bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Consume into the raw packet bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Lower `Raw → Coded` by **dispatching** to the fused kernel over the zero-copy
/// window views. This is the DAG's `lower()`: a dispatch to
/// `compress_with_mode_views`, never a reimplementation of the DSP — so the output
/// is byte-identical to the shipped kernel.
///
/// ```
/// use lamquant_core::stage::{lower_encode, lower_decode, Raw};
/// use lamquant_core::lpc::LpcMode;
/// use abir::{Abir, Eeg, ModalitySource};
///
/// let sig = vec![vec![10i64, -20, 30, -40, 50, -60, 70, -80]; 4];
/// let raw: Raw<Eeg> =
///     Raw::new(Abir::from_channels_i64(sig, 250.0).into_modality::<Eeg>(ModalitySource::Manual));
/// let coded = lower_encode(&raw, 0, LpcMode::Fixed).unwrap();
/// let back = lower_decode(&coded, 250.0).unwrap();
/// assert_eq!(back.abir().n_channels(), 4);
/// ```
pub fn lower_encode<M: Modality>(
    raw: &Raw<M>,
    noise_bits: u8,
    mode: LpcMode,
) -> LmlResult<Coded<M>> {
    let n = raw.abir.n_samples();
    let views = raw.abir.window_views(0, n);
    let refs: Vec<&[i64]> = views.iter().map(|c| c.as_ref()).collect();
    let bytes = compress_with_mode_views(&refs, noise_bits, mode)?;
    Ok(Coded { bytes, _m: PhantomData })
}

/// Lower `Coded → Raw` (decode), restoring the modality type `M`. `sample_rate` is
/// supplied by the caller: the LML1 packet does not carry it (it lives in the BCS1
/// container header), so the decode is parameterized on it. Provenance is stamped
/// `Manual` — the modality is asserted by the caller's type argument `M`.
pub fn lower_decode<M: Modality>(coded: &Coded<M>, sample_rate: f64) -> LmlResult<Raw<M>> {
    let channels = decompress(coded.bytes())?;
    let abir = Abir::from_channels_i64(channels, sample_rate).into_modality::<M>(ModalitySource::Manual);
    Ok(Raw::new(abir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use abir::Eeg;

    fn synth(n_ch: usize, t: usize) -> Vec<Vec<i64>> {
        (0..n_ch)
            .map(|c| (0..t).map(|i| (((i * 3 + c * 7) % 512) as i64 - 256) * 40).collect())
            .collect()
    }

    fn eeg_raw(sig: Vec<Vec<i64>>) -> Raw<Eeg> {
        Raw::new(Abir::from_channels_i64(sig, 250.0).into_modality::<Eeg>(ModalitySource::Manual))
    }

    #[test]
    fn lower_encode_is_byte_identical_to_the_fused_kernel() {
        for &(n_ch, t) in &[(1usize, 100usize), (4, 2500), (8, 313)] {
            let sig = synth(n_ch, t);
            let raw = eeg_raw(sig.clone());
            for mode in [LpcMode::Fixed, LpcMode::Anytime { max_order: 16, deadline: None }] {
                let coded = lower_encode(&raw, 0, mode).expect("lower_encode");
                let views: Vec<&[i64]> = sig.iter().map(|c| c.as_slice()).collect();
                let fused = compress_with_mode_views(&views, 0, mode).expect("fused");
                assert_eq!(coded.bytes(), fused.as_slice(), "lower_encode != fused ({n_ch}x{t})");
            }
        }
    }

    #[test]
    fn round_trip_through_the_typed_endpoints() {
        let sig = synth(4, 2500);
        let raw = eeg_raw(sig.clone());
        // Both the fixed and the (reproducible, deadline-free) anytime path.
        for mode in [LpcMode::Fixed, LpcMode::Anytime { max_order: 16, deadline: None }] {
            let coded = lower_encode(&raw, 0, mode).expect("encode");
            let back = lower_decode(&coded, 250.0).expect("decode");
            assert_eq!(back.abir().n_channels(), 4);
            assert_eq!(back.abir().n_samples(), 2500);
            // Sample-level fidelity (the LML floor is lossless).
            let n = back.abir().n_samples();
            let got: Vec<Vec<i64>> = back
                .abir()
                .window_views(0, n)
                .iter()
                .map(|c| c.as_ref().to_vec())
                .collect();
            assert_eq!(got, sig, "typed-endpoint round-trip lost samples ({mode:?})");
        }
    }

    #[test]
    fn coded_phantom_is_zero_overhead() {
        // `M` on Coded is a compile-time-only tag: same size as the bytes it wraps.
        assert_eq!(
            core::mem::size_of::<Coded<Eeg>>(),
            core::mem::size_of::<Vec<u8>>()
        );
    }
}
