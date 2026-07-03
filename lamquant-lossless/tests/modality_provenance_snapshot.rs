//! ADR 0074 · Track I — modality-provenance neutrality gate.
//!
//! The entire per-dataset ingest config rests on ONE fact: declaring a dataset's
//! modality changes only the BCS1 header provenance bytes (6 = `modality_tag`,
//! 7 = `modality_source`) and NOTHING in the compressed payload. This gate pins
//! it: encoding the same signal as `Untyped` vs `Eeg` vs `Ecg` (all `Manual`
//! source) yields byte streams that differ at EXACTLY byte 6, with everything at
//! offset ≥ 40 (the metadata + window index + LML packets + footer) byte-identical.
//!
//! So authoritative typing is **byte-neutral for every existing corpus** by
//! construction — the whole reason the ingest config is safe to add.
#![cfg(feature = "archive")]

use abir::{Abir, Ecg, Eeg, Modality, ModalitySource, Untyped};
use lamquant_core::abir_container::write_abir_to_vec;
use lamquant_core::lpc::LpcMode;

fn synth() -> Vec<Vec<i64>> {
    (0..8)
        .map(|c| (0..2000).map(|i| (((i * 5 + c * 11) % 512) as i64 - 256) * 30).collect())
        .collect()
}

fn encode<M: Modality>(abir: &Abir<M>) -> Vec<u8> {
    write_abir_to_vec(abir, 250.0, 256, 0, "{}", LpcMode::Fixed, None, None).expect("write_abir")
}

#[test]
fn modality_typing_touches_only_header_byte_6_and_never_the_payload() {
    let sig = synth();
    let untyped: Abir<Untyped> = Abir::from_channels_i64(sig.clone(), 250.0);
    let eeg: Abir<Eeg> =
        Abir::from_channels_i64(sig.clone(), 250.0).into_modality::<Eeg>(ModalitySource::Manual);
    let ecg: Abir<Ecg> =
        Abir::from_channels_i64(sig, 250.0).into_modality::<Ecg>(ModalitySource::Manual);

    let b_untyped = encode(&untyped);
    let b_eeg = encode(&eeg);
    let b_ecg = encode(&ecg);

    let manual = ModalitySource::Manual.to_u8();
    // Provenance bytes reflect the declared modality; source is Manual throughout.
    assert_eq!((b_untyped[6], b_untyped[7]), (Untyped::TAG, manual), "untyped provenance");
    assert_eq!((b_eeg[6], b_eeg[7]), (Eeg::TAG, manual), "eeg provenance");
    assert_eq!((b_ecg[6], b_ecg[7]), (Ecg::TAG, manual), "ecg provenance");

    // THE LOAD-BEARING ASSERTION: typing changes EXACTLY byte 6 (the tag) — same
    // length, identical everywhere else (byte 7 is Manual for all three).
    assert_eq!(b_eeg.len(), b_untyped.len());
    assert_eq!(b_ecg.len(), b_untyped.len());
    for i in 0..b_untyped.len() {
        if i == 6 {
            continue;
        }
        assert_eq!(b_eeg[i], b_untyped[i], "eeg differs from untyped at byte {i} (only byte 6 may differ)");
        assert_eq!(b_ecg[i], b_untyped[i], "ecg differs from untyped at byte {i} (only byte 6 may differ)");
    }
    // Explicit payload-neutrality: offset ≥ 40 is byte-identical regardless of modality.
    assert_eq!(&b_eeg[40..], &b_untyped[40..], "payload must be byte-identical regardless of modality");
    assert_eq!(&b_ecg[40..], &b_untyped[40..], "payload must be byte-identical regardless of modality");
}
