//! Phase 2 acceptance gate (ADR 0054): **one functional battery, both formats.**
//!
//! The same parametrized battery — lossless (WP0, bit-exact), bounded-MAE
//! (near-lossless, δ-bound), and target-BPS (WP1–WP8, rate ceiling) — runs over
//! BOTH the LML floor (`LmlCodec`, integer) and the LMO ceiling (`LmoCodec`,
//! the Phase-2 parity re-encode). This is the ADR 0052 "one test battery over
//! both formats" property and the ADR 0054 Phase-2 gate.
//!
//! Requires the `encode` feature (the host LMO encoder). Run with:
//!   cargo test -p lamquant-optimum --features encode --test dual_format_battery
#![cfg(feature = "encode")]

use lamquant_lml_mcu::codec::{self, Codec, Format, LmlCodec, Mode};
use lamquant_lml_optimum::{decode_any, LmoCodec};

/// Smooth low-freq + a little high-freq + periodic spikes, so the wavelet
/// subbands carry real, allocatable energy (mirrors the core BPS property test).
fn make_signal(n_ch: usize, t: usize, seed: i64) -> Vec<Vec<i64>> {
    (0..n_ch)
        .map(|c| {
            let ph = seed.wrapping_add(c as i64 * 911);
            (0..t)
                .map(|i| {
                    let i = i as i64;
                    let lo = (((i + ph) as f64 * 0.05).sin() * 3000.0) as i64;
                    let hi = (((i + ph) as f64 * 0.9).sin() * 250.0) as i64;
                    let spike = if (i + ph) % 101 == 0 { 1200 } else { 0 };
                    lo + hi + spike
                })
                .collect()
        })
        .collect()
}

/// The two codecs under test, behind the shared `Codec` seam.
fn codecs() -> [(&'static str, Format, Box<dyn Codec>); 2] {
    [
        ("LML", Format::Lml, Box::new(LmlCodec)),
        ("LMO", Format::Lmo, Box::new(LmoCodec)),
    ]
}

#[test]
fn wp0_lossless_bit_exact_both_formats() {
    let sig = make_signal(6, 1024, 7);
    for (name, fmt, codec) in codecs() {
        let stream = codec.encode(&sig, Mode::Lossless).expect("encode");
        assert_eq!(codec::peek_format(&stream), Some(fmt), "{name} magic");
        let back = codec.decode(&stream).expect("decode");
        assert_eq!(back, sig, "{name}: WP0 must be bit-exact");
    }
}

#[test]
fn bounded_mae_delta_honored_both_formats() {
    let sig = make_signal(4, 2048, 21);
    for delta in [0u64, 4, 16] {
        for (name, fmt, codec) in codecs() {
            let stream = codec.encode(&sig, Mode::BoundedMae(delta)).expect("encode");
            assert_eq!(codec::peek_format(&stream), Some(fmt), "{name} magic");
            let back = codec.decode(&stream).expect("decode");
            for (co, cr) in sig.iter().zip(back.iter()) {
                for (o, r) in co.iter().zip(cr.iter()) {
                    assert!(
                        (o - r).unsigned_abs() <= delta,
                        "{name}: |{o}-{r}| exceeds δ={delta}"
                    );
                }
            }
        }
    }
}

#[test]
fn target_bps_rate_ceiling_both_formats() {
    let (n_ch, t) = (8usize, 2560usize);
    let nm = (n_ch * t) as f64;
    let sig = make_signal(n_ch, t, 99);
    for target in [4.0f64, 3.0, 2.0] {
        for (name, fmt, codec) in codecs() {
            let stream = codec.encode(&sig, Mode::TargetBps(target)).expect("encode");
            assert_eq!(codec::peek_format(&stream), Some(fmt), "{name} magic");
            let bps = stream.len() as f64 * 8.0 / nm;
            // +10% tolerance for header overhead on a finite window (same bound
            // the core target_bps_property test uses); LMO's 7-byte container
            // header is sub-0.01 BPS here.
            assert!(
                bps <= target * 1.10,
                "{name}: target {target} BPS achieved {bps:.3} exceeds ceiling+10%"
            );
            let back = codec.decode(&stream).expect("decode");
            assert_eq!(back.len(), n_ch, "{name}: channel count preserved");
            assert_eq!(back[0].len(), t, "{name}: sample count preserved");
        }
    }
}

/// ADR 0054 lever 2: LMO `TargetBps` runs both the 5/3 floor and the 9/7 ratio
/// attack and keeps the lower-PRD one. The auto-pick guarantee: the LMO stream's
/// reconstruction is **never meaningfully worse** than the raw 5/3 PCRD floor at
/// the matched rate ceiling — and `decode_any` round-trips whichever transform
/// (incl. a `transform_id=1` / 9/7 body) was chosen.
#[test]
fn lmo_target_bps_auto_pick_never_worse_than_5_3_floor() {
    fn prd(orig: &[Vec<i64>], recon: &[Vec<i64>]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (o, r) in orig.iter().zip(recon) {
            let m = o.iter().sum::<i64>() as f64 / o.len().max(1) as f64;
            for (a, b) in o.iter().zip(r) {
                let e = (*a - *b) as f64;
                num += e * e;
                den += (*a as f64 - m).powi(2);
            }
        }
        if den == 0.0 {
            0.0
        } else {
            100.0 * (num / den).sqrt()
        }
    }

    let sig = make_signal(8, 2560, 99);
    for target in [3.0f64, 2.0, 1.5] {
        // Raw 5/3 floor (the LML codec's TargetBps path).
        let floor = LmlCodec.encode(&sig, Mode::TargetBps(target)).unwrap();
        let floor_prd = prd(&sig, &LmlCodec.decode(&floor).unwrap());

        // LMO auto-pick. decode_any must round-trip whichever transform won.
        let lmo = LmoCodec.encode(&sig, Mode::TargetBps(target)).unwrap();
        let lmo_recon = decode_any(&lmo).unwrap();
        assert_eq!(lmo_recon.len(), sig.len());
        let lmo_prd = prd(&sig, &lmo_recon);

        assert!(
            lmo_prd <= floor_prd * 1.001,
            "target {target}: LMO auto-pick PRD {lmo_prd:.3} worse than 5/3 floor {floor_prd:.3}"
        );
    }
}

/// ADR 0054 Lever C: on correlated multichannel data the LMO `Lossless` auto-pick
/// selects the Optimum-lossless body (`transform_id=2`, cross-channel spatial
/// prediction), is bit-exact, and is no larger than the 5/3 floor.
#[test]
fn lmo_lossless_picks_crosschan_on_correlated_and_is_bit_exact() {
    // Channels = gain·(shared base) + small per-channel detail ⇒ real cross-channel
    // redundancy (unlike make_signal's phase-decorrelated channels).
    let t = 4096usize;
    let base: Vec<i64> = (0..t)
        .map(|i| ((i as f64 * 0.05).sin() * 3000.0) as i64)
        .collect();
    let sig: Vec<Vec<i64>> = (0..8)
        .map(|c| {
            let g = 0.6 + 0.1 * c as f64;
            (0..t)
                .map(|i| {
                    (g * base[i] as f64) as i64 + (((i + c * 7) as f64 * 0.9).sin() * 120.0) as i64
                })
                .collect()
        })
        .collect();

    let lmo = LmoCodec.encode(&sig, Mode::Lossless).unwrap();
    assert_eq!(codec::peek_format(&lmo), Some(Format::Lmo));
    assert_eq!(
        lmo[6], 2,
        "correlated channels should select transform_id=2 (Optimum-lossless)"
    );

    assert_eq!(
        decode_any(&lmo).unwrap(),
        sig,
        "id=2 must be bit-exact lossless"
    );

    let floor = LmlCodec.encode(&sig, Mode::Lossless).unwrap();
    assert!(
        lmo.len() <= floor.len(),
        "auto-pick: LMO {} must be ≤ floor {}",
        lmo.len(),
        floor.len()
    );
}

#[test]
fn universal_dispatch_routes_both_and_core_reports_not_installed() {
    let sig = make_signal(3, 512, 5);
    let lml_stream = LmlCodec.encode(&sig, Mode::Lossless).unwrap();
    let lmo_stream = LmoCodec.encode(&sig, Mode::Lossless).unwrap();

    // The full dispatch (Desktop / LMO-installed) decodes BOTH formats.
    assert_eq!(decode_any(&lml_stream).unwrap(), sig, "dispatch: LML");
    assert_eq!(decode_any(&lmo_stream).unwrap(), sig, "dispatch: LMO");

    // The core-only dispatch (the Firmware view, no LMO linked) decodes LML but
    // reports the typed "not installed" for an LMO stream — never a mis-parse.
    assert_eq!(codec::decode(&lml_stream).unwrap(), sig, "core: LML");
    match codec::decode(&lmo_stream) {
        Err(codec::CodecError::OptimumNotInstalled) => {}
        other => panic!("core decode of LMO must be OptimumNotInstalled, got {other:?}"),
    }
}
