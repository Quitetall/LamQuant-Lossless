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
            // the core target_bps_property test uses); LMO's 6-byte container
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
