//! Track 2 P2: target-BPS rate-controlled lossy mode (ADR 0051).
//!
//! Contract: `compress_target_bps(signal, target_bps)` must produce a packet
//! whose size is at/under the target bits-per-sample (within a small tolerance
//! for header overhead on short inputs), decode to the right shape, and trade
//! distortion for rate monotonically (a higher BPS budget => lower PRD).
//! Unlike bounded-MAE there is no hard error bound — this minimizes distortion
//! subject to a RATE ceiling (the H.BWC WP1..WP8 competition tier).

use lamquant_lml_mcu::lml::{compress_target_bps, decompress};
use lamquant_lml_mcu::lpc::LpcMode;

fn make_signal(n_ch: usize, t: usize, seed: i64) -> Vec<Vec<i64>> {
    (0..n_ch)
        .map(|c| {
            let ph = seed.wrapping_add(c as i64 * 911);
            (0..t)
                .map(|i| {
                    let i = i as i64;
                    // smooth low-freq component + a little high-freq + spikes,
                    // so the wavelet subbands carry real, allocatable energy.
                    let lo = (((i + ph) as f64 * 0.05).sin() * 3000.0) as i64;
                    let hi = (((i + ph) as f64 * 0.9).sin() * 250.0) as i64;
                    let spike = if (i + ph) % 101 == 0 { 1200 } else { 0 };
                    lo + hi + spike
                })
                .collect()
        })
        .collect()
}

fn prd(orig: &[Vec<i64>], recon: &[Vec<i64>]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (o, r) in orig.iter().zip(recon.iter()) {
        let m = o.iter().sum::<i64>() as f64 / o.len().max(1) as f64;
        for (a, b) in o.iter().zip(r.iter()) {
            let e = (*a - *b) as f64;
            num += e * e;
            den += (*a as f64 - m) * (*a as f64 - m);
        }
    }
    if den == 0.0 {
        0.0
    } else {
        100.0 * (num / den).sqrt()
    }
}

#[test]
fn target_bps_meets_rate_ceiling() {
    let signal = make_signal(8, 2560, 99);
    let nm = 8 * 2560;
    // Above the entropy-coder floor, the rate controller must hit the ceiling.
    // Golomb-Rice has a hard ~1 bit/symbol floor on the zero-heavy quantized
    // streams, so sub-~1.5 BPS targets are not yet reachable — that's exactly
    // what ADR 0051 track-2 P3 (zero-RLE / rANS) lifts. Until then, those
    // targets get best-effort (driven to the floor), still decodable.
    const GOLOMB_FLOOR_BPS: f64 = 1.5;
    for &target in &[4.0f64, 3.0, 2.0, 1.5] {
        let bytes = compress_target_bps(&signal, target, LpcMode::default())
            .unwrap_or_else(|e| panic!("encode failed target={}: {:?}", target, e));
        let bps = bytes.len() as f64 * 8.0 / nm as f64;
        assert!(
            bps <= target * 1.10,
            "target {} BPS: achieved {:.3} BPS exceeds ceiling+10%",
            target,
            bps
        );
        let recon = decompress(&bytes).unwrap();
        assert_eq!(recon.len(), 8);
        assert_eq!(recon[0].len(), 2560);
    }
    // Sub-floor targets: best-effort, must reach at least the floor and decode.
    for &target in &[1.0f64, 0.75] {
        let bytes = compress_target_bps(&signal, target, LpcMode::default()).unwrap();
        let bps = bytes.len() as f64 * 8.0 / nm as f64;
        assert!(
            bps <= GOLOMB_FLOOR_BPS,
            "sub-floor target {}: best-effort BPS {:.3} should reach the ~{} floor",
            target,
            bps,
            GOLOMB_FLOOR_BPS
        );
        let recon = decompress(&bytes).unwrap();
        assert_eq!(recon.len(), 8);
    }
}

#[test]
fn target_bps_higher_budget_lower_distortion() {
    let signal = make_signal(8, 2560, 7);
    let prd_low = {
        let b = compress_target_bps(&signal, 1.0, LpcMode::default()).unwrap();
        prd(&signal, &decompress(&b).unwrap())
    };
    let prd_high = {
        let b = compress_target_bps(&signal, 4.0, LpcMode::default()).unwrap();
        prd(&signal, &decompress(&b).unwrap())
    };
    assert!(
        prd_high <= prd_low,
        "higher BPS budget must not increase PRD: 4.0→{:.2}% vs 1.0→{:.2}%",
        prd_high,
        prd_low
    );
}
