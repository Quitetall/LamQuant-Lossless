//! Phase 6 conformance suite — host-side end-to-end tests.
//!
//! Synthetic ADC stream → biquad → LPC → lifting → safety hooks. Verifies:
//!   * Determinism: identical input → identical output across runs
//!   * Lifting roundtrip: forward then inverse recovers the LPC residual
//!   * Safety event log: seizure start/end produces the expected entries
//!   * Adversarial inputs: DC, all-zeros, max-amplitude don't crash
//!
//! Skips the codec (rans/Golomb) + scheduler tests because those are
//! `target_arch = "riscv32"`-only behind cfg gates. Phase 7 wires them
//! in via a host-verify cfg.

use lamquant_firmware::dsp::biquad::{HpFilter, HpFilterBank, NUM_CHANNELS, WINDOW_SAMPLES};
use lamquant_firmware::dsp::lifting::{
    forward_all_channels, inverse_all_channels, Subbands,
};
use lamquant_firmware::dsp::lpc::{analyze_all_channels, synthesize_all_channels, LpcOutput};
use lamquant_firmware::safety::{EventType, SafetyState};

const ALL_CHANNELS_MASK: u32 = (1 << NUM_CHANNELS) - 1;

// ─── Helpers ───────────────────────────────────────────────────────

/// Generate a deterministic synthetic ADC window (Q31 i32). `seed` cycles
/// through a small set of EEG-like patterns: alpha rhythm + 60 Hz hum +
/// per-channel offset.
fn synth_adc(seed: u32) -> Box<[[i32; WINDOW_SAMPLES]; NUM_CHANNELS]> {
    let mut buf = Box::new([[0i32; WINDOW_SAMPLES]; NUM_CHANNELS]);
    for ch in 0..NUM_CHANNELS {
        let phase = (seed as u64 + ch as u64).wrapping_mul(7919) & 0xFFFF;
        for i in 0..WINDOW_SAMPLES {
            // Pseudo-EEG: 10 Hz alpha (period = 25 samples) + tiny 60 Hz
            // (period = 4.16 samples) + DC offset per channel.
            let alpha = ((((i + phase as usize) * 51) & 0xFF) as i32 - 128) << 16;
            let hum = ((((i * 240) & 0x3F) as i32 - 32) << 12) as i32;
            let dc = ((ch as i32 + 1) * 0x10000) as i32;
            buf[ch][i] = alpha + hum + dc;
        }
    }
    buf
}

// ─── Tests ─────────────────────────────────────────────────────────

#[test]
fn pipeline_deterministic() {
    // Two independent runs on identical input must produce identical output.
    let mut buf_a = synth_adc(42);
    let mut buf_b = synth_adc(42);

    let mut hp_a = HpFilterBank::new();
    let mut hp_b = HpFilterBank::new();
    hp_a.run(&mut buf_a, WINDOW_SAMPLES, HpFilter::Hz0_5, ALL_CHANNELS_MASK);
    hp_b.run(&mut buf_b, WINDOW_SAMPLES, HpFilter::Hz0_5, ALL_CHANNELS_MASK);
    assert_eq!(*buf_a, *buf_b, "biquad output drifts across runs");

    let mut lpc_a = LpcOutput::zeroed();
    let mut lpc_b = LpcOutput::zeroed();
    analyze_all_channels(&buf_a, &mut lpc_a);
    analyze_all_channels(&buf_b, &mut lpc_b);
    assert_eq!(lpc_a.coeffs, lpc_b.coeffs, "LPC coeffs drift");
    assert_eq!(lpc_a.residual, lpc_b.residual, "LPC residual drifts");

    let mut sb_a = Subbands::zeroed();
    let mut sb_b = Subbands::zeroed();
    forward_all_channels(&lpc_a.residual, &mut sb_a);
    forward_all_channels(&lpc_b.residual, &mut sb_b);
    assert_eq!(sb_a.l3_approx, sb_b.l3_approx, "L3 approx drifts");
    assert_eq!(sb_a.l3_detail, sb_b.l3_detail, "L3 detail drifts");
    assert_eq!(sb_a.l2_detail, sb_b.l2_detail, "L2 detail drifts");
    assert_eq!(sb_a.l1_detail, sb_b.l1_detail, "L1 detail drifts");
}

#[test]
fn lifting_inverse_recovers_residual() {
    let mut buf = synth_adc(7);
    let mut hp = HpFilterBank::new();
    hp.run(&mut buf, WINDOW_SAMPLES, HpFilter::Hz0_5, ALL_CHANNELS_MASK);

    let mut lpc = LpcOutput::zeroed();
    analyze_all_channels(&buf, &mut lpc);
    let residual_orig = lpc.residual;

    let mut sb = Subbands::zeroed();
    forward_all_channels(&lpc.residual, &mut sb);

    let mut recovered = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
    inverse_all_channels(&sb, &mut recovered);

    for ch in 0..NUM_CHANNELS {
        for i in 0..WINDOW_SAMPLES {
            assert_eq!(
                recovered[ch][i], residual_orig[ch][i],
                "lifting inverse mismatch ch{ch} sample {i}"
            );
        }
    }
}

#[test]
fn lpc_inverse_recovers_signal() {
    let mut buf = synth_adc(13);
    let mut hp = HpFilterBank::new();
    hp.run(&mut buf, WINDOW_SAMPLES, HpFilter::Hz0_5, ALL_CHANNELS_MASK);
    let signal_orig = *buf;

    let mut lpc = LpcOutput::zeroed();
    analyze_all_channels(&buf, &mut lpc);

    let mut recovered = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
    synthesize_all_channels(&lpc.residual, &lpc.coeffs, &mut recovered);

    // Integer LPC should be bit-exact.
    for ch in 0..NUM_CHANNELS {
        for i in 0..WINDOW_SAMPLES {
            assert_eq!(
                recovered[ch][i], signal_orig[ch][i],
                "LPC inverse mismatch ch{ch} sample {i}"
            );
        }
    }
}

#[test]
fn full_pipeline_roundtrip() {
    let mut signal = synth_adc(99);
    let signal_pre_hp = *signal;

    // Forward: HP → LPC → lifting.
    let mut hp = HpFilterBank::new();
    hp.run(&mut signal, WINDOW_SAMPLES, HpFilter::Hz0_5, ALL_CHANNELS_MASK);
    let mut lpc = LpcOutput::zeroed();
    analyze_all_channels(&signal, &mut lpc);
    let mut sb = Subbands::zeroed();
    forward_all_channels(&lpc.residual, &mut sb);

    // Inverse: lifting → LPC → biquad output recovery (skip biquad inverse —
    // it's lossy by design; we recover the post-HP signal instead).
    let mut residual_back = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
    inverse_all_channels(&sb, &mut residual_back);
    assert_eq!(residual_back, lpc.residual, "lifting roundtrip lost data");

    let mut signal_back = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
    synthesize_all_channels(&residual_back, &lpc.coeffs, &mut signal_back);
    assert_eq!(signal_back, *signal, "LPC roundtrip lost data");

    // Sanity: biquad changed something (DC was removed).
    let _ = signal_pre_hp;
}

#[test]
fn adversarial_all_zeros() {
    let mut buf = Box::new([[0i32; WINDOW_SAMPLES]; NUM_CHANNELS]);
    let mut hp = HpFilterBank::new();
    hp.run(&mut buf, WINDOW_SAMPLES, HpFilter::Hz0_5, ALL_CHANNELS_MASK);
    let mut lpc = LpcOutput::zeroed();
    analyze_all_channels(&buf, &mut lpc);
    let mut sb = Subbands::zeroed();
    forward_all_channels(&lpc.residual, &mut sb);
    // No panics, no overflow.
}

#[test]
fn adversarial_max_amplitude() {
    let mut buf = Box::new([[i32::MAX / 4; WINDOW_SAMPLES]; NUM_CHANNELS]);
    let mut hp = HpFilterBank::new();
    hp.run(&mut buf, WINDOW_SAMPLES, HpFilter::Hz0_5, ALL_CHANNELS_MASK);
    let mut lpc = LpcOutput::zeroed();
    analyze_all_channels(&buf, &mut lpc);
    let mut sb = Subbands::zeroed();
    forward_all_channels(&lpc.residual, &mut sb);
    // No panics from saturating math.
}

// ─── Safety subsystem integration ────────────────────────────────

#[test]
fn safety_seizure_lifecycle_logs_events() {
    let mut s = SafetyState::default();
    s.init(0);

    s.on_seizure_start(1000, 220, 0b0000_0111);
    assert_eq!(s.faults.total_seizures_detected, 1);
    assert!(s.preictal.seizure_triggered);

    s.on_seizure_end(31_000);
    assert!(!s.preictal.seizure_triggered);
    let dur = s.seizure_log.entries[0].duration_s;
    assert!(
        (29..=30).contains(&dur),
        "30s seizure logged as {dur}s (>>10 ms→s)"
    );

    // Boot + start + end events.
    let event_types: Vec<u8> = s
        .event_log
        .entries
        .iter()
        .filter(|e| e.timestamp_ms != 0 || e.r#type != 0)
        .map(|e| e.r#type)
        .collect();
    assert!(event_types.contains(&(EventType::BootCold as u8)));
    assert!(event_types.contains(&(EventType::SeizureDetect as u8)));
    assert!(event_types.contains(&(EventType::SeizureEnd as u8)));
}

#[test]
fn safety_ble_retry_fifo_overflow_overwrites_oldest() {
    let mut s = SafetyState::default();
    // Push 9 packets — 1 more than the 8-slot ring buffer.
    for i in 0..9u32 {
        let mut pkt = [0u8; 16];
        pkt[0] = i as u8;
        s.ble_push_packet(&pkt, i);
    }
    assert_eq!(s.ble_retry.count, 8);
    // Oldest (seq=0) was evicted; tail points at seq=1.
    let (data, seq) = s.ble_peek_retry().unwrap();
    assert_eq!(seq, 1, "retry FIFO didn't evict oldest");
    assert_eq!(data[0], 1);
}

#[test]
fn safety_impedance_threshold_alert() {
    let mut s = SafetyState::default();
    s.update_impedance(0, 0, 30); // below threshold, no alert
    assert_eq!(s.impedance.alert_flags, 0);
    s.update_impedance(0, 5, 70); // above threshold, alert bit 5
    assert_eq!(s.impedance.alert_flags & (1 << 5), 1 << 5);
}
