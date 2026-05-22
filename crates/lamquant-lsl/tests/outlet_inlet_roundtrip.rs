//! Phase 3.5 integration test: outlet → in-process inlet roundtrip.
//!
//! Spawns a thread that publishes a tiny known signal via an
//! `lsl::StreamOutlet`. The main test thread subscribes via
//! `lamquant_lsl::Inlet` and pulls the same samples back. Verifies:
//!
//!   1. Stream discovery by name works on a same-host LSL network.
//!   2. Sample values round-trip int32 → int32 with bit-exact
//!      fidelity.
//!   3. Sample order matches publishing order (sequential push +
//!      sequential pull preserve LSL's FIFO semantics).
//!
//! Requires the `liblsl` Cargo feature. Without it the test
//! conditional-compiles away. Liblsl initialises a multicast UDP
//! discovery service on construction; in CI environments where
//! that's blocked the test fails gracefully via the inlet resolve
//! timeout.

#![cfg(feature = "liblsl")]

use lamquant_lsl::Inlet;

#[test]
fn outlet_to_inlet_roundtrip() {
    // Build a deterministic 3-channel × 16-sample test signal.
    let n_ch = 3;
    let n_samples = 16;
    let signal: Vec<Vec<i32>> = (0..n_samples)
        .map(|t| {
            (0..n_ch)
                .map(|ch| (t as i32) * 10 + ch as i32)
                .collect()
        })
        .collect();

    // Unique stream name so concurrent test runs don't clash.
    let unique_id = format!(
        "lamquant-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    // `lsl::StreamOutlet` isn't `Send`, so construct it inside
    // the publisher thread. The StreamInfo arguments are owned
    // primitives we move across the boundary instead.
    let publisher_id = unique_id.clone();
    let signal_owned = signal.clone();
    let publisher = std::thread::spawn(move || {
        use lsl::Pushable;
        let info = lsl::StreamInfo::new(
            &publisher_id,
            "EEG-test",
            n_ch as u32,
            100.0,
            lsl::ChannelFormat::Int32,
            &publisher_id,
        )
        .expect("StreamInfo::new");
        let outlet = lsl::StreamOutlet::new(&info, 0, 360).expect("StreamOutlet::new");
        // Give the inlet a moment to attach before publishing.
        std::thread::sleep(std::time::Duration::from_millis(150));
        for sample in &signal_owned {
            outlet.push_sample(sample).expect("push_sample");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        // Keep the outlet alive until the inlet has pulled all
        // samples. Dropping the outlet unregisters the stream
        // and would race with the inlet's final pull_sample.
        std::thread::sleep(std::time::Duration::from_secs(2));
    });

    // Resolve + connect. 5 s timeout is generous for the
    // unicast discovery on localhost.
    let inlet = Inlet::resolve_by_name(&unique_id, 5.0).expect("resolve");
    assert_eq!(inlet.channel_count(), n_ch);
    assert_eq!(inlet.nominal_srate(), 100.0);

    // Pull all the samples back.
    let mut received: Vec<Vec<i32>> = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let (sample, _ts) = inlet.pull_sample(2.0).expect("pull_sample");
        received.push(sample);
    }

    publisher.join().expect("publisher join");

    // Compare bit-exact.
    assert_eq!(received, signal, "outlet → inlet round-trip mismatch");
}
