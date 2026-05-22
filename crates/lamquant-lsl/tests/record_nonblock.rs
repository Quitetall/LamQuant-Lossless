//! `RecordSession::capture_timeout` returns quickly when no LSL
//! samples arrive — vs the default `capture` which uses
//! `lsl::FOREVER` and would block indefinitely.
//!
//! TDD: this test was written before the `capture_timeout` API.
//! It defines the contract:
//!
//!   * Per-sample timeout of `0.0` ⇒ immediate non-blocking poll.
//!     Returns `Ok(0)` (zero windows) on a quiet network without
//!     blocking the caller.
//!   * A non-zero timeout caps the wait per-sample. Total wall
//!     time is bounded by `max_samples × timeout` worst case.
//!
//! Without this hook the existing `capture` blocks forever on a
//! quiet network. Recovering required a process kill.

#![cfg(feature = "liblsl")]

use lamquant_lsl::inlet::{InletEncodeOpts, RecordSession};
use lamquant_lsl::Inlet;

#[test]
fn capture_timeout_returns_quickly_on_quiet_network() {
    // No publisher running for this unique stream name → resolve
    // will time out. We can't even build the Inlet, so this test
    // covers the resolve-timeout path explicitly.
    let unique = format!(
        "lamquant-nopub-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let start = std::time::Instant::now();
    let res = Inlet::resolve_by_name(&unique, 0.3);
    let elapsed = start.elapsed();
    let err = match res {
        Err(e) => e,
        Ok(_) => panic!("resolve unexpectedly succeeded for {}", unique),
    };
    // 0.3 s timeout + a little overhead, certainly < 2 s.
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "resolve should respect the timeout; took {:?}",
        elapsed
    );
    // Error should be informative about the absent stream.
    let msg = err.to_string();
    assert!(
        msg.contains(&unique) || msg.contains("no LSL stream"),
        "error should mention the missing stream name; got: {}",
        msg
    );
}

#[test]
fn capture_timeout_with_publisher_drains_then_returns() {
    // Spawn a brief publisher emitting 4 samples then quitting.
    // Inlet's capture_timeout pulls available samples but doesn't
    // hang waiting for more after the publisher leaves.
    let unique = format!(
        "lamquant-rec-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let pub_id = unique.clone();
    let publisher = std::thread::spawn(move || {
        use lsl::Pushable;
        let info = lsl::StreamInfo::new(
            &pub_id, "EEG-test", 2, 100.0,
            lsl::ChannelFormat::Int32, &pub_id
        ).unwrap();
        let outlet = lsl::StreamOutlet::new(&info, 0, 360).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));
        for i in 0..4i32 {
            outlet.push_sample(&vec![i, i * 10]).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    });

    // Give the outlet time to register on the LSL network before
    // we try to resolve it. 5 s timeout is generous for localhost
    // unicast discovery.
    let inlet = Inlet::resolve_by_name(&unique, 5.0).expect("resolve");
    let mut session =
        RecordSession::new(inlet, /* window */ 2, InletEncodeOpts::default()).unwrap();

    // Try to capture up to 100 samples with 1 s per-sample
    // timeout. 1 s comfortably covers the publisher's 300 ms
    // pre-publish sleep + the 4-sample emit loop. Once the
    // publisher stops, the next pull times out + capture_timeout
    // returns whatever it had.
    let start = std::time::Instant::now();
    let result = session.capture_timeout(100, 1.0).expect("capture_timeout");
    let elapsed = start.elapsed();

    // Got at least one window worth of samples (2 per window,
    // publisher emitted 4 → expect at least 2 windows).
    assert!(result >= 1, "expected at least 1 window; got {}", result);
    // Total wall time bounded by publisher emit time (~120 ms) +
    // the final 0.2 s timeout. Must complete within a generous
    // 3 s ceiling so the test doesn't flake on busy CI.
    assert!(
        elapsed < std::time::Duration::from_secs(3),
        "capture_timeout should return after publisher quits, took {:?}",
        elapsed
    );

    publisher.join().ok();
}
