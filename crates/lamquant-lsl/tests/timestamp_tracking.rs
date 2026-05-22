//! Phase 4.y — RecordSession captures LSL timestamps from every
//! pull so the recording end-time + per-sample interval metadata
//! can flow into the `.lml` container.
//!
//! TDD contract:
//!
//!   * `first_lsl_timestamp()` returns None before any sample,
//!     Some(t) after at least one successful pull.
//!   * `last_lsl_timestamp()` mirrors first_lsl_timestamp on the
//!     first sample, advances as samples accumulate.
//!   * Both clear on `finish()` consumption (the session is moved).
//!   * Timestamps are monotonic across same-publisher samples
//!     (LSL's design: publishers tag with their local clock; the
//!     `time_correction()` machinery handles cross-host skew).

#![cfg(feature = "liblsl")]

use lamquant_lsl::inlet::{InletEncodeOpts, RecordSession};
use lamquant_lsl::Inlet;

#[test]
fn session_timestamps_track_with_pulls() {
    let unique = format!(
        "lamquant-ts-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let pub_id = unique.clone();
    let publisher = std::thread::spawn(move || {
        use lsl::Pushable;
        let info = lsl::StreamInfo::new(
            &pub_id,
            "EEG-test",
            2,
            100.0,
            lsl::ChannelFormat::Int32,
            &pub_id,
        )
        .unwrap();
        let outlet = lsl::StreamOutlet::new(&info, 0, 360).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(200));
        for i in 0..6i32 {
            outlet.push_sample(&vec![i, i * 2]).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    });

    let inlet = Inlet::resolve_by_name(&unique, 5.0).expect("resolve");
    let mut session =
        RecordSession::new(inlet, /* window */ 2, InletEncodeOpts::default()).unwrap();

    // Before any pull both timestamps are None.
    assert_eq!(session.first_lsl_timestamp(), None);
    assert_eq!(session.last_lsl_timestamp(), None);

    let captured = session.capture_timeout(6, 1.0).expect("capture");
    assert!(captured >= 1, "expected at least 1 window; got {}", captured);

    let first = session.first_lsl_timestamp().expect("first_ts set");
    let last = session.last_lsl_timestamp().expect("last_ts set");
    // Monotonic: last >= first (LSL publisher's clock is monotonic
    // per process).
    assert!(
        last >= first,
        "last_ts ({}) should be >= first_ts ({})",
        last,
        first
    );
    // Sample count vs nominal_srate sanity: 6 samples at 100 Hz
    // ⇒ spread ~ 50 ms. Allow a wide window for OS scheduler
    // jitter on the publisher's emit loop (10 ms sleeps).
    let spread = last - first;
    assert!(
        spread >= 0.0 && spread < 5.0,
        "timestamp spread ({}) should be a few hundred ms, got {} s",
        spread,
        spread
    );

    publisher.join().ok();
}
