//! ADR 0024 Phase 6.e — async StreamOutlet via actor pattern.
//!
//! `lsl::StreamOutlet` isn't `Send`, so wrapping it in
//! `Arc<Outlet>` for `tokio::task::spawn_blocking` doesn't compile.
//! The actor pattern fixes it: a dedicated OS thread owns the
//! outlet for its lifetime; the async caller sends commands +
//! awaits responses over channels. liblsl's network I/O stays on
//! that thread; tokio coordinates from the outside.
//!
//! TDD contract:
//!
//!   * `OutletActor::spawn_from_lml(path).await` returns when
//!     the outlet is constructed + ready.
//!   * `OutletActor::push_all().await` runs to completion, returns
//!     the sample count.
//!   * Multiple actors can be `tokio::join!`ed without deadlock.
//!   * `OutletActor::shutdown().await` joins the OS thread cleanly.

#![cfg(feature = "async")]

use lamquant_lsl::OutletActor;

fn write_tiny_lml(path: &std::path::Path, n_ch: usize, t: usize) {
    use lamquant_core::container;
    use lamquant_core::lpc::LpcMode;
    let signal: Vec<Vec<i64>> = (0..n_ch)
        .map(|ch| (0..t as i64).map(|i| (i + ch as i64) % 50).collect())
        .collect();
    let meta = format!(
        r#"{{"sample_rate":100.0,"n_channels":{},"signal_sha256":"actor","channels":[{}],"phys_dim":"uV","duration_s":{}}}"#,
        n_ch,
        (0..n_ch).map(|i| format!("\"ch{}\"", i)).collect::<Vec<_>>().join(","),
        t as f64 / 100.0,
    );
    container::write_file_with_mode(path, &signal, 100.0, t, 0, &meta, LpcMode::default())
        .expect("write_file");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn actor_single_push_all() {
    let tmp = tempfile::tempdir().unwrap();
    let lml = tmp.path().join("a.lml");
    write_tiny_lml(&lml, 2, 25);

    let actor = OutletActor::spawn_from_lml(
        lml.clone(),
        Some("ActorTest".into()),
        lamquant_lsl::Rate::Burst,
    )
    .await
    .expect("spawn");
    let pushed = actor.push_all().await.expect("push_all");
    assert_eq!(pushed, 25);
    actor.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn actor_three_parallel() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.lml");
    let b = tmp.path().join("b.lml");
    let c = tmp.path().join("c.lml");
    write_tiny_lml(&a, 1, 10);
    write_tiny_lml(&b, 2, 15);
    write_tiny_lml(&c, 3, 20);

    let (act_a, act_b, act_c) = tokio::join!(
        OutletActor::spawn_from_lml(a, Some("A".into()), lamquant_lsl::Rate::Burst),
        OutletActor::spawn_from_lml(b, Some("B".into()), lamquant_lsl::Rate::Burst),
        OutletActor::spawn_from_lml(c, Some("C".into()), lamquant_lsl::Rate::Burst),
    );
    let act_a = act_a.expect("a");
    let act_b = act_b.expect("b");
    let act_c = act_c.expect("c");

    let (ra, rb, rc) = tokio::join!(
        act_a.push_all(),
        act_b.push_all(),
        act_c.push_all(),
    );
    assert_eq!(ra.unwrap(), 10);
    assert_eq!(rb.unwrap(), 15);
    assert_eq!(rc.unwrap(), 20);

    tokio::join!(
        act_a.shutdown(),
        act_b.shutdown(),
        act_c.shutdown(),
    );
}

/// Codec-independent path: build an outlet straight from channel
/// labels + sample rate (no `.lml`), then push a live chunk. This is
/// the decoded-buffer / live-stream entry point Vision uses to bridge
/// decoded EEG to LSL consumers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn actor_from_channels_push_chunk() {
    let channels = vec!["Fp1".to_string(), "F7".to_string(), "T3".to_string()];
    let actor = OutletActor::spawn_from_channels(
        channels,
        256.0,
        "DecodedStream".into(),
        "lamquant:test-decoded".into(),
    )
    .await
    .expect("spawn_from_channels");

    // 5 time-steps × 3 channels.
    let chunk: Vec<Vec<i32>> = (0..5)
        .map(|t| vec![t, t + 10, t + 20])
        .collect();
    let pushed = actor.push_chunk(chunk).await.expect("push_chunk");
    assert_eq!(pushed, 5);

    // Empty chunk is a no-op.
    let none = actor.push_chunk(Vec::new()).await.expect("empty chunk");
    assert_eq!(none, 0);

    actor.shutdown().await.expect("shutdown");
}

/// A mis-shaped chunk (wrong channel count) must return a clean
/// `Err`, NOT panic the worker thread (the underlying
/// `lsl::StreamOutlet` would `assert_eq!`-panic without the
/// Rust-side guard). After the error the actor must still be usable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn actor_push_chunk_rejects_wrong_width() {
    let channels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let actor = OutletActor::spawn_from_channels(
        channels,
        100.0,
        "WidthGuard".into(),
        "lamquant:test-width".into(),
    )
    .await
    .expect("spawn");

    // 3-channel outlet, but a 2-wide sample row.
    let bad: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5]];
    let err = actor.push_chunk(bad).await;
    assert!(err.is_err(), "mis-shaped chunk must error, not panic");

    // Actor still alive: a correctly-shaped chunk now succeeds.
    let good: Vec<Vec<i32>> = vec![vec![1, 2, 3]];
    assert_eq!(actor.push_chunk(good).await.expect("good chunk"), 1);

    actor.shutdown().await.expect("shutdown");
}
