//! LamQuant ↔ Lab Streaming Layer (LSL) integration.
//!
//! World-class real-time bridge between the LamQuant lossless codec
//! and the de-facto neuroscience real-time data network. Two
//! directions:
//!
//!   * **Outlet** (`outlet.rs`) — read a `.lml` archive, decode
//!     samples, publish as an `lsl::StreamOutlet` so hospital tools
//!     (LabRecorder, NeuroPype, OpenViBE, OpenBCI GUI, MATLAB LSL
//!     toolbox, pylsl) can subscribe.
//!
//!   * **Inlet** (`inlet.rs`) — subscribe to a live LSL stream,
//!     buffer samples into the codec's window-sized chunks, encode
//!     to `.lml` on disk. Live recording of clinical sessions
//!     straight into our compressed format.
//!
//! ## Feature flags
//!
//! Two Cargo features control what's compiled in:
//!
//!   * `liblsl` (off by default) — pulls in the `lsl` Rust crate
//!     and links against the system liblsl C library. Required for
//!     actual LSL network I/O. Default-off so the crate lives
//!     cleanly in the workspace's default-members without
//!     requiring liblsl on every dev machine. When OFF, the outlet
//!     constructors return [`LslIntegrationError::FeatureDisabled`]
//!     with a clear install hint.
//!
//!   * `async` (off by default, requires `liblsl`) — adds
//!     `OutletAsync` / `InletAsync` async wrappers via
//!     `tokio::task::spawn_blocking`. The sync core is always the
//!     recommended path for max real-time pacing accuracy
//!     (`tokio::time::sleep` has ~1 ms granularity, sync
//!     `Instant::now` + spin-yield reaches microseconds).
//!
//! ## Sync core + async wrappers
//!
//! liblsl is a sync C library; its native API blocks on
//! `push_sample` / `pull_sample`. Our sync core (`Outlet`, `Inlet`)
//! mirrors that and reaches microsecond-accurate real-time pacing
//! via `std::time::Instant`-based scheduling. The optional async
//! layer is convenience for multi-stream daemons / TUI integrations,
//! not a capability gate.
//!
//! ## Stream identity + metadata
//!
//! Streams identify themselves via `source_id` derived deterministic-
//! ally from the source `.lml` file's signal SHA-256 (see
//! `stream_id.rs`). Channel labels, units, and types from the EDF
//! signal headers propagate into the LSL `StreamInfo` XML so
//! LabRecorder shows "Fp1-F7" rather than "ch0" out of the box
//! (`metadata.rs`).
//!
//! ## Real-time pacing
//!
//! `pacing.rs` schedules `push_sample` calls at the declared sample
//! rate using a monotonic clock. Replay modes:
//!   * `Rate::RealTime` — match the source's nominal sample rate
//!     exactly (default).
//!   * `Rate::Burst` — push as-fast-as-possible (sanity checks,
//!     batch processing pipelines).
//!   * `Rate::Multiplier(x)` — playback at `x` × real time.
//!
//! ## Wire format
//!
//! All LSL types map cleanly: EDF int24 → `cf_int32`,
//! synthesised i16 (ASCII ingest) → `cf_int32`, float sources →
//! `cf_float32`. Timestamps are LSL microsecond floats anchored to
//! `EdfData::start_seconds_since_unix_epoch + n / sample_rate`.

pub mod metadata_lite;
pub mod stream_id;

// Sync core APIs — only compiled when liblsl is linked. Without
// the feature, callers see [`LslIntegrationError::FeatureDisabled`]
// from the helper constructors in this module.
#[cfg(feature = "liblsl")]
pub mod metadata;
#[cfg(feature = "liblsl")]
pub mod outlet;

// `outlet_async.rs` was deleted: liblsl's `lsl::StreamOutlet` is not
// `Send`, so wrapping in `Arc<Outlet>` for `tokio::task::spawn_blocking`
// dispatch fails the trait bound. Multi-stream concurrency happens via
// `tokio::task::spawn_blocking` directly with the sync `Outlet` — each
// task owns its outlet on its own blocking worker. See
// `tests/multi_stream_async.rs` for the canonical pattern.

// Phase 3 — Inlet (LSL → .lml). The offline pieces (SampleBuffer +
// encode_window) are always-on; the actual liblsl subscription
// arrives in Phase 4 behind the `liblsl` feature.
pub mod inlet;

// Phase 2 — Pacer primitive (Outlet uses an inline anchored pacer
// for Phase 1; this module gets the full implementation later).
pub mod pacing;

mod error;

pub use error::LslIntegrationError;
pub use stream_id::stream_id_from_lml;

#[cfg(feature = "liblsl")]
pub use inlet::{Inlet, RecordSession};
#[cfg(feature = "liblsl")]
pub use metadata::stream_info_from_lml;
#[cfg(feature = "liblsl")]
pub use outlet::{Outlet, Rate};
