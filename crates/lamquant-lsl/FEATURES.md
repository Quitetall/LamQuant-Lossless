# `lamquant-lsl` — feature inventory

World-class Lab Streaming Layer integration for LamQuant. Two-way
bridge between the lossless `.lml` codec + the de-facto neuroscience
real-time data network. Every entry below is shipped + tested.

## Status snapshot (2026-05-22)

| build | tests | result |
| ----- | ----: | ------ |
| `cargo test -p lamquant-lsl` (default) | 28 | green |
| `cargo test -p lamquant-lsl --features liblsl` | 45 | green |
| `cargo test -p lamquant-lsl --features async` | 52 | green |

## Cargo features

| feature | default | pulls in | purpose |
| ------- | :-----: | -------- | ------- |
| `liblsl` | off | `lsl` (git master), system liblsl 1.16.2 | Real LSL network I/O |
| `async` | off | `liblsl`, `tokio` (rt, time, sync) | Async wrappers + actor pattern |

## Public API

### Sync core (default + `liblsl`)

```rust
// Replay .lml as LSL outlet
use lamquant_lsl::{Outlet, Rate};
let outlet = Outlet::from_lml_with_rate(path, Some("MyStream"), Rate::RealTime)?;
outlet.push_all()?;

// Record LSL → .lml
use lamquant_lsl::Inlet;
let inlet = Inlet::resolve_by_name("BioSemi", 5.0)?;
let (sample, ts) = inlet.pull_sample(1.0)?;

// Recording session w/ codec windows
use lamquant_lsl::{RecordSession, inlet::InletEncodeOpts};
let mut session = RecordSession::new(inlet, 2500, InletEncodeOpts::default())?;
session.capture_timeout(1000, 0.2)?;    // non-blocking variant
let first = session.first_lsl_timestamp();    // Option<f64>
let last = session.last_lsl_timestamp();      // Option<f64>
let bytes = session.finish()?;
```

### Real-time pacing (default, no LSL required)

```rust
use lamquant_lsl::pacing::{Pacer, PaceRate};
let mut p = Pacer::new(256.0, PaceRate::RealTime);
loop {
    p.await_next();           // blocks until next sample due
    if p.should_emit_now() {  // non-blocking probe
        emit_sample();
    }
    p.pause();                // pause/resume for interactive replay
    p.resume();
}
```

### XDF export (default, no LSL required)

```rust
use lamquant_lsl::xdf::{write_xdf_from_lml, write_xdf_from_lml_opts,
                       write_xdf_multistream, XdfOpts};

// Simple single-stream
write_xdf_from_lml(&lml, &xdf)?;

// Rich options: timestamps + clock offsets + multi-stream
let opts = XdfOpts::default()
    .with_timestamps(true)
    .with_timestamp_anchor(1.7e9)
    .with_clock_offsets(&[(0.0, 0.001), (5.0, 0.0015)]);
write_xdf_multistream(&[lml_a, lml_b], &xdf, opts)?;
```

### Async actor (`async`)

```rust
use lamquant_lsl::{OutletActor, Rate};
let actor = OutletActor::spawn_from_lml(path, None, Rate::Burst).await?;
let pushed = actor.push_all().await?;
actor.shutdown().await?;

// Compose with tokio::join! for multi-stream daemons
let (a, b, c) = tokio::join!(
    actor_a.push_all(), actor_b.push_all(), actor_c.push_all()
);
```

### CLI binaries (`--bins`, requires `liblsl`)

```bash
# Replay an .lml as an LSL outlet
lml-stream recording.lml --rate 2.0 --name "MyAmp"

# Record an LSL stream into a .lml
lml-record --name BioSemi --out session.lml --max-samples 100000 --window 2500

# List visible LSL streams on the network
lml-discover --timeout 2.0
```

## Shipped features by phase

### Phase 1 — Foundation
- `Outlet` (sync) wrapping `lsl::StreamOutlet`
- `Rate { RealTime, Burst, Multiplier(f64) }`
- `StreamSpec` builder from EDF/LMA metadata (`metadata_lite.rs`)
- `stream_info_from_lml` populating full LSL StreamInfo with
  per-channel labels/units/types in XML
- Deterministic `stream_id_from_lml` → `lamquant:<16hex>` source_id
  derived from `signal_sha256`
- Anchored-Instant pacer inline in `push_all` (no drift)
- 18 unit tests (stream_id, metadata, channel-format)

### Phase 1.5 — Build
- Switched `lsl` dep from crates.io `0.1.1` to liblsl-rust master
  (bundles liblsl 1.16.2 — compiles on modern C++ toolchains)
- StreamInfo XML walk uses `mut` per the newer lsl crate API
- System install path documented (`paru -S liblsl`, `apt`, `brew`,
  manual cmake)

### Phase 2 — Standalone Pacer (`pacing.rs`)
- `Pacer { Real-time/Burst/Multiplier }` separate from outlet
- `await_next()` blocking + `should_emit_now()` non-blocking probe
- `pause()` / `resume()` for interactive replay UIs
- Anchored-Instant scheduling (cumulative drift bounded)
- 8 unit tests (real-time tolerance, burst, pause/resume,
  monotonic counter)

### Phase 3 — Inlet offline (`inlet.rs`)
- `SampleBuffer` for codec-window accumulation, sample-major →
  channel-major transpose on flush
- `drain_padded` for partial-window flushes (zero-padded)
- `encode_window` thin wrapper around `lml::compress_with_mode`
- 6 unit tests (validation, flush, drain, codec roundtrip)

### Phase 3.5 — Live Inlet (`inlet.rs::live`)
- `Inlet::resolve_by_name(name, timeout)` + `Inlet::from_info`
- Re-fetches full StreamInfo via `inner.info(FOREVER)` so the
  publisher's actual `channel_format` is used for dispatch (the
  discovery beacon may carry a placeholder)
- `pull_sample` dispatches on channel_format; routes int8/16/32
  to `Vec<i32>`; rejects float/double for the lossless flow
- `RecordSession` orchestrates inlet → SampleBuffer → codec
  flushes → list of encoded windows
- Outlet → Inlet roundtrip integration test (3-channel, 16-sample,
  bit-exact)

### Phase 4 — CLI (`src/bin/*.rs`, requires `liblsl`)
- `lml-stream <path.lml> [--rate x] [--burst] [--name]`
- `lml-record --name <s> --out <path> [--max-samples N] [--window N]
  [--timeout sec]`
- `lml-discover [--timeout sec]`
- Three crate-local binaries (no cycle with `lamquant-core`)
- 4 smoke tests (usage / help / discover-empty)

### Phase 4.x — Non-blocking capture
- `RecordSession::capture_timeout(max_samples, per_sample_timeout)`
- Treats liblsl's "empty sample Vec" as a clean timeout signal
- 2 integration tests (quiet network, publisher disconnect)

### Phase 4.y — Timestamp tracking
- `RecordSession::first_lsl_timestamp()` / `last_lsl_timestamp()`
- Captures per-sample LSL timestamps for downstream metadata
- 1 integration test (monotonic, bounded)

### Phase 4.z — Multi-stream daemon
- Demonstration test: 3 concurrent `Outlet`s via
  `tokio::task::spawn_blocking` on a `multi_thread` runtime
- Deadlock detection via `tokio::time::timeout` wrapper
- 1 integration test

### Phase 6 — XDF export (`xdf.rs`)
- Phase 6.a single-stream XDF writer (FileHeader + StreamHeader +
  Samples + StreamFooter chunks)
- Phase 6.b per-sample timestamps via `XdfOpts::with_timestamps`
- Phase 6.c ClockOffset chunks (tag 4) via
  `XdfOpts::with_clock_offsets`
- Phase 6.d `write_xdf_multistream(&[lml...], xdf, opts)` for
  multi-stream files
- Deterministic `stream_id_to_u32` via SHA-256 over source_id
- Atomic write (tmp + rename)
- 8 XDF tests (basic shape + samples + timestamps + clockoffset +
  multistream + 3 unit tests)

### Phase 6.e — Async OutletActor (`outlet_actor.rs`, `async`)
- Actor pattern: dedicated OS thread owns the non-Send
  `lsl::StreamOutlet`; async caller talks via tokio mpsc/oneshot
- `OutletActor::spawn_from_lml(path, name, rate).await -> Self`
- `OutletActor::push_all().await -> Result<usize>`
- `OutletActor::shutdown().await` joins the worker thread cleanly
- Drop joins as a fallback (no leaked threads)
- 2 integration tests (single + 3-parallel via `tokio::join!`)

### Phase 7 — Polish
- ADR 0024 at `decisions/0024-lsl-integration.md`
- README per-platform install steps
- Example `examples/replay_lml.rs`
- This `FEATURES.md`

## Out of scope (deferred or wontfix)

| item | why |
| ---- | --- |
| `lsl-core` pure-Rust backend | Alpha 0.0.1, wire-format compatibility unverified vs C liblsl reference |
| `OutletAsync` via `spawn_blocking` | Impossible: `lsl::StreamOutlet` is `!Send` |
| Per-stream divergent `XdfOpts` in multi-stream | Single-opt is sufficient for current use cases |
| XDF Boundary chunks (tag 5) | Optional for receivers; future work |
| Cross-host clock-correction in inlet | LSL's `time_correction()` API not wrapped yet |
| Stream metadata sync (StreamInfo updates mid-recording) | Rare in practice; future work |
| `lml-cli` as separate crate | Avoiding it for now; CLI bins live in `lamquant-lsl` |

## Implementation notes

- Hybrid sync core + opt-in async wrappers. Sync is the
  recommended path for max real-time pacing accuracy
  (microseconds vs tokio::time::sleep's 1 ms).
- `lsl::StreamOutlet` + `lsl::StreamInlet` are NOT `Send`. The
  actor pattern (Phase 6.e) is the canonical async strategy.
- liblsl signals timeout by returning an empty sample Vec
  (not `Err(Timeout)`). All wrappers handle this correctly.
- All XDF chunks emit length prefix per the three-tier
  XDF spec (1/4/8 byte forms).
- Stream IDs: `lamquant:<16hex>` for LSL `source_id` (string)
  + SHA-256-derived u32 for XDF internal stream ID.
- Atomic file writes via tmp + rename throughout.

## How to install

```bash
# 1. System liblsl (network library)
paru -S liblsl                 # Arch / CachyOS (AUR, ~1.16.2)
sudo apt install liblsl-dev    # Debian / Ubuntu
brew install lsl               # macOS

# 2. lamquant-lsl binaries
cargo install --path crates/lamquant-lsl --features liblsl --bins
# → installs lml-stream, lml-record, lml-discover

# 3. Quick smoke
lml-discover --timeout 1.0
```
