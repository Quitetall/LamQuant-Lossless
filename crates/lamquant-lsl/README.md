# lamquant-lsl

World-class Lab Streaming Layer (LSL) integration for the LamQuant
lossless EEG codec. Two-way bridge between the codec's `.lml`
container and the de-facto neuroscience real-time data network.

  * **Outlet**: read a `.lml` archive, decode the signal, publish it
    as an LSL outlet that LabRecorder, NeuroPype, OpenViBE, OpenBCI
    GUI, MATLAB LSL toolbox, pylsl, and every other LSL consumer
    can subscribe to.

  * **Inlet**: subscribe to a live LSL stream, buffer samples into
    the codec's window-sized chunks, encode straight to `.lml` on
    disk. Live recording of clinical sessions into the LamQuant
    compressed format.

## Quick start

```toml
[dependencies]
lamquant-lsl = { path = "../crates/lamquant-lsl", features = ["liblsl"] }
```

```rust
use lamquant_lsl::{Outlet, Rate};
use std::path::Path;

let outlet = Outlet::from_lml_with_rate(
    Path::new("recording.lml"),
    Some("MySource"),
    Rate::RealTime,
)?;
let pushed = outlet.push_all()?;
println!("pushed {} samples", pushed);
```

## Features

| Feature  | Default | Purpose                                                              |
| -------- | :-----: | -------------------------------------------------------------------- |
| `liblsl` |   off   | Pull in the `lsl` Rust crate; requires the C liblsl library.        |
| `async`  |   off   | `OutletAsync` via `tokio::task::spawn_blocking`. Implies `liblsl`. |

Without `liblsl` the crate still compiles and provides the
`StreamSpec` + `stream_id` + JSON-parsing utilities. The actual
LSL I/O constructors return
`LslIntegrationError::FeatureDisabled` with an install hint.

## liblsl install

The `lsl-sys 0.1.1` crate (transitively pulled in via `--features
liblsl`) bundles a vendored liblsl source tree that **fails to
compile on modern C++ toolchains** for two reasons:

  1. Modern glibc (≥ 2.34) makes `PTHREAD_STACK_MIN` a runtime
     function call rather than a preprocessor macro; the bundled
     boost::thread's `#if PTHREAD_STACK_MIN > 0` fails to parse.
  2. Modern libstdc++ (C++17 / C++20) tightens template-resolution
     rules; the bundled boost::bimaps template fails to compile.

Workaround until lsl-sys upstream is patched: install liblsl
system-wide and the cmake build won't be needed for most use
cases. Each platform:

  * **Arch / CachyOS**: AUR package — `yay -S liblsl` (or
    `paru -S liblsl`). No `extra` repo package as of writing.
  * **Debian / Ubuntu**: `apt install liblsl-dev`.
  * **macOS**: `brew install lsl`.
  * **Windows**: build from source via cmake — see
    https://github.com/sccn/liblsl.
  * **From source** (any platform):
    ```bash
    git clone https://github.com/sccn/liblsl
    cd liblsl && mkdir build && cd build
    cmake .. -DLSL_UNIXFOLDERS=ON
    cmake --build . --config Release
    sudo cmake --install .
    sudo ldconfig    # Linux
    ```

After install, build with the feature:

```bash
cargo build -p lamquant-lsl --features liblsl
```

## Real-time pacing

Three replay modes (`Rate` enum):

  * `Rate::RealTime` — match the source's nominal sample rate.
    Default. Microsecond-accurate via `std::time::Instant`.
  * `Rate::Burst` — push as-fast-as-possible. No pacing.
  * `Rate::Multiplier(x)` — `x` × real-time playback. `2.0` = 2×
    speed, `0.5` = half speed, `0.0` = burst.

## Sync core + optional async

liblsl is sync C. Our sync core (`Outlet`) mirrors it and reaches
microsecond pacing accuracy. The opt-in `async` feature adds
`OutletAsync` wrappers via `tokio::task::spawn_blocking` —
convenient for multi-stream daemons or TUI event loops, but
not as accurate (`tokio::time::sleep` has ~1 ms granularity).

```rust
// Sync — recommended for single-stream replay at any sample rate.
let outlet = Outlet::from_lml(Path::new("x.lml"), None)?;
outlet.push_all()?;

// Async — recommended for multi-stream daemons or cancellation
// via tokio::select!.
#[cfg(feature = "async")]
{
    let outlet = OutletAsync::from_lml(Path::new("x.lml"), None).await?;
    outlet.push_all().await?;
}
```

## Stream identity

LSL identifies streams by `source_id`. We derive it deterministic-
ally from the LML container's `signal_sha256` field, prefixed with
`lamquant:`. Same file → same UID → LabRecorder dedup behaves
correctly across replay restarts.

## Channel metadata

EDF signal headers (channel labels, units, transducer types,
prefilter strings) survive the round-trip. The LSL StreamInfo's
XML description includes a `<channels>` block with one
`<channel>{label, unit, type}` entry per channel. LabRecorder's
source picker shows the real channel names ("Fp1-F7", "ECG",
etc.) out of the box, not "ch0".

## Roadmap

  * Phase 1 (shipped): outlet + metadata + stream_id + sync core
  * Phase 2 (shipped): standalone Pacer primitive
  * Phase 3 (shipped): Inlet (LSL → `.lml` live recording) —
    `inlet::{Inlet, RecordSession, SampleBuffer}`
  * Phase 4 (shipped): CLI binaries `lml-stream`, `lml-record`,
    `lml-discover` (built with `--features liblsl`)
  * Phase 5 (planned): pylsl interop tests, LabRecorder field test
  * Phase 6 (shipped): XDF interop (`xdf::write_xdf_from_lml`,
    `write_xdf_multistream` — pure Rust, no liblsl dep)
  * Phase 7 (planned): ADR 0024 + production polish

  See `FEATURES.md` for the full per-module shipped-feature
  inventory (the authoritative status snapshot).

## License

GPL-3.0-or-later (same as the wider LamQuant project).
