# Crate Classification for 8-Repo Decomposition

## Task 1: `lamquant-history`

### Purpose
Shared history.json reader/writer for all three LamQuant front-ends (Rust TUI, Tauri GUI, Python TUI). 

**Functionality:**
- Reads/writes a canonical `history.json` file using platform-specific paths (Linux XDG, macOS, Windows)
- Tracks recent input/output file paths (20 each, circular)
- Records recent operations in a rolling 50-entry log with result status (ok/error/cancelled/partial)
- Implements resume markers: `interrupted`, `last_op`, `last_input`, `last_output` for resuming interrupted workflows
- OS-level advisory locking (fcntl on Unix, LockFileEx on Windows) to prevent concurrent-writer corruption
- Merges concurrent writes: union of recent paths, append-dedup of operations
- Zero external dependencies: only serde/serde_json

### Public API
- `History` struct with methods: `load()`, `load_from(path)`, `save()`, `save_to(path)`
- Mutators: `add_input()`, `add_output()`, `record_op()`, `mark_running()`, `mark_complete()`
- `HistoryOp` and `RecentPaths` types for serialization
- `history_path()` function for platform-aware path resolution

### Why `lamquant-core` depends on it
The Rust TUI (in `lamquant-core/src/tui/`) uses `lamquant-history` to:
- Display recent file paths in the file browser
- Show operation history in the UI
- Resume interrupted encodes/decodes (Phase 6 resume panel)
The dependency is **optional**, gated by the `host` feature (firmware never reads user history).

### Feature gating in lamquant-core/Cargo.toml
Yes, both `lamquant-history` and `lamquant-ops` are in the `host` feature list. Neither appears in the base codec build. Firmware never uses them.

### Split behavior: lamquant-common vs lamquant-lossless
If `lamquant-core` splits:
- `lamquant-history` is TUI/front-end infrastructure → goes to **common** (shared across all UIs)
- The actual history.json file is user-facing state → infrastructure crate

### Could BLUT/Firmware-TUI/Vision-TUI all use it?
Yes, all three front-ends (Rust TUI, Python TUI, Tauri GUI) already use this same file. Vision-TUI and Firmware-TUI (if built) would benefit from resumable state. Pure infrastructure.

### Destination Recommendation: **LamQuant-Lossless (inside workspace)**

**Rationale:**
1. **UI parity spec defines it:** `specs/history-schema.json` is canonical; all three UIs must read/write the same file
2. **Not training/vision-specific:** history is for codec operations (encode/decode/verify)
3. **Pure infrastructure:** no codec math, no hardware
4. **Shared across Lossless + Neural:** both neural and lossless codecs write to the same history.json
5. **Belongs in Lossless workspace** alongside `lamquant-ops` (which also serves all front-ends)
6. **Zero deps:** safe to include in any workspace

**Why NOT elsewhere:**
- Not a training/ML crate → LamQuant-Neural doesn't own it
- Not a hardware crate → LamQuant-Firmware doesn't own it
- Not vision-specific → LamQuant-Vision uses it but doesn't own it
- Public library? No—it's internal UI infrastructure
- Standalone infrastructure crate? No—tightly coupled to UI parity spec

### Risk if mis-classified
**HIGH RISK if moved to Codec:** Would split history between codecs (nightmare for UX). User edits history in Tauri GUI with lossless codec, switches to neural codec in Python TUI—history is gone.
**HIGH RISK if moved to Vision:** Vision doesn't own codec operation history.
**LOW RISK here:** All front-ends converge on Lossless workspace.

---

## Task 2: `lamquant-ipc-types`

### Purpose
MCU ↔ host IPC protocol types for the firmware communication layer. NOT Tauri IPC.

**Functionality:**
- Defines `PostcardEnvelope`: a versioned wire format wrapper for MCU-host messages
- Defines `MsgKind` enum: discriminator for message type (Ping, Pong, Status, EncodedWindow, ActivityMap, Command, CommandAck, Log)
- Binary serialization via `postcard` (no-std compatible)
- Supports `defmt-format` feature for embedded logging without dragging payload bytes
- Envelope versioning strategy: `version` bumped only for framing changes, not per-message-kind additions
- Max payload: 240 bytes (fits in USB bulk-out packets after COBS framing)
- `no_std` capable (firmware side) + `host` feature for serde_json bridge (Python/Tauri side)

### Public API
- `PostcardEnvelope` struct with `empty()`, `with_payload()` constructors
- `MsgKind` enum (Ping, Pong, Status, EncodedWindow, ActivityMap, Command, CommandAck, Log)
- Constants: `MAX_PAYLOAD = 240`, `ENVELOPE_VERSION = 1`
- `EnvelopeError` for serialization failures

### Who uses it
1. **lamquant-firmware** (`comms.rs`) — directly uses `PostcardEnvelope` and `MsgKind` for all MCU-host messaging (encodes/decodes with postcard)
2. **gui/src-tauri** (serial.rs) — consumes MCU messages for:
   - Device impedance readings
   - EEG signal streaming (EncodedWindow frames)
   - Activity maps from SNN inference
   - Commands sent to MCU
3. **BLUT** (training daemon, future) — would communicate with MCU for distributed training telemetry

### Is this Tauri ↔ codec IPC?
**NO.** This is MCU ↔ host IPC (firmware communication). Tauri itself doesn't use this protocol. The GUI uses it when interfacing with the RP2350 firmware over USB/BLE. Tauri is just the HTTP desktop framework; the real protocol is `PostcardEnvelope`.

### Destination Recommendation: **LamQuant-Firmware (PUBLIC)**

**Rationale:**
1. **Firmware-owned protocol:** defines the wire contract between MCU and host
2. **Versioned for forward-compat:** the version field and `#[non_exhaustive]` design anticipates future hardware variants
3. **Multi-consumer:** firmware, GUI, BLUT all read the same enum
4. **Not codec-specific:** independent of LML/LMQ compression
5. **Hardware abstraction:** belongs in Firmware repo alongside the firmware source
6. **PUBLIC distribution:** other hardware projects could reuse this protocol
7. **Already no-std:** can be vendored into firmware builds without bloat

**Why NOT elsewhere:**
- Not a codec type → doesn't belong in Lossless
- Not training → doesn't belong in Neural
- Not Vision-specific → Vision uses it but doesn't own the protocol
- Codec doesn't interact directly → the GUI layer is the consumer

### Risk if mis-classified
**HIGH RISK if stays in core:** Core is lossless codec math. MCU protocol is orthogonal.
**HIGH RISK if goes to Codec workspace:** Would create a hard codec ↔ firmware dependency that complicates versioning.
**MEDIUM RISK if goes to Vision:** Vision consumes MCU data, but Vision doesn't define the protocol.
**LOW RISK here:** Firmware owns the protocol definition; hosts consume it.

---

## Task 3: `lamquant-ops`

### Purpose
Shared op-runner contract for all three front-ends. Generic subprocess execution + remote dispatch via SSH transport.

**Functionality (6 modules):**

1. **`lib.rs`**: `OpEvent` enum — JSON-line wire format for operation progress
   - `Started { op_id, total }`
   - `Progress { current, total, message }`
   - `FileDone { path, success, ms, cr, bytes_in/out, samples, duration_s, n_channels, sample_rate, sha256, n_windows }`
   - `Done { message }`, `Error { message }`, `Log { message }`
   - Canonical schema at `specs/op-events.schema.json`

2. **`runner.rs`**: Spawn subprocesses (`spawn_lml`, `spawn_command`, `spawn_blut`)
   - Streams stdout/stderr as `OpEvent`s
   - Bounded termination (SIGKILL → poll `try_wait()` with timeout)
   - Returns `OpHandle` for cancellation
   - Generic over `OpEventSink` trait

3. **`sink.rs`**: `OpEventSink` trait + concrete impls
   - `MpscSink` (in-process Rust TUI)
   - `OpProgressSnapshot` (Tauri GUI's shared state)
   - `bounded_channel` helper

4. **`launcher.rs`**: Op ID → command mapping table
   - Codec ops: encode, decode, verify, info, stats, archive, extract, recover, export_*
   - Eagle: eagle_quick, eagle_lqs_{l,c,m,a}, eagle_perf, eagle_rd, eagle_h2h
   - Firmware: firmware_export_weights, firmware_configure, firmware_build, firmware_flash
   - Training: train_encoder, train_snn, train_tnn, train_resume
   - Setup/diagnostics: setup_pip, test_conformance, etc.

5. **`transport.rs`**: Remote peer dispatch
   - `Peer`, `TransportKind` (SSH, future gRPC/QUIC)
   - `Transport` trait: `verify()`, `health()`, `stage_input()`, `dispatch()`, `cancel()`
   - `SshTransport` implementation (rsync + remote command execution)

6. **`op_spec.rs`**: Op metadata (name, description, input/output constraints)

### Public API
Re-exported at crate root:
- `OpEvent` and `OpState` enums
- `OpEventSink`, `MpscSink`, `OpProgressSnapshot` for sink implementations
- `launcher()`, `blut_launcher()` for op ID → command lookup
- `spawn_lml()`, `spawn_command()`, `spawn_blut()`, `OpHandle` for execution
- `Peer`, `Transport`, `SshTransport` for remote dispatch

### Why lamquant-core depends on it TWICE
- **Regular dep** in `[dependencies]` (host feature): TUI uses `lamquant_ops::OpEvent` types + `runner` for spawning operations
- **Dev-dep** in `[dev-dependencies]`: integration tests spawn real operations and parse `OpEvent`s

### Why lamquant-gui depends on it
Tauri GUI uses `lamquant-ops`:
- `OpEvent` types for progress rendering
- Custom `TauriSink` (defined in `gui/src-tauri/src/op.rs`) updates shared state + emits Tauri events
- Frontend polls `op_snapshot` every 200ms for progress
- Same op-runner contract as Rust TUI for consistency

### Is it pure infrastructure?
**YES.** `lamquant-ops` contains:
- Generic subprocess / remote-dispatch machinery (not codec-specific)
- Progress event types (language-neutral JSON schema)
- Transport abstraction (SSH transport, future gRPC)
- Op ID table (references `specs/ui-parity.md`, which defines the contract)

**NO codec logic:** never calls `lamquant_core::encode()` or decoder. It only spawns `lml` as a subprocess.
**NO codec-specific types:** `OpEvent` is generic over op_id strings, not sealed to LML/LMQ variants.

### Destination Recommendation: **LamQuant-Lossless (inside workspace) OR promoted to shared infrastructure crate**

**Primary: Stay in Lossless workspace**

**Rationale:**
1. **All three front-ends converge on one spec:** `specs/op-events.schema.json` (parity test enforces this)
2. **Not codec-specific:** infrastructure, not math. But ops are Lossless-centric today (op ID table references lossless codec ops)
3. **Shared across Neural:** neural codec will use the same op-runner contract (same `OpEvent` types)
4. **Workspace member simplicity:** lives alongside `lamquant-history` (which it's tightly coupled to in TUI use)
5. **Single source of truth:** op ID table in `specs/ui-parity.md` + `launcher.rs` must stay synchronized

**Secondary: Could be promoted to **infrastructure-only** crate (hypothetical future)**

If the op-runner were completely divorced from codec specifics (e.g., op_id table moved to a config file instead of hardcoded), it could become:
- A separate public crate (LamQuantOps)
- Vendored by BLUT, Vision, Firmware-TUI without pulling codec

But today, the op ID table ties it to the lossless codec ops. Decoupling would require:
- `launcher.rs` becoming a plugin system (registry-based lookup)
- `op_spec.rs` becoming data-driven (YAML/JSON, not enum)

**Why NOT elsewhere:**
- Not owned by Neural (lossless ops dominate the table)
- Not owned by Vision (Vision is one consumer, not the owner)
- Not owned by Firmware (firmware ops are a small slice)
- Not owned by Training (training ops are a small slice)
- Public? No—it's internal LamQuant infrastructure

### Risk if mis-classified
**HIGH RISK if moved to Vision:** Vision is a consumer; Vision doesn't own codec operations.
**HIGH RISK if moved to Codec:** If split between LML-ops and LMQ-ops, the parity test breaks (all three UIs expect one schema).
**MEDIUM RISK if promoted prematurely:** Op ID table is hardcoded; premature decoupling would lose single-source-of-truth.
**LOW RISK here:** Infrastructure crate in Lossless workspace, alongside history (which TUI consumes with ops).

---

## Summary Classification Table

| Crate | Destination | Why | Risk if mis-classified |
|---|---|---|---|
| `lamquant-history` | **LamQuant-Lossless (workspace member)** | Canonical history.json defined in specs; all three UIs read/write same file; tightly coupled to UI parity spec; pure infrastructure | HIGH: splitting history per codec breaks resume state for users |
| `lamquant-ipc-types` | **LamQuant-Firmware (PUBLIC workspace member)** | MCU↔host protocol types; firmware-owned wire contract; versioned for forward-compat; multi-consumer (firmware, GUI, BLUT); independent of codec logic; no-std capable | HIGH: moving to core creates unnecessary codec↔firmware dependency; moving to Vision mis-attributes ownership |
| `lamquant-ops` | **LamQuant-Lossless (workspace member)** or future infrastructure crate | Generic subprocess + remote-dispatch machinery; all three UIs converge on `OpEvent` schema; op ID table references lossless ops (ui-parity.md); parity test enforces single source of truth | HIGH: splitting per codec breaks parity contract; MEDIUM: promoting prematurely loses hardcoded table value |

---

## Decomposition Constraints

1. **Do NOT split `lamquant-history`** — all three UIs must read/write the same file. Create symmetric access (Lossless workspace can re-export if Neural codecs also write to it).

2. **`lamquant-ipc-types` belongs in Firmware** — but remains usable by GUI and BLUT via transitive dependency. Firmware is the protocol owner; hosts are consumers.

3. **`lamquant-ops` must stay unified** — three front-ends, one op-events schema, one parity test. If ever decomposed, it's as a data-driven registry, not code duplication.

4. **All three are infrastructure, not codec logic** — none of them will change in response to codec algorithm updates (compression ratio, bit width, etc.). They change only when UI/protocol contracts evolve.

