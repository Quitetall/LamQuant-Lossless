# Operational

> Long-running daemons, systemd integration, observability, cloud
> object stores. Most of this surface is feature-gated behind
> `--features async` (and `--features s3` for cloud).

If you're running LamQuant on a single laptop encoding files on
demand, you can skip this bucket. It's for the corpus-ingest /
clinical-pipeline / cloud-backup case where files arrive continuously
and the codec runs as a long-lived process.

## At a glance

| Feature | Command / flag | Status | First shipped | Notes |
|---|---|---|---|---|
| Directory watch daemon | `lml watch` | shipped | v1.0 (6.4) | Bounded mpsc; drop-oldest WARN; SIGINT stop |
| systemd unit (watch) | `dist/systemd/lml-watch.service` | shipped | v1.0 (6.5) | `Type=simple`, sandboxed |
| systemd unit (lmafs auto-mount) | `installer/lmafs.service` | shipped | v1.1 (U) | Per-user template; opt-in |
| Webhook callbacks | `lml notify` | shipped | v1.0 (6.6) | op_id idempotency key, exp backoff |
| Prometheus metrics | `lml metrics` | shipped | v1.0 (8.1) | Text-exposition HTTP endpoint |
| Tracing spans per phase | (automatic) | shipped | v1.0 (8.2) | `info_span!` at encode_one / decode_one_to_raw / etc. |
| HTTP fetch | `lml fetch` | shipped | v1.0 (6.2) | rustls-tls; refuses non-http(s) |
| S3 read | `--features s3` | shipped | v1.0 (6.2) | `async_io::fetch_s3` via aws-sdk-s3 |
| S3 write | `--features s3` | shipped | v1.0 (6.3) | `async_io::put_s3` with optional If-Match |
| S3 ETag conditional PUT | `--features s3` | shipped | v1.0 (6.7) | Deterministic tiebreak |
| tokio + async wrappers | `--features async` | shipped | v1.0 (6.1) | `compress_async` / `decompress_async` via `spawn_blocking` |
| `--continue-on-error` | (batch ops) | shipped | v1.0 | Per-file failure logged, batch continues |
| `--force` | (most writers) | shipped | v1.0 | Overwrite-or-refuse semantics |

All async features require the binary be built with `--features async`.
Stock release artifacts include it; default `cargo build` does not.

## Commands

### `lml watch`

Watch a directory for new EDF/BDF files and auto-encode them to LML
in real time. Bounded mpsc channel between the filesystem watcher and
the encoder worker — on backpressure (consumer slower than producer),
the oldest queued item is dropped with a WARN log entry. Bible R33:
no unbounded queues, no silent message loss.

Stops on SIGINT. Designed to run under systemd / Docker.

Synopsis:
```
lml watch <INPUT_DIR> -o <OUTPUT_DIR> [--queue-cap N]
                                       [--noise-bits N]
                                       [--window-size N]
                                       [--sample-rate HZ]
```

Examples:
```
# Default: 256-entry queue, lossless, 2500-sample windows
lml watch /var/ingest -o /backup/lml

# Custom queue capacity for high-throughput corpora
lml watch /var/ingest -o /backup/lml --queue-cap 1024
```

Requires `--features async`.

### `lml fetch`

HTTP(S) fetch a remote file. rustls-tls only — refuses non-http(s)
schemes (no `file://`, `ftp://`, etc.).

Synopsis:
```
lml fetch <URL> -o <OUTPUT> [--max-bytes N] [--force]
```

Example:
```
lml fetch https://example.org/recording.lma -o local.lma
```

Default `--max-bytes` is 8 GiB. Bigger files require an explicit cap
to avoid an OOM footgun on accidental download of a multi-TB asset.

For S3, the binary needs `--features s3`. Pass `s3://bucket/key` as
the URL. Uses `aws-sdk-s3` (rustls).

### `lml notify`

POST a webhook callback. op_id idempotency key prevents duplicate
delivery when the network drops mid-request and the client retries.
Exponential backoff between retries (default max 3).

Synopsis:
```
lml notify <URL> --op <VERB> [--op-id <ID>]
                              [--source-path <PATH>]
                              [--output-path <PATH>]
                              [--content-sha256 <HEX>]
                              [--bytes N] [--max-retries N]
```

Example:
```
lml notify https://internal.example.org/lml-hook \
  --op encode --op-id "$(uuidgen)" \
  --source-path recording.edf \
  --output-path recording.lma \
  --content-sha256 "$(sha256sum recording.lma | cut -d' ' -f1)" \
  --bytes "$(stat -c%s recording.lma)"
```

The receiver is expected to dedupe on `op_id` — if the same id
arrives twice with the same payload, it's the same operation, not two.

### `lml metrics`

Serve Prometheus text-exposition counters on `<bind>/metrics`. Useful
for dashboards watching a long-running `lml watch` daemon.

Synopsis:
```
lml metrics [--bind <HOST:PORT>]
```

Default bind is `127.0.0.1:9100`. Hand-rolled HTTP/1.1 (no full HTTP
server dep). Counters cover: files encoded, bytes in/out, encode
latency histograms, failures by type.

```
curl http://127.0.0.1:9100/metrics
```

Scrape with Prometheus, render with Grafana. Standard practice.

## systemd integration

### `dist/systemd/lml-watch.service`

```ini
[Unit]
Description=LamQuant directory watcher
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/lml watch /var/ingest -o /var/backup/lml
Restart=on-failure
RestartSec=5s
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Sandboxing: `NoNewPrivileges`, `ProtectSystem=full`, `ProtectHome`,
`PrivateTmp`. The watch daemon does not need elevated privileges.

### `installer/lmafs.service` (per-user template)

For mounting `.lma` archives at login. Per-user unit template
`lmafs@.service` documented in
[`../INSTALL_FILE_MANAGER_INTEGRATION.md`](../INSTALL_FILE_MANAGER_INTEGRATION.md).
Briefly:

```sh
systemctl --user enable --now lmafs@recording.service
# Mounts ~/lmafs/recording.lma at ~/lmafs/recording/
```

See [OS Integration](./07-os-integration.md) for the broader mount
flow.

## Tracing spans

`tracing::info_span!` instrumentation at every codec phase:

| Span | Where |
|---|---|
| `encode_one` | `lamquant-core/src/bin/lml.rs` per-file encode |
| `decode_one_to_raw` | per-file decode |
| `cmd_archive` | `lml archive` directory pack |
| `cmd_extract` | `lml extract` directory unpack |
| `lma::pack_archive` | LMA wire write |
| `lma::unpack_archive` | LMA wire read |

Set `RUST_LOG=lamquant_core=info` to see span enter/exit. With
`-vvv` (TRACE level), every window emits a span. Plugs into
distributed-tracing backends via `tracing-opentelemetry` (caller wires
it up — not in the default build).

## Async wrappers (`--features async`)

The blocking codec runs inside `tokio::task::spawn_blocking`, so it
can be composed with other async work:

```rust
let bytes = lamquant_core::async_io::compress_async(input).await?;
let signal = lamquant_core::async_io::decompress_async(bytes).await?;
let body = lamquant_core::async_io::fetch_url("https://...").await?;
```

S3 (`--features s3`): `fetch_s3` / `put_s3`. ETag conditional PUT
prevents the "two writers, last-write-wins" footgun:

```rust
async_io::put_s3("s3://bucket/key", body, Some(expected_etag)).await?;
//                                          └── If-Match: <etag>
// AWS returns 412 if the current ETag doesn't match → caller retries
// with the new ETag (or surfaces a conflict).
```

## Batch-op semantics

| Flag | Default | Behavior |
|---|---|---|
| `--continue-on-error` | implicit | Per-file failure → log + continue; batch summary reports the count |
| `--fail-fast` | off | Per-file failure → abort the rest of the batch |
| `--force` | off | Overwrite-or-refuse on most write commands; encode/decode/export silent-overwrite for CI back-compat |
| `--skip-existing` | off | Skip files whose output already exists (resume support) |

The encoder uses an `AtomicBool` `FAIL_FAST_FLAG` consulted by the
rayon worker pool — when one worker errors and `--fail-fast` is set,
peer workers short-circuit on the next poll.

## Flags

### `lml watch`

| Flag | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output <DIR>` | path | (required) | Output `.lml` directory |
| `--queue-cap <N>` | usize | 256 | Watcher → encoder queue capacity |
| `--noise-bits <N>` | u8 | 0 | Strip N LSBs |
| `--window-size <N>` | usize | 2500 | Samples per compression window |
| `--sample-rate <HZ>` | f64 | 250.0 | Source sample rate (Hz) — for assumed `.raw` files |

### `lml fetch`

| Flag | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output <PATH>` | path | (required) | Output file |
| `--max-bytes <N>` | u64 | 8589934592 (8 GiB) | Refuse fetches larger than this |
| `--force` | bool | false | Overwrite existing output |

### `lml notify`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--op <VERB>` | string | — | Operation verb |
| `--op-id <ID>` | string | timestamp-derived | Idempotency key |
| `--source-path <PATH>` | string | "" | Source path field |
| `--output-path <PATH>` | string | "" | Output path field |
| `--content-sha256 <HEX>` | string | "" | SHA-256 hex (or "" if N/A) |
| `--bytes <N>` | u64 | 0 | Bytes count |
| `--max-retries <N>` | u32 | 3 | Max retries on transient failure |

### `lml metrics`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--bind <HOST:PORT>` | string | `127.0.0.1:9100` | Bind address |

## Error cases

| Trigger | Behavior |
|---|---|
| `lml watch` queue full | Drop-oldest WARN log entry, continue |
| `lml watch` SIGINT | Graceful shutdown; in-flight encode finishes, no new work picked up |
| `lml fetch` non-http(s) URL | Refuse |
| `lml fetch` body exceeds `--max-bytes` | Abort the download mid-stream |
| `lml notify` POST 5xx | Exponential backoff retry up to `--max-retries` |
| `lml notify` POST persistent failure | Exit nonzero after final retry; caller decides whether to alert |
| S3 ETag mismatch on conditional PUT | `412 Precondition Failed`; surface to caller |
| Async feature missing | `lml: unknown subcommand 'watch'` (compiled-out subcommands) |

## Related

- **Other buckets**:
  - [Compression](./01-compression.md) — `lml watch` is encode-on-arrival
  - [Verification](./03-verification.md) — `lml notify` payloads can include the archive SHA
  - [OS Integration](./07-os-integration.md) — `lmafs.service` per-user systemd template
  - [CLI UX](./11-cli-ux.md) — `--quiet` / `-v` / `--emit-json-events` for daemonised use
  - [Build / Release](./10-build-release.md) — `--features async` / `--features s3` lanes
- **Source files**:
  - `lamquant-core/src/bin/lml.rs:666` — `Watch`
  - `lamquant-core/src/bin/lml.rs:688` — `Fetch`
  - `lamquant-core/src/bin/lml.rs:704` — `Notify`
  - `lamquant-core/src/bin/lml.rs:656` — `Metrics`
  - `lamquant-core/src/async_io.rs` — async wrappers, S3, fetch, watch dir
  - `dist/systemd/lml-watch.service` — systemd unit template
- **Tests**:
  - `tests/integration/test_watch_daemon.py`
  - `tests/integration/test_notify_idempotent.py`
  - `tests/integration/test_metrics_endpoint.py`
  - `tests/integration/test_s3_roundtrip.py` (feature-gated)
- **Commits**:
  - Phase 6.1 — tokio + async wrappers
  - Phase 6.4 — `lml watch` daemon
  - Phase 6.5 — systemd unit
  - Phase 6.6 — webhook callbacks
  - Phase 6.7 — S3 ETag conditional PUT
  - Phase 8.1 — Prometheus metrics endpoint
  - Phase 8.2 — tracing spans per phase
- **Cross-cutting docs**:
  - [`../FEATURES.md`](../FEATURES.md) §7 (async/network), §8 (observability)
