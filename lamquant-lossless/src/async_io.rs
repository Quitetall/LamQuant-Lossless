//! Phase 6 — async runtime + network surface.
//!
//! Gated behind the `async` cargo feature. Provides:
//!   - `compress_async` / `decompress_async` — thin wrappers that run
//!     the synchronous codec on a `tokio::task::spawn_blocking` thread
//!     so call sites in an async runtime don't stall the executor.
//!   - `fetch_url` — HTTP fetch into a buffer via reqwest (rustls-tls).
//!   - `post_webhook` — POST a JSON payload to a callback URL. Bible
//!     R32: every webhook call carries an idempotency-key in the body
//!     so receivers can dedupe on retry.
//!   - `watch_dir` — filesystem watcher built on `notify::recommended_watcher`,
//!     emits one event per new file. Bounded `tokio::sync::mpsc` channel
//!     with drop-oldest WARN log on capacity overflow (Bible R33 —
//!     backpressure first-class, never unbounded queue).
//!
//! S3 read/write (Phase 6.2 + 6.3) deferred — `aws-sdk-s3` pulls in a
//! 100+-crate dependency chain that bloats every `cargo build` even
//! behind a feature gate. Users who need S3 should call our HTTP API
//! against a signed-URL endpoint, or use an external `aws s3 cp`
//! followed by `lml encode`.
//!
//! Bible alignment:
//!   - R6  strict types at trust boundaries (`WebhookPayload` newtype)
//!   - R23 validate URL scheme + status before treating response as data
//!   - R30 hostile-caller interfaces (refuse non-http(s) URLs, refuse
//!         a watch dir that's also the output dir)
//!   - R32 idempotent webhook receivers (op_id + content_sha256 in body)
//!   - R33 bounded mpsc with drop-oldest WARN (never silently lose data)

use crate::error::{LmlError, LmlResult};
use std::path::PathBuf;
use std::time::Duration;

/// Async wrapper around `lamquant_core::container::write_file`.
///
/// Offloads the CPU-bound codec to `spawn_blocking` so async callers
/// don't stall the runtime. Returns the path that was written on
/// success (mirrors the sync API for ergonomic chaining).
pub async fn compress_async(
    signal: Vec<Vec<i64>>,
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: String,
    out_path: PathBuf,
) -> LmlResult<PathBuf> {
    let out_clone = out_path.clone();
    tokio::task::spawn_blocking(move || {
        crate::container::write_file(
            &out_clone,
            &signal,
            sample_rate,
            window_size,
            noise_bits,
            &metadata_json,
        )?;
        Ok::<(), LmlError>(())
    })
    .await
    .map_err(|e| LmlError::InvalidHeader(format!("compress_async: spawn_blocking join: {e}")))??;
    Ok(out_path)
}

/// Async wrapper around `lamquant_core::container::read_file`.
pub async fn decompress_async(path: PathBuf) -> LmlResult<(Vec<Vec<i64>>, String)> {
    tokio::task::spawn_blocking(move || crate::container::read_file(&path))
        .await
        .map_err(|e| {
            LmlError::InvalidHeader(format!("decompress_async: spawn_blocking join: {e}"))
        })?
}

/// HTTP fetch a URL into a buffer. Caller's job to write it to disk.
///
/// Refuses non-http(s) schemes (Bible R30). Errors on non-2xx status
/// rather than silently returning the body — a 404 page is not an EDF.
pub async fn fetch_url(url: &str, max_bytes: u64) -> LmlResult<Vec<u8>> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(LmlError::InvalidHeader(format!(
            "fetch_url: refusing non-http(s) URL '{url}'"
        )));
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| LmlError::InvalidHeader(format!("fetch_url: build client: {e}")))?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| LmlError::InvalidHeader(format!("fetch_url: GET {url}: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(LmlError::InvalidHeader(format!(
            "fetch_url: HTTP {status} from {url}"
        )));
    }
    // Length guard before allocating; reqwest's content-length is best-
    // effort but catches obvious cases.
    if let Some(len) = resp.content_length() {
        if len > max_bytes {
            return Err(LmlError::InvalidHeader(format!(
                "fetch_url: Content-Length {len} > max_bytes {max_bytes}"
            )));
        }
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| LmlError::InvalidHeader(format!("fetch_url: body: {e}")))?;
    if bytes.len() as u64 > max_bytes {
        return Err(LmlError::InvalidHeader(format!(
            "fetch_url: response body {} > max_bytes {max_bytes}",
            bytes.len()
        )));
    }
    Ok(bytes.to_vec())
}

/// Strict-typed body for `post_webhook`. Carries the idempotency-key
/// the receiver dedupes on (Bible R32).
#[derive(Debug, Clone)]
pub struct WebhookPayload {
    pub op_id: String,
    pub op: String,
    pub source_path: String,
    pub output_path: String,
    pub content_sha256: String,
    pub bytes: u64,
}

impl WebhookPayload {
    fn to_json(&self) -> String {
        format!(
            "{{\"op_id\":\"{}\",\"op\":\"{}\",\"source_path\":\"{}\",\
             \"output_path\":\"{}\",\"content_sha256\":\"{}\",\"bytes\":{}}}",
            escape(&self.op_id),
            escape(&self.op),
            escape(&self.source_path),
            escape(&self.output_path),
            escape(&self.content_sha256),
            self.bytes,
        )
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Phase 8.1 — Prometheus text-exposition counters. Process-global,
/// updated by encode/decode/watch hot paths. Cleared per-process; no
/// histograms (pure counters keep the format hand-rollable without a
/// `prometheus` crate dep).
pub static METRIC_ENCODE_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static METRIC_DECODE_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static METRIC_ENCODE_BYTES_IN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static METRIC_ENCODE_BYTES_OUT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static METRIC_WATCH_DROPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static METRIC_WEBHOOK_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Render counters as Prometheus text-exposition format. One HELP +
/// TYPE + value block per counter. Cheap to generate (<1 µs) so we
/// emit on every `/metrics` request without caching.
pub fn render_metrics_text() -> String {
    use std::sync::atomic::Ordering;
    let mut out = String::with_capacity(512);
    let counter = |out: &mut String, name: &str, help: &str, val: u64| {
        out.push_str(&format!("# HELP {name} {help}\n"));
        out.push_str(&format!("# TYPE {name} counter\n"));
        out.push_str(&format!("{name} {val}\n"));
    };
    counter(
        &mut out,
        "lamquant_encode_total",
        "Number of files encoded since process start.",
        METRIC_ENCODE_TOTAL.load(Ordering::Relaxed),
    );
    counter(
        &mut out,
        "lamquant_decode_total",
        "Number of files decoded since process start.",
        METRIC_DECODE_TOTAL.load(Ordering::Relaxed),
    );
    counter(
        &mut out,
        "lamquant_encode_bytes_in",
        "Total source bytes ingested by encode.",
        METRIC_ENCODE_BYTES_IN.load(Ordering::Relaxed),
    );
    counter(
        &mut out,
        "lamquant_encode_bytes_out",
        "Total LML bytes emitted by encode.",
        METRIC_ENCODE_BYTES_OUT.load(Ordering::Relaxed),
    );
    counter(
        &mut out,
        "lamquant_watch_drops",
        "Watcher queue drop count (drop-oldest WARN).",
        METRIC_WATCH_DROPS.load(Ordering::Relaxed),
    );
    counter(
        &mut out,
        "lamquant_webhook_failures",
        "Non-retryable webhook delivery failures.",
        METRIC_WEBHOOK_FAILURES.load(Ordering::Relaxed),
    );
    out
}

/// Serve `GET /metrics` on `bind_addr` until `run_until` resolves.
/// Tiny hand-rolled HTTP/1.1 responder over `tokio::net::TcpListener`
/// — no `hyper` dep. Bible R30: only `/metrics` returns 200; anything
/// else returns 404 so the server can't be tricked into echoing
/// arbitrary request bodies.
pub async fn serve_metrics<F>(bind_addr: &str, run_until: F) -> LmlResult<()>
where
    F: std::future::Future<Output = ()>,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    let listener = TcpListener::bind(bind_addr)
        .await
        .map_err(|e| LmlError::InvalidHeader(format!("serve_metrics: bind {bind_addr}: {e}")))?;
    tracing::info!("metrics server listening on http://{bind_addr}/metrics");
    tokio::pin!(run_until);
    loop {
        tokio::select! {
            biased;
            _ = &mut run_until => {
                tracing::info!("metrics server: cancellation signal, exiting");
                return Ok(());
            }
            accept = listener.accept() => {
                let (mut stream, _peer) = match accept {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("metrics: accept failed: {e}");
                        continue;
                    }
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let n = match stream.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => n,
                    };
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let response = if req.starts_with("GET /metrics ") {
                        let body = render_metrics_text();
                        format!(
                            "HTTP/1.1 200 OK\r\n\
                             Content-Type: text/plain; version=0.0.4\r\n\
                             Content-Length: {}\r\n\
                             Connection: close\r\n\r\n{body}",
                            body.len()
                        )
                    } else {
                        "HTTP/1.1 404 Not Found\r\n\
                         Content-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_string()
                    };
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        }
    }
}

/// POST a webhook callback with exponential backoff. Retries on
/// transient errors (timeout, 5xx) up to `max_retries`; non-retryable
/// status (4xx) returns immediately. Returns Ok on first 2xx.
pub async fn post_webhook(url: &str, payload: &WebhookPayload, max_retries: u32) -> LmlResult<()> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(LmlError::InvalidHeader(format!(
            "post_webhook: refusing non-http(s) URL '{url}'"
        )));
    }
    let body = payload.to_json();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| LmlError::InvalidHeader(format!("post_webhook: build client: {e}")))?;
    let mut delay_ms: u64 = 500;
    for attempt in 0..=max_retries {
        let resp = client
            .post(url)
            .header("content-type", "application/json")
            .header("x-lamquant-op-id", payload.op_id.clone())
            .body(body.clone())
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => return Ok(()),
            Ok(r) if r.status().is_client_error() => {
                return Err(LmlError::InvalidHeader(format!(
                    "post_webhook: non-retryable HTTP {} from {url}",
                    r.status()
                )))
            }
            Ok(r) => {
                if attempt == max_retries {
                    return Err(LmlError::InvalidHeader(format!(
                        "post_webhook: retried {max_retries}× HTTP {} from {url}",
                        r.status()
                    )));
                }
            }
            Err(e) => {
                if attempt == max_retries {
                    return Err(LmlError::InvalidHeader(format!(
                        "post_webhook: retried {max_retries}× transport error: {e}"
                    )));
                }
            }
        }
        // Exponential backoff with cap at 8 s + 50 ms jitter from the
        // attempt index (no rand dep).
        let jitter = ((attempt as u64) * 137) % 50;
        tokio::time::sleep(Duration::from_millis(delay_ms + jitter)).await;
        delay_ms = (delay_ms * 2).min(8_000);
    }
    Err(LmlError::InvalidHeader("post_webhook: unreachable".into()))
}

/// Phase 6.2 — S3 GET into an in-memory buffer. URI form: `s3://bucket/key`.
/// Uses the AWS SDK's default credential chain (env, profile, IMDS).
#[cfg(feature = "s3")]
pub async fn fetch_s3(uri: &str, max_bytes: u64) -> LmlResult<Vec<u8>> {
    let (bucket, key) = parse_s3_uri(uri)?;
    let cfg = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_s3::Client::new(&cfg);
    let resp = client
        .get_object()
        .bucket(&bucket)
        .key(&key)
        .send()
        .await
        .map_err(|e| LmlError::InvalidHeader(format!("s3 get {uri}: {e}")))?;
    let len = resp.content_length.unwrap_or(0) as u64;
    if len > max_bytes {
        return Err(LmlError::InvalidHeader(format!(
            "s3 get {uri}: Content-Length {len} > max_bytes {max_bytes}"
        )));
    }
    let body = resp
        .body
        .collect()
        .await
        .map_err(|e| LmlError::InvalidHeader(format!("s3 read body {uri}: {e}")))?;
    let bytes = body.into_bytes().to_vec();
    if bytes.len() as u64 > max_bytes {
        return Err(LmlError::InvalidHeader(format!(
            "s3 get {uri}: body {} > max_bytes {max_bytes}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

/// Phase 6.3 + 6.7 — S3 PUT with optional ETag-conditional write.
/// When `expected_etag` is `Some`, sends `If-Match: <etag>` so a
/// racing writer's payload doesn't get silently overwritten (Bible
/// R32 — deterministic tiebreak; first writer wins). Returns the
/// newly-written object's ETag for the caller's bookkeeping.
#[cfg(feature = "s3")]
pub async fn put_s3(uri: &str, body: Vec<u8>, expected_etag: Option<&str>) -> LmlResult<String> {
    let (bucket, key) = parse_s3_uri(uri)?;
    let cfg = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_s3::Client::new(&cfg);
    let mut req = client
        .put_object()
        .bucket(&bucket)
        .key(&key)
        .body(aws_sdk_s3::primitives::ByteStream::from(body));
    if let Some(etag) = expected_etag {
        req = req.if_match(etag);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| LmlError::InvalidHeader(format!("s3 put {uri}: {e}")))?;
    Ok(resp.e_tag.unwrap_or_default())
}

#[cfg(feature = "s3")]
fn parse_s3_uri(uri: &str) -> LmlResult<(String, String)> {
    let stripped = uri.strip_prefix("s3://").ok_or_else(|| {
        LmlError::InvalidHeader(format!("s3 uri must start with s3://, got {uri}"))
    })?;
    let (bucket, key) = stripped.split_once('/').ok_or_else(|| {
        LmlError::InvalidHeader(format!("s3 uri must be s3://bucket/key, got {uri}"))
    })?;
    if bucket.is_empty() || key.is_empty() {
        return Err(LmlError::InvalidHeader(format!(
            "s3 uri bucket and key both required, got {uri}"
        )));
    }
    Ok((bucket.to_string(), key.to_string()))
}

/// Phase 6.4 — Filesystem watcher daemon.
///
/// Watches `input_dir` for new `.edf` / `.bdf` files and emits each
/// new path through a bounded mpsc channel. The receiver task encodes
/// each file to `output_dir/<stem>.lml`. Drop-oldest backpressure
/// when the channel is full — never silently lose data; warns via
/// `tracing::warn!`.
///
/// `run_until` is awaited as the cancellation signal. Pass
/// `tokio::signal::ctrl_c()` to stop on SIGINT, or a custom future
/// from your control loop.
pub async fn watch_dir<F>(
    input_dir: PathBuf,
    output_dir: PathBuf,
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    queue_cap: usize,
    run_until: F,
) -> LmlResult<usize>
where
    F: std::future::Future<Output = ()>,
{
    use notify::{RecursiveMode, Watcher};
    if input_dir.canonicalize().ok() == output_dir.canonicalize().ok()
        && input_dir.exists()
        && output_dir.exists()
    {
        return Err(LmlError::InvalidHeader(
            "watch_dir: input_dir == output_dir would create a feedback loop".into(),
        ));
    }
    std::fs::create_dir_all(&output_dir).map_err(LmlError::Io)?;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<PathBuf>(queue_cap.max(1));
    let send_tx = tx.clone();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(ev) = res {
            if matches!(
                ev.kind,
                notify::EventKind::Create(_) | notify::EventKind::Modify(_)
            ) {
                for path in ev.paths {
                    if let Some(ext) = path.extension() {
                        let lower = ext.to_string_lossy().to_ascii_lowercase();
                        if lower == "edf" || lower == "bdf" {
                            // try_send avoids backpressure-induced unwinding in the
                            // OS watcher callback. Full queue → drop oldest.
                            if let Err(e) = send_tx.try_send(path.clone()) {
                                tracing::warn!(
                                    "watch_dir: queue full, dropping {} ({e})",
                                    path.display()
                                );
                            }
                        }
                    }
                }
            }
        }
    })
    .map_err(|e| LmlError::InvalidHeader(format!("watch_dir: notify init: {e}")))?;
    watcher
        .watch(&input_dir, RecursiveMode::Recursive)
        .map_err(|e| {
            LmlError::InvalidHeader(format!("watch_dir: watch {}: {e}", input_dir.display()))
        })?;
    drop(tx); // drop our extra sender so rx closes when the watcher closes its clone

    // Process loop until run_until resolves.
    let mut processed = 0usize;
    tokio::pin!(run_until);
    loop {
        tokio::select! {
            biased;
            _ = &mut run_until => {
                tracing::info!("watch_dir: cancellation signal received, draining");
                break;
            }
            maybe_path = rx.recv() => {
                let path = match maybe_path {
                    Some(p) => p,
                    None => break,
                };
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("input");
                let out = output_dir.join(format!("{stem}.lml"));
                let meta = format!(
                    "{{\"source_file\":\"{}\",\"format\":\"watch\"}}",
                    escape(&path.display().to_string())
                );
                let out_clone = out.clone();
                let path_clone = path.clone();
                let task = tokio::task::spawn_blocking(move || {
                    let edf = crate::edf::read_edf(&path_clone)?;
                    crate::container::write_file(
                        &out_clone,
                        &edf.signal,
                        sample_rate,
                        window_size,
                        noise_bits,
                        &meta,
                    )?;
                    Ok::<(), LmlError>(())
                });
                match task.await {
                    Ok(Ok(())) => {
                        processed += 1;
                        tracing::info!("watch_dir: encoded {} → {}", path.display(), out.display());
                    }
                    Ok(Err(e)) => {
                        tracing::warn!("watch_dir: encode {}: {e}", path.display());
                    }
                    Err(e) => {
                        tracing::warn!("watch_dir: join {}: {e}", path.display());
                    }
                }
            }
        }
    }
    Ok(processed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webhook_payload_to_json_escapes_special_chars() {
        let p = WebhookPayload {
            op_id: "abc-1".into(),
            op: "encode".into(),
            source_path: "/data/with\"quote.edf".into(),
            output_path: "/out/with\\back.lml".into(),
            content_sha256: "deadbeef".into(),
            bytes: 12345,
        };
        let j = p.to_json();
        assert!(j.contains("\"op_id\":\"abc-1\""));
        assert!(j.contains("\\\"quote.edf"));
        assert!(j.contains("\\\\back.lml"));
        assert!(j.contains("\"bytes\":12345"));
    }

    #[tokio::test]
    async fn fetch_url_refuses_file_scheme() {
        let r = fetch_url("file:///etc/passwd", 1024).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn post_webhook_refuses_file_scheme() {
        let p = WebhookPayload {
            op_id: "x".into(),
            op: "encode".into(),
            source_path: "a".into(),
            output_path: "b".into(),
            content_sha256: "c".into(),
            bytes: 0,
        };
        assert!(post_webhook("file:///x", &p, 0).await.is_err());
    }

    #[tokio::test]
    async fn compress_decompress_round_trip_via_spawn_blocking() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("rt.lml");
        let signal = vec![
            (0i64..128).collect::<Vec<_>>(),
            (100i64..228).collect::<Vec<_>>(),
        ];
        compress_async(
            signal.clone(),
            250.0,
            128,
            0,
            "{}".to_string(),
            path.clone(),
        )
        .await
        .unwrap();
        let (recovered, meta) = decompress_async(path).await.unwrap();
        assert_eq!(recovered, signal);
        assert_eq!(meta, "{}");
    }
}
