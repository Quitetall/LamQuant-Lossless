//! I/O abstractions for the host side of the LML codec.
//!
//! Why: every encoder/decoder/container entry point used to take `&Path`
//! or `&[u8]`. That blocks stdin/stdout piping, S3/HTTP sources, future
//! async, and the upcoming daemon mode. This module introduces the
//! traits + locator enum that those features sit on; the existing
//! Path-based entry points become 3-line wrappers in subsequent commits.
//!
//! Bible alignment:
//!   - R6  Strict types at the boundary — `Locator` is an enum, not a
//!     `String`; misuse fails at compile time.
//!   - R26 Trait bounds catch misuse at compile time, not runtime.
//!   - R30 Hostile-caller interface — every fallible op returns
//!     `LmlResult<T>`; unsupported schemes return an explicit "wait for
//!     Phase 6" error rather than silently mis-routing.
//!   - R33 Backpressure first-class — `LmlSink::before_window` lets slow
//!     downstream block the encoder before another packet allocates.

use std::io::{self, BufReader, BufWriter, IsTerminal, Read, Seek, Write};
use std::path::{Path, PathBuf};

use crate::error::{LmlError, LmlResult};

/// Source of LML bytes (or pre-LML signal samples on the encode side).
///
/// Blanket-implemented for every `R: Read`, so any `BufReader<File>`,
/// `Cursor<Vec<u8>>`, `Stdin`, or future async-adapted reader plugs in
/// with no boilerplate.
///
/// **Tradeoff** — the blanket impl always returns `None` for both
/// `len_hint` and `try_seek`. The codec hot path takes generic
/// `R: Read + Seek` directly when random access is required and
/// monomorphises against the concrete source type, so the trait-method
/// defaults are intentional dead ends. If a future phase needs the
/// trait-object path to expose seeking, the migration is to remove the
/// blanket impl and write per-source impls — purely additive change
/// (see plan R6 risk note).
///
/// Capability matrix: file = `Read + Seek + len_hint`; stdin = `Read`
/// only; S3 with HEAD = `Read + len_hint` (no Seek).
pub trait LmlSource: Read {
    /// Total length if cheaply known. `None` for pure streams (stdin,
    /// network pipe). The encoder uses it to pre-size buffers.
    fn len_hint(&self) -> Option<u64> {
        None
    }

    /// Returns a `Seek` handle if random access is supported, else
    /// `None`. Used by the seek-table reader (Phase 0.7) to short-circuit
    /// scanning the window-length index.
    fn try_seek(&mut self) -> Option<&mut dyn Seek> {
        None
    }
}

impl<R: Read> LmlSource for R {}

/// Sink for LML bytes (compressed output or decoded signal samples on
/// the decode side).
///
/// Blanket-implemented for every `W: Write`. The `before_window` hook is
/// the backpressure attachment point: a slow sink (S3 multipart upload,
/// daemon mailbox, network pipe) can block here before the encoder
/// allocates the next packet. The default impl is a no-op so existing
/// sinks (`BufWriter<File>`, `Vec<u8>`) keep their zero-cost behaviour.
pub trait LmlSink: Write {
    /// Hint to the sink that a packet of approximately `approx_bytes` is
    /// about to be written. Returning `Err` aborts the encode cleanly
    /// (the partial output file is removed by atomic rename, never
    /// committed). Default: no-op.
    fn before_window(&mut self, _approx_bytes: usize) -> io::Result<()> {
        Ok(())
    }
}

impl<W: Write> LmlSink for W {}

/// Where bytes come from / go to. The CLI parses argv `-` as `Stdio`,
/// `scheme://...` as `Url`, everything else as `Path`.
///
/// Bible R6: strict-typed, so a misrouted CLI arg is a compile-time
/// shape error in the dispatcher, not a runtime "path doesn't exist".
#[derive(Debug, Clone)]
pub enum Locator {
    /// File path on local filesystem.
    Path(PathBuf),
    /// stdin (when used as source) or stdout (when used as sink).
    Stdio,
    /// Remote URL. `s3://`, `http(s)://` land behind `feature = "async"`
    /// in Phase 6; other schemes return an explicit error rather than
    /// silently failing.
    Url(String),
}

impl Locator {
    /// Parse a CLI argument into a Locator.
    ///
    /// Routing rules:
    ///   - `-` → `Stdio`
    ///   - `<scheme>://...` where scheme is RFC 3986 valid → `Url`
    ///   - everything else → `Path`
    pub fn parse(arg: &str) -> Self {
        // Empty string is a clap edge case (e.g. `lml encode ""`). Route
        // to an explicit empty `PathBuf` so the caller can refuse with
        // a clean message at open time, rather than silently treating
        // it as cwd or surfacing a confusing "is a directory" later.
        if arg.is_empty() {
            return Self::Path(PathBuf::new());
        }
        if arg == "-" {
            return Self::Stdio;
        }
        if let Some(colon_slash) = arg.find("://") {
            let scheme = &arg[..colon_slash];
            // RFC 3986 scheme: ALPHA *(ALPHA / DIGIT / "+" / "-" / ".")
            let valid = !scheme.is_empty()
                && scheme.bytes().enumerate().all(|(i, b)| {
                    if i == 0 {
                        b.is_ascii_alphabetic()
                    } else {
                        b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.')
                    }
                });
            if valid {
                return Self::Url(arg.to_string());
            }
            // Malformed scheme — fall through and treat as Path so the
            // user gets a "no such file" error instead of a confusing
            // "unsupported URL scheme".
        }
        Self::Path(PathBuf::from(arg))
    }

    /// Open this locator as a Read source.
    ///
    /// Returns `Box<dyn LmlSource + Send>` so the CLI dispatch layer can
    /// hold heterogeneous sources in one Vec. The hot path inside the
    /// codec uses monomorphised generic `R: Read` parameters to bypass
    /// the virtual call.
    pub fn open_source(&self) -> LmlResult<Box<dyn LmlSource + Send>> {
        match self {
            Self::Path(p) => {
                let f = std::fs::File::open(p)?;
                Ok(Box::new(BufReader::new(f)))
            }
            Self::Stdio => Ok(Box::new(io::stdin())),
            Self::Url(u) => Err(LmlError::InvalidHeader(format!(
                "Locator::Url not yet supported: {u}. Enable feature \
                 \"async\" for s3:// / http(s):// (Phase 6)."
            ))),
        }
    }

    /// Open this locator as a Write sink. Creates parent directories on
    /// `Path` variants so callers don't have to. See `open_source` for
    /// dispatch rationale.
    pub fn open_sink(&self) -> LmlResult<Box<dyn LmlSink + Send>> {
        match self {
            Self::Path(p) => {
                // `Path::new("foo.lml").parent()` returns `Some("")`,
                // and `Path::new("/foo.lml").parent()` returns `Some("/")`.
                // Skip the empty case (cwd, no need to create) and let
                // the OS reject the root case as already-exists.
                if let Some(parent) = p.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent)?;
                    }
                }
                let f = std::fs::File::create(p)?;
                Ok(Box::new(BufWriter::new(f)))
            }
            Self::Stdio => Ok(Box::new(io::stdout())),
            Self::Url(u) => Err(LmlError::InvalidHeader(format!(
                "Locator::Url not yet supported: {u}. Enable feature \
                 \"async\" for s3:// / http(s):// (Phase 6)."
            ))),
        }
    }

    /// Returns true if this locator refers to a terminal device (for
    /// stdio) or `false` otherwise. Used by the CLI in Phase 1.5 to
    /// refuse binary-to-tty without `--force-binary-tty`.
    pub fn is_tty_sink(&self) -> bool {
        match self {
            Self::Stdio => io::stdout().is_terminal(),
            Self::Path(_) | Self::Url(_) => false,
        }
    }

    /// As above for the source side.
    pub fn is_tty_source(&self) -> bool {
        match self {
            Self::Stdio => io::stdin().is_terminal(),
            Self::Path(_) | Self::Url(_) => false,
        }
    }

    /// Path accessor — `None` for non-`Path` variants. Used by call
    /// sites that need a path for filesystem-only operations (mtime
    /// preservation, atomic rename) and fall back to a deterministic
    /// behaviour for stdio/url.
    pub fn as_path(&self) -> Option<&Path> {
        match self {
            Self::Path(p) => Some(p),
            _ => None,
        }
    }
}

// `pub(crate)` so other test modules in this crate (e.g. lml::tests)
// can reuse the `ByteAtATime` partial-read adapter.
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    /// Reader that yields one byte per `read()` call. Forces caller
    /// loops — `BufReader<File>` and `Cursor<Vec<u8>>` both satisfy in
    /// one go, so a real partial-read test bed needs this adapter.
    pub(crate) struct ByteAtATime<'a> {
        src: &'a [u8],
    }

    impl<'a> ByteAtATime<'a> {
        pub fn new(src: &'a [u8]) -> Self {
            Self { src }
        }
    }

    impl<'a> Read for ByteAtATime<'a> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.src.is_empty() || buf.is_empty() {
                return Ok(0);
            }
            buf[0] = self.src[0];
            self.src = &self.src[1..];
            Ok(1)
        }
    }

    #[test]
    fn cursor_satisfies_lmlsource_blanket() {
        let data = vec![1u8, 2, 3, 4, 5];
        let mut src = Cursor::new(&data);
        let mut buf = [0u8; 5];
        fn take_source<S: LmlSource>(s: &mut S, buf: &mut [u8]) -> usize {
            s.read(buf).unwrap()
        }
        let n = take_source(&mut src, &mut buf);
        assert_eq!(n, 5);
        assert_eq!(buf, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn vec_satisfies_lmlsink_blanket() {
        let mut sink: Vec<u8> = Vec::new();
        fn take_sink<S: LmlSink>(s: &mut S, bytes: &[u8]) {
            s.before_window(bytes.len()).unwrap();
            s.write_all(bytes).unwrap();
        }
        take_sink(&mut sink, b"hello");
        assert_eq!(sink, b"hello");
    }

    #[test]
    fn partial_read_adapter_yields_one_byte_at_a_time() {
        let data = vec![0xAAu8, 0xBB, 0xCC];
        let mut src = ByteAtATime::new(&data);
        let mut buf = [0u8; 3];
        let n = src.read(&mut buf).unwrap();
        assert_eq!(n, 1, "ByteAtATime must yield exactly one byte per read()");
        assert_eq!(buf[0], 0xAA);
        // Caller must loop — second call yields next byte.
        let n2 = src.read(&mut buf).unwrap();
        assert_eq!(n2, 1);
        assert_eq!(buf[0], 0xBB);
    }

    #[test]
    fn locator_parse_path() {
        match Locator::parse("/tmp/foo.lml") {
            Locator::Path(p) => assert_eq!(p, PathBuf::from("/tmp/foo.lml")),
            other => panic!("expected Path, got {other:?}"),
        }
    }

    #[test]
    fn locator_parse_relative_path() {
        match Locator::parse("foo.lml") {
            Locator::Path(p) => assert_eq!(p, PathBuf::from("foo.lml")),
            other => panic!("expected Path, got {other:?}"),
        }
    }

    #[test]
    fn locator_parse_stdio_dash() {
        assert!(matches!(Locator::parse("-"), Locator::Stdio));
    }

    #[test]
    fn locator_parse_s3_url() {
        match Locator::parse("s3://bucket/key.lml") {
            Locator::Url(u) => assert_eq!(u, "s3://bucket/key.lml"),
            other => panic!("expected Url, got {other:?}"),
        }
    }

    #[test]
    fn locator_parse_http_url() {
        assert!(matches!(
            Locator::parse("https://example.com/data.lml"),
            Locator::Url(_)
        ));
    }

    #[test]
    fn locator_parse_file_url() {
        assert!(matches!(
            Locator::parse("file:///tmp/x.lml"),
            Locator::Url(_)
        ));
    }

    #[test]
    fn locator_parse_empty_string_routes_to_empty_path() {
        // Empty clap arg must NOT silently become Path("") which would
        // open the cwd as a file later. It routes to an explicit empty
        // PathBuf so the caller can refuse with a clean message.
        match Locator::parse("") {
            Locator::Path(p) => assert_eq!(p, PathBuf::new()),
            other => panic!("expected empty Path, got {other:?}"),
        }
    }

    #[test]
    fn locator_parse_malformed_scheme_falls_to_path() {
        // Scheme can't start with digit per RFC 3986; treat as a weird
        // path so the user gets "no such file" not "unsupported URL".
        match Locator::parse("123://foo") {
            Locator::Path(_) => {}
            other => panic!("expected Path fallback, got {other:?}"),
        }
    }

    #[test]
    fn locator_url_open_source_errors_with_guidance() {
        let loc = Locator::Url("s3://bucket/key".into());
        // `Box<dyn LmlSource>` isn't `Debug`, so `unwrap_err` won't
        // compile; match the Result instead.
        let msg = match loc.open_source() {
            Err(e) => format!("{e}"),
            Ok(_) => panic!("Url variant must error until Phase 6 lands"),
        };
        assert!(
            msg.contains("not yet supported") && msg.contains("Phase 6"),
            "error message must guide user to the eventual feature, got: {msg}"
        );
    }

    #[test]
    fn locator_path_open_source_reads_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello").unwrap();
        let loc = Locator::Path(tmp.path().to_path_buf());
        let mut src = loc.open_source().unwrap();
        let mut buf = Vec::new();
        src.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"hello");
    }

    #[test]
    fn locator_path_open_sink_writes_and_creates_parents() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let nested = tmp_dir.path().join("a").join("b").join("out.lml");
        let loc = Locator::Path(nested.clone());
        {
            let mut sink = loc.open_sink().unwrap();
            sink.write_all(b"payload").unwrap();
        } // drop flushes BufWriter
        let read_back = std::fs::read(&nested).unwrap();
        assert_eq!(read_back, b"payload");
    }

    #[test]
    fn locator_as_path() {
        let p = PathBuf::from("/tmp/x.lml");
        assert_eq!(Locator::Path(p.clone()).as_path(), Some(p.as_path()));
        assert_eq!(Locator::Stdio.as_path(), None);
        assert_eq!(
            Locator::Url("s3://x/y".into()).as_path(),
            None,
            "Url variant must not pretend to have a filesystem path"
        );
    }

    /// Verify the trait-object Send bound holds for every concrete
    /// source/sink we hand out. Compile-time check; if a future
    /// implementor accidentally drops Send, this fails to build.
    #[test]
    fn boxed_source_and_sink_are_send() {
        fn requires_send<T: Send>(_: T) {}
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"x").unwrap();
        let loc = Locator::Path(tmp.path().to_path_buf());
        requires_send(loc.open_source().unwrap());
        let tmp_sink = tempfile::NamedTempFile::new().unwrap();
        let sink_loc = Locator::Path(tmp_sink.path().to_path_buf());
        requires_send(sink_loc.open_sink().unwrap());
    }
}
