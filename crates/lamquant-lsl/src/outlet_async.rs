//! Async wrapper around the sync `Outlet` core, gated by the
//! `async` Cargo feature. Built on `tokio::task::spawn_blocking`
//! so the underlying `lsl::StreamOutlet` (sync C library) stays
//! on a blocking-threadpool worker while the caller's async task
//! awaits completion.
//!
//! When to use the async wrapper:
//!   * Multi-stream daemon (server hosts N replay outlets concurrently)
//!   * Cancellation via `tokio::select!` between the playback task
//!     + a shutdown signal
//!   * TUI / GUI event-loop integration where blocking is forbidden
//!
//! When to stay on the sync core (`outlet::Outlet`):
//!   * Single-stream replay
//!   * High-rate sources (> 256 Hz) where the sync pacer's
//!     microsecond accuracy matters
//!   * Minimal-dep deployments
//!
//! liblsl itself is sync. The async layer is convenience, not
//! capability — every async operation is exactly its sync sibling
//! wrapped in spawn_blocking.

use crate::error::LslIntegrationError;
use crate::outlet::{Outlet, Rate};
use std::path::Path;

/// Async wrapper around [`Outlet`].
///
/// Construction runs on a blocking thread; runtime operations
/// (`push_all_async`) hand off the inner sync push loop to a
/// blocking-threadpool worker. The wrapper owns the `Outlet` and
/// won't allow concurrent pushes — call into a single instance
/// from a single task, mirror the sync API's `&self` contract.
pub struct OutletAsync {
    inner: std::sync::Arc<Outlet>,
}

impl OutletAsync {
    /// Build an outlet async-ly from an `.lml` file. The actual
    /// liblsl outlet creation is sync; `spawn_blocking` wraps it.
    pub async fn from_lml(
        lml_path: &Path,
        name: Option<&str>,
    ) -> Result<Self, LslIntegrationError> {
        Self::from_lml_with_rate(lml_path, name, Rate::RealTime).await
    }

    /// Build an outlet async-ly with an explicit rate.
    pub async fn from_lml_with_rate(
        lml_path: &Path,
        name: Option<&str>,
        rate: Rate,
    ) -> Result<Self, LslIntegrationError> {
        let path = lml_path.to_path_buf();
        let name_owned = name.map(|s| s.to_string());
        let inner = tokio::task::spawn_blocking(move || {
            Outlet::from_lml_with_rate(&path, name_owned.as_deref(), rate)
        })
        .await
        .map_err(|e| LslIntegrationError::Other(format!("spawn_blocking join: {}", e)))??;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Drain every sample to the LSL outlet. Returns the number
    /// of samples pushed. Underlying push loop runs on a blocking
    /// threadpool worker; the awaiting task can be cancelled via
    /// `tokio::select!` against a shutdown signal.
    pub async fn push_all(&self) -> Result<usize, LslIntegrationError> {
        let outlet = std::sync::Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || outlet.push_all())
            .await
            .map_err(|e| LslIntegrationError::Other(format!("spawn_blocking join: {}", e)))?
    }

    /// Number of samples buffered, ready to push.
    pub fn sample_count(&self) -> usize {
        self.inner.sample_count()
    }

    /// LSL nominal sample rate, as reported in the StreamInfo.
    pub fn nominal_srate(&self) -> f64 {
        self.inner.nominal_srate()
    }
}
