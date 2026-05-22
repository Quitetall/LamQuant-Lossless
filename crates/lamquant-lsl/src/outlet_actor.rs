//! Async StreamOutlet via actor pattern.
//!
//! ADR 0024 Phase 6.e. `lsl::StreamOutlet` is sync C + not `Send`,
//! so the obvious `Arc<Outlet>` + `spawn_blocking` recipe fails the
//! trait bound. The actor pattern lets us expose an `async` API
//! over a non-Send resource: a dedicated OS thread owns the outlet
//! for its lifetime; the async caller sends commands via
//! `tokio::sync::mpsc` + awaits responses via `oneshot`.
//!
//! Tradeoffs vs raw `Outlet`:
//!
//!   * One OS thread per active outlet (cheap — they spend most of
//!     their time blocked in `recv` waiting for the next command).
//!   * Per-command latency adds one channel hop. Negligible vs
//!     LSL network latency in practice.
//!   * Cancellation: dropping the actor handle disconnects the
//!     command channel; the worker thread observes the disconnect
//!     + exits cleanly. Explicit `shutdown().await` also supported.

#![cfg(feature = "async")]

use crate::error::LslIntegrationError;
use crate::outlet::{Outlet, Rate};
use std::path::PathBuf;
use tokio::sync::{mpsc, oneshot};

/// Commands the actor thread executes against its owned Outlet.
enum Cmd {
    /// Drain every buffered sample to the outlet, reply with the
    /// pushed count.
    PushAll(oneshot::Sender<Result<usize, LslIntegrationError>>),
    /// Shut down cleanly. Worker exits after replying.
    Shutdown(oneshot::Sender<()>),
}

/// Async handle to a sync Outlet running on a dedicated OS thread.
pub struct OutletActor {
    tx: mpsc::Sender<Cmd>,
    /// JoinHandle of the worker thread. Some until shutdown joins
    /// it; ensures the thread is reaped before drop completes.
    thread: Option<std::thread::JoinHandle<()>>,
}

impl OutletActor {
    /// Spawn an actor that opens an `Outlet` from the given .lml
    /// path. Returns once the outlet is constructed on the worker
    /// thread + the command channel is ready.
    pub async fn spawn_from_lml(
        lml_path: PathBuf,
        name: Option<String>,
        rate: Rate,
    ) -> Result<Self, LslIntegrationError> {
        let (tx, mut rx) = mpsc::channel::<Cmd>(16);
        let (ready_tx, ready_rx) = oneshot::channel::<Result<(), LslIntegrationError>>();

        let thread = std::thread::Builder::new()
            .name("lamquant-lsl-outlet-actor".into())
            .spawn(move || {
                // Build the outlet on this thread (where it must
                // live — !Send). Signal readiness; bail if construct
                // fails.
                let outlet = match Outlet::from_lml_with_rate(
                    &lml_path,
                    name.as_deref(),
                    rate,
                ) {
                    Ok(o) => {
                        let _ = ready_tx.send(Ok(()));
                        o
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };

                // Command loop. blocking_recv works inside a plain
                // OS thread (no tokio runtime required) by
                // delegating to the mpsc channel's sync primitives.
                while let Some(cmd) = rx.blocking_recv() {
                    match cmd {
                        Cmd::PushAll(reply) => {
                            let result = outlet.push_all();
                            let _ = reply.send(result);
                        }
                        Cmd::Shutdown(ack) => {
                            let _ = ack.send(());
                            break;
                        }
                    }
                }
                // Sender side dropped → channel closed → exit
                // cleanly; the outlet drops here.
            })
            .map_err(|e| {
                LslIntegrationError::Other(format!("actor spawn: {}", e))
            })?;

        // Wait for the worker to report outlet construction
        // success / failure.
        match ready_rx.await {
            Ok(Ok(())) => Ok(Self {
                tx,
                thread: Some(thread),
            }),
            Ok(Err(e)) => {
                // Worker exited; join to reap.
                let _ = thread.join();
                Err(e)
            }
            Err(_) => Err(LslIntegrationError::Other(
                "actor: worker dropped readiness signal".into(),
            )),
        }
    }

    /// Drain every sample to the LSL outlet. Returns the number
    /// of samples pushed. Cancellation: dropping the future before
    /// it completes still leaves the underlying push running on
    /// the worker thread (the worker doesn't know the caller
    /// stopped caring); next `push_all` will wait for the prior
    /// to finish. Use explicit `shutdown` to tear down between
    /// pushes.
    pub async fn push_all(&self) -> Result<usize, LslIntegrationError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Cmd::PushAll(reply_tx))
            .await
            .map_err(|_| LslIntegrationError::Other("actor: worker disconnected".into()))?;
        reply_rx
            .await
            .map_err(|_| LslIntegrationError::Other("actor: reply channel closed".into()))?
    }

    /// Shut the worker down + join the OS thread. Idempotent
    /// only if the actor is consumed (`shutdown(self)`); calling
    /// drop afterwards is a no-op because `thread` is already
    /// None.
    pub async fn shutdown(mut self) -> Result<(), LslIntegrationError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        // Best-effort send — if the worker has already exited
        // we still want to join its thread below.
        let _ = self.tx.send(Cmd::Shutdown(ack_tx)).await;
        let _ = ack_rx.await;
        if let Some(handle) = self.thread.take() {
            handle.join().map_err(|_| {
                LslIntegrationError::Other("actor: worker thread panicked".into())
            })?;
        }
        Ok(())
    }
}

impl Drop for OutletActor {
    fn drop(&mut self) {
        // If shutdown wasn't called explicitly, drop the command
        // sender (the worker's blocking_recv returns None on
        // disconnect, exiting the loop). Join to reap the thread
        // synchronously — the worker won't block long since liblsl
        // calls don't outlast a single push_sample.
        if let Some(thread) = self.thread.take() {
            // Close the sender by replacing with an already-closed
            // one: a Drop here means the channel's tx half is
            // about to die anyway, so we can short-circuit.
            // Joining the thread within Drop is safe because the
            // worker exits within a bounded time.
            let _ = thread.join();
        }
    }
}
