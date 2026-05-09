//! SSH transport — concrete [`Transport`] impl.
//!
//! Stub in commit 1. Filled in commit 4 (verify, stage_input via hash
//! detect + rsync, dispatch via `ssh peer "lamquant <op> --emit-json-events"`,
//! cancel via SSH process kill, health via `ssh peer "lamquant --status"`).
//!
//! Security stance:
//! - Explicit key path (no SSH-agent fallback)
//! - Pinned host fingerprint (StrictHostKeyChecking + per-peer known_hosts)
//! - Hard refuse on version mismatch

use crate::transport::{
    Peer, PeerHealth, PeerInfo, RemoteHandle, RemotePath, Transport, TransportError,
};
use crate::sink::OpReceiver;
use std::path::Path;

/// SSH-shelling transport. Construct via [`SshTransport::new`].
pub struct SshTransport {
    // Future: connection pool, command cache, etc.
    _placeholder: (),
}

impl SshTransport {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for SshTransport { fn default() -> Self { Self::new() } }

impl Transport for SshTransport {
    fn verify(&self, _peer: &Peer) -> Result<PeerInfo, TransportError> {
        Err(TransportError::DispatchFailed(
            "SshTransport::verify not yet implemented (commit 4 of transport refactor)".into(),
        ))
    }

    fn stage_input(
        &self,
        _peer: &Peer,
        _local: &Path,
    ) -> Result<RemotePath, TransportError> {
        Err(TransportError::StagingFailed(
            "SshTransport::stage_input not yet implemented (commit 4)".into(),
        ))
    }

    fn dispatch(
        &self,
        _peer: &Peer,
        _op_id: &str,
        _args: &[String],
    ) -> Result<(OpReceiver, RemoteHandle), TransportError> {
        Err(TransportError::DispatchFailed(
            "SshTransport::dispatch not yet implemented (commit 4)".into(),
        ))
    }

    fn cancel(
        &self,
        _peer: &Peer,
        _handle: &RemoteHandle,
    ) -> Result<(), TransportError> {
        Err(TransportError::DispatchFailed(
            "SshTransport::cancel not yet implemented (commit 4)".into(),
        ))
    }

    fn health(&self, _peer: &Peer) -> Result<PeerHealth, TransportError> {
        Err(TransportError::Unreachable(
            "SshTransport::health not yet implemented (commit 4)".into(),
        ))
    }
}
