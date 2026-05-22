//! Firmware-side wrapper around `lamquant-ipc-types::PostcardEnvelope`.
//!
//! Cat A7 step 2 (2026-05-22). All MCU→host status / event traffic must
//! flow through this module so the wire envelope stays the single
//! source of truth — `version`, `seq`, `kind`, `payload` — for every
//! message on the USB or BLE link.
//!
//! Encode path:
//!   1. caller builds typed `Status` / `EncodedWindow` / etc. payload
//!   2. `encode_status` / `encode_window` packs it into a fresh
//!      envelope with the next sequence number
//!   3. result is a `[u8]` slice into the caller's scratch buffer
//!      ready to hand to the link layer (`transport::usb` /
//!      `transport::ble`) for COBS framing + bulk-OUT send
//!
//! Decode path (host → MCU control plane):
//!   1. link layer hands raw post-COBS bytes
//!   2. `decode_envelope` returns the typed envelope or a typed error
//!   3. dispatcher reads `envelope.kind` + `envelope.payload`
//!
//! Sequence numbering uses a per-MCU monotonic counter wrapping at
//! `u16::MAX` — sufficient at 1 Hz status + 0.1 Hz window traffic.

#![allow(dead_code)] // until transport layer wires in.

use core::sync::atomic::{AtomicU16, Ordering};

pub use lamquant_ipc_types::{
    EnvelopeError, MsgKind, PostcardEnvelope, ENVELOPE_VERSION, MAX_PAYLOAD,
};

/// Monotonic sequence number for envelopes leaving this MCU. Wraps at
/// `u16::MAX`.
static SEQ_NEXT: AtomicU16 = AtomicU16::new(0);

/// Atomically allocate the next sequence number.
#[inline]
pub fn next_seq() -> u16 {
    SEQ_NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Top-level error type for the comms layer. `Postcard` wraps the
/// upstream ser/de error; `Envelope` wraps the builder error.
#[derive(Debug)]
#[non_exhaustive]
pub enum CommsError {
    Envelope(EnvelopeError),
    Postcard(postcard::Error),
}

impl From<EnvelopeError> for CommsError {
    fn from(e: EnvelopeError) -> Self {
        CommsError::Envelope(e)
    }
}
impl From<postcard::Error> for CommsError {
    fn from(e: postcard::Error) -> Self {
        CommsError::Postcard(e)
    }
}

/// Encode an empty envelope (Ping / Pong style) into `out`. Returns the
/// number of bytes written.
pub fn encode_empty(kind: MsgKind, out: &mut [u8]) -> Result<usize, CommsError> {
    let env = PostcardEnvelope::empty(next_seq(), kind);
    let n = postcard::to_slice(&env, out)?.len();
    Ok(n)
}

/// Encode an envelope wrapping `payload` bytes into `out`. Returns the
/// number of wire bytes written.
pub fn encode_with_payload(
    kind: MsgKind,
    payload: &[u8],
    out: &mut [u8],
) -> Result<usize, CommsError> {
    let env = PostcardEnvelope::with_payload(next_seq(), kind, payload)?;
    let n = postcard::to_slice(&env, out)?.len();
    Ok(n)
}

/// Decode a wire envelope. Returns the parsed envelope on success.
///
/// **Forward-compat policy** (see `lamquant_ipc_types::MsgKind`): an
/// envelope whose `version > ENVELOPE_VERSION` is rejected; one whose
/// `version <` is accepted as best-effort and the caller chooses what
/// to do. An unknown `MsgKind` discriminator becomes a
/// `CommsError::Postcard` — the link stays up; only the offending
/// envelope is dropped.
pub fn decode_envelope(wire: &[u8]) -> Result<PostcardEnvelope, CommsError> {
    let (env, _rest): (PostcardEnvelope, _) = postcard::take_from_bytes(wire)?;
    Ok(env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_ping() {
        let mut buf = [0u8; 32];
        let n = encode_empty(MsgKind::Ping, &mut buf).expect("encode");
        let env = decode_envelope(&buf[..n]).expect("decode");
        assert_eq!(env.version, ENVELOPE_VERSION);
        assert_eq!(env.kind, MsgKind::Ping);
        assert!(env.payload.is_empty());
    }

    #[test]
    fn encode_decode_with_payload() {
        let mut buf = [0u8; 128];
        let payload = b"battery 87% temp 28C";
        let n = encode_with_payload(MsgKind::Status, payload, &mut buf).expect("encode");
        let env = decode_envelope(&buf[..n]).expect("decode");
        assert_eq!(env.kind, MsgKind::Status);
        assert_eq!(env.payload.as_slice(), payload);
    }

    #[test]
    fn seq_increments() {
        let s1 = next_seq();
        let s2 = next_seq();
        assert_eq!(s2.wrapping_sub(s1), 1);
    }

    #[test]
    fn oversize_payload_rejected() {
        let mut buf = [0u8; 512];
        let huge = vec![0u8; MAX_PAYLOAD + 1];
        let err = encode_with_payload(MsgKind::Log, &huge, &mut buf)
            .expect_err("must reject");
        assert!(matches!(err, CommsError::Envelope(EnvelopeError::PayloadTooLarge { .. })));
    }
}
