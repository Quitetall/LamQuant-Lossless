//! Shared MCU↔host IPC types (Cat A7 — 2026-05-21).
//!
//! `PostcardEnvelope` is the single versioned wrapper that crosses the
//! USB/BLE link. The `version` byte lives **at envelope level** (not per
//! message), so adding a new `MsgKind` variant is forward-compatible
//! without bumping the wire version. Bumping `version` is reserved for
//! framing / COBS-layer changes only.
//!
//! Wire encoding:
//!   - `postcard::to_slice(&envelope, &mut buf)` on send
//!   - COBS framing in the link layer (separate concern; not in this
//!     crate to avoid pulling `cobs` into firmware unless needed)
//!   - `postcard::from_bytes::<PostcardEnvelope>(...)` on receive

#![cfg_attr(not(any(test, feature = "host")), no_std)]
#![deny(unsafe_code)]

use heapless::Vec as HVec;
use serde::{Deserialize, Serialize};

/// Maximum inline payload bytes a single envelope can carry. Picked to
/// fit comfortably inside a 256-byte USB bulk-out packet after COBS
/// expansion (worst-case +1 byte per 254). Raise carefully — increases
/// `static` RAM use on the firmware side.
pub const MAX_PAYLOAD: usize = 240;

/// Current wire-format version. Bump only on framing-layer breaking
/// changes (envelope reshape, COBS swap, length-prefix change).
pub const ENVELOPE_VERSION: u8 = 1;

/// Message discriminator.
///
/// **Forward-compatibility policy** — postcard's default `Deserialize`
/// for an enum rejects unknown discriminants with `Err`. That means
/// adding a new `MsgKind` variant **does** require a sender/receiver
/// rebuild; the wire is NOT forward-compatible by default. Two
/// canonical update paths:
///
/// 1. **Coordinated bump** — bump `ENVELOPE_VERSION`, ship both ends
///    together. Decoders MAY accept lower-version envelopes via
///    back-compat shims; they MUST reject higher-version envelopes.
///    Use this for any new variant or any payload-shape change.
/// 2. **`Log` fallback** — send the new event as a `Log` payload
///    (free-form bytes). Old receivers still parse the envelope; the
///    payload bytes are forwarded to the operator without
///    interpretation. Use this only for transient observability
///    additions, not control-plane traffic.
///
/// Variant order MUST NOT change without an `ENVELOPE_VERSION` bump —
/// postcard encodes the discriminator as a varint over the declaration
/// order. (lamu review fix on 88b7868.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "defmt-format", derive(defmt::Format))]
#[repr(u8)]
pub enum MsgKind {
    /// Heartbeat / liveness ping (payload empty).
    Ping = 0,
    /// Pong reply (payload empty).
    Pong = 1,
    /// Periodic device status (battery, temp, electrode state).
    Status = 2,
    /// Encoded EEG window from MCU → host.
    EncodedWindow = 3,
    /// Activity-map summary (post-SNN inference).
    ActivityMap = 4,
    /// Command (host → MCU). Payload is a `CommandKind` byte + args.
    Command = 5,
    /// Acknowledgement (MCU → host) for a previously received Command.
    CommandAck = 6,
    /// Free-form structured log line (DEBUG/INFO/WARN/ERROR + bytes).
    Log = 7,
}

/// Single envelope that wraps every MCU↔host postcard message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "defmt-format", derive(defmt::Format))]
pub struct PostcardEnvelope {
    /// Must equal `ENVELOPE_VERSION`. Decoder MUST reject envelopes
    /// with a higher version (forward-incompatibility) but MAY accept
    /// lower-version envelopes if it has back-compat shims.
    pub version: u8,
    /// Sequence number — wraps at `u16::MAX`. Used for matching
    /// `Command` ↔ `CommandAck` pairs and for retry detection.
    pub seq: u16,
    /// Discriminator for the payload bytes.
    pub kind: MsgKind,
    /// Postcard-encoded payload bytes for the chosen `kind`.
    /// Empty slice = no payload (e.g. `Ping`).
    pub payload: HVec<u8, MAX_PAYLOAD>,
}

/// Errors that `PostcardEnvelope` constructors can produce. Distinct
/// from postcard's own ser/de errors — those bubble up untouched at
/// the wire layer. (lamu review fix on 88b7868.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt-format", derive(defmt::Format))]
pub enum EnvelopeError {
    /// Caller passed more than `MAX_PAYLOAD` bytes. Chunk over
    /// multiple envelopes or compress before send.
    PayloadTooLarge { len: usize, max: usize },
}

impl PostcardEnvelope {
    /// Build an envelope with an empty payload (Ping / Pong style).
    pub fn empty(seq: u16, kind: MsgKind) -> Self {
        Self {
            version: ENVELOPE_VERSION,
            seq,
            kind,
            payload: HVec::new(),
        }
    }

    /// Build an envelope from caller-provided payload bytes. Returns
    /// `Err(EnvelopeError::PayloadTooLarge)` if `bytes.len() > MAX_PAYLOAD`
    /// — caller must chunk or compress.
    pub fn with_payload(seq: u16, kind: MsgKind, bytes: &[u8])
        -> Result<Self, EnvelopeError>
    {
        let mut payload: HVec<u8, MAX_PAYLOAD> = HVec::new();
        payload.extend_from_slice(bytes)
            .map_err(|_| EnvelopeError::PayloadTooLarge {
                len: bytes.len(),
                max: MAX_PAYLOAD,
            })?;
        Ok(Self {
            version: ENVELOPE_VERSION,
            seq,
            kind,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_roundtrip() {
        let env = PostcardEnvelope::empty(42, MsgKind::Ping);
        let mut buf = [0u8; 256];
        let used = postcard::to_slice(&env, &mut buf).expect("encode");
        let (decoded, _rest): (PostcardEnvelope, _) =
            postcard::take_from_bytes(used).expect("decode");
        assert_eq!(decoded.version, ENVELOPE_VERSION);
        assert_eq!(decoded.seq, 42);
        assert_eq!(decoded.kind, MsgKind::Ping);
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn payload_roundtrip() {
        let payload = b"hello postcard envelope wire format";
        let env = PostcardEnvelope::with_payload(7, MsgKind::Log, payload).expect("build");
        let mut buf = [0u8; 256];
        let used = postcard::to_slice(&env, &mut buf).expect("encode");
        let (decoded, _): (PostcardEnvelope, _) =
            postcard::take_from_bytes(used).expect("decode");
        assert_eq!(decoded.payload.as_slice(), payload);
    }

    #[test]
    fn over_max_payload_rejected() {
        let big = vec![0u8; MAX_PAYLOAD + 1];
        let err = PostcardEnvelope::with_payload(0, MsgKind::Status, &big)
            .expect_err("oversize must reject");
        assert_eq!(
            err,
            EnvelopeError::PayloadTooLarge {
                len: MAX_PAYLOAD + 1,
                max: MAX_PAYLOAD,
            }
        );
    }
}
