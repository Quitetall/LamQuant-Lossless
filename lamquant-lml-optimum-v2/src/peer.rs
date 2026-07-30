//! Stable production facade for the frozen generation-v4 peer portfolio.
//!
//! Research encoders remain available in [`crate::mix1`]. Production callers
//! should bind to this module so candidate experiments do not become part of
//! their API contract.

use crate::mix1::{Mix1Codec, Mix1Decoded};
use crate::OptimumV2Error;

pub const PEER_KERNEL_ID: &str = "org.quitetall.lamquant.lml-optimum-v2.peer-v4";
pub const PEER_MAX_CHANNELS: usize = crate::mix1::MAX_CHANNELS;
pub const PEER_MAX_SAMPLES: usize = crate::mix1::MAX_SAMPLES;
pub const PEER_MAX_VALUES: usize = crate::mix1::MAX_VALUES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerEncodeContext {
    pub sample_rate_mhz: u32,
    pub bit_depth: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerPacketKind {
    Mix1,
    Multivariate,
    HierarchicalMultivariate,
    ChannelContext,
    CommonMode,
    PermutedCommonMode,
    AdaptivePermuted,
    CompactCommon,
    Alias,
    Bitplane,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerEncodeReport {
    pub selected_kind: PeerPacketKind,
    pub packet_bytes: usize,
    pub source_value_count: usize,
    pub source_bytes: usize,
    pub channel_count: usize,
    pub sample_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerEncodedWindow {
    pub packet: Vec<u8>,
    pub report: PeerEncodeReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerDecodedWindow {
    pub samples: Vec<Vec<i64>>,
    pub context: PeerEncodeContext,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PeerCodec;

impl PeerCodec {
    pub fn encode_window(
        &self,
        signal: &[Vec<i64>],
        context: PeerEncodeContext,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        Mix1Codec.encode_best_peer_window(signal, context.sample_rate_mhz, context.bit_depth)
    }

    pub fn encode_window_with_report(
        &self,
        signal: &[Vec<i64>],
        context: PeerEncodeContext,
    ) -> Result<PeerEncodedWindow, OptimumV2Error> {
        let packet = self.encode_window(signal, context)?;
        let channel_count = signal.len();
        let sample_count = signal.first().map_or(0, Vec::len);
        let source_value_count = channel_count
            .checked_mul(sample_count)
            .ok_or_else(|| input_error("peer source value count overflows"))?;
        let source_bytes = source_value_count
            .checked_mul(core::mem::size_of::<i32>())
            .ok_or_else(|| input_error("peer source byte count overflows"))?;
        let report = PeerEncodeReport {
            selected_kind: packet_kind(&packet)?,
            packet_bytes: packet.len(),
            source_value_count,
            source_bytes,
            channel_count,
            sample_count,
        };
        Ok(PeerEncodedWindow { packet, report })
    }

    pub fn decode_window(&self, packet: &[u8]) -> Result<PeerDecodedWindow, OptimumV2Error> {
        let decoded = Mix1Codec.decode_window(packet)?;
        Ok(decoded.into())
    }

    pub fn inspect_packet(&self, packet: &[u8]) -> Result<PeerEncodeReport, OptimumV2Error> {
        let decoded = Mix1Codec.decode_window(packet)?;
        let selected_kind = packet_kind(packet)?;
        let channel_count = decoded.samples.len();
        let sample_count = decoded.samples.first().map_or(0, Vec::len);
        let source_value_count = channel_count
            .checked_mul(sample_count)
            .ok_or_else(|| packet_error("peer source value count overflows"))?;
        let source_bytes = source_value_count
            .checked_mul(core::mem::size_of::<i32>())
            .ok_or_else(|| packet_error("peer source byte count overflows"))?;
        Ok(PeerEncodeReport {
            selected_kind,
            packet_bytes: packet.len(),
            source_value_count,
            source_bytes,
            channel_count,
            sample_count,
        })
    }
}

impl From<Mix1Decoded> for PeerDecodedWindow {
    fn from(decoded: Mix1Decoded) -> Self {
        Self {
            samples: decoded.samples,
            context: PeerEncodeContext {
                sample_rate_mhz: decoded.sample_rate_mhz,
                bit_depth: decoded.bit_depth,
            },
        }
    }
}

fn packet_kind(packet: &[u8]) -> Result<PeerPacketKind, OptimumV2Error> {
    if packet.get(..4) != Some(&b"OV2P"[..]) {
        return Err(packet_error("peer packet magic is invalid"));
    }
    let graph_offset = match packet.get(4) {
        Some(2) => 72,
        Some(3) => 40,
        Some(4) => 24,
        _ => return Err(packet_error("peer packet version is unsupported")),
    };
    let magic = packet
        .get(graph_offset..graph_offset + 4)
        .ok_or_else(|| packet_error("peer packet graph magic is truncated"))?;
    match magic {
        b"MIX1" => Ok(PeerPacketKind::Mix1),
        b"MMV1" => Ok(PeerPacketKind::Multivariate),
        b"MCH1" => Ok(PeerPacketKind::HierarchicalMultivariate),
        b"MCX1" => Ok(PeerPacketKind::ChannelContext),
        b"MQX1" => Ok(PeerPacketKind::CommonMode),
        b"MPX1" => Ok(PeerPacketKind::PermutedCommonMode),
        b"APX1" => Ok(PeerPacketKind::AdaptivePermuted),
        b"BQX1" => Ok(PeerPacketKind::CompactCommon),
        b"ALX1" => Ok(PeerPacketKind::Alias),
        b"BLX1" => Ok(PeerPacketKind::Bitplane),
        _ => Err(packet_error("peer packet graph magic is unsupported")),
    }
}

fn input_error(message: &str) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}

fn packet_error(message: &str) -> OptimumV2Error {
    OptimumV2Error::InvalidPacket(message.into())
}

#[cfg(test)]
mod tests {
    use super::{PeerCodec, PeerPacketKind};
    use crate::crc32c;
    use crate::mix1::Mix1Codec;

    #[test]
    fn inspector_accepts_compact_v3_header_at_its_canonical_offset() {
        let signal = vec![
            (0..64).map(|sample| i64::from(sample) - 32).collect(),
            (0..64)
                .map(|sample| i64::from((sample * 3) % 41) - 20)
                .collect(),
        ];
        let full = Mix1Codec
            .encode_window(&signal, 256_000, 16, 4)
            .expect("encode v2-header packet");
        assert_eq!(full[4], 2);

        let mut compact = full[..36].to_vec();
        compact[4] = 3;
        compact.extend_from_slice(&0_u32.to_le_bytes());
        compact.extend_from_slice(&full[72..]);
        let checksum = crc32c(&compact);
        compact[36..40].copy_from_slice(&checksum.to_le_bytes());

        let report = PeerCodec
            .inspect_packet(&compact)
            .expect("inspect canonical v3 packet");
        assert_eq!(report.selected_kind, PeerPacketKind::Mix1);
        assert_eq!(report.channel_count, 2);
        assert_eq!(report.sample_count, 64);
    }
}
