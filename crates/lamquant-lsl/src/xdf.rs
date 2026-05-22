//! XDF export. Reads a `.lml` archive, writes out a `.xdf` file
//! consumable by LabRecorder, OpenViBE, and other XDF tools.
//!
//! XDF spec: https://github.com/sccn/xdf/wiki/Specifications
//!
//! Minimal single-stream writer for Phase 6. The XDF format is
//! richer than what we emit (multi-stream files, ClockOffset
//! chunks, Boundary markers, per-chunk timestamps) — those land
//! incrementally as the LamQuant ↔ LSL bridge grows past
//! single-recording replay.
//!
//! Chunk format:
//!   [length_prefix: 1 or 8 bytes][tag: u16 LE][content: bytes]
//!
//! Length prefix:
//!   * 0xFE then 1 byte if length ≤ 255
//!   * 0xFF then 8 byte little-endian u64 otherwise
//!   * 0xFD then 4 byte little-endian u32 for 256..=u32::MAX
//!
//! Tags we emit:
//!   * 1 — FileHeader (XML, info envelope)
//!   * 2 — StreamHeader (XML, channel metadata)
//!   * 3 — Samples (binary)
//!   * 6 — StreamFooter (XML, sample count + timestamps)

use crate::error::LslIntegrationError;
use crate::metadata_lite::{stream_spec_from_lml, ChannelFormatLite};
use std::io::Write;

/// XDF chunk tags. See specification linked at module level.
#[repr(u16)]
#[derive(Clone, Copy)]
enum ChunkTag {
    FileHeader = 1,
    StreamHeader = 2,
    Samples = 3,
    StreamFooter = 6,
}

/// Write the per-chunk variable-length prefix. Matches the XDF
/// spec's three-tier encoding (1/4/8 byte length).
fn write_length_prefix(out: &mut Vec<u8>, len: u64) {
    if len <= u8::MAX as u64 {
        out.push(0x01);
        out.push(len as u8);
    } else if len <= u32::MAX as u64 {
        out.push(0x04);
        out.extend_from_slice(&(len as u32).to_le_bytes());
    } else {
        out.push(0x08);
        out.extend_from_slice(&len.to_le_bytes());
    }
}

/// Write a complete chunk: length prefix + tag + content.
fn write_chunk(out: &mut Vec<u8>, tag: ChunkTag, content: &[u8]) {
    // Length includes the 2-byte tag field per the XDF spec.
    let total_len = (content.len() + 2) as u64;
    write_length_prefix(out, total_len);
    out.extend_from_slice(&(tag as u16).to_le_bytes());
    out.extend_from_slice(content);
}

/// XML escape a string for safe embedding in stream headers.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

/// Build the StreamHeader XML body. Mirrors the
/// `lsl::StreamInfo`'s `to_xml` output that LabRecorder + other
/// XDF consumers parse.
fn build_stream_header_xml(
    name: &str,
    stream_type: &str,
    channel_count: u32,
    nominal_srate: f64,
    channel_format: ChannelFormatLite,
    source_id: &str,
    channel_labels: &[String],
    channel_unit: &str,
) -> String {
    let fmt = match channel_format {
        ChannelFormatLite::Int32 => "int32",
        ChannelFormatLite::Float32 => "float32",
    };
    let mut channels_xml = String::new();
    for (i, label) in channel_labels.iter().enumerate() {
        let display = if label.is_empty() {
            format!("ch{}", i)
        } else {
            label.clone()
        };
        channels_xml.push_str(&format!(
            "<channel><label>{}</label><unit>{}</unit><type>EEG</type></channel>",
            xml_escape(&display),
            xml_escape(channel_unit),
        ));
    }
    format!(
        "<?xml version=\"1.0\"?>\
         <info>\
         <name>{}</name>\
         <type>{}</type>\
         <channel_count>{}</channel_count>\
         <nominal_srate>{}</nominal_srate>\
         <channel_format>{}</channel_format>\
         <source_id>{}</source_id>\
         <desc><channels>{}</channels></desc>\
         </info>",
        xml_escape(name),
        xml_escape(stream_type),
        channel_count,
        nominal_srate,
        fmt,
        xml_escape(source_id),
        channels_xml,
    )
}

fn build_file_header_xml() -> String {
    "<?xml version=\"1.0\"?>\
     <info>\
     <version>1.0</version>\
     <writer>lamquant-lsl</writer>\
     </info>"
        .to_string()
}

fn build_stream_footer_xml(sample_count: u64, first_ts: f64, last_ts: f64) -> String {
    format!(
        "<?xml version=\"1.0\"?>\
         <info>\
         <sample_count>{}</sample_count>\
         <first_timestamp>{}</first_timestamp>\
         <last_timestamp>{}</last_timestamp>\
         </info>",
        sample_count, first_ts, last_ts,
    )
}

/// XDF writer options. Backwards-compatible — `Default` matches
/// the Phase 6.a writer's behaviour (no per-sample timestamps).
#[derive(Debug, Clone, Default)]
pub struct XdfOpts {
    /// If true, emit per-sample timestamps in Samples chunks
    /// (flag byte = 1, followed by `f64 LE` timestamp per sample).
    /// If false (default), emit flag = 0 — readers infer timing
    /// from `nominal_srate`.
    pub per_sample_timestamps: bool,
    /// First sample's timestamp (LSL epoch seconds). Subsequent
    /// samples are `anchor + i / nominal_srate`. Default 0.0.
    pub timestamp_anchor: f64,
}

impl XdfOpts {
    pub fn with_timestamps(mut self, on: bool) -> Self {
        self.per_sample_timestamps = on;
        self
    }
    pub fn with_timestamp_anchor(mut self, anchor: f64) -> Self {
        self.timestamp_anchor = anchor;
        self
    }
}

/// Build a Samples chunk payload. XDF Samples chunks have the
/// format:
///   [stream_id: u32 LE][num_samples: variable][per-sample data]
///
/// Per-sample data: [timestamp_present_flag: u8][timestamp: f64
/// LE if flag][channel_count * value]
fn build_samples_chunk(
    stream_id: u32,
    samples: &[Vec<i32>],
    timestamps: Option<&[f64]>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + samples.len() * 16);
    out.extend_from_slice(&stream_id.to_le_bytes());
    write_length_prefix(&mut out, samples.len() as u64);
    for (i, sample) in samples.iter().enumerate() {
        if let Some(ts) = timestamps {
            // flag = 1 → next 8 bytes are the f64 LE timestamp.
            out.push(1u8);
            let t = ts.get(i).copied().unwrap_or(0.0);
            out.extend_from_slice(&t.to_le_bytes());
        } else {
            out.push(0u8);
        }
        for &v in sample {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Convenience: write XDF with default options.
pub fn write_xdf_from_lml(
    lml_path: &std::path::Path,
    xdf_path: &std::path::Path,
) -> Result<(), LslIntegrationError> {
    write_xdf_from_lml_opts(lml_path, xdf_path, XdfOpts::default())
}

/// Write a `.xdf` file from a `.lml` source. Single-stream;
/// reads the LML container's signal + metadata, transposes to
/// per-sample channel rows, writes XDF chunks. `opts` controls
/// per-sample timestamp emission (Phase 6.b).
pub fn write_xdf_from_lml_opts(
    lml_path: &std::path::Path,
    xdf_path: &std::path::Path,
    opts: XdfOpts,
) -> Result<(), LslIntegrationError> {
    // Load the codec-decoded signal + container metadata.
    let (signal_i64, _meta) =
        lamquant_core::container::read_file(lml_path).map_err(LslIntegrationError::LmlDecode)?;
    let n_ch = signal_i64.len();
    let n_samples = signal_i64.first().map(|c| c.len()).unwrap_or(0);

    // Pull metadata fields via the lite parser (no liblsl dep).
    let source_id = crate::stream_id::stream_id_from_lml(lml_path)?;
    let spec = stream_spec_from_lml(lml_path, None, Some("EEG"), &source_id)?;

    // Transpose channel-major → sample-major i32 vectors.
    let samples_i32: Vec<Vec<i32>> = (0..n_samples)
        .map(|t| {
            (0..n_ch)
                .map(|ch| {
                    signal_i64[ch]
                        .get(t)
                        .copied()
                        .unwrap_or(0)
                        .clamp(i32::MIN as i64, i32::MAX as i64) as i32
                })
                .collect()
        })
        .collect();

    let mut out: Vec<u8> = Vec::with_capacity(2048 + samples_i32.len() * n_ch * 4);
    // Magic — XDF files start with "XDF:" (4 bytes).
    out.extend_from_slice(b"XDF:");

    // FileHeader chunk (tag 1).
    let header_xml = build_file_header_xml();
    write_chunk(&mut out, ChunkTag::FileHeader, header_xml.as_bytes());

    // StreamHeader chunk (tag 2). XDF Stream IDs are u32; we
    // hash the source_id to a stable u32 so the same .lml always
    // yields the same stream_id in the output.
    let stream_id: u32 = stream_id_to_u32(&source_id);
    let stream_header_xml = build_stream_header_xml(
        &spec.name,
        &spec.stream_type,
        spec.channel_count,
        spec.nominal_srate,
        spec.channel_format,
        &spec.source_id,
        &spec.channel_labels,
        &spec.channel_unit,
    );
    // StreamHeader format: [stream_id: u32 LE][xml: bytes].
    let mut sh_payload = Vec::with_capacity(4 + stream_header_xml.len());
    sh_payload.extend_from_slice(&stream_id.to_le_bytes());
    sh_payload.extend_from_slice(stream_header_xml.as_bytes());
    write_chunk(&mut out, ChunkTag::StreamHeader, &sh_payload);

    // Samples chunk (tag 3). For Phase 6 emit one big Samples
    // chunk; large recordings should chunk into ~1-second slices
    // (future work).
    let timestamps: Option<Vec<f64>> = if opts.per_sample_timestamps {
        let step = if spec.nominal_srate > 0.0 {
            1.0 / spec.nominal_srate
        } else {
            0.0
        };
        Some(
            (0..samples_i32.len())
                .map(|i| opts.timestamp_anchor + i as f64 * step)
                .collect(),
        )
    } else {
        None
    };
    let samples_chunk = build_samples_chunk(stream_id, &samples_i32, timestamps.as_deref());
    write_chunk(&mut out, ChunkTag::Samples, &samples_chunk);

    // StreamFooter chunk (tag 6).
    let footer_xml = build_stream_footer_xml(
        n_samples as u64,
        0.0,
        if spec.nominal_srate > 0.0 {
            n_samples as f64 / spec.nominal_srate
        } else {
            0.0
        },
    );
    let mut sf_payload = Vec::with_capacity(4 + footer_xml.len());
    sf_payload.extend_from_slice(&stream_id.to_le_bytes());
    sf_payload.extend_from_slice(footer_xml.as_bytes());
    write_chunk(&mut out, ChunkTag::StreamFooter, &sf_payload);

    // Write atomically: write to tmp + rename.
    let parent = xdf_path.parent().unwrap_or(std::path::Path::new("."));
    let tmp_path = parent.join(format!(
        ".{}.tmp.xdf",
        xdf_path.file_name().unwrap_or_default().to_string_lossy()
    ));
    {
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(&out)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp_path, xdf_path)?;
    Ok(())
}

/// Hash an LSL source_id (string) to a stable u32 stream ID for
/// the XDF file. XDF uses u32 stream IDs internally even though
/// LSL's source_id is a string.
fn stream_id_to_u32(source_id: &str) -> u32 {
    use sha2::Digest;
    let h = sha2::Sha256::digest(source_id.as_bytes());
    u32::from_le_bytes([h[0], h[1], h[2], h[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml_escape_basic() {
        assert_eq!(xml_escape("a<b>c&d\"e'f"), "a&lt;b&gt;c&amp;d&quot;e&apos;f");
        assert_eq!(xml_escape("normal"), "normal");
    }

    #[test]
    fn length_prefix_tiers() {
        let mut out = Vec::new();
        write_length_prefix(&mut out, 10);
        assert_eq!(out, vec![0x01, 0x0a]);
        out.clear();
        write_length_prefix(&mut out, 1_000_000);
        assert_eq!(out[0], 0x04);
        out.clear();
        write_length_prefix(&mut out, u64::MAX);
        assert_eq!(out[0], 0x08);
    }

    #[test]
    fn stream_id_to_u32_is_deterministic() {
        let a = stream_id_to_u32("lamquant:abc");
        let b = stream_id_to_u32("lamquant:abc");
        assert_eq!(a, b);
        let c = stream_id_to_u32("lamquant:xyz");
        assert_ne!(a, c);
    }
}
