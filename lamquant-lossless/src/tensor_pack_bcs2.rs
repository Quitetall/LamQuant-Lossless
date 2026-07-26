//! The training window pack on the BCS2 `TRAINING_*` wire.
//!
//! `LQTP1` (see [`crate::tensor_pack`]) is a bespoke 64-byte header followed by
//! fixed-stride block-floating-point records. ADR 0139 contract 5 wants one
//! native wire family, and ADR 0144 already registered six `TRAINING_*` profiles
//! with a sealed catalog, per-row identity and a validated frame closure — which
//! is what a pack has been approximating with a `manifest_sha256` field and an
//! implicit row order.
//!
//! # What the mapping had to get right
//!
//! A pack does not store windows as the real-valued arrays they are. Each window
//! is block-floating-point encoded: a per-channel `f32` scale plus integer
//! mantissas, a third the size of raw `f32` and exactly what a GPU wants to load
//! before dequantising itself.
//!
//! Expressing that required a choice, and two of the three options were lies:
//!
//! - Declare each row to be *integer mantissas*. Loses the fact that a window is
//!   real-valued and hides the scales entirely — a consumer reading the catalog
//!   would see `I16 [channels, samples]` and have no way to know it was looking
//!   at a quantised view, or where the scales went.
//! - Declare each row `F32` while storing the record. Breaks frame closure, and
//!   the catalog would assert an identity the frame does not have.
//! - Declare the row's honest logical content (`F32 [channels, samples]`) and
//!   record separately that the frame carries a BFP encoding, naming the
//!   capability that recovers it. This is what the row-encoding declaration in
//!   `abir-training` exists for, and it is what this module uses.
//!
//! The third is not merely tidier. Mantissas are plausible-looking integers: a
//! consumer that read them as amplitudes would train on numerically wrong data
//! with nothing raising an error anywhere, producing a model that is merely bad
//! rather than obviously broken. `CAP_LAMQUANT_BFP_V1` turns that into a refusal
//! at the envelope.
//!
//! # What this module does not do
//!
//! It does not retire `LQTP1`. [`crate::tensor_pack`] still writes and reads it,
//! no existing pack is rewritten, and conversion is opt-in and writes a new file
//! beside the source. Retirement is a later phase, gated on this path first being
//! measured superior.

use crate::tensor_pack::{PackDtype, PackError, PackHeader, PackReader, LQTP_HEADER_LEN};
use semantic_abir::{payload_content_id, ByteOrder, ContentId, ElementType};
use semantic_abir_bcs::{ResourceBounds, SemanticPayloadFrame, CAP_LAMQUANT_BFP_V1};
use semantic_abir_training::{
    encode_snapshot, ContentKey, TrainingProfile, TrainingRow, TrainingRowEncoding,
    TrainingSnapshot, TrainingWindowStore,
};
use std::path::Path;

type Error = Box<dyn std::error::Error + Send + Sync>;

/// Capabilities a LamQuant training consumer offers for pack rows.
pub const READER_CAPABILITIES: u64 = CAP_LAMQUANT_BFP_V1;

/// Domain separator for the synthetic content keys this module derives.
///
/// A pack's rows are positional — row `i` is manifest entry `i` — so their keys
/// must be derived from something stable rather than invented. Deriving them
/// from the manifest hash plus the row index makes two packs built from the same
/// manifest produce the same keys, and makes a pack built from a *different*
/// manifest produce different ones, which is the property `manifest_sha256`
/// existed to enforce.
const KEY_DOMAIN: &[u8] = b"org.quitetall.lamquant.training-pack-v1\0";

fn derive_key(manifest_sha256: &[u8; 32], kind: &str, index: u64) -> ContentKey {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(KEY_DOMAIN);
    hasher.update(manifest_sha256);
    hasher.update(kind.as_bytes());
    hasher.update(index.to_le_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    ContentKey::new(ContentId::from_bytes(digest))
}

/// Dequantise one BFP record into its logical `f32` window, row-major
/// `[channels, samples]`.
///
/// This is the reference definition of what `CAP_LAMQUANT_BFP_V1` means. It is
/// used to derive each row's logical identity at write time, so the catalog's
/// claim about a row is computed from the same arithmetic a reader will apply
/// rather than asserted independently of it.
pub fn dequantize_record(
    record: &[u8],
    dtype: PackDtype,
    n_channels: usize,
    window_len: usize,
) -> Result<Vec<f32>, PackError> {
    let scale_bytes = n_channels * 4;
    let expected = scale_bytes + n_channels * window_len * dtype.mant_size();
    if record.len() != expected {
        return Err(PackError::ShapeMismatch(format!(
            "record is {} bytes; expected {expected}",
            record.len()
        )));
    }
    let mut values = Vec::with_capacity(n_channels * window_len);
    for channel in 0..n_channels {
        let scale = f32::from_le_bytes([
            record[channel * 4],
            record[channel * 4 + 1],
            record[channel * 4 + 2],
            record[channel * 4 + 3],
        ]);
        let mant_size = dtype.mant_size();
        let base = scale_bytes + channel * window_len * mant_size;
        for sample in 0..window_len {
            let at = base + sample * mant_size;
            let value = match dtype {
                PackDtype::Int8 => (record[at] as i8) as f32 * scale,
                PackDtype::Int16 => i16::from_le_bytes([record[at], record[at + 1]]) as f32 * scale,
                // Scale is 1.0 by construction for F32 packs; multiplying anyway
                // would turn a corrupt scale into silently wrong values instead
                // of leaving the exact bytes exact.
                PackDtype::F32 => {
                    f32::from_le_bytes([record[at], record[at + 1], record[at + 2], record[at + 3]])
                }
            };
            values.push(value);
        }
    }
    Ok(values)
}

fn logical_bytes_of(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Build the sealed catalog and payload frames for a pack's records.
fn build_snapshot<'a>(
    header: &PackHeader,
    records: &'a [Vec<u8>],
    profile: TrainingProfile,
) -> Result<(TrainingSnapshot, Vec<SemanticPayloadFrame<'a>>), Error> {
    let mut rows = Vec::with_capacity(records.len());
    let mut frames = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let values = dequantize_record(record, header.dtype, header.n_channels, header.window_len)?;
        let logical = logical_bytes_of(&values);
        let stored_payload = ContentKey::new(payload_content_id(ElementType::U8, record));
        rows.push(TrainingRow {
            byte_order: ByteOrder::Little,
            encoding: Some(TrainingRowEncoding {
                capabilities: CAP_LAMQUANT_BFP_V1,
                stored_element: ElementType::U8,
                stored_bytes: record.len() as u64,
                stored_payload,
            }),
            group: derive_key(&header.manifest_sha256, "group", index as u64),
            label: derive_key(&header.manifest_sha256, "label", index as u64),
            logical_bytes: logical.len() as u64,
            logical_id: derive_key(&header.manifest_sha256, "row", index as u64),
            payload: ContentKey::new(payload_content_id(ElementType::F32, &logical)),
            element: ElementType::F32,
            shape: vec![header.n_channels as u64, header.window_len as u64],
            split: derive_key(&header.manifest_sha256, "split", index as u64),
        });
        frames.push(SemanticPayloadFrame::encoded(
            ElementType::U8,
            record,
            CAP_LAMQUANT_BFP_V1,
        ));
    }
    // Rows must be ordered by logical id for the sealed catalog.
    rows.sort_by_key(|row| row.logical_id);

    let snapshot = TrainingSnapshot::seal(
        vec![derive_key(&header.manifest_sha256, "dataset", 0)],
        derive_key(&header.manifest_sha256, "spec", 0),
        profile,
        rows,
        derive_key(&header.manifest_sha256, "decision-log", 0),
    )?;
    Ok((snapshot, frames))
}

fn bounds_for(records: usize, largest: usize) -> ResourceBounds {
    let headroom = |value: usize| -> u32 {
        u32::try_from(value.saturating_mul(2).saturating_add(1 << 20)).unwrap_or(u32::MAX)
    };
    let mut bounds = ResourceBounds::default();
    bounds.max_index_entries = bounds
        .max_index_entries
        .max(u32::try_from(records.saturating_add(16)).unwrap_or(u32::MAX));
    bounds.max_frame_bytes = bounds.max_frame_bytes.max(headroom(largest));
    bounds.max_catalog_bytes = bounds.max_catalog_bytes.max(headroom(records * 1024));
    bounds
}

/// Convert an `LQTP1` pack into a BCS2 training snapshot.
///
/// Non-destructive: the source pack is read and left exactly as it is. Records
/// are carried across verbatim — they are the artifact the training loop mmaps,
/// and re-encoding them would change the numbers a model trains on.
pub fn convert_pack(
    pack_path: &Path,
    output_path: &Path,
    profile: TrainingProfile,
) -> Result<u64, Error> {
    let reader = PackReader::open(pack_path, None)?;
    let header = reader.header().clone();
    let mut records = Vec::with_capacity(header.n_windows);
    for row in 0..header.n_windows {
        // `window_raw` lends the scale and mantissa spans separately; the frame
        // is their concatenation, which is exactly the pack's on-disk record.
        let (scales, mantissas) = reader.window_raw(row)?;
        let mut record = Vec::with_capacity(scales.len() + mantissas.len());
        record.extend_from_slice(scales);
        record.extend_from_slice(mantissas);
        records.push(record);
    }
    let (snapshot, frames) = build_snapshot(&header, &records, profile)?;
    let bounds = bounds_for(records.len(), header.record_stride);
    let artifact = encode_snapshot(&snapshot, &frames, bounds)?;
    std::fs::write(output_path, &artifact)?;
    Ok(artifact.len() as u64)
}

/// A pack read back from a BCS2 training snapshot.
pub struct SnapshotPack {
    bytes: Vec<u8>,
}

impl SnapshotPack {
    pub fn open(path: &Path) -> Result<Self, Error> {
        Ok(Self {
            bytes: std::fs::read(path)?,
        })
    }

    fn store(&self) -> Result<TrainingWindowStore<'_>, Error> {
        let bounds = bounds_for(0, self.bytes.len());
        Ok(TrainingWindowStore::open_with_capabilities(
            &self.bytes,
            READER_CAPABILITIES,
            bounds,
        )?)
    }

    /// Number of windows.
    pub fn len(&self) -> Result<usize, Error> {
        Ok(self.store()?.rows().len())
    }

    pub fn is_empty(&self) -> Result<bool, Error> {
        Ok(self.len()? == 0)
    }

    /// The stored BFP records, in catalog order.
    ///
    /// This is the zero-copy path the training loop wants: the bytes are the
    /// pack's records, and the caller dequantises on device.
    pub fn records(&self) -> Result<Vec<Vec<u8>>, Error> {
        Ok(self
            .store()?
            .rows()
            .map(|row| row.bytes().to_vec())
            .collect())
    }

    /// Every window dequantised to its logical `f32` array.
    pub fn windows(&self, dtype: PackDtype) -> Result<Vec<Vec<f32>>, Error> {
        let store = self.store()?;
        let mut out = Vec::with_capacity(store.rows().len());
        for row in store.rows() {
            let shape = row.shape();
            if shape.len() != 2 {
                return Err(format!("row shape {shape:?} is not [channels, samples]").into());
            }
            out.push(dequantize_record(
                row.bytes(),
                dtype,
                shape[0] as usize,
                shape[1] as usize,
            )?);
        }
        Ok(out)
    }
}

/// True when `path` holds a BCS2 artifact rather than an `LQTP1` pack.
pub fn is_snapshot(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic).is_ok() && magic == semantic_abir_bcs::BCS2_MAGIC
}

/// Byte length of one BFP record for this shape — the pack's record stride.
pub fn record_stride(dtype: PackDtype, n_channels: usize, window_len: usize) -> usize {
    PackHeader::record_stride(dtype, n_channels, window_len)
}

/// The fixed header length of the container this module replaces.
pub const LEGACY_HEADER_LEN: usize = LQTP_HEADER_LEN;
