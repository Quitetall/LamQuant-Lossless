//! HDF5 / NWB integer-signal reader → LML ingest (ADR 0051 Track 3, Phase 1).
//!
//! **Host-only** (never in the no_std firmware build): pulls in libhdf5 via
//! `hdf5-metno`. This is the read half of "own the NWB flank H.BWC is not
//! addressing" — NWB (HDF5 + schema) is the AI/BCI/iEEG research container, a
//! format H.BWC structurally cannot touch.
//!
//! The reader walks an HDF5/NWB file, extracts every integer-typed 1-D/2-D
//! dataset (NWB `ElectricalSeries/data` falls out naturally), widens each to
//! the codec's `i64` channel-major form, and records exactly enough metadata
//! (`h5_path`, on-disk width/signedness, original shape, orientation) to
//! reconstruct the dataset losslessly.
//!
//! Float / non-integer datasets are intentionally **not** returned here: LML is
//! integer-only (ADR 0051 line 83 lists float roundtrip as a separate, later
//! item), so the ingest caller stores those byte-exact instead of through LML.

use crate::error::{LmlError, LmlResult};
use hdf5_metno::types::{IntSize, TypeDescriptor};
use hdf5_metno::{Dataset, File, Group};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

/// One integer time-series dataset extracted from an HDF5/NWB file, in the
/// codec's channel-major `i64` form, plus the metadata needed to put it back
/// exactly as it was found.
#[derive(Debug, Clone)]
pub struct H5IntSignal {
    /// Full HDF5 path of the dataset, e.g. `/acquisition/ElectricalSeries/data`.
    pub h5_path: String,
    /// Channel-major signal `[n_ch][t]` — each inner `Vec` is one channel's
    /// time series (the codec's prediction axis).
    pub signal: Vec<Vec<i64>>,
    /// On-disk integer width in bytes: 1, 2, 4, or 8.
    pub int_bytes: u8,
    /// On-disk signedness (`false` ⇒ the source was an unsigned integer type).
    pub signed: bool,
    /// `true` when the source was a 2-D dataset stored time-major (rows = time,
    /// the NWB `ElectricalSeries` convention) and we transposed it to
    /// channel-major. `false` for 1-D datasets (a single channel, no transpose).
    pub time_major: bool,
    /// Original HDF5 dataset shape (pre-transpose), so a writer can restore the
    /// exact dimensionality and orientation.
    pub orig_shape: Vec<usize>,
}

/// Map an `hdf5_metno` error into the codec's error type with context.
fn h5<T>(r: hdf5_metno::Result<T>, ctx: &str) -> LmlResult<T> {
    r.map_err(|e| LmlError::InvalidHeader(format!("HDF5 {ctx}: {e}")))
}

#[inline]
fn int_bytes_of(sz: IntSize) -> u8 {
    match sz {
        IntSize::U1 => 1,
        IntSize::U2 => 2,
        IntSize::U4 => 4,
        IntSize::U8 => 8,
    }
}

/// Recursively collect every dataset reachable from `group` (depth-first).
fn collect_datasets(group: &Group, out: &mut Vec<Dataset>) -> LmlResult<()> {
    for ds in h5(group.datasets(), "datasets")? {
        out.push(ds);
    }
    for g in h5(group.groups(), "groups")? {
        collect_datasets(&g, out)?;
    }
    Ok(())
}

/// Read one dataset into channel-major `i64`. Generic over the concrete on-disk
/// element type `T`; `widen` lifts each value to `i64` (fallible only for the
/// `u64` case, where a value can exceed `i64::MAX`).
///
/// 2-D datasets are treated as `(time, channel)` (NWB `ElectricalSeries`
/// convention) and transposed to channel-major; 1-D datasets become a single
/// channel. Returns `(signal, time_major)`.
fn build<T>(
    ds: &Dataset,
    shape: &[usize],
    widen: impl Fn(T) -> LmlResult<i64>,
) -> LmlResult<(Vec<Vec<i64>>, bool)>
where
    T: hdf5_metno::H5Type + Copy,
{
    if shape.len() == 1 {
        let a = h5(ds.read_1d::<T>(), "read_1d")?;
        let mut ch = Vec::with_capacity(a.len());
        for &v in a.iter() {
            ch.push(widen(v)?);
        }
        Ok((vec![ch], false))
    } else {
        // shape == [d0 = time, d1 = channels]; channel-major sig[c][t] = a[[t, c]].
        let a = h5(ds.read_2d::<T>(), "read_2d")?;
        let (d0, d1) = (shape[0], shape[1]);
        let mut sig: Vec<Vec<i64>> = (0..d1).map(|_| Vec::with_capacity(d0)).collect();
        for t in 0..d0 {
            for (c, ch) in sig.iter_mut().enumerate() {
                ch.push(widen(a[[t, c]])?);
            }
        }
        Ok((sig, true))
    }
}

/// Extract one integer dataset, or `None` if it is not an integer 1-D/2-D
/// dataset (float / compound / string / scalar / >2-D are skipped — the caller
/// stores those byte-exact).
fn read_int_dataset(ds: &Dataset) -> LmlResult<Option<H5IntSignal>> {
    let descriptor = h5(h5(ds.dtype(), "dtype")?.to_descriptor(), "to_descriptor")?;
    let (int_bytes, signed) = match descriptor {
        TypeDescriptor::Integer(sz) => (int_bytes_of(sz), true),
        TypeDescriptor::Unsigned(sz) => (int_bytes_of(sz), false),
        _ => return Ok(None),
    };

    let shape = ds.shape();
    if shape.is_empty() || shape.len() > 2 || shape.contains(&0) {
        return Ok(None);
    }

    let (signal, time_major) = match (int_bytes, signed) {
        (1, true) => build::<i8>(ds, &shape, |v| Ok(v as i64))?,
        (2, true) => build::<i16>(ds, &shape, |v| Ok(v as i64))?,
        (4, true) => build::<i32>(ds, &shape, |v| Ok(v as i64))?,
        (8, true) => build::<i64>(ds, &shape, Ok)?,
        (1, false) => build::<u8>(ds, &shape, |v| Ok(v as i64))?,
        (2, false) => build::<u16>(ds, &shape, |v| Ok(v as i64))?,
        (4, false) => build::<u32>(ds, &shape, |v| Ok(v as i64))?,
        (8, false) => build::<u64>(ds, &shape, |v| {
            i64::try_from(v).map_err(|_| {
                LmlError::InvalidHeader(
                    "u64 dataset value exceeds i64 range; not LML-representable".into(),
                )
            })
        })?,
        _ => return Ok(None),
    };

    Ok(Some(H5IntSignal {
        h5_path: ds.name(),
        signal,
        int_bytes,
        signed,
        time_major,
        orig_shape: shape,
    }))
}

/// Open an HDF5/NWB file and return every integer 1-D/2-D dataset, widened to
/// channel-major `i64`. Float / non-integer datasets are omitted by design.
///
/// The order is deterministic (depth-first over `hdf5-metno`'s name-sorted
/// member iteration), so a downstream ingest manifest is stable.
pub fn read_int_signals(path: &Path) -> LmlResult<Vec<H5IntSignal>> {
    let file = h5(File::open(path), "open")?;
    let mut datasets = Vec::new();
    collect_datasets(&file, &mut datasets)?;

    let mut out = Vec::new();
    for ds in &datasets {
        if let Some(sig) = read_int_dataset(ds)? {
            out.push(sig);
        }
    }
    Ok(out)
}

// ── Zero-skeleton NWB ⇄ ABIR (ADR 0051 Track 3, Phase B) ─────────────────────
//
// Ingest an NWB/HDF5 file into canonical ABIR without a fragile structural
// transcoder. Integer datasets become typed Tensor atoms. A content-bound
// source capsule carries the original file with those datasets **zeroed** (a
// "skeleton"), preserving groups, attributes, float/compound datasets, and
// object references. Reconstruction writes tensor values back into the
// skeleton.

/// Source-capsule key: original HDF5 with integer datasets zeroed.
const SKEL_KEY: &str = "nwb_skeleton";
/// Source-capsule key: JSON `[NwbSlot]` mapping tensors into the skeleton.
const SLOTS_KEY: &str = "nwb_slots";

/// One integer dataset's placement in the skeleton and ABIR tensor catalog.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct NwbAtomId(semantic_abir::ObjectId<semantic_abir::AtomTag>);

impl Serialize for NwbAtomId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.to_bytes().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NwbAtomId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes = <[u8; 16]>::deserialize(deserializer)?;
        Ok(Self(semantic_abir::ObjectId::from_bytes(bytes)))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NwbSlot {
    h5_path: String,
    int_bytes: u8,
    signed: bool,
    orig_shape: Vec<usize>,
    time_major: bool,
    atom_id: NwbAtomId,
}

/// Write `flat` (i64, row-major over the dataset's dims) into `ds`, narrowed to
/// the dataset's on-disk integer type. Two's-complement low bytes are identical
/// for signed/unsigned, so `as` narrowing is exact for values that fit.
fn write_flat_i64(ds: &Dataset, int_bytes: u8, signed: bool, flat: &[i64]) -> LmlResult<()> {
    macro_rules! w {
        ($t:ty) => {{
            let v: Vec<$t> = flat.iter().map(|&x| x as $t).collect();
            h5(ds.write_raw(&v), "write_raw")?;
        }};
    }
    match (int_bytes, signed) {
        (1, true) => w!(i8),
        (2, true) => w!(i16),
        (4, true) => w!(i32),
        (8, true) => w!(i64),
        (1, false) => w!(u8),
        (2, false) => w!(u16),
        (4, false) => w!(u32),
        (8, false) => w!(u64),
        _ => {
            return Err(LmlError::InvalidHeader(format!(
                "unsupported int width {int_bytes}"
            )))
        }
    }
    Ok(())
}

/// Channel-major `[n_ch][t]` → flat row-major over `shape` (the dataset's
/// storage order), matching `write_raw`.
fn flatten_slot(chs: &[Vec<i64>], shape: &[usize], time_major: bool) -> Vec<i64> {
    if shape.len() == 1 {
        return chs.first().cloned().unwrap_or_default();
    }
    let (t, c) = (shape[0], shape[1]);
    let mut flat = Vec::with_capacity(t * c);
    if time_major {
        // dataset is (T, C): flat[t*C + c] = chs[c][t]
        for ti in 0..t {
            for ch in chs.iter().take(c) {
                flat.push(ch[ti]);
            }
        }
    } else {
        for ch in chs.iter().take(c) {
            flat.extend_from_slice(&ch[..t]);
        }
    }
    flat
}

struct PreparedTensor {
    slot: NwbSlot,
    bytes: Vec<u8>,
    content_id: semantic_abir::ContentId,
}

fn semantic_id<T>(seed: &[u8; 32], domain: &[u8], index: u64) -> semantic_abir::ObjectId<T> {
    let mut hasher = Sha256::new();
    hasher.update(b"lamquant.nwb.abir.v1\0");
    hasher.update(seed);
    hasher.update(domain);
    hasher.update(index.to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    semantic_abir::ObjectId::from_bytes(bytes)
}

fn i64_payload_bytes(signal: &[Vec<i64>], shape: &[usize], time_major: bool) -> Vec<u8> {
    let flat = flatten_slot(signal, shape, time_major);
    let mut bytes = Vec::with_capacity(flat.len().saturating_mul(8));
    for value in flat {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn invalid(message: impl Into<String>) -> LmlError {
    LmlError::InvalidHeader(message.into())
}

/// Read HDF5/NWB integer datasets as canonical ABIR Tensor atoms.
///
/// Arbitrary HDF5 integer arrays are not falsely labeled as one uniform-rate
/// signal. Exact source structure remains recoverable through content-bound
/// skeleton and slot-map capsules.
pub fn read_dataset(
    path: &Path,
) -> LmlResult<semantic_abir::OpenedDataset<semantic_abir::InMemoryPayloadAccess>> {
    let sigs = read_int_signals(path)?;
    if sigs.is_empty() {
        return Err(invalid(
            "no little-endian integer datasets found to compress",
        ));
    }

    let mut seed_hasher = Sha256::new();
    let mut prepared = Vec::with_capacity(sigs.len());
    for (index, signal) in sigs.into_iter().enumerate() {
        let bytes = i64_payload_bytes(&signal.signal, &signal.orig_shape, signal.time_major);
        let content_id = semantic_abir::payload_content_id(semantic_abir::ElementType::I64, &bytes);
        seed_hasher.update(signal.h5_path.as_bytes());
        seed_hasher.update([signal.int_bytes, u8::from(signal.signed)]);
        seed_hasher.update(content_id.to_bytes());
        for extent in &signal.orig_shape {
            seed_hasher.update((*extent as u64).to_le_bytes());
        }
        prepared.push(PreparedTensor {
            slot: NwbSlot {
                h5_path: signal.h5_path,
                int_bytes: signal.int_bytes,
                signed: signal.signed,
                orig_shape: signal.orig_shape,
                time_major: signal.time_major,
                atom_id: NwbAtomId(semantic_abir::ObjectId::from_bytes([0; 16])),
            },
            bytes,
            content_id,
        });
        seed_hasher.update((index as u64).to_le_bytes());
    }
    let seed: [u8; 32] = seed_hasher.finalize().into();
    let dataset_id = semantic_id(&seed, b"dataset", 0);
    let recording_id = semantic_id(&seed, b"recording", 0);
    let stream_id = semantic_id(&seed, b"stream", 0);
    for (index, tensor) in prepared.iter_mut().enumerate() {
        tensor.slot.atom_id = NwbAtomId(semantic_id::<semantic_abir::AtomTag>(
            &seed,
            b"tensor",
            index as u64,
        ));
    }
    let slots = prepared
        .iter()
        .map(|tensor| tensor.slot.clone())
        .collect::<Vec<_>>();

    // Build the zeroed skeleton: copy the file, overwrite each integer dataset
    // with zeros, read the bytes back. The temp file is removed on drop.
    let skel = tempfile::Builder::new()
        .prefix("lml_nwb_skel_")
        .suffix(".h5")
        .tempfile()
        .map_err(LmlError::Io)?;
    std::fs::copy(path, skel.path()).map_err(LmlError::Io)?;
    {
        let f = h5(File::open_rw(skel.path()), "open_rw skeleton")?;
        for slot in &slots {
            let ds = h5(f.dataset(&slot.h5_path), "skeleton dataset")?;
            let n: usize = slot.orig_shape.iter().product();
            write_flat_i64(&ds, slot.int_bytes, slot.signed, &vec![0i64; n])?;
        }
    }
    let skel_bytes = std::fs::read(skel.path()).map_err(LmlError::Io)?;

    let slots_json = serde_json::to_vec(&slots)
        .map_err(|e| LmlError::InvalidHeader(format!("slot encode: {e}")))?;

    let atom_ids = slots.iter().map(|slot| slot.atom_id.0).collect::<Vec<_>>();
    let mut draft = semantic_abir::DatasetDraft::new(dataset_id);
    let mut recording = semantic_abir::Recording::new(recording_id, vec![stream_id]);
    recording.add_source_key(
        semantic_abir::SourceKey::new("source.format", "NWB")
            .map_err(|_| invalid("invalid static NWB source key"))?,
    );
    if let Some(name) = path.file_name() {
        recording.add_source_key(
            semantic_abir::SourceKey::new("source.file", name.to_string_lossy())
                .map_err(|_| invalid("invalid NWB source filename"))?,
        );
    }
    draft.add_recording(recording);
    draft.add_stream(semantic_abir::Stream::new(
        stream_id,
        recording_id,
        semantic_abir::ConceptId::new("nwb:modality/integer-dataset")
            .map_err(|error| invalid(error.to_string()))?,
        atom_ids,
        None,
        None,
        None,
    ));

    let mut access = semantic_abir::InMemoryPayloadAccess::new();
    for tensor in prepared {
        let shape = tensor
            .slot
            .orig_shape
            .iter()
            .map(|extent| u64::try_from(*extent).map_err(|_| invalid("NWB extent exceeds u64")))
            .collect::<LmlResult<Vec<_>>>()?;
        let axes = shape
            .iter()
            .enumerate()
            .map(|(axis, extent)| {
                Ok(semantic_abir::SemanticAxis::new(
                    semantic_abir::ConceptId::new(format!("nwb:axis/dimension-{axis}"))
                        .map_err(|error| invalid(error.to_string()))?,
                    *extent,
                ))
            })
            .collect::<LmlResult<Vec<_>>>()?;
        let logical_bytes =
            u64::try_from(tensor.bytes.len()).map_err(|_| invalid("NWB payload too large"))?;
        let descriptor = semantic_abir::PayloadDescriptor::new(
            tensor.content_id,
            logical_bytes,
            semantic_abir::ElementType::I64,
            semantic_abir::ByteOrder::Little,
            shape,
            semantic_abir::Layout::DenseRowMajor,
            None,
            Some("application/x-nwb-integer-tensor".into()),
        );
        access.insert(tensor.content_id, tensor.bytes);
        draft.add_atom(semantic_abir::Atom::Tensor(semantic_abir::Tensor::new(
            tensor.slot.atom_id.0,
            semantic_abir::Presence::Present,
            Some(descriptor),
            axes,
        )));
    }

    for (index, (key, bytes, media_type)) in [
        (SKEL_KEY, skel_bytes, "application/x-hdf5"),
        (SLOTS_KEY, slots_json, "application/json"),
    ]
    .into_iter()
    .enumerate()
    {
        let content_id =
            semantic_abir::payload_content_id(semantic_abir::ElementType::Bytes, &bytes);
        access.insert(content_id, bytes);
        draft.add_source_capsule(semantic_abir::SourceCapsule::new(
            semantic_abir::SourceKey::new(format!("source.sidecar.{index}"), key)
                .map_err(|_| invalid("invalid static NWB capsule key"))?,
            content_id,
            Some(media_type),
        ));
    }

    let dataset = draft
        .validate(semantic_abir::ValidationLimits::default())
        .map_err(|report| invalid(format!("NWB ABIR validation failed: {report:?}")))?;
    Ok(semantic_abir::OpenedDataset::new(dataset, access))
}

fn capsule_bytes<'a>(
    opened: &'a semantic_abir::OpenedDataset<semantic_abir::InMemoryPayloadAccess>,
    key: &str,
) -> LmlResult<&'a [u8]> {
    let capsule = opened
        .dataset()
        .source_capsules()
        .iter()
        .find(|capsule| capsule.source().value() == key)
        .ok_or_else(|| invalid(format!("NWB dataset missing {key} source capsule")))?;
    opened
        .access()
        .bytes(capsule.content_id())
        .ok_or_else(|| invalid(format!("NWB dataset missing {key} payload bytes")))
}

/// Reconstruct an HDF5/NWB file from ABIR tensors and retained source capsules.
pub fn write_dataset(
    opened: &semantic_abir::OpenedDataset<semantic_abir::InMemoryPayloadAccess>,
    out: &Path,
) -> LmlResult<()> {
    let skel = capsule_bytes(opened, SKEL_KEY)?;
    let slots_blob = capsule_bytes(opened, SLOTS_KEY)?;
    let slots: Vec<NwbSlot> = serde_json::from_slice(slots_blob)
        .map_err(|e| LmlError::InvalidHeader(format!("slot decode: {e}")))?;

    std::fs::write(out, skel).map_err(LmlError::Io)?;
    let f = h5(File::open_rw(out), "open_rw output")?;
    for slot in &slots {
        let atom_id = slot.atom_id.0;
        let view = opened
            .tensor_view(atom_id)
            .map_err(|error| invalid(format!("NWB tensor {}: {error}", slot.h5_path)))?;
        if view.descriptor().element() != semantic_abir::ElementType::I64 {
            return Err(invalid(format!(
                "NWB tensor {} is not widened i64",
                slot.h5_path
            )));
        }
        let chunks = view.bytes().chunks_exact(8);
        if !chunks.remainder().is_empty() {
            return Err(invalid(format!(
                "NWB tensor {} byte length is not divisible by 8",
                slot.h5_path
            )));
        }
        let flat = chunks
            .map(|chunk| {
                let mut bytes = [0_u8; 8];
                bytes.copy_from_slice(chunk);
                i64::from_le_bytes(bytes)
            })
            .collect::<Vec<_>>();
        let expected = slot
            .orig_shape
            .iter()
            .try_fold(1_usize, |total, extent| total.checked_mul(*extent));
        if expected != Some(flat.len()) {
            return Err(invalid(format!(
                "NWB tensor {} has {} values, expected {:?}",
                slot.h5_path,
                flat.len(),
                expected
            )));
        }
        let ds = h5(f.dataset(&slot.h5_path), "output dataset")?;
        write_flat_i64(&ds, slot.int_bytes, slot.signed, &flat)?;
    }
    Ok(())
}
