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
use crate::source::{
    SemanticChannelMapping, SemanticFidelityReport, SemanticMappingReport, SemanticRead,
};
use hdf5_metno::types::{IntSize, TypeDescriptor};
use hdf5_metno::{Dataset, File, Group};
use semantic_abir as semantic;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;

/// Private parser result for one integer HDF5/NWB dataset.
#[derive(Debug, Clone)]
struct H5IntSignal {
    /// Full HDF5 path of the dataset, e.g. `/acquisition/ElectricalSeries/data`.
    h5_path: String,
    /// Channel-major signal `[n_ch][t]` — each inner `Vec` is one channel's
    /// time series (the codec's prediction axis).
    signal: Vec<Vec<i64>>,
    /// On-disk integer width in bytes: 1, 2, 4, or 8.
    int_bytes: u8,
    /// On-disk signedness (`false` ⇒ the source was an unsigned integer type).
    signed: bool,
    /// `true` when the source was a 2-D dataset stored time-major (rows = time,
    /// the NWB `ElectricalSeries` convention) and we transposed it to
    /// channel-major. `false` for 1-D datasets (a single channel, no transpose).
    time_major: bool,
    /// Original HDF5 dataset shape (pre-transpose), so a writer can restore the
    /// exact dimensionality and orientation.
    orig_shape: Vec<usize>,
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
fn read_int_signals(path: &Path) -> LmlResult<Vec<H5IntSignal>> {
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

// ── Zero-skeleton NWB ⇄ ABIR tensors (ADR 0051 Track 3, Phase B) ──────────────
//
// Ingest NWB/HDF5 into ABIR without a fragile structural transcoder. Tensor
// atoms carry integer datasets; a capsule carries the original file with those
// datasets
// **zeroed** (a "skeleton"). Zeros compress to ~nothing, so the skeleton adds
// little, yet it is a real HDF5 file — every group, attribute, float/compound
// dataset, and object reference survives untouched. Reconstruction writes the
// LML-decoded values back into the skeleton's (zeroed) datasets. The result is
// data-identical to the original, with no structural modelling and no
// double-storage of the signal.

/// Sidecar key: the original HDF5 with its integer datasets zeroed.
const SKEL_KEY: &str = "nwb_skeleton";
/// Sidecar key: JSON `[NwbSlot]` describing how to split `signal` back into the
/// skeleton's integer datasets.
const SLOTS_KEY: &str = "nwb_slots";

/// One integer dataset's placement: where it lives in the skeleton and which
/// span of mapped ABIR Tensor channels reconstructs it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NwbSlot {
    h5_path: String,
    int_bytes: u8,
    signed: bool,
    orig_shape: Vec<usize>,
    time_major: bool,
    /// First channel index in the mapped ABIR Tensor sequence.
    first_ch: usize,
    /// Number of channels this dataset contributes.
    n_ch: usize,
}

fn checked_narrow<T>(flat: &[i64], target: &str) -> LmlResult<Vec<T>>
where
    T: TryFrom<i64>,
{
    flat.iter()
        .copied()
        .map(|value| {
            T::try_from(value).map_err(|_| {
                LmlError::InvalidHeader(format!(
                    "NWB tensor value {value} is outside source {target} range"
                ))
            })
        })
        .collect()
}

/// Write `flat` (i64, row-major over the dataset's dims) into `ds`, narrowed to
/// the dataset's on-disk integer type without truncation.
fn write_flat_i64(ds: &Dataset, int_bytes: u8, signed: bool, flat: &[i64]) -> LmlResult<()> {
    macro_rules! w {
        ($t:ty) => {{
            let v: Vec<$t> = checked_narrow(flat, stringify!($t))?;
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
fn flatten_slot(chs: &[Vec<i64>], shape: &[usize], time_major: bool) -> LmlResult<Vec<i64>> {
    if shape.len() == 1 {
        if time_major || chs.len() != 1 || chs[0].len() != shape[0] {
            return Err(LmlError::InvalidHeader(
                "NWB rank-1 slot shape does not match its ABIR tensor".into(),
            ));
        }
        return Ok(chs[0].clone());
    }
    if shape.len() != 2 || !time_major {
        return Err(LmlError::InvalidHeader(
            "NWB slot shape/orientation is not canonical".into(),
        ));
    }
    let (t, c) = (shape[0], shape[1]);
    if chs.len() != c || chs.iter().any(|channel| channel.len() != t) {
        return Err(LmlError::InvalidHeader(
            "NWB rank-2 slot shape does not match its ABIR tensors".into(),
        ));
    }
    let capacity = t
        .checked_mul(c)
        .ok_or_else(|| LmlError::InvalidHeader("NWB slot extent overflow".into()))?;
    let mut flat = Vec::with_capacity(capacity);
    // dataset is (T, C): flat[t*C + c] = chs[c][t]
    for ti in 0..t {
        for ch in chs {
            flat.push(ch[ti]);
        }
    }
    Ok(flat)
}

/// Read HDF5/NWB integer datasets into one validated ABIR dataset.
///
/// Each flattened channel is a Tensor atom, so differing source shapes and
/// lengths remain honest rather than being forced through a uniform signal
/// carrier. A zeroed HDF5 skeleton and slot map remain content-addressed source
/// capsules for exact reconstruction.
pub fn read_semantic(path: &Path) -> LmlResult<SemanticRead> {
    let sigs = read_int_signals(path)?;
    if sigs.is_empty() {
        return Err(LmlError::InvalidHeader(
            "no little-endian integer datasets found to compress".into(),
        ));
    }

    let mut signal: Vec<Vec<i64>> = Vec::new();
    let mut slots: Vec<NwbSlot> = Vec::new();
    for s in sigs {
        let first_ch = signal.len();
        let n_ch = s.signal.len();
        slots.push(NwbSlot {
            h5_path: s.h5_path,
            int_bytes: s.int_bytes,
            signed: s.signed,
            orig_shape: s.orig_shape,
            time_major: s.time_major,
            first_ch,
            n_ch,
        });
        signal.extend(s.signal);
    }

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
    lower_nwb_tensors(path, signal, &slots, skel_bytes, slots_json)
}

/// Reconstruct HDF5/NWB from validated Tensor atoms and bound source capsules.
/// Data remains identical; HDF5 byte layout may differ.
pub fn write_semantic(read: &SemanticRead, out: &Path) -> LmlResult<()> {
    let skel = source_capsule_bytes(read, "source.nwb.capsule.0", SKEL_KEY, "application/x-hdf5")?;
    let slots_blob =
        source_capsule_bytes(read, "source.nwb.capsule.1", SLOTS_KEY, "application/json")?;
    let slots: Vec<NwbSlot> = serde_json::from_slice(slots_blob)
        .map_err(|e| LmlError::InvalidHeader(format!("slot decode: {e}")))?;
    let signal = signal_channels(read)?;

    let mut expected_first_channel = 0_usize;
    let mut dataset_paths = BTreeSet::new();
    let mut prepared = Vec::with_capacity(slots.len());
    for slot in &slots {
        if slot.first_ch != expected_first_channel
            || slot.n_ch == 0
            || !dataset_paths.insert(slot.h5_path.as_str())
        {
            return Err(LmlError::InvalidHeader(
                "NWB slot map is overlapping, gapped, empty, or duplicate".into(),
            ));
        }
        let end = slot
            .first_ch
            .checked_add(slot.n_ch)
            .ok_or_else(|| LmlError::InvalidHeader("NWB slot channel span overflow".into()))?;
        if end > signal.len() {
            return Err(LmlError::InvalidHeader(format!(
                "slot {} channel span {}..{} exceeds signal ({})",
                slot.h5_path,
                slot.first_ch,
                end,
                signal.len()
            )));
        }
        let chs = &signal[slot.first_ch..end];
        let flat = flatten_slot(chs, &slot.orig_shape, slot.time_major)?;
        prepared.push((slot, flat));
        expected_first_channel = end;
    }
    if expected_first_channel != signal.len() {
        return Err(LmlError::InvalidHeader(
            "NWB slot map does not cover every ABIR tensor".into(),
        ));
    }

    std::fs::write(out, skel).map_err(LmlError::Io)?;
    let f = h5(File::open_rw(out), "open_rw output")?;
    for (slot, flat) in prepared {
        let ds = h5(f.dataset(&slot.h5_path), "output dataset")?;
        write_flat_i64(&ds, slot.int_bytes, slot.signed, &flat)?;
    }
    Ok(())
}

/// Materialize NWB Tensor atoms as channel-major i64 vectors.
fn signal_channels(read: &SemanticRead) -> LmlResult<Vec<Vec<i64>>> {
    let mut signal = Vec::with_capacity(read.mapping.channels.len());
    for channel in &read.mapping.channels {
        let view = read
            .opened
            .block_view(channel.atom_id)
            .map_err(|error| LmlError::InvalidHeader(format!("NWB tensor payload: {error:?}")))?;
        let descriptor = view.descriptor();
        if descriptor.element() != semantic::ElementType::I64
            || descriptor.byte_order() != semantic::ByteOrder::Little
            || view.bytes().len() % 8 != 0
        {
            return Err(LmlError::InvalidHeader(
                "NWB tensor payload is not dense little-endian i64".into(),
            ));
        }
        signal.push(
            view.bytes()
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("exact chunk")))
                .collect(),
        );
    }
    Ok(signal)
}

fn source_capsule_bytes<'a>(
    read: &'a SemanticRead,
    namespace: &str,
    value: &str,
    media_type: &str,
) -> LmlResult<&'a [u8]> {
    let mut matches = read
        .opened
        .dataset()
        .source_capsules()
        .iter()
        .filter(|capsule| {
            capsule.source().namespace() == namespace
                && capsule.source().value() == value
                && capsule.media_type() == Some(media_type)
        });
    let capsule = matches.next().ok_or_else(|| {
        LmlError::InvalidHeader(format!("ABIR missing {namespace}:{value} source capsule"))
    })?;
    if matches.next().is_some() {
        return Err(LmlError::InvalidHeader(format!(
            "ABIR repeats {namespace}:{value} source capsule"
        )));
    }
    read.opened
        .access()
        .payload_bytes(capsule.content_id())
        .ok_or_else(|| {
            LmlError::InvalidHeader(format!("ABIR missing {namespace}:{value} capsule payload"))
        })
}

fn lower_nwb_tensors(
    path: &Path,
    signal: Vec<Vec<i64>>,
    slots: &[NwbSlot],
    skeleton: Vec<u8>,
    slots_json: Vec<u8>,
) -> LmlResult<SemanticRead> {
    let mut digest = Sha256::new();
    digest.update(b"org.quitetall.lamquant.nwb-tensors-v1\0");
    digest.update(&slots_json);
    let mut payloads = semantic::InMemoryPayloadAccess::new();
    let mut channel_payloads = Vec::with_capacity(signal.len());
    for channel in &signal {
        let mut bytes = Vec::with_capacity(channel.len().saturating_mul(8));
        for value in channel {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let content_id = semantic::payload_content_id(semantic::ElementType::I64, &bytes);
        digest.update(content_id.as_bytes());
        payloads.insert(content_id, bytes);
        channel_payloads.push(content_id);
    }
    let seed: [u8; 32] = digest.finalize().into();
    let dataset_id = nwb_id(&seed, b"dataset", 0);
    let recording_id = nwb_id(&seed, b"recording", 0);
    let stream_id = nwb_id(&seed, b"stream", 0);
    let basis_id = nwb_id(&seed, b"basis", 0);
    let mut draft = semantic::DatasetDraft::new(dataset_id);
    let mut recording = semantic::Recording::new(recording_id, vec![stream_id]);
    recording.add_source_key(
        semantic::SourceKey::new("source.format", "NWB")
            .map_err(|error| LmlError::InvalidHeader(error.to_string()))?,
    );
    recording.add_source_key(
        semantic::SourceKey::new(
            "source.file",
            path.file_name()
                .map(|value| value.to_string_lossy().into_owned())
                .unwrap_or_default(),
        )
        .map_err(|error| LmlError::InvalidHeader(error.to_string()))?,
    );
    draft.add_recording(recording);

    let mut atom_ids = Vec::with_capacity(signal.len());
    let mut mappings = Vec::with_capacity(signal.len());
    let mut channel_specs = Vec::with_capacity(signal.len());
    for (index, (channel, content_id)) in signal
        .iter()
        .zip(channel_payloads.iter().copied())
        .enumerate()
    {
        let atom_id = nwb_id(&seed, b"tensor", index as u64);
        let extent = u64::try_from(channel.len())
            .map_err(|_| LmlError::InvalidHeader("NWB tensor extent exceeds u64".into()))?;
        let logical_bytes = extent
            .checked_mul(8)
            .ok_or_else(|| LmlError::InvalidHeader("NWB tensor byte count overflow".into()))?;
        draft.add_atom(semantic::Atom::Tensor(semantic::Tensor::new(
            atom_id,
            semantic::Presence::Present,
            Some(semantic::PayloadDescriptor::new(
                content_id,
                logical_bytes,
                semantic::ElementType::I64,
                semantic::ByteOrder::Little,
                vec![extent],
                semantic::Layout::DenseRowMajor,
                None,
                None,
            )),
            vec![semantic::SemanticAxis::new(
                semantic::ConceptId::new("abir:axis/value").expect("static concept is canonical"),
                extent,
            )],
        )));
        atom_ids.push(atom_id);
        mappings.push(SemanticChannelMapping {
            index,
            atom_id,
            content_id,
        });
        let slot = slots
            .iter()
            .find(|slot| index >= slot.first_ch && index < slot.first_ch + slot.n_ch)
            .ok_or_else(|| LmlError::InvalidHeader("NWB channel has no slot mapping".into()))?;
        let mut spec = semantic::ChannelSpec::new(
            semantic::ConceptId::new(format!("source:nwb/channel/{index}"))
                .map_err(|error| LmlError::InvalidHeader(error.to_string()))?,
        );
        for (namespace, value) in [
            ("source.nwb.h5-path", slot.h5_path.clone()),
            (
                "source.nwb.original-shape",
                slot.orig_shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("x"),
            ),
            ("source.nwb.time-major", slot.time_major.to_string()),
        ] {
            spec = spec.with_source_key(
                semantic::SourceKey::new(namespace, value)
                    .map_err(|error| LmlError::InvalidHeader(error.to_string()))?,
            );
        }
        channel_specs.push(spec);
    }
    draft.add_stream(semantic::Stream::new(
        stream_id,
        recording_id,
        semantic::ConceptId::new("abir:modality/unknown").expect("static concept is canonical"),
        atom_ids,
        None,
        Some(basis_id),
        None,
    ));
    draft.add_channel_basis(semantic::ChannelBasis::new(
        basis_id,
        channel_specs,
        semantic::ReferenceKind::Unknown,
    ));

    for (index, (key, bytes, media_type)) in [
        (SKEL_KEY, skeleton, "application/x-hdf5"),
        (SLOTS_KEY, slots_json, "application/json"),
    ]
    .into_iter()
    .enumerate()
    {
        let content_id = semantic::payload_content_id(semantic::ElementType::Bytes, &bytes);
        payloads.insert(content_id, bytes);
        draft.add_source_capsule(semantic::SourceCapsule::new(
            semantic::SourceKey::new(format!("source.nwb.capsule.{index}"), key)
                .map_err(|error| LmlError::InvalidHeader(error.to_string()))?,
            content_id,
            Some(media_type),
        ));
    }
    let dataset = draft
        .validate(semantic::ValidationLimits::default())
        .map_err(|report| {
            LmlError::InvalidHeader(format!("NWB semantic validation failed: {report:?}"))
        })?;
    Ok(SemanticRead::from_opened_dataset(
        semantic::OpenedDataset::new(dataset, payloads),
        SemanticMappingReport {
            source_format: "NWB".into(),
            recording_count: 1,
            stream_count: 1,
            channel_count: mappings.len(),
            source_capsule_count: 2,
            channels: mappings,
            events: Vec::new(),
        },
        SemanticFidelityReport {
            sample_values_exact: true,
            sample_rate_exact: false,
            channel_order_exact: true,
            labels_exact: false,
            physical_ranges_preserved_as_source_keys: false,
            calibration_promoted: false,
            source_capsules_content_bound: true,
        },
    ))
}

fn nwb_id<T>(seed: &[u8; 32], domain: &[u8], index: u64) -> semantic::ObjectId<T> {
    let mut digest = Sha256::new();
    digest.update(b"org.quitetall.lamquant.nwb-object-v1\0");
    digest.update(seed);
    digest.update(domain);
    digest.update(index.to_le_bytes());
    let bytes: [u8; 32] = digest.finalize().into();
    let mut id = [0_u8; 16];
    id.copy_from_slice(&bytes[..16]);
    semantic::ObjectId::from_bytes(id)
}

#[cfg(test)]
mod tests {
    use super::{checked_narrow, flatten_slot};

    #[test]
    fn flatten_slot_restores_time_major_storage() {
        let channels = vec![vec![1, 2, 3], vec![10, 20, 30]];
        assert_eq!(
            flatten_slot(&channels, &[3, 2], true).unwrap(),
            vec![1, 10, 2, 20, 3, 30]
        );
    }

    #[test]
    fn flatten_slot_rejects_shape_and_orientation_mismatch() {
        assert!(flatten_slot(&[vec![1, 2]], &[2], true).is_err());
        assert!(flatten_slot(&[vec![1], vec![2]], &[2, 2], true).is_err());
        assert!(flatten_slot(&[vec![1, 2]], &[2, 1], false).is_err());
        assert!(flatten_slot(&[vec![1, 2]], &[1, 1, 2], true).is_err());
    }

    #[test]
    fn integer_writeback_rejects_lossy_narrowing() {
        assert_eq!(
            checked_narrow::<i16>(&[i64::from(i16::MIN), 0, i64::from(i16::MAX)], "i16").unwrap(),
            [i16::MIN, 0, i16::MAX]
        );
        assert!(checked_narrow::<i16>(&[i64::from(i16::MAX) + 1], "i16").is_err());
        assert!(checked_narrow::<u8>(&[-1], "u8").is_err());
        assert!(checked_narrow::<u8>(&[256], "u8").is_err());
    }
}
