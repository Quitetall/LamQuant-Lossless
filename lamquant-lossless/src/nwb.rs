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
use crate::source::{SidecarBlob, SignalBundle, SourceMetadata};
use hdf5_metno::types::{IntSize, TypeDescriptor};
use hdf5_metno::{Dataset, File, Group};
use serde::{Deserialize, Serialize};
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
    if shape.is_empty() || shape.len() > 2 || shape.iter().any(|&d| d == 0) {
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

// ── Zero-skeleton NWB ⇄ SignalBundle (ADR 0051 Track 3, Phase B) ──────────────
//
// Ingest an NWB/HDF5 into the codec's `SignalBundle` IR without a fragile
// structural transcoder. The trick: the bundle's `signal` carries the integer
// datasets (→ LML); a sidecar carries the original file with those datasets
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
/// span of `SignalBundle.signal` channels reconstructs it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NwbSlot {
    h5_path: String,
    int_bytes: u8,
    signed: bool,
    orig_shape: Vec<usize>,
    time_major: bool,
    /// First channel index in `SignalBundle.signal` for this dataset.
    first_ch: usize,
    /// Number of channels this dataset contributes.
    n_ch: usize,
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
        _ => return Err(LmlError::InvalidHeader(format!("unsupported int width {int_bytes}"))),
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

/// Read an HDF5/NWB file into a [`SignalBundle`]: integer datasets become
/// `signal` (for LML), and a zeroed copy of the whole file plus a slot map go
/// into the sidecar (see module note). The original is never mutated.
///
/// Note: the bundle may be *ragged* (integer datasets of differing length), so
/// callers must NOT assume `SignalBundle::validate`'s equal-length invariant.
pub fn read_bundle(path: &Path) -> LmlResult<SignalBundle> {
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

    let n = signal.len();
    let slots_json = serde_json::to_vec(&slots)
        .map_err(|e| LmlError::InvalidHeader(format!("slot encode: {e}")))?;
    Ok(SignalBundle {
        signal,
        sample_rate: 0.0,
        channels: (0..n).map(|i| format!("ch{i}")).collect(),
        phys_min: vec![0.0; n],
        phys_max: vec![0.0; n],
        duration_s: 0.0,
        metadata: SourceMetadata {
            source_file: path.display().to_string(),
            format: "NWB".into(),
            patient_id: String::new(),
            recording_info: String::new(),
            startdate: String::new(),
            phys_dim: String::new(),
        },
        sidecar: vec![
            SidecarBlob { key: SKEL_KEY.into(), bytes: skel_bytes, aux: None },
            SidecarBlob { key: SLOTS_KEY.into(), bytes: slots_json, aux: None },
        ],
    })
}

/// Reconstruct an HDF5/NWB file at `out` from a [`SignalBundle`] produced by
/// [`read_bundle`]: write the skeleton, then refill each integer dataset with
/// its `signal` channels. Data-identical to the original (HDF5 byte layout may
/// differ, as HDF5 permits).
pub fn write_bundle(bundle: &SignalBundle, out: &Path) -> LmlResult<()> {
    let skel = bundle
        .sidecar_first(SKEL_KEY)
        .ok_or_else(|| LmlError::InvalidHeader("bundle missing nwb_skeleton sidecar".into()))?;
    let slots_blob = bundle
        .sidecar_first(SLOTS_KEY)
        .ok_or_else(|| LmlError::InvalidHeader("bundle missing nwb_slots sidecar".into()))?;
    let slots: Vec<NwbSlot> = serde_json::from_slice(&slots_blob.bytes)
        .map_err(|e| LmlError::InvalidHeader(format!("slot decode: {e}")))?;

    std::fs::write(out, &skel.bytes).map_err(LmlError::Io)?;
    let f = h5(File::open_rw(out), "open_rw output")?;
    for slot in &slots {
        let end = slot.first_ch + slot.n_ch;
        if end > bundle.signal.len() {
            return Err(LmlError::InvalidHeader(format!(
                "slot {} channel span {}..{} exceeds signal ({})",
                slot.h5_path,
                slot.first_ch,
                end,
                bundle.signal.len()
            )));
        }
        let chs = &bundle.signal[slot.first_ch..end];
        let flat = flatten_slot(chs, &slot.orig_shape, slot.time_major);
        let ds = h5(f.dataset(&slot.h5_path), "output dataset")?;
        write_flat_i64(&ds, slot.int_bytes, slot.signed, &flat)?;
    }
    Ok(())
}
