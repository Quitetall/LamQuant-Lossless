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
