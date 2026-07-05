use abir::{
    name_for_tag, Abir, Accel, Bcs1Header, Channel, Column, Ecg, Ecog, Eeg, Emg, Eog, Ieeg,
    Modality, ModalitySource, Other, Resp, Seeg, Untyped, BCS1_MAGIC,
};
use std::sync::Arc;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict};

fn extract_bytes(data: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = data.downcast::<PyBytes>() {
        Ok(b.as_bytes().to_vec())
    } else if let Ok(b) = data.downcast::<PyByteArray>() {
        Ok(unsafe { b.as_bytes().to_vec() })
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected bytes or bytearray",
        ))
    }
}

/// Map an LMA read error to the right Python exception via the classifier that
/// lives in `lml::lma` (co-located with the error strings — single source of
/// truth, MiMo #16). Replaces the substring-matching that had been duplicated,
/// with divergent substring sets, across the LMA pyfunctions. `NotFound` →
/// `KeyError`, `InvalidArchive` (bad magic / unsupported / corrupt / oversized
/// manifest) → `ValueError`, everything else → `IOError`.
fn lma_err_to_py(e: Box<dyn std::error::Error + Send + Sync>) -> PyErr {
    let msg = e.to_string();
    match lml::lma::classify_error(&*e) {
        lml::lma::LmaErrorKind::NotFound => pyo3::exceptions::PyKeyError::new_err(msg),
        lml::lma::LmaErrorKind::InvalidArchive => pyo3::exceptions::PyValueError::new_err(msg),
        lml::lma::LmaErrorKind::Io => pyo3::exceptions::PyIOError::new_err(msg),
    }
}

#[pyfunction]
fn golomb_encode_dense<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let slice = coeffs.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array must be contiguous: {e}"))
    })?;
    let result = lml::golomb::encode_dense(slice).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("{}", e))
    })?;
    Ok(PyBytes::new(py, &result))
}

#[pyfunction]
fn golomb_decode_dense<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    offset: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, usize)> {
    let bytes = extract_bytes(data)?;
    let (values, consumed) = lml::golomb::decode_dense(&bytes, offset)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    Ok((PyArray1::from_vec(py, values), consumed))
}

#[pyfunction]
fn rans_encode<'py>(
    py: Python<'py>,
    symbols: PyReadonlyArray1<'py, i64>,
    freq: PyReadonlyArray1<'py, i32>,
    start: PyReadonlyArray1<'py, i32>,
    m: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let sym = symbols.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array must be contiguous: {e}"))
    })?;
    let f = freq.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array must be contiguous: {e}"))
    })?;
    let s = start.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array must be contiguous: {e}"))
    })?;
    let result = lml::rans::encode(sym, f, s, m)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    Ok(PyBytes::new(py, &result))
}

#[pyfunction]
fn rans_decode<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    freq: PyReadonlyArray1<'py, i32>,
    start: PyReadonlyArray1<'py, i32>,
    cum2sym: PyReadonlyArray1<'py, i32>,
    m: u64,
    n_symbols: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let bytes = extract_bytes(data)?;
    let f = freq.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array must be contiguous: {e}"))
    })?;
    let s = start.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array must be contiguous: {e}"))
    })?;
    let c = cum2sym.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array must be contiguous: {e}"))
    })?;
    let result = lml::rans::decode(&bytes, f, s, c, m, n_symbols)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    Ok(PyArray1::from_vec(py, result))
}

/// Compress [n_ch][T] signal → LML1 packet bytes.
///
/// Raises `ValueError` on invalid header dimensions (Fix-C3).
#[pyfunction]
fn lml_compress<'py>(
    py: Python<'py>,
    signal: Vec<Vec<i64>>,
    noise_bits: u8,
) -> PyResult<Bound<'py, PyBytes>> {
    let result = lml::lml::compress(&signal, noise_bits)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &result))
}

/// Decompress LML1 packet bytes → list of list of i64.
#[pyfunction]
fn lml_decompress(data: &[u8]) -> PyResult<Vec<Vec<i64>>> {
    lml::lml::decompress(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Write LML container file from signal.
#[pyfunction]
fn container_write(
    path: &str,
    signal: Vec<Vec<i64>>,
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
) -> PyResult<()> {
    lml::container::write_file(
        std::path::Path::new(path),
        &signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
    )
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(())
}

/// Read LML container file → (signal, metadata_json).
#[pyfunction]
fn container_read(path: &str) -> PyResult<(Vec<Vec<i64>>, String)> {
    lml::container::read_file(std::path::Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Read LML container from in-memory bytes → (signal, metadata_json).
///
/// Used by the LMA-direct training dataloader: the .lml bytes are
/// extracted from an LMA archive via `lma_read_entry` and decoded
/// here without a tempfile round-trip.
#[pyfunction]
fn container_read_bytes(data: &[u8]) -> PyResult<(Vec<Vec<i64>>, String)> {
    lml::container::read_bytes(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Parse the LML/BCS1 container header from in-memory bytes — metadata
/// only, no window decode.
///
/// Returns `(metadata_json, n_ch, n_windows, total_samples,
/// window_size)`. Cheap: a fixed-size header read (legacy 32/20/18-byte
/// LML1 or the 40-byte BCS1 header — `lml::container::parse_header`
/// dispatches on magic, task #34) + UTF-8 metadata parse. Use this when
/// you need only the calibration / sample-rate dict, not the signal —
/// saves the ~5 MB / ~50 ms cost of decoding a window just for its
/// metadata side-channel.
#[pyfunction]
fn container_metadata(data: &[u8]) -> PyResult<(String, usize, usize, usize, usize)> {
    let header = lml::container::parse_header(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((
        header.metadata,
        header.n_ch,
        header.n_windows,
        header.total_samples,
        header.window_size,
    ))
}

/// Decode entire LML container into a `[n_ch, total_samples]` float32
/// PyArray, applying per-channel digital→physical calibration in Rust.
///
/// `calib_f32` must be a contiguous `[n_ch, 4]` array carrying
/// `(dig_min, dig_max, phys_min, phys_max)` per channel in that
/// order, dtype float32.
///
/// Returns `(signal[n_ch, total_samples] float32, metadata_json,
/// n_windows)`.
#[pyfunction]
fn container_read_phys_f32<'py>(
    py: Python<'py>,
    data: &[u8],
    calib_f32: PyReadonlyArray1<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, String, usize)> {
    let dims = lml::container::parse_header(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let n_ch = dims.n_ch;
    let total = dims.total_samples;

    let calib_slice = calib_f32.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("calib not contiguous: {e:?}"))
    })?;
    if calib_slice.len() != n_ch * 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "calib length {} != n_ch*4 ({})",
            calib_slice.len(),
            n_ch * 4
        )));
    }

    let arr = PyArray2::<f32>::zeros(py, [n_ch, total], false);
    // Safe: arr just allocated, no aliasing, GIL held, contiguous
    // C-order from `zeros`. GIL must stay held while the &mut [f32]
    // aliases the PyArray2 heap buffer (no py.allow_threads here).
    let header = unsafe {
        let out = arr.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("PyArray slice: {e:?}"))
        })?;
        lml::container::read_bytes_into_f32_calibrated(data, out, calib_slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };

    Ok((arr, header.metadata, header.n_windows))
}

/// Selected-channel calibrated f32 decode (ADR 0075 A2). Decodes only `channel_mask`'s
/// channels into a fresh `[n_sel, total]` f32 array (in selected order), skipping the
/// full `[n_ch, total]` array `container_read_phys_f32` builds AND the downstream
/// channel-select copy. `calib_f32` is `[n_sel*4]` in selected order; `channel_mask[sel]`
/// is the source channel index. Returns `(array[n_sel, total], metadata, n_windows)`.
#[pyfunction]
fn container_read_phys_selected<'py>(
    py: Python<'py>,
    data: &[u8],
    calib_f32: PyReadonlyArray1<'py, f32>,
    channel_mask: Vec<u16>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, String, usize)> {
    let dims = lml::container::parse_header(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let total = dims.total_samples;
    let n_sel = channel_mask.len();

    let calib_slice = calib_f32.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("calib not contiguous: {e:?}"))
    })?;
    if calib_slice.len() != n_sel * 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "calib length {} != n_sel*4 ({})",
            calib_slice.len(),
            n_sel * 4
        )));
    }

    let arr = PyArray2::<f32>::zeros(py, [n_sel, total], false);
    // Safe: freshly allocated, no aliasing, GIL held — the &mut [f32] aliases the
    // PyArray2 heap buffer, so no py.allow_threads while it is live.
    let header = unsafe {
        let out = arr.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("PyArray slice: {e:?}"))
        })?;
        lml::container::read_bytes_into_f32_calibrated_selected(data, out, calib_slice, &channel_mask)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };

    Ok((arr, header.metadata, header.n_windows))
}

/// mmap + selected decode in one call (ADR 0075 A3). Opens the `.lma` as a memory map,
/// borrows the entry's compressed bytes (NO ~1.4 GB copy — contrast `lma_read_entry` +
/// `container_read_phys_f32`), and decodes only `channel_mask`'s channels into
/// `[n_sel, total]` f32. The mmap borrow never crosses to Python. This is what the
/// canonical training decode calls to collapse the fallback peak: it skips both the
/// compressed-bytes copy and the all-channel `[n_ch, total]` array. `channel_mask[sel]
/// == 65535` (u16::MAX) marks a missing target → that output row is zero-filled.
#[pyfunction]
fn lma_mmap_read_phys_selected<'py>(
    py: Python<'py>,
    lma_path: &str,
    entry_name: &str,
    calib_f32: PyReadonlyArray1<'py, f32>,
    channel_mask: Vec<u16>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, String, usize)> {
    let archive =
        lml::lma::MmapArchive::open(std::path::Path::new(lma_path)).map_err(lma_err_to_py)?;
    let data = archive.entry_bytes(entry_name).map_err(lma_err_to_py)?;
    let dims = lml::container::parse_header(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let total = dims.total_samples;
    let n_sel = channel_mask.len();

    let calib_slice = calib_f32.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("calib not contiguous: {e:?}"))
    })?;
    if calib_slice.len() != n_sel * 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "calib length {} != n_sel*4 ({})",
            calib_slice.len(),
            n_sel * 4
        )));
    }

    let arr = PyArray2::<f32>::zeros(py, [n_sel, total], false);
    // Safe: freshly allocated, no aliasing, GIL held while the &mut [f32] aliases the
    // PyArray2 heap buffer (no allow_threads). `archive` outlives the borrowed `data`.
    let header = unsafe {
        let out = arr.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("PyArray slice: {e:?}"))
        })?;
        lml::container::read_bytes_into_f32_calibrated_selected(data, out, calib_slice, &channel_mask)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };

    Ok((arr, header.metadata, header.n_windows))
}

/// mmap header/metadata peek (ADR 0075 A3): open the `.lma` map and parse just an entry's
/// container header + metadata (touches only the header pages — no decode, no full-entry
/// copy). Returns `(metadata_json, n_ch, n_windows, total_samples, window_size)`, the same
/// shape as `container_metadata`, so the caller can resolve channels + build calibration
/// BEFORE decoding the selected channels from the same map.
#[pyfunction]
fn lma_mmap_entry_metadata(
    lma_path: &str,
    entry_name: &str,
) -> PyResult<(String, usize, usize, usize, usize)> {
    let archive =
        lml::lma::MmapArchive::open(std::path::Path::new(lma_path)).map_err(lma_err_to_py)?;
    let data = archive.entry_bytes(entry_name).map_err(lma_err_to_py)?;
    let header = lml::container::parse_header(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((
        header.metadata,
        header.n_ch,
        header.n_windows,
        header.total_samples,
        header.window_size,
    ))
}

/// Random-access read of one window from an in-memory LML container.
///
/// Returns `(window[n_ch, window_size_actual] int64, metadata_json,
/// n_windows)`. The trailing window may be shorter than `window_size`
/// if `total_samples` is not a multiple of it.
#[pyfunction]
fn container_read_window_np<'py>(
    py: Python<'py>,
    data: &[u8],
    window_idx: usize,
) -> PyResult<(Bound<'py, PyArray2<i64>>, String, usize)> {
    let (window, header) = lml::container::read_window_from_bytes(data, window_idx)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let n_ch = window.len();
    let t = window.first().map(|c| c.len()).unwrap_or(0);
    let arr = PyArray2::<i64>::zeros(py, [n_ch, t], false);
    // Safe: arr was just allocated, no aliasing, GIL held, layout
    // is contiguous C-order from `zeros`.
    unsafe {
        let slice = arr.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("PyArray slice: {e:?}"))
        })?;
        for (ch, row) in window.iter().enumerate() {
            let copy = row.len().min(t);
            let dst_off = ch * t;
            slice[dst_off..dst_off + copy].copy_from_slice(&row[..copy]);
        }
    }
    Ok((arr, header.metadata, header.n_windows))
}

/// Read a single named entry from an LMA archive without unpacking.
///
/// Returns the decompressed bytes for the entry:
///   - `store` → raw stored bytes (use this for `<stem>.lml` entries)
///   - `secondary`/`zstd` → zstd-decompressed bytes
///
/// Raises `IOError` on missing archive / read failure, `KeyError` if
/// the entry name is not in the manifest, `ValueError` on a corrupt
/// archive.
#[pyfunction]
fn lma_read_entry<'py>(
    py: Python<'py>,
    archive_path: &str,
    entry_name: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = lml::lma::read_entry(std::path::Path::new(archive_path), entry_name)
        .map_err(lma_err_to_py)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Batch ranged-header read across many entries in one LMA archive.
///
/// Parses the archive index ONCE, then reads only the first
/// `n_bytes` of each named entry's payload — for raw-stored tiers
/// only (`lml` / `store`). Returns a list aligned 1:1 with `names`:
///   - `bytes` — entry found, raw tier, prefix read OK
///   - `None` — not in manifest, compressed entry, or out-of-bounds
///
/// Raises `IOError` on missing archive / read failure, `ValueError`
/// on a corrupt archive.
#[pyfunction]
fn lma_entry_headers<'py>(
    py: Python<'py>,
    archive_path: &str,
    names: Vec<String>,
    n_bytes: usize,
) -> PyResult<Vec<Option<Bound<'py, PyBytes>>>> {
    let results = lml::lma::read_entry_headers_path(
        std::path::Path::new(archive_path),
        &names,
        n_bytes,
    )
    .map_err(lma_err_to_py)?;
    Ok(results
        .into_iter()
        .map(|opt| opt.map(|b| PyBytes::new(py, &b)))
        .collect())
}

/// Write a channel-agnostic neural container (.lmq / LMQC) carrying the
/// per-recording montage so the decoder can reconstruct N channels
/// off-device. `coords`: flat row-major `[N*3]` f32 (NaN = unknown) or None;
/// `channels`: per-channel names or None; `payload`: encoded latent bytes;
/// `payload_kind`: 0 = fp16-latent (1 = FSQ tokens, reserved).
#[pyfunction]
#[pyo3(signature = (path, n_channels, latent_c, latent_t, sample_rate,
                    window_samples, payload_kind, payload, coords=None, channels=None))]
#[allow(clippy::too_many_arguments)]
fn write_ca_lmq(
    path: &str,
    n_channels: u16,
    latent_c: u16,
    latent_t: u16,
    sample_rate: u16,
    window_samples: u32,
    payload_kind: u8,
    payload: &[u8],
    coords: Option<Vec<f32>>,
    channels: Option<Vec<String>>,
) -> PyResult<()> {
    let buf = lml::lmqc::encode_lmqc(
        n_channels, latent_c, latent_t, sample_rate, window_samples,
        payload_kind, coords.as_deref(), channels.as_deref(), payload,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("LMQC encode: {:?}", e)))?;
    std::fs::write(path, &buf)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Read an LMQC `.lmq` container → dict with montage + payload. Verifies
/// magic + CRC. `coords` is a flat `[N*3]` list (or None); `channels` is a
/// `list[str]` (or None); `payload` is bytes (caller decodes per
/// `payload_kind`). Raises on corruption / bad magic / version.
#[pyfunction]
fn read_ca_lmq<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let buf = std::fs::read(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let c = lml::lmqc::decode_lmqc(&buf)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("LMQC decode: {:?}", e)))?;
    let d = PyDict::new(py);
    d.set_item("version", c.version)?;
    d.set_item("n_channels", c.n_channels)?;
    d.set_item("latent_c", c.latent_c)?;
    d.set_item("latent_t", c.latent_t)?;
    d.set_item("sample_rate", c.sample_rate)?;
    d.set_item("window_samples", c.window_samples)?;
    d.set_item("payload_kind", c.payload_kind)?;
    d.set_item("coords", c.coords)?;
    d.set_item("channels", c.channels)?;
    d.set_item("payload", PyBytes::new(py, &c.payload))?;
    Ok(d)
}

/// A typed, verifier-checked handle to a decoded ABIR recording (ADR 0069 S7a).
///
/// Wraps `Abir<Untyped>` — the born-typed modality lives in the recording's
/// provenance (`prov.tag`), assigned at read time by label inference. The
/// modality accessors (`.eeg()`, `.ecg()`, …) are the sanctioned way to get
/// sample arrays out: each runs the ABIR verifier's `try_into_modality` check
/// and raises `ValueError` on a modality mismatch, so an ECG recording can
/// never be pulled into an EEG training path (and vice versa) by accident. The
/// one modality-BLIND escape hatch, `.samples_i64()`, is explicit and named.
///
/// NB (S7a scope): a mixed-modality (PSG) file types as its majority modality;
/// per-modality sub-views (`view::<M>()`, ADR 0069 criterion (b)) are deferred,
/// so `.eeg()` on a PSG file would return every channel. Single-modality files
/// (the common case) are fully enforced.
#[pyclass]
struct PyAbir {
    inner: Abir<Untyped>,
}

impl PyAbir {
    /// Materialize a recording as a `[n_ch, n_samples]` int64 PyArray. The
    /// modality marker `M` is only a compile-time witness that the caller
    /// already passed the verifier — the samples are modality-blind (the same
    /// `window_views` egress the encoder uses).
    fn to_i64_array<'py, M: Modality>(
        py: Python<'py>,
        abir: &Abir<M>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let n_ch = abir.n_channels();
        let t = abir.n_samples();
        let arr = PyArray2::<i64>::zeros(py, [n_ch, t], false);
        let cows = abir.window_views(0, t);
        // Only `as_slice_mut` is the unsafe intrinsic: safe here because `arr`
        // was just allocated (no aliasing), the GIL is held, and `zeros` gives a
        // contiguous C-order buffer. The fill loop below operates on the safe
        // `&mut [i64]`.
        let slice = unsafe { arr.as_slice_mut() }.map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("PyArray slice: {e:?}"))
        })?;
        for (ch, cow) in cows.iter().enumerate() {
            let row = cow.as_ref();
            // All channels of a verified `Abir` share `n_samples`, so `copy == t`
            // in practice; the `.min` is a defensive clamp (never truncates real,
            // uniform-length data) so a malformed handle can't index OOB.
            let copy = row.len().min(t);
            let dst = ch * t;
            slice[dst..dst + copy].copy_from_slice(&row[..copy]);
        }
        Ok(arr)
    }

    /// Verifier-checked promotion + materialization for one modality. Clones the
    /// inner `Abir` (pymethods take `&self`; `try_into_modality` consumes),
    /// checks `prov.tag == M::TAG`, runs the full `verify()`, then materializes.
    fn typed<'py, M: Modality>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let typed = self
            .inner
            .clone()
            .try_into_modality::<M>()
            .map_err(|(_orig, e)| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "ABIR modality mismatch: recording is not {} ({e})",
                    M::NAME
                ))
            })?;
        typed
            .verify()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Self::to_i64_array(py, &typed)
    }
}

#[pymethods]
impl PyAbir {
    /// The born-typed modality name ("eeg"/"ecg"/…/"untyped"), from provenance.
    /// Uses `abir::name_for_tag` (single source of truth) — an unknown
    /// tag reports "unknown" rather than a stale hard-coded name.
    fn modality(&self) -> &'static str {
        name_for_tag(self.inner.provenance().tag).unwrap_or("unknown")
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAbir(modality={:?}, n_channels={}, n_samples={}, sample_rate={})",
            self.modality(),
            self.inner.n_channels(),
            self.inner.n_samples(),
            self.inner.sample_rate,
        )
    }

    /// How the modality was decided ("channel_label"/"format_declared"/"manual").
    fn modality_source(&self) -> &'static str {
        match self.inner.provenance().source {
            ModalitySource::ChannelLabel => "channel_label",
            ModalitySource::FormatDeclared => "format_declared",
            ModalitySource::Manual => "manual",
        }
    }

    fn n_channels(&self) -> usize {
        self.inner.n_channels()
    }

    fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    fn sample_rate(&self) -> f64 {
        self.inner.sample_rate
    }

    fn channels(&self) -> Vec<String> {
        self.inner
            .channels
            .iter()
            .map(|c| c.label.to_string())
            .collect()
    }

    /// EEG samples `[n_ch, n_samples]` int64 — raises `ValueError` if the
    /// recording is not born-typed EEG. Same shape for every modality accessor.
    fn eeg<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.typed::<Eeg>(py)
    }
    fn ieeg<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.typed::<Ieeg>(py)
    }
    fn ecog<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.typed::<Ecog>(py)
    }
    fn seeg<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.typed::<Seeg>(py)
    }
    fn ecg<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.typed::<Ecg>(py)
    }
    fn emg<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.typed::<Emg>(py)
    }
    fn eog<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.typed::<Eog>(py)
    }
    fn resp<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.typed::<Resp>(py)
    }
    fn accel<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.typed::<Accel>(py)
    }
    fn other<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.typed::<Other>(py)
    }

    /// The one sanctioned modality-BLIND egress: samples `[n_ch, n_samples]`
    /// int64 with no modality check. Named + explicit so it can never happen by
    /// accident — the trust model lives at the typed accessors above.
    fn samples_i64<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        Self::to_i64_array(py, &self.inner)
    }
}

/// Build a born-typed `PyAbir` from a decoded container's raw bytes + `(signal,
/// metadata_json)`. Channel labels come from the metadata JSON; sample_rate is
/// taken from the authoritative BCS1 header when present (else the JSON);
/// modality is inferred from the labels (`with_inferred_modality`) — the same
/// deterministic inference the reader ran at born-typing, so `.eeg()`/`.ecg()`
/// check against the modality the file was written as.
fn build_pyabir(data: &[u8], signal: Vec<Vec<i64>>, metadata_json: &str) -> PyResult<PyAbir> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("metadata JSON: {e}")))?;
    // sample_rate: prefer the authoritative BCS1 header field (write_abir stores
    // it there as milli-Hz, NOT in the JSON), fall back to the metadata JSON
    // (legacy LML1 carries it only there), then validate. A zero/absent rate is a
    // broken container, not a usable default — it would silently break any
    // Hz-normalizing downstream pass (e.g. S7b resampling).
    let header_sr = if data.len() >= 4 && data[0..4] == *BCS1_MAGIC {
        Bcs1Header::parse(data)
            .ok()
            .map(|h| h.sample_rate_mhz as f64 / 1000.0)
    } else {
        None
    };
    let sample_rate = header_sr
        .filter(|r| *r > 0.0)
        .or_else(|| meta.get("sample_rate").and_then(|v| v.as_f64()).filter(|r| *r > 0.0))
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "container has no valid sample_rate (absent/zero in both the BCS1 header and metadata JSON)",
            )
        })?;
    let labels: Vec<String> = meta
        .get("channels")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|x| x.as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default();
    // Optional per-channel physical calibration; the i64 samples are the codec
    // currency and are unaffected, so absent → 0.0 (matches `from_channels_i64`).
    let phys_min = meta.get("phys_min").and_then(|v| v.as_array());
    let phys_max = meta.get("phys_max").and_then(|v| v.as_array());
    let at = |arr: Option<&Vec<serde_json::Value>>, i: usize| -> f64 {
        arr.and_then(|a| a.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0)
    };

    // Build labeled `Channel`s from the metadata (not `from_channels_i64`, which
    // leaves labels empty) so `.channels()` round-trips and born-typed inference
    // reads the same labels the reader used at write time.
    let n_samples = signal.first().map(|c| c.len()).unwrap_or(0);
    let channels: Vec<Channel> = signal
        .into_iter()
        .enumerate()
        .map(|(i, ch)| Channel {
            label: labels
                .get(i)
                .map(|s| Arc::from(s.as_str()))
                .unwrap_or_else(|| Arc::from("")),
            data: Column::I64(Arc::from(ch)),
            phys_min: at(phys_min, i),
            phys_max: at(phys_max, i),
        })
        .collect();

    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    let inner =
        Abir::from_parts(channels, sample_rate, n_samples).with_inferred_modality(&label_refs, None);
    Ok(PyAbir { inner })
}

/// Read an LML/BCS1 container file → a typed `PyAbir` (ADR 0069 S7a). Unlike
/// `container_read` (a raw `(signal, metadata)` tuple), this returns a handle
/// whose `.eeg()`/`.ecg()` accessors enforce modality at the boundary.
#[pyfunction]
fn container_read_abir(path: &str) -> PyResult<PyAbir> {
    // Read the bytes once, decode via the BCS1-aware `read_bytes` dispatch, and
    // keep the bytes so `build_pyabir` can read the authoritative header sample_rate.
    let data = std::fs::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let (signal, metadata) = lml::container::read_bytes(&data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    build_pyabir(&data, signal, &metadata)
}

/// Read an LML/BCS1 container from in-memory bytes → a typed `PyAbir` (ADR 0069
/// S7a). BCS1-aware via the L9/#34 magic dispatch.
#[pyfunction]
fn container_read_bytes_abir(data: &[u8]) -> PyResult<PyAbir> {
    let (signal, metadata) = lml::container::read_bytes(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    build_pyabir(data, signal, &metadata)
}

/// ADR 0069 S7b: the LMQ training normalization DSP, in Rust. Takes the
/// already-channel-selected `[n_ch, T]` float32 signal (channel-select is a
/// pure gather with no parity concern, so it stays in Python) + its original
/// sample rate, and returns the `[n_ch, T']` float32 array `decode_lma_signal`
/// produces: resample→250 → 0.5 Hz zero-phase HP → Q31 → the f32 round-trip.
///
/// Returns `None` for an all-flat signal (matches the Python `max_abs < 1e-12`
/// skip). Raises `NotImplementedError` for a sample rate that needs the FFT
/// resample branch (not yet ported) so the caller can fall back to the Python
/// path. Bit-exact to `decode_lma_signal`'s tail (parity-gated in
/// `lamquant-lossless/tests/normalize_parity.rs`).
#[pyfunction]
fn normalize_eeg_f32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    orig_sr: f64,
) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
    let arr = data.as_array();
    let (n_ch, t) = (arr.shape()[0], arr.shape()[1]);
    let signal: Vec<Vec<f64>> = (0..n_ch)
        .map(|c| (0..t).map(|j| arr[[c, j]] as f64).collect())
        .collect();

    match lml::normalize::normalize_eeg_signal_f32(&signal, orig_sr) {
        Ok(Some(out)) => {
            let n_out = out.len();
            let t_out = out.first().map(|r| r.len()).unwrap_or(0);
            let arr = PyArray2::<f32>::zeros(py, [n_out, t_out], false);
            // Safe: freshly allocated, GIL held, contiguous C-order.
            let slice = unsafe { arr.as_slice_mut() }.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("PyArray slice: {e:?}"))
            })?;
            for (c, row) in out.iter().enumerate() {
                slice[c * t_out..c * t_out + row.len()].copy_from_slice(row);
            }
            Ok(Some(arr))
        }
        Ok(None) => Ok(None),
        Err(lml::normalize::NormalizeError::FftResampleUnsupported { orig_sr }) => {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "resample {orig_sr} Hz → 250 Hz needs the scipy FFT branch (not ported); \
                 fall back to the Python normalization for this rate"
            )))
        }
    }
}

/// LQTP1 training-pack writer (ADR 0075 B2). Streams BFP-quantized `[n_channels,
/// window_len]` f32 windows to a pack file in manifest order; `finish()` atomically
/// renames the `.partial` into place. `manifest_sha256` binds the pack to the ordered
/// index the trainer will verify.
#[pyclass]
struct PyPackWriter {
    inner: Option<lml::tensor_pack::PackWriter>,
    n_channels: usize,
    window_len: usize,
}

#[pymethods]
impl PyPackWriter {
    #[new]
    fn new(
        path: &str,
        dtype: &str,
        n_channels: usize,
        window_len: usize,
        n_windows: usize,
        manifest_sha256: &[u8],
    ) -> PyResult<Self> {
        let dt = lml::tensor_pack::PackDtype::parse(dtype).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("bad pack dtype '{dtype}' (int8|int16|f32)"))
        })?;
        if manifest_sha256.len() != 32 {
            return Err(pyo3::exceptions::PyValueError::new_err("manifest_sha256 must be 32 bytes"));
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(manifest_sha256);
        let w = lml::tensor_pack::PackWriter::create(
            std::path::Path::new(path),
            dt,
            n_channels,
            window_len,
            n_windows,
            hash,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: Some(w), n_channels, window_len })
    }

    /// Append one `[n_channels, window_len]` f32 window at the next row (BFP-quantized).
    fn write_window(&mut self, x: PyReadonlyArray2<f32>) -> PyResult<()> {
        let w = self
            .inner
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer already finished"))?;
        let (r, c) = x.as_array().dim();
        if r != self.n_channels || c != self.window_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "window shape [{r}, {c}] != [{}, {}]",
                self.n_channels, self.window_len
            )));
        }
        let slice = x
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("window not contiguous: {e:?}")))?;
        w.write_window(slice).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Flush, fsync, and atomically finalize the pack. The writer is consumed.
    fn finish(&mut self) -> PyResult<()> {
        let w = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer already finished"))?;
        w.finish().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

/// LQTP1 training-pack reader (ADR 0075 B3). Memory-maps the pack read-only (shared across
/// DataLoader workers) and verifies the fail-closed `expected_sha256`. `dequantize(row)`
/// returns a `[n_channels, window_len]` f32 window; the mmap pages are demand-loaded, so
/// resident RSS is only the touched windows.
#[pyclass]
struct PyPackReader {
    inner: lml::tensor_pack::PackReader,
}

#[pymethods]
impl PyPackReader {
    #[new]
    #[pyo3(signature = (path, expected_sha256=None))]
    fn new(path: &str, expected_sha256: Option<&[u8]>) -> PyResult<Self> {
        let exp = match expected_sha256 {
            Some(b) => {
                if b.len() != 32 {
                    return Err(pyo3::exceptions::PyValueError::new_err("expected_sha256 must be 32 bytes"));
                }
                let mut h = [0u8; 32];
                h.copy_from_slice(b);
                Some(h)
            }
            None => None,
        };
        let r = lml::tensor_pack::PackReader::open(std::path::Path::new(path), exp)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: r })
    }

    #[getter]
    fn n_windows(&self) -> usize {
        self.inner.n_windows()
    }

    #[getter]
    fn n_channels(&self) -> usize {
        self.inner.header().n_channels
    }

    #[getter]
    fn window_len(&self) -> usize {
        self.inner.header().window_len
    }

    /// Dequantize window `row` to a `[n_channels, window_len]` f32 array.
    fn dequantize<'py>(&self, py: Python<'py>, row: usize) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let v = self
            .inner
            .dequantize_window(row)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let n_ch = self.inner.header().n_channels;
        let t = self.inner.header().window_len;
        let arr = PyArray2::<f32>::zeros(py, [n_ch, t], false);
        // Safe: freshly allocated, GIL held, no aliasing; v is exactly n_ch*t long.
        unsafe {
            let out = arr
                .as_slice_mut()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("PyArray slice: {e:?}")))?;
            out.copy_from_slice(&v);
        }
        Ok(arr)
    }
}

#[pymodule]
fn lamquant_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(golomb_encode_dense, m)?)?;
    m.add_function(wrap_pyfunction!(golomb_decode_dense, m)?)?;
    m.add_function(wrap_pyfunction!(rans_encode, m)?)?;
    m.add_function(wrap_pyfunction!(rans_decode, m)?)?;
    m.add_function(wrap_pyfunction!(lml_compress, m)?)?;
    m.add_function(wrap_pyfunction!(lml_decompress, m)?)?;
    m.add_function(wrap_pyfunction!(container_write, m)?)?;
    m.add_function(wrap_pyfunction!(container_read, m)?)?;
    m.add_function(wrap_pyfunction!(container_read_bytes, m)?)?;
    m.add_class::<PyAbir>()?;
    m.add_class::<PyPackWriter>()?;
    m.add_class::<PyPackReader>()?;
    m.add_function(wrap_pyfunction!(container_read_abir, m)?)?;
    m.add_function(wrap_pyfunction!(container_read_bytes_abir, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_eeg_f32, m)?)?;
    m.add_function(wrap_pyfunction!(container_read_window_np, m)?)?;
    m.add_function(wrap_pyfunction!(container_read_phys_f32, m)?)?;
    m.add_function(wrap_pyfunction!(container_read_phys_selected, m)?)?;
    m.add_function(wrap_pyfunction!(lma_mmap_read_phys_selected, m)?)?;
    m.add_function(wrap_pyfunction!(lma_mmap_entry_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(container_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(lma_read_entry, m)?)?;
    m.add_function(wrap_pyfunction!(lma_entry_headers, m)?)?;
    m.add_function(wrap_pyfunction!(write_ca_lmq, m)?)?;
    m.add_function(wrap_pyfunction!(read_ca_lmq, m)?)?;
    Ok(())
}
