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
    let result = lml::golomb::encode_dense(slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
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
    lml::lml::decompress(data).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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

/// Parse the current ABIR/BCS2 LML bundle header from in-memory bytes — metadata
/// only, no window decode.
///
/// Returns `(metadata_json, n_ch, n_windows, total_samples,
/// window_size)`. Use this when
/// you need only the calibration / sample-rate dict, not the signal —
/// saves the ~5 MB / ~50 ms cost of decoding a window just for its
/// metadata side-channel.
#[pyfunction]
fn container_metadata(data: &[u8]) -> PyResult<(String, usize, usize, usize, usize)> {
    let header = lml::container::parse_header(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((
        header.metadata,
        header.n_channels,
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
    let n_ch = dims.n_channels;
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
        lml::container::read_bytes_into_f32_calibrated_selected(
            data,
            out,
            calib_slice,
            &channel_mask,
        )
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
        lml::container::read_bytes_into_f32_calibrated_selected(
            data,
            out,
            calib_slice,
            &channel_mask,
        )
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
        header.n_channels,
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
    let results =
        lml::lma::read_entry_headers_path(std::path::Path::new(archive_path), &names, n_bytes)
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
        n_channels,
        latent_c,
        latent_t,
        sample_rate,
        window_samples,
        payload_kind,
        coords.as_deref(),
        channels.as_deref(),
        payload,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("LMQC encode: {:?}", e)))?;
    std::fs::write(path, &buf).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Read an LMQC `.lmq` container → dict with montage + payload. Verifies
/// magic + CRC. `coords` is a flat `[N*3]` list (or None); `channels` is a
/// `list[str]` (or None); `payload` is bytes (caller decodes per
/// `payload_kind`). Raises on corruption / bad magic / version.
#[pyfunction]
fn read_ca_lmq<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let buf =
        std::fs::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
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
            pyo3::exceptions::PyValueError::new_err(format!(
                "bad pack dtype '{dtype}' (int8|int16|f32)"
            ))
        })?;
        if manifest_sha256.len() != 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "manifest_sha256 must be 32 bytes",
            ));
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
        Ok(Self {
            inner: Some(w),
            n_channels,
            window_len,
        })
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
        let slice = x.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("window not contiguous: {e:?}"))
        })?;
        w.write_window(slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Flush, fsync, and atomically finalize the pack. The writer is consumed.
    fn finish(&mut self) -> PyResult<()> {
        let w = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer already finished"))?;
        w.finish()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "expected_sha256 must be 32 bytes",
                    ));
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
            let out = arr.as_slice_mut().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("PyArray slice: {e:?}"))
            })?;
            out.copy_from_slice(&v);
        }
        Ok(arr)
    }
}

// ───────────────── ADR 0114 N2: the Neural Evidence Graph handle ─────────────────

use lamquant_neg::class::{
    tag_is_evidence, Action, Derived, EpistemicClass, Estimated, Generated, Hypothesis, Measured,
    Outcome,
};
use lamquant_neg::{
    EdgeClass, NegGraph, Node, NodeId, NodePayload, NodeRecord, Provenance, Uncertainty,
};

/// A Neural Evidence Graph exposed to Python (ADR 0114 N2). The typed read
/// accessors (`.measured()/.estimated()/.generated()`) raise `ValueError` on an
/// epistemic-class mismatch — the runtime twin of the Rust compile-time barrier,
/// so a Python consumer asking for measured evidence can never be handed a
/// generated node. Construction is class-tagged; verification + content
/// addressing run in Rust (the letter of the invariant is not re-implemented in
/// Python).
#[pyclass]
struct PyNeg {
    inner: NegGraph,
}

impl PyNeg {
    fn node_to_dict<'py>(py: Python<'py>, rec: &NodeRecord) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("id", rec.id.as_str())?;
        d.set_item("class", rec.class_name().unwrap_or("unknown"))?;
        d.set_item("content_ref", rec.payload.content_ref.clone())?;
        d.set_item("summary", rec.payload.summary.clone())?;
        d.set_item("producer", rec.provenance.producer.clone())?;
        let parents: Vec<String> = rec.provenance.parents.iter().map(|p| p.0.clone()).collect();
        d.set_item("parents", parents)?;
        d.set_item(
            "is_evidence",
            tag_is_evidence(rec.class_tag).unwrap_or(false),
        )?;
        if let Some(u) = &rec.uncertainty {
            d.set_item("uncertainty_metric", u.metric.clone())?;
            d.set_item("uncertainty_value", u.value)?;
        }
        Ok(d)
    }

    /// The verified typed boundary: return the node as a dict iff its class is
    /// `C`, else raise `ValueError` (missing node → `KeyError`).
    fn typed_view<'py, C: EpistemicClass>(
        &self,
        py: Python<'py>,
        id: String,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rec = self
            .inner
            .get(&NodeId(id.clone()))
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("no such node {id}")))?;
        rec.view::<C>()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Self::node_to_dict(py, rec)
    }
}

#[pymethods]
impl PyNeg {
    #[new]
    fn new() -> Self {
        PyNeg {
            inner: NegGraph::new(),
        }
    }

    /// Add a typed node; returns its content-address id. `class` ∈
    /// measured/derived/estimated/generated/hypothesis/action/outcome. The node
    /// is *born* into its class in Rust — Python cannot cast one class to another.
    #[pyo3(signature = (class, producer, content_ref=None, summary=None, parents=None, note=None, uncertainty_metric=None, uncertainty_value=None))]
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        class: &str,
        producer: String,
        content_ref: Option<String>,
        summary: Option<String>,
        parents: Option<Vec<String>>,
        note: Option<String>,
        uncertainty_metric: Option<String>,
        uncertainty_value: Option<f64>,
    ) -> PyResult<String> {
        let payload = NodePayload {
            content_ref,
            summary,
        };
        let mut prov = Provenance::from_parents(
            producer,
            parents
                .unwrap_or_default()
                .into_iter()
                .map(NodeId)
                .collect(),
        );
        prov.note = note;
        let unc = match (uncertainty_metric, uncertainty_value) {
            (Some(metric), Some(value)) => Some(Uncertainty { metric, value }),
            (None, None) => None,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "uncertainty needs both metric and value",
                ))
            }
        };
        let id = match class {
            "measured" => self
                .inner
                .add_node(Node::<Measured>::new(payload, prov, unc)),
            "derived" => self
                .inner
                .add_node(Node::<Derived>::new(payload, prov, unc)),
            "estimated" => self
                .inner
                .add_node(Node::<Estimated>::new(payload, prov, unc)),
            "generated" => self
                .inner
                .add_node(Node::<Generated>::new(payload, prov, unc)),
            "hypothesis" => self
                .inner
                .add_node(Node::<Hypothesis>::new(payload, prov, unc)),
            "action" => self.inner.add_node(Node::<Action>::new(payload, prov, unc)),
            "outcome" => self
                .inner
                .add_node(Node::<Outcome>::new(payload, prov, unc)),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown epistemic class {other:?}"
                )))
            }
        };
        Ok(id.0)
    }

    /// Add a typed edge (`from`/`to` are content-address ids).
    fn add_edge(&mut self, from: String, to: String, edge_class: &str) -> PyResult<()> {
        let ec = match edge_class {
            "deterministic-transform" => EdgeClass::DeterministicTransform,
            "probabilistic-inference" => EdgeClass::ProbabilisticInference,
            "temporal-dependence" => EdgeClass::TemporalDependence,
            "spatial-correspondence" => EdgeClass::SpatialCorrespondence,
            "causal-intervention" => EdgeClass::CausalIntervention,
            "calibration-dependency" => EdgeClass::CalibrationDependency,
            "provenance-dependency" => EdgeClass::ProvenanceDependency,
            "uncertainty-propagation" => EdgeClass::UncertaintyPropagation,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown edge class {other:?}"
                )))
            }
        };
        self.inner.add_edge(NodeId(from), NodeId(to), ec);
        Ok(())
    }

    /// Materialize every node's provenance parents as provenance-dependency edges.
    fn materialize_provenance_edges(&mut self) {
        self.inner.materialize_provenance_edges();
    }

    /// Integrity check; raises `ValueError` listing every violation (fail-closed).
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(|errs| {
            let msg = errs
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            pyo3::exceptions::PyValueError::new_err(format!("NEG verify failed: {msg}"))
        })
    }

    /// The graph's own content address (stable across insertion order).
    fn content_address(&self) -> String {
        self.inner.content_address()
    }

    /// Deterministic canonical JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Parse a graph from JSON (does not verify — call `.verify()` after).
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        NegGraph::from_json(s)
            .map(|inner| PyNeg { inner })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Typed accessor — raises unless the node's class is `measured`.
    fn measured<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyDict>> {
        self.typed_view::<Measured>(py, id)
    }
    /// Typed accessor — raises unless the node's class is `estimated`.
    fn estimated<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyDict>> {
        self.typed_view::<Estimated>(py, id)
    }
    /// Typed accessor — raises unless the node's class is `generated`.
    fn generated<'py>(&self, py: Python<'py>, id: String) -> PyResult<Bound<'py, PyDict>> {
        self.typed_view::<Generated>(py, id)
    }

    /// Whether the node may be treated as measured evidence (fail-closed on
    /// unknown class). The type-erased twin of `EpistemicClass::IS_EVIDENCE`.
    fn is_evidence(&self, id: String) -> PyResult<bool> {
        let rec = self
            .inner
            .get(&NodeId(id))
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("no such node"))?;
        tag_is_evidence(rec.class_tag)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unknown epistemic class"))
    }

    fn __len__(&self) -> usize {
        self.inner.nodes.len()
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
    m.add_class::<PyNeg>()?;
    m.add_class::<PyPackWriter>()?;
    m.add_class::<PyPackReader>()?;
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
