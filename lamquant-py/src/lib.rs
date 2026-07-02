use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
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
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("not found") {
                pyo3::exceptions::PyKeyError::new_err(msg)
            } else if msg.contains("Not an LMA archive")
                || msg.contains("not supported")
                || msg.contains("Manifest")
                || msg.contains("exceeds")
            {
                pyo3::exceptions::PyValueError::new_err(msg)
            } else {
                pyo3::exceptions::PyIOError::new_err(msg)
            }
        })?;
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
    .map_err(|e| {
        let msg = e.to_string();
        if msg.contains("Not an LMA archive")
            || msg.contains("not supported")
            || msg.contains("Manifest")
            || msg.contains("exceeds")
            || msg.contains("corrupt")
            || msg.contains("too small")
        {
            pyo3::exceptions::PyValueError::new_err(msg)
        } else {
            pyo3::exceptions::PyIOError::new_err(msg)
        }
    })?;
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
    m.add_function(wrap_pyfunction!(container_read_window_np, m)?)?;
    m.add_function(wrap_pyfunction!(container_read_phys_f32, m)?)?;
    m.add_function(wrap_pyfunction!(container_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(lma_read_entry, m)?)?;
    m.add_function(wrap_pyfunction!(lma_entry_headers, m)?)?;
    m.add_function(wrap_pyfunction!(write_ca_lmq, m)?)?;
    m.add_function(wrap_pyfunction!(read_ca_lmq, m)?)?;
    Ok(())
}
