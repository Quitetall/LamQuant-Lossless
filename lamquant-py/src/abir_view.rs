use numpy::{ndarray::Array2, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};

use lml::source::{from_uniform_signal_view, SourceMetadata};

fn invalid(error: impl ToString) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

fn owned_bytes(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    if let Ok(bytes) = data.downcast::<PyBytes>() {
        return Ok(bytes.clone().unbind());
    }
    if let Ok(bytes) = data.downcast::<PyByteArray>() {
        // SAFETY: GIL is held and bytes are copied into immutable Python
        // storage before the bytearray can be observed or mutated again.
        return Ok(PyBytes::new(py, unsafe { bytes.as_bytes() }).unbind());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected bytes or bytearray",
    ))
}

/// Read-only Python ownership boundary for one authenticated ABIR LML dataset.
///
/// `numpy_view()` and `lml_bytes()` return the same owned Python objects on
/// every call. Consumers can therefore take zero-copy views after the initial
/// authenticated decode without retaining Rust borrows or native parser types.
#[pyclass(name = "AbirDatasetView", frozen)]
pub(crate) struct PyAbirDatasetView {
    encoded: Py<PyBytes>,
    signal: Py<PyArray2<i64>>,
    canonical_json: Vec<u8>,
    content_id: String,
    metadata_json: String,
    packet_sample_counts: Vec<usize>,
    n_channels: usize,
    total_samples: usize,
    sample_rate_hz: f64,
}

impl PyAbirDatasetView {
    fn from_owned_lml(py: Python<'_>, encoded: Py<PyBytes>) -> PyResult<Self> {
        let (
            canonical_json,
            content_id,
            metadata_json,
            packet_sample_counts,
            n_channels,
            total_samples,
            sample_rate_hz,
            flat,
        ) = {
            let opened = lml::container::open(encoded.bind(py).as_bytes()).map_err(invalid)?;
            let header = lml::container::header(&opened).map_err(invalid)?;
            let canonical_json =
                semantic_abir::canonical_debug_json(opened.dataset()).map_err(invalid)?;
            let content_id = semantic_abir::logical_content_id(opened.dataset())
                .map_err(invalid)?
                .to_string();
            let flat = opened
                .signal()
                .iter()
                .flat_map(|channel| channel.iter().copied())
                .collect::<Vec<_>>();
            (
                canonical_json,
                content_id,
                header.metadata,
                opened.packet_sample_counts().to_vec(),
                header.n_channels,
                header.total_samples,
                header.sample_rate_hz,
                flat,
            )
        };

        let signal = Array2::from_shape_vec((n_channels, total_samples), flat)
            .map_err(|error| invalid(format!("decoded ABIR matrix shape is invalid: {error}")))?;
        let signal = PyArray2::from_owned_array(py, signal);
        signal.call_method1("setflags", (false,))?;
        let signal = signal.unbind();
        Ok(Self {
            encoded,
            signal,
            canonical_json,
            content_id,
            metadata_json,
            packet_sample_counts,
            n_channels,
            total_samples,
            sample_rate_hz,
        })
    }

    fn encode_numpy(
        py: Python<'_>,
        signal: PyReadonlyArray2<'_, i64>,
        sample_rate_hz: f64,
        window_size: Option<usize>,
        metadata_json: Option<&str>,
        channel_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let shape = signal.shape();
        let n_channels = shape[0];
        let total_samples = shape[1];
        if n_channels == 0 || total_samples == 0 {
            return Err(invalid("signal must have shape [channels>0, samples>0]"));
        }
        lamquant_abir_codec::validate_lml_signal_shape(n_channels, total_samples)
            .map_err(invalid)?;
        if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
            return Err(invalid("sample_rate_hz must be finite and positive"));
        }
        if window_size == Some(0) {
            return Err(invalid("window_size must be positive"));
        }
        if let Some(channels) = channel_names.as_ref() {
            if channels.len() != n_channels {
                return Err(invalid(format!(
                    "channel_names length {} does not match signal channels {n_channels}",
                    channels.len()
                )));
            }
        }
        let metadata_json = metadata_json.unwrap_or("{}");
        let limits = semantic_abir::ValidationLimits::default();
        if metadata_json.len() > limits.max_metadata_bytes {
            return Err(invalid("metadata_json exceeds ABIR metadata limit"));
        }
        let metadata = serde_json::from_str::<serde_json::Value>(metadata_json)
            .map_err(|error| invalid(format!("metadata_json is invalid JSON: {error}")))?;
        if !metadata.is_object() {
            return Err(invalid("metadata_json must encode a JSON object"));
        }
        let contiguous = signal.as_slice().map_err(|error| {
            invalid(format!(
                "signal must be a C-contiguous int64 array: {error}"
            ))
        })?;
        let signal = contiguous
            .chunks_exact(total_samples)
            .map(<[i64]>::to_vec)
            .collect::<Vec<_>>();
        let channels = channel_names
            .unwrap_or_else(|| (0..n_channels).map(|index| format!("ch{index}")).collect());
        let physical_min = signal
            .iter()
            .map(|channel| channel.iter().copied().min().unwrap_or(0) as f64)
            .collect();
        let physical_max = signal
            .iter()
            .map(|channel| channel.iter().copied().max().unwrap_or(0) as f64)
            .collect();
        let semantic = from_uniform_signal_view(
            &signal,
            sample_rate_hz,
            channels,
            physical_min,
            physical_max,
            total_samples as f64 / sample_rate_hz,
            SourceMetadata {
                source_file: String::new(),
                format: "Python/NumPy".into(),
                patient_id: String::new(),
                recording_info: metadata_json.into(),
                startdate: String::new(),
                phys_dim: "digital".into(),
            },
            limits,
        )
        .map_err(invalid)?;
        let mut options = lml::container::LmlEncodeOptions::default();
        if let Some(window_size) = window_size {
            options.window_size = window_size;
        }
        let encoded = lml::container::encode_from_signal_with_options(
            semantic.opened.dataset(),
            &signal,
            options,
        )
        .map_err(invalid)?
        .into_bytes();
        Self::from_owned_lml(py, PyBytes::new(py, &encoded).unbind())
    }
}

#[pymethods]
impl PyAbirDatasetView {
    /// Open and authenticate current BCS2 LML bytes as an ABIR dataset view.
    #[staticmethod]
    fn from_lml(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_owned_lml(py, owned_bytes(py, data)?)
    }

    /// Construct validated ABIR semantics from a contiguous `[channel, sample]`
    /// int64 NumPy array, then encode and reopen the canonical BCS2 LML form.
    #[staticmethod]
    #[pyo3(signature = (signal, sample_rate_hz, window_size=None, metadata_json=None, channel_names=None))]
    fn from_numpy(
        py: Python<'_>,
        signal: PyReadonlyArray2<'_, i64>,
        sample_rate_hz: f64,
        window_size: Option<usize>,
        metadata_json: Option<&str>,
        channel_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        Self::encode_numpy(
            py,
            signal,
            sample_rate_hz,
            window_size,
            metadata_json,
            channel_names,
        )
    }

    /// Pointer-stable, zero-copy NumPy view of authenticated decoded samples.
    fn numpy_view(&self, py: Python<'_>) -> Py<PyArray2<i64>> {
        self.signal.clone_ref(py)
    }

    /// Original authenticated BCS2 LML object; no byte copy.
    fn lml_bytes(&self, py: Python<'_>) -> Py<PyBytes> {
        self.encoded.clone_ref(py)
    }

    /// Canonical RFC 8785 ABIR semantic document.
    fn canonical_json<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.canonical_json)
    }

    #[getter]
    fn content_id(&self) -> &str {
        &self.content_id
    }

    #[getter]
    fn metadata_json(&self) -> &str {
        &self.metadata_json
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.n_channels, self.total_samples)
    }

    #[getter]
    fn n_channels(&self) -> usize {
        self.n_channels
    }

    #[getter]
    fn total_samples(&self) -> usize {
        self.total_samples
    }

    #[getter]
    fn n_windows(&self) -> usize {
        self.packet_sample_counts.len()
    }

    #[getter]
    fn packet_sample_counts(&self) -> Vec<usize> {
        self.packet_sample_counts.clone()
    }

    #[getter]
    fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    fn payload_pointer(&self, py: Python<'_>) -> usize {
        self.signal.bind(py).data() as usize
    }

    fn __repr__(&self) -> String {
        format!(
            "AbirDatasetView(content_id='{}', shape=({}, {}), sample_rate_hz={})",
            self.content_id, self.n_channels, self.total_samples, self.sample_rate_hz
        )
    }
}

/// Convenience entrypoint matching `AbirDatasetView.from_lml`.
#[pyfunction]
pub(crate) fn open_abir(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyAbirDatasetView> {
    PyAbirDatasetView::from_owned_lml(py, owned_bytes(py, data)?)
}
