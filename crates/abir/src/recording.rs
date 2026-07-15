//! Recording graph primitives for ABIR2.
//!
//! This module is intentionally minimal and immutable-by-default after build:
//! builders are mutable, `Recording` is an immutable view over `Arc`-backed data.

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;

/// Rational number utility for sample rates and timing math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rational {
    numerator: u64,
    denominator: u64,
}

impl Rational {
    /// Construct a reduced rational. Returns `None` when denominator is zero.
    pub fn new(numerator: u64, denominator: u64) -> Option<Self> {
        if denominator == 0 {
            return None;
        }

        let g = gcd_u64(numerator, denominator);
        Some(Self {
            numerator: numerator / g,
            denominator: denominator / g,
        })
    }

    /// Numerator in reduced form.
    pub const fn numerator(&self) -> u64 {
        self.numerator
    }

    /// Denominator in reduced form.
    pub const fn denominator(&self) -> u64 {
        self.denominator
    }
}

fn gcd_u64(a: u64, b: u64) -> u64 {
    let mut x = a;
    let mut y = b;
    while y != 0 {
        let r = x % y;
        x = y;
        y = r;
    }
    if x == 0 {
        1
    } else {
        x
    }
}

/// Immutable identifier backed by string bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModalityId {
    id: Arc<str>,
}

impl ModalityId {
    /// Construct from any string-like value.
    pub fn new(value: impl AsRef<str>) -> Self {
        Self {
            id: Arc::from(value.as_ref()),
        }
    }

    /// Access the ID string.
    pub fn as_str(&self) -> &str {
        &self.id
    }

    /// Canonical EEG ID.
    pub fn eeg() -> Self {
        Self::new("eeg")
    }

    /// Canonical ECG ID.
    pub fn ecg() -> Self {
        Self::new("ecg")
    }
}

/// Unit identifier with tiny UCUM-oriented construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Unit {
    system: Arc<str>,
    value: Arc<str>,
}

impl Unit {
    /// Construct UCUM unit.
    pub fn ucum(unit: impl AsRef<str>) -> Self {
        Self {
            system: Arc::from("ucum"),
            value: Arc::from(unit.as_ref()),
        }
    }

    /// Access the unit string.
    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Access the unit system string if present.
    pub fn system(&self) -> &str {
        &self.system
    }
}

/// Scalar width tagging for stored samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleDtype {
    I16,
    I32,
}

/// Immutable sample buffer for signal series.
#[derive(Clone, Debug)]
pub enum SampleBuffer {
    /// Signed 16-bit samples.
    I16(Arc<[i16]>),
    /// Signed 32-bit samples.
    I32(Arc<[i32]>),
}

impl SampleBuffer {
    /// Construct a shared 16-bit buffer.
    pub fn from_i16(data: Arc<[i16]>) -> Self {
        Self::I16(data)
    }

    /// Construct a shared 32-bit buffer.
    pub fn from_i32(data: Arc<[i32]>) -> Self {
        Self::I32(data)
    }

    /// Sample count.
    pub fn len(&self) -> usize {
        match self {
            Self::I16(buf) => buf.len(),
            Self::I32(buf) => buf.len(),
        }
    }

    /// Whether this buffer contains no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stored sample width.
    pub fn dtype(&self) -> SampleDtype {
        match self {
            Self::I16(_) => SampleDtype::I16,
            Self::I32(_) => SampleDtype::I32,
        }
    }
}

/// Immutable recording identity keys.
#[derive(Clone, Debug)]
pub struct RecordingIdentity {
    subject: Arc<str>,
    session: Option<Arc<str>>,
    run: Option<Arc<str>>,
}

impl RecordingIdentity {
    /// Construct identity with optional session/run.
    pub fn new(subject: impl AsRef<str>, session: Option<&str>, run: Option<&str>) -> Self {
        Self {
            subject: Arc::from(subject.as_ref()),
            session: session.map(Arc::from),
            run: run.map(Arc::from),
        }
    }

    /// Subject value.
    pub fn subject(&self) -> &str {
        &self.subject
    }

    /// Optional session value.
    pub fn session(&self) -> Option<&str> {
        self.session.as_deref()
    }

    /// Optional run value.
    pub fn run(&self) -> Option<&str> {
        self.run.as_deref()
    }
}

/// Channel identifier + metadata.
#[derive(Clone, Debug)]
pub struct ChannelDescriptor {
    id: Arc<str>,
    label: Arc<str>,
    modality: ModalityId,
    unit: Unit,
}

impl ChannelDescriptor {
    /// Construct a channel descriptor.
    pub fn new(
        id: impl AsRef<str>,
        label: impl AsRef<str>,
        modality: ModalityId,
        unit: Unit,
    ) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            label: Arc::from(label.as_ref()),
            modality,
            unit,
        }
    }

    /// Channel ID string.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Human-readable channel label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Channel modality.
    pub fn modality(&self) -> &ModalityId {
        &self.modality
    }

    /// Channel engineering unit.
    pub fn unit(&self) -> &Unit {
        &self.unit
    }
}

/// Sample axis description.
#[derive(Clone, Debug)]
pub enum TimeAxis {
    /// Evenly spaced clock ticks.
    Uniform {
        /// Clock identifier.
        clock_id: Arc<str>,
        /// Tick at first sample.
        start_tick: i64,
        /// Sample rate denominator/numerator pair.
        sample_rate: Rational,
    },
    /// Explicit tick list per sample.
    Explicit {
        /// Clock identifier.
        clock_id: Arc<str>,
        /// Ticks for every sample.
        ticks: Arc<[i64]>,
    },
}

impl TimeAxis {
    /// Construct uniform axis.
    pub fn uniform(clock_id: impl AsRef<str>, start_tick: i64, sample_rate: Rational) -> Self {
        Self::Uniform {
            clock_id: Arc::from(clock_id.as_ref()),
            start_tick,
            sample_rate,
        }
    }

    /// Construct explicit axis.
    pub fn explicit(clock_id: impl AsRef<str>, ticks: Arc<[i64]>) -> Self {
        Self::Explicit {
            clock_id: Arc::from(clock_id.as_ref()),
            ticks,
        }
    }

    /// Optional sample rate for uniform axes.
    pub fn sample_rate(&self) -> Option<Rational> {
        match self {
            Self::Uniform { sample_rate, .. } => Some(*sample_rate),
            Self::Explicit { .. } => None,
        }
    }

    /// Clock identifier shared with related axes and events.
    pub fn clock_id(&self) -> &str {
        match self {
            Self::Uniform { clock_id, .. } | Self::Explicit { clock_id, .. } => clock_id,
        }
    }

    /// Tick for the first uniformly sampled value.
    pub fn start_tick(&self) -> Option<i64> {
        match self {
            Self::Uniform { start_tick, .. } => Some(*start_tick),
            Self::Explicit { .. } => None,
        }
    }

    /// Explicit timestamp ticks, when present.
    pub fn explicit_ticks(&self) -> Option<&[i64]> {
        match self {
            Self::Explicit { ticks, .. } => Some(ticks),
            Self::Uniform { .. } => None,
        }
    }

    fn explicit_tick_count(&self) -> Option<usize> {
        match self {
            Self::Explicit { ticks, .. } => Some(ticks.len()),
            Self::Uniform { .. } => None,
        }
    }
}

/// Channel timeseries and samples.
#[derive(Clone, Debug)]
pub struct SignalSeries {
    channel: ChannelDescriptor,
    time_axis: TimeAxis,
    samples: SampleBuffer,
}

impl SignalSeries {
    /// Construct a series.
    pub fn new(channel: ChannelDescriptor, time_axis: TimeAxis, samples: SampleBuffer) -> Self {
        Self {
            channel,
            time_axis,
            samples,
        }
    }

    /// Channel descriptor.
    pub fn channel(&self) -> &ChannelDescriptor {
        &self.channel
    }

    /// Time axis.
    pub fn time_axis(&self) -> &TimeAxis {
        &self.time_axis
    }

    /// Samples.
    pub fn samples(&self) -> &SampleBuffer {
        &self.samples
    }

    /// Sample count.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether this series contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Collection of one modality stream.
#[derive(Clone, Debug)]
pub struct SignalStream {
    id: Arc<str>,
    modality: ModalityId,
    series: Vec<SignalSeries>,
}

impl SignalStream {
    /// Construct stream with empty series list.
    pub fn new(id: impl AsRef<str>, modality: ModalityId) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            modality,
            series: Vec::new(),
        }
    }

    /// Add one series (builder-style, immutable on return).
    pub fn with_series(mut self, series: SignalSeries) -> Self {
        self.series.push(series);
        self
    }

    /// Stream identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Stream modality.
    pub fn modality(&self) -> &ModalityId {
        &self.modality
    }

    /// Series list.
    pub fn series(&self) -> &[SignalSeries] {
        &self.series
    }
}

/// Errors for recording construction and validation.
#[derive(Debug)]
pub enum RecordingError {
    /// Duplicate stream ID in builder input.
    DuplicateStreamId(Arc<str>),
    /// Duplicate channel ID observed during verification.
    DuplicateChannelId(Arc<str>),
    /// Explicit timestamps don't match sample count.
    ExplicitTimestampCountMismatch {
        /// Stream ID.
        stream_id: Arc<str>,
        /// Channel ID.
        channel_id: Arc<str>,
        /// Explicit ticks count.
        explicit_ticks: usize,
        /// Sample count.
        sample_count: usize,
    },
}

impl core::fmt::Display for RecordingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicateStreamId(id) => write!(f, "duplicate signal stream id '{id}'"),
            Self::DuplicateChannelId(id) => {
                write!(f, "duplicate channel id '{id}' across streams")
            }
            Self::ExplicitTimestampCountMismatch {
                stream_id,
                channel_id,
                explicit_ticks,
                sample_count,
            } => write!(
                f,
                "explicit timestamp count mismatch in stream '{stream_id}', channel '{channel_id}' ({explicit_ticks} ticks, {sample_count} samples)"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RecordingError {}

/// Mutable builder for an immutable recording.
#[derive(Clone, Debug)]
pub struct RecordingBuilder {
    identity: RecordingIdentity,
    signal_streams: Vec<SignalStream>,
}

impl RecordingBuilder {
    /// Start a new builder.
    pub fn new(identity: RecordingIdentity) -> Self {
        Self {
            identity,
            signal_streams: Vec::new(),
        }
    }

    /// Append a stream if and only if stream ID is unique.
    pub fn add_signal_stream(&mut self, stream: SignalStream) -> Result<(), RecordingError> {
        if self.signal_streams.iter().any(|s| s.id() == stream.id()) {
            return Err(RecordingError::DuplicateStreamId(Arc::from(stream.id())));
        }
        self.signal_streams.push(stream);
        Ok(())
    }

    /// Finalize an immutable recording and validate integrity.
    pub fn freeze(self) -> Result<Recording, RecordingError> {
        let recording = Recording {
            identity: self.identity,
            signal_streams: self.signal_streams.into_iter().collect::<Vec<_>>().into(),
        };
        recording.verify()?;
        Ok(recording)
    }
}

/// Immutable recording container.
#[derive(Clone, Debug)]
pub struct Recording {
    identity: RecordingIdentity,
    signal_streams: Arc<[SignalStream]>,
}

impl Recording {
    /// Constructed only via [`RecordingBuilder`], immutable after freeze.
    pub fn identity(&self) -> &RecordingIdentity {
        &self.identity
    }

    /// Frozen streams.
    pub fn signal_streams(&self) -> &[SignalStream] {
        &self.signal_streams
    }

    /// Filter streams by modality.
    pub fn streams_by_modality(&self, modality: &ModalityId) -> Vec<&SignalStream> {
        self.signal_streams
            .iter()
            .filter(|stream| stream.modality() == modality)
            .collect()
    }

    /// Cross-stream checks.
    pub fn verify(&self) -> Result<(), RecordingError> {
        let mut seen_channels: Vec<Arc<str>> = Vec::new();

        for stream in self.signal_streams.iter() {
            for series in stream.series() {
                if let Some(count) = series.time_axis().explicit_tick_count() {
                    if count != series.len() {
                        return Err(RecordingError::ExplicitTimestampCountMismatch {
                            stream_id: Arc::from(stream.id()),
                            channel_id: Arc::from(series.channel().id()),
                            explicit_ticks: count,
                            sample_count: series.len(),
                        });
                    }
                }

                if seen_channels
                    .iter()
                    .any(|existing| existing.as_ref() == series.channel().id())
                {
                    return Err(RecordingError::DuplicateChannelId(Arc::from(
                        series.channel().id(),
                    )));
                }
                seen_channels.push(Arc::from(series.channel().id()));
            }
        }

        Ok(())
    }
}
