//! Semantic graph types for ABIR2.
//
//! These are immutable value types with private fields and read-only accessors.
//! Construction happens through explicit constructors; validation is performed by
//! recording freeze/verify.

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;

/// Rational number utility for timing rates.
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

/// Qualified name with namespace and local string.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QualifiedName {
    namespace: Arc<str>,
    local: Arc<str>,
}

impl QualifiedName {
    /// Construct a qualified name.
    pub fn new(namespace: impl AsRef<str>, local: impl AsRef<str>) -> Self {
        Self {
            namespace: Arc::from(namespace.as_ref()),
            local: Arc::from(local.as_ref()),
        }
    }

    /// Namespace component.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Local component.
    pub fn local(&self) -> &str {
        &self.local
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

    /// Access unit string.
    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Access unit system string.
    pub fn system(&self) -> &str {
        &self.system
    }
}

/// Scalar width tagging for stored samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleDtype {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    F32,
    F64,
}

/// Immutable sample buffer for signal series.
#[derive(Clone, Debug)]
pub enum SampleBuffer {
    /// Signed 8-bit samples.
    I8(Arc<[i8]>),
    /// Unsigned 8-bit samples.
    U8(Arc<[u8]>),
    /// Signed 16-bit samples.
    I16(Arc<[i16]>),
    /// Unsigned 16-bit samples.
    U16(Arc<[u16]>),
    /// Signed 32-bit samples.
    I32(Arc<[i32]>),
    /// Unsigned 32-bit samples.
    U32(Arc<[u32]>),
    /// Signed 64-bit samples.
    I64(Arc<[i64]>),
    /// IEEE-754 binary32 samples.
    F32(Arc<[f32]>),
    /// IEEE-754 binary64 samples.
    F64(Arc<[f64]>),
}

impl SampleBuffer {
    /// Construct a shared signed 8-bit buffer.
    pub fn from_i8(data: Arc<[i8]>) -> Self {
        Self::I8(data)
    }

    /// Construct a shared unsigned 8-bit buffer.
    pub fn from_u8(data: Arc<[u8]>) -> Self {
        Self::U8(data)
    }

    /// Construct a shared 16-bit buffer.
    pub fn from_i16(data: Arc<[i16]>) -> Self {
        Self::I16(data)
    }

    /// Construct a shared unsigned 16-bit buffer.
    pub fn from_u16(data: Arc<[u16]>) -> Self {
        Self::U16(data)
    }

    /// Construct a shared 32-bit buffer.
    pub fn from_i32(data: Arc<[i32]>) -> Self {
        Self::I32(data)
    }

    /// Construct a shared unsigned 32-bit buffer.
    pub fn from_u32(data: Arc<[u32]>) -> Self {
        Self::U32(data)
    }

    /// Construct a shared signed 64-bit buffer.
    pub fn from_i64(data: Arc<[i64]>) -> Self {
        Self::I64(data)
    }

    /// Construct a shared binary32 buffer.
    pub fn from_f32(data: Arc<[f32]>) -> Self {
        Self::F32(data)
    }

    /// Construct a shared binary64 buffer.
    pub fn from_f64(data: Arc<[f64]>) -> Self {
        Self::F64(data)
    }

    /// Sample count.
    pub fn len(&self) -> usize {
        match self {
            Self::I8(buf) => buf.len(),
            Self::U8(buf) => buf.len(),
            Self::I16(buf) => buf.len(),
            Self::U16(buf) => buf.len(),
            Self::I32(buf) => buf.len(),
            Self::U32(buf) => buf.len(),
            Self::I64(buf) => buf.len(),
            Self::F32(buf) => buf.len(),
            Self::F64(buf) => buf.len(),
        }
    }

    /// Whether this buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stored sample width.
    pub fn dtype(&self) -> SampleDtype {
        match self {
            Self::I8(_) => SampleDtype::I8,
            Self::U8(_) => SampleDtype::U8,
            Self::I16(_) => SampleDtype::I16,
            Self::U16(_) => SampleDtype::U16,
            Self::I32(_) => SampleDtype::I32,
            Self::U32(_) => SampleDtype::U32,
            Self::I64(_) => SampleDtype::I64,
            Self::F32(_) => SampleDtype::F32,
            Self::F64(_) => SampleDtype::F64,
        }
    }

    /// Borrow binary32 samples without copying.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values),
            _ => None,
        }
    }

    /// Borrow binary64 samples without copying.
    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            Self::F64(values) => Some(values),
            _ => None,
        }
    }
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClockKind {
    Device,
    UnixUtc,
    Relative,
    Other(QualifiedName),
}

#[derive(Clone, Debug)]
pub struct Clock {
    id: Arc<str>,
    kind: ClockKind,
    tick_rate: Rational,
}

impl Clock {
    /// Construct a clock.
    pub fn new(id: impl AsRef<str>, kind: ClockKind, tick_rate: Rational) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            kind,
            tick_rate,
        }
    }

    /// Clock id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Clock kind.
    pub fn kind(&self) -> &ClockKind {
        &self.kind
    }

    /// Tick rate.
    pub fn tick_rate(&self) -> Rational {
        self.tick_rate
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

    /// Clock identifier.
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
}

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

    /// Add one series (builder-style).
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

#[derive(Clone, Debug)]
pub struct Event {
    id: Arc<str>,
    clock_id: Arc<str>,
    tick: i64,
    label: QualifiedName,
    properties: PropertyBag,
}

impl Event {
    /// Construct an event.
    pub fn new(
        id: impl AsRef<str>,
        clock_id: impl AsRef<str>,
        tick: i64,
        label: QualifiedName,
    ) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            clock_id: Arc::from(clock_id.as_ref()),
            tick,
            label,
            properties: PropertyBag::new(Vec::new()),
        }
    }

    /// Event id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Clock id.
    pub fn clock_id(&self) -> &str {
        &self.clock_id
    }

    /// Tick.
    pub fn tick(&self) -> i64 {
        self.tick
    }

    /// Label.
    pub fn label(&self) -> &QualifiedName {
        &self.label
    }

    /// Properties.
    pub fn properties(&self) -> &PropertyBag {
        &self.properties
    }

    /// Set properties.
    pub fn with_properties(mut self, properties: PropertyBag) -> Self {
        self.properties = properties;
        self
    }
}

#[derive(Clone, Debug)]
pub struct Interval {
    id: Arc<str>,
    clock_id: Arc<str>,
    start_tick: i64,
    end_tick: i64,
    label: QualifiedName,
}

impl Interval {
    /// Construct interval.
    pub fn new(
        id: impl AsRef<str>,
        clock_id: impl AsRef<str>,
        start_tick: i64,
        end_tick: i64,
        label: QualifiedName,
    ) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            clock_id: Arc::from(clock_id.as_ref()),
            start_tick,
            end_tick,
            label,
        }
    }

    /// ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Clock ID.
    pub fn clock_id(&self) -> &str {
        &self.clock_id
    }

    /// Start tick.
    pub fn start_tick(&self) -> i64 {
        self.start_tick
    }

    /// End tick.
    pub fn end_tick(&self) -> i64 {
        self.end_tick
    }

    /// Label.
    pub fn label(&self) -> &QualifiedName {
        &self.label
    }
}

#[derive(Clone, Debug)]
pub struct TableColumn {
    name: QualifiedName,
    value_type: ValueType,
    values: Arc<[Value]>,
}

impl TableColumn {
    /// Construct a typed table column.
    pub fn new(name: QualifiedName, value_type: ValueType, values: Arc<[Value]>) -> Self {
        Self {
            name,
            value_type,
            values,
        }
    }

    /// Column name.
    pub fn name(&self) -> &QualifiedName {
        &self.name
    }

    /// Declared value type.
    pub fn value_type(&self) -> ValueType {
        self.value_type
    }

    /// Column values.
    pub fn values(&self) -> &[Value] {
        &self.values
    }
}

#[derive(Clone, Debug)]
pub struct Table {
    id: Arc<str>,
    columns: Vec<TableColumn>,
}

impl Table {
    /// Construct a table.
    pub fn new(id: impl AsRef<str>) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            columns: Vec::new(),
        }
    }

    /// Add a column (builder-style).
    pub fn with_column(mut self, column: TableColumn) -> Self {
        self.columns.push(column);
        self
    }

    /// ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Columns.
    pub fn columns(&self) -> &[TableColumn] {
        &self.columns
    }

    /// Column count.
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Row count; zero when there are no columns.
    pub fn row_count(&self) -> usize {
        self.columns.first().map(|c| c.values.len()).unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDataType {
    I8,
    U8,
    U16,
    I32,
    U32,
    I64,
    F32,
    F64,
    I16,
}

#[derive(Clone, Debug)]
pub enum TensorBuffer {
    I8(Arc<[i8]>),
    U8(Arc<[u8]>),
    F32(Arc<[f32]>),
    F64(Arc<[f64]>),
    I16(Arc<[i16]>),
    U16(Arc<[u16]>),
    I32(Arc<[i32]>),
    U32(Arc<[u32]>),
    I64(Arc<[i64]>),
}

impl TensorBuffer {
    /// Construct from signed 8-bit values.
    pub fn from_i8(data: Arc<[i8]>) -> Self {
        Self::I8(data)
    }

    /// Construct from unsigned 8-bit values.
    pub fn from_u8(data: Arc<[u8]>) -> Self {
        Self::U8(data)
    }

    /// Construct from f32 values.
    pub fn from_f32(data: Arc<[f32]>) -> Self {
        Self::F32(data)
    }

    /// Construct from binary64 values.
    pub fn from_f64(data: Arc<[f64]>) -> Self {
        Self::F64(data)
    }

    /// Construct from i16 values.
    pub fn from_i16(data: Arc<[i16]>) -> Self {
        Self::I16(data)
    }

    /// Construct from unsigned 16-bit values.
    pub fn from_u16(data: Arc<[u16]>) -> Self {
        Self::U16(data)
    }

    /// Construct from signed 32-bit values.
    pub fn from_i32(data: Arc<[i32]>) -> Self {
        Self::I32(data)
    }

    /// Construct from unsigned 32-bit values.
    pub fn from_u32(data: Arc<[u32]>) -> Self {
        Self::U32(data)
    }

    /// Construct from signed 64-bit values.
    pub fn from_i64(data: Arc<[i64]>) -> Self {
        Self::I64(data)
    }

    /// Element count.
    pub fn len(&self) -> usize {
        match self {
            Self::I8(buf) => buf.len(),
            Self::U8(buf) => buf.len(),
            Self::F32(buf) => buf.len(),
            Self::F64(buf) => buf.len(),
            Self::I16(buf) => buf.len(),
            Self::U16(buf) => buf.len(),
            Self::I32(buf) => buf.len(),
            Self::U32(buf) => buf.len(),
            Self::I64(buf) => buf.len(),
        }
    }

    /// Whether the tensor buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dtype.
    pub fn dtype(&self) -> TensorDataType {
        match self {
            Self::I8(_) => TensorDataType::I8,
            Self::U8(_) => TensorDataType::U8,
            Self::F32(_) => TensorDataType::F32,
            Self::F64(_) => TensorDataType::F64,
            Self::I16(_) => TensorDataType::I16,
            Self::U16(_) => TensorDataType::U16,
            Self::I32(_) => TensorDataType::I32,
            Self::U32(_) => TensorDataType::U32,
            Self::I64(_) => TensorDataType::I64,
        }
    }

    /// Borrow binary64 values without copying.
    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            Self::F64(values) => Some(values),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    id: Arc<str>,
    shape: Arc<[u64]>,
    buffer: TensorBuffer,
}

impl Tensor {
    /// Construct tensor.
    pub fn new(id: impl AsRef<str>, shape: Arc<[u64]>, buffer: TensorBuffer) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            shape,
            buffer,
        }
    }

    /// ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Tensor shape.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Buffer.
    pub fn buffer(&self) -> &TensorBuffer {
        &self.buffer
    }
}

#[derive(Clone, Debug)]
pub struct CoordinateFrame {
    id: Arc<str>,
    dimension: usize,
    system: QualifiedName,
}

impl CoordinateFrame {
    /// Construct coordinate frame.
    pub fn new(id: impl AsRef<str>, dimension: usize, system: QualifiedName) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            dimension,
            system,
        }
    }

    /// ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Coordinate system.
    pub fn system(&self) -> &QualifiedName {
        &self.system
    }
}

#[derive(Clone, Debug)]
pub struct CoordinatePoint {
    id: Arc<str>,
    frame_id: Arc<str>,
    object_id: Arc<str>,
    values: Arc<[f64]>,
    unit: Unit,
}

impl CoordinatePoint {
    /// Construct a coordinate point.
    pub fn new(
        id: impl AsRef<str>,
        frame_id: impl AsRef<str>,
        object_id: impl AsRef<str>,
        values: Arc<[f64]>,
        unit: Unit,
    ) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            frame_id: Arc::from(frame_id.as_ref()),
            object_id: Arc::from(object_id.as_ref()),
            values,
            unit,
        }
    }

    /// ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Frame id.
    pub fn frame_id(&self) -> &str {
        &self.frame_id
    }

    /// Object id.
    pub fn object_id(&self) -> &str {
        &self.object_id
    }

    /// Coordinate values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Unit.
    pub fn unit(&self) -> &Unit {
        &self.unit
    }
}

#[derive(Clone, Debug)]
pub struct Property {
    name: QualifiedName,
    value: Value,
}

impl Property {
    /// Construct a property.
    pub fn new(name: QualifiedName, value: Value) -> Self {
        Self { name, value }
    }

    /// Property name.
    pub fn name(&self) -> &QualifiedName {
        &self.name
    }

    /// Property value.
    pub fn value(&self) -> &Value {
        &self.value
    }
}

#[derive(Clone, Debug)]
pub struct PropertyBag {
    properties: Arc<[Property]>,
}

impl PropertyBag {
    /// Construct property bag.
    pub fn new(properties: Vec<Property>) -> Self {
        Self {
            properties: properties.into(),
        }
    }

    /// Access property by qualified name.
    pub fn get(&self, name: &QualifiedName) -> Option<&Value> {
        self.properties
            .iter()
            .find(|property| property.name() == name)
            .map(|property| &property.value)
    }

    /// All properties in source order.
    pub fn properties(&self) -> &[Property] {
        &self.properties
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(u64),
    Rational(Rational),
    Text(Arc<str>),
    Bytes(Arc<[u8]>),
    List(Arc<[Value]>),
    Record(PropertyBag),
}

impl Value {
    /// Construct text value.
    pub fn text(value: impl AsRef<str>) -> Self {
        Self::Text(Arc::from(value.as_ref()))
    }

    /// Construct bytes value.
    pub fn bytes(value: Arc<[u8]>) -> Self {
        Self::Bytes(value)
    }

    /// Construct an exact rational value.
    pub fn rational(value: Rational) -> Self {
        Self::Rational(value)
    }

    /// Construct a list value.
    pub fn list(values: Arc<[Value]>) -> Self {
        Self::List(values)
    }

    /// Construct a nested record value.
    pub fn record(properties: PropertyBag) -> Self {
        Self::Record(properties)
    }

    /// F64 accessor.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }

    /// Text accessor.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(v) => Some(v),
            _ => None,
        }
    }

    /// Bytes accessor.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(v) => Some(v),
            _ => None,
        }
    }

    /// Declared scalar type, or `None` for a missing value.
    pub fn value_type(&self) -> Option<ValueType> {
        match self {
            Self::Null => None,
            Self::Bool(_) => Some(ValueType::Bool),
            Self::I64(_) => Some(ValueType::I64),
            Self::U64(_) => Some(ValueType::U64),
            Self::F64(_) => Some(ValueType::F64),
            Self::Rational(_) => Some(ValueType::Rational),
            Self::Text(_) => Some(ValueType::Text),
            Self::Bytes(_) => Some(ValueType::Bytes),
            Self::List(_) => Some(ValueType::List),
            Self::Record(_) => Some(ValueType::Record),
        }
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::F64(value.to_bits())
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueType {
    Null,
    Bool,
    I64,
    U64,
    F64,
    Rational,
    Text,
    Bytes,
    List,
    Record,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticDisposition {
    Exact,
    Normalized,
    PreservedAsExtension,
    Approximated,
    Dropped,
}

#[derive(Clone, Debug)]
pub struct LossReceipt {
    id: Arc<str>,
    label: QualifiedName,
    disposition: SemanticDisposition,
    extension: Option<QualifiedName>,
    details: Arc<str>,
}

impl LossReceipt {
    pub fn new(
        id: impl AsRef<str>,
        label: QualifiedName,
        disposition: SemanticDisposition,
        extension: Option<QualifiedName>,
        details: impl AsRef<str>,
    ) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            label,
            disposition,
            extension,
            details: Arc::from(details.as_ref()),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn label(&self) -> &QualifiedName {
        &self.label
    }

    pub fn disposition(&self) -> SemanticDisposition {
        self.disposition
    }

    pub fn extension(&self) -> Option<&QualifiedName> {
        self.extension.as_ref()
    }

    pub fn details(&self) -> &str {
        &self.details
    }
}

#[derive(Clone, Debug)]
pub struct ProvenanceActivity {
    id: Arc<str>,
    activity: QualifiedName,
    software: Arc<str>,
    inputs: Vec<Arc<str>>,
    outputs: Vec<Arc<str>>,
}

impl ProvenanceActivity {
    pub fn new(id: impl AsRef<str>, activity: QualifiedName, software: impl AsRef<str>) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            activity,
            software: Arc::from(software.as_ref()),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn with_input(mut self, input: impl AsRef<str>) -> Self {
        self.inputs.push(Arc::from(input.as_ref()));
        self
    }

    pub fn with_output(mut self, output: impl AsRef<str>) -> Self {
        self.outputs.push(Arc::from(output.as_ref()));
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn activity(&self) -> &QualifiedName {
        &self.activity
    }

    pub fn software(&self) -> &str {
        &self.software
    }

    pub fn inputs(&self) -> &[Arc<str>] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Arc<str>] {
        &self.outputs
    }

    pub fn inputs_vec(&self) -> &Vec<Arc<str>> {
        &self.inputs
    }

    pub fn outputs_vec(&self) -> &Vec<Arc<str>> {
        &self.outputs
    }
}

#[derive(Clone, Debug)]
pub struct Attachment {
    id: Arc<str>,
    media_type: Arc<str>,
    bytes: Arc<[u8]>,
}

impl Attachment {
    pub fn new(id: impl AsRef<str>, media_type: impl AsRef<str>, bytes: Arc<[u8]>) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            media_type: Arc::from(media_type.as_ref()),
            bytes,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReferenceNodeKind {
    Channel,
    PhysicalReference,
    DerivedReference,
    Ground,
    Other(QualifiedName),
}

#[derive(Clone, Debug)]
pub struct ReferenceNode {
    id: Arc<str>,
    kind: ReferenceNodeKind,
}

impl ReferenceNode {
    pub fn new(id: impl AsRef<str>, kind: ReferenceNodeKind) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            kind,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn kind(&self) -> &ReferenceNodeKind {
        &self.kind
    }
}

#[derive(Clone, Debug)]
pub struct ReferenceEdge {
    id: Arc<str>,
    from_node: Arc<str>,
    to_node: Arc<str>,
    label: QualifiedName,
}

impl ReferenceEdge {
    pub fn new(
        id: impl AsRef<str>,
        from_node: impl AsRef<str>,
        to_node: impl AsRef<str>,
        label: QualifiedName,
    ) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            from_node: Arc::from(from_node.as_ref()),
            to_node: Arc::from(to_node.as_ref()),
            label,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn from(&self) -> &str {
        &self.from_node
    }

    pub fn to(&self) -> &str {
        &self.to_node
    }

    pub fn label(&self) -> &QualifiedName {
        &self.label
    }
}

#[derive(Clone, Debug)]
pub struct Relationship {
    id: Arc<str>,
    subject: Arc<str>,
    predicate: QualifiedName,
    object: Arc<str>,
}

impl Relationship {
    pub fn new(
        id: impl AsRef<str>,
        subject: impl AsRef<str>,
        predicate: QualifiedName,
        object: impl AsRef<str>,
    ) -> Self {
        Self {
            id: Arc::from(id.as_ref()),
            subject: Arc::from(subject.as_ref()),
            predicate,
            object: Arc::from(object.as_ref()),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn subject(&self) -> &str {
        &self.subject
    }

    pub fn predicate(&self) -> &QualifiedName {
        &self.predicate
    }

    pub fn object(&self) -> &str {
        &self.object
    }
}

impl core::fmt::Display for ValueType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let text = match self {
            Self::Null => "null",
            Self::Bool => "bool",
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::F64 => "f64",
            Self::Rational => "rational",
            Self::Text => "text",
            Self::Bytes => "bytes",
            Self::List => "list",
            Self::Record => "record",
        };
        f.write_str(text)
    }
}
