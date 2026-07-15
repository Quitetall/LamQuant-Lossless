//! Recording graph primitives for ABIR2.
//!
//! This module is intentionally minimal and immutable-by-default after build:
//! builders are mutable, `Recording` is an immutable view over `Arc`-backed data.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::semantic::{
    Attachment, Clock, CoordinateFrame, CoordinatePoint, Event, Interval, LossReceipt, ModalityId,
    PropertyBag, ProvenanceActivity, QualifiedName, RecordingIdentity, ReferenceEdge,
    ReferenceNode, Relationship, SignalStream, Table, Tensor, ValueType,
};

/// Errors for recording construction and validation.
#[derive(Debug, Clone)]
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
    /// Explicit timestamps must be nondecreasing.
    ExplicitTimestampOrderViolation {
        /// Stream ID.
        stream_id: Arc<str>,
        /// Channel ID.
        channel_id: Arc<str>,
    },
    /// Stream and channel modality mismatch.
    StreamChannelModalityMismatch {
        /// Stream ID.
        stream_id: Arc<str>,
        /// Channel ID.
        channel_id: Arc<str>,
    },
    /// Unresolved clock ID.
    UnknownClockId {
        /// Clock ID.
        clock_id: Arc<str>,
    },
    /// Unresolved node ID.
    UnknownNodeId {
        /// Node ID.
        node_id: Arc<str>,
    },
    /// Duplicate node ID.
    DuplicateNodeId {
        /// Node ID.
        node_id: Arc<str>,
    },
    /// Duplicate property names in a property bag.
    DuplicatePropertyName {
        /// Property name.
        name: QualifiedName,
    },
    /// Interval bounds invalid.
    InvalidIntervalBounds {
        /// Interval ID.
        interval_id: Arc<str>,
        /// Start tick.
        start_tick: i64,
        /// End tick.
        end_tick: i64,
    },
    /// Table column lengths mismatch.
    TableColumnLengthMismatch {
        /// Table ID.
        table_id: Arc<str>,
        /// First length.
        first_len: usize,
        /// Second length.
        second_len: usize,
    },
    /// Table value type mismatch.
    TableColumnTypeMismatch {
        /// Table ID.
        table_id: Arc<str>,
        /// Column local name.
        column: Arc<str>,
        /// Expected value type.
        expected: ValueType,
        /// Actual value type.
        actual: ValueType,
    },
    /// A table contains the same qualified column name more than once.
    DuplicateTableColumnName {
        /// Table ID.
        table_id: Arc<str>,
        /// Repeated qualified name.
        name: QualifiedName,
    },
    /// Tensor shape and buffer mismatch.
    TensorElementCountMismatch {
        /// Tensor ID.
        tensor_id: Arc<str>,
        /// Total elements from shape.
        expected: u64,
        /// Actual number of elements.
        actual: usize,
    },
    /// Coordinate dimension mismatch.
    CoordinateDimensionMismatch {
        /// Coordinate ID.
        coordinate_id: Arc<str>,
        /// Frame ID.
        frame_id: Arc<str>,
        /// Required dimension.
        expected: usize,
        /// Observed dimension.
        actual: usize,
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
            Self::ExplicitTimestampOrderViolation {
                stream_id,
                channel_id,
            } => write!(
                f,
                "explicit timestamps not nondecreasing in stream '{stream_id}', channel '{channel_id}'"
            ),
            Self::StreamChannelModalityMismatch {
                stream_id,
                channel_id,
            } => write!(
                f,
                "signal channel '{channel_id}' modality must match stream '{stream_id}'"
            ),
            Self::UnknownClockId { clock_id } => write!(f, "unknown clock id '{clock_id}'"),
            Self::UnknownNodeId { node_id } => write!(f, "unknown node id '{node_id}'"),
            Self::DuplicateNodeId { node_id } => write!(f, "duplicate node id '{node_id}'"),
            Self::DuplicatePropertyName { name } => {
                write!(f, "duplicate property name '{}:{}'", name.namespace(), name.local())
            }
            Self::InvalidIntervalBounds {
                interval_id,
                start_tick,
                end_tick,
            } => write!(
                f,
                "invalid interval bounds for '{interval_id}' ({start_tick}..{end_tick})"
            ),
            Self::TableColumnLengthMismatch {
                table_id,
                first_len,
                second_len,
            } => write!(
                f,
                "table '{table_id}' ragged columns ({first_len} != {second_len})"
            ),
            Self::TableColumnTypeMismatch {
                table_id,
                column,
                expected,
                actual,
            } => write!(
                f,
                "table '{table_id}' column '{column}' type mismatch: expected {expected}, found {actual}"
            ),
            Self::DuplicateTableColumnName { table_id, name } => write!(
                f,
                "table '{table_id}' repeats column '{}:{}'",
                name.namespace(),
                name.local()
            ),
            Self::TensorElementCountMismatch {
                tensor_id,
                expected,
                actual,
            } => write!(
                f,
                "tensor '{tensor_id}' element-count mismatch (expected {expected}, actual {actual})"
            ),
            Self::CoordinateDimensionMismatch {
                coordinate_id,
                frame_id,
                expected,
                actual,
            } => write!(
                f,
                "coordinate '{coordinate_id}' dimension mismatch for frame '{frame_id}' ({expected} != {actual})"
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
    clocks: Vec<Clock>,
    signal_streams: Vec<SignalStream>,
    events: Vec<Event>,
    intervals: Vec<Interval>,
    tables: Vec<Table>,
    tensors: Vec<Tensor>,
    coordinate_frames: Vec<CoordinateFrame>,
    coordinates: Vec<CoordinatePoint>,
    reference_nodes: Vec<ReferenceNode>,
    reference_edges: Vec<ReferenceEdge>,
    relationships: Vec<Relationship>,
    attachments: Vec<Attachment>,
    provenance: Vec<ProvenanceActivity>,
    loss_receipts: Vec<LossReceipt>,
    extensions: PropertyBag,
}

impl RecordingBuilder {
    /// Start a new builder.
    pub fn new(identity: RecordingIdentity) -> Self {
        Self {
            identity,
            clocks: Vec::new(),
            signal_streams: Vec::new(),
            events: Vec::new(),
            intervals: Vec::new(),
            tables: Vec::new(),
            tensors: Vec::new(),
            coordinate_frames: Vec::new(),
            coordinates: Vec::new(),
            reference_nodes: Vec::new(),
            reference_edges: Vec::new(),
            relationships: Vec::new(),
            attachments: Vec::new(),
            provenance: Vec::new(),
            loss_receipts: Vec::new(),
            extensions: PropertyBag::new(Vec::new()),
        }
    }

    /// Add a clock.
    pub fn add_clock(&mut self, clock: Clock) -> Result<(), RecordingError> {
        self.clocks.push(clock);
        Ok(())
    }

    /// Add a stream.
    pub fn add_signal_stream(&mut self, stream: SignalStream) -> Result<(), RecordingError> {
        if self
            .signal_streams
            .iter()
            .any(|existing| existing.id() == stream.id())
        {
            return Err(RecordingError::DuplicateStreamId(Arc::from(stream.id())));
        }
        self.signal_streams.push(stream);
        Ok(())
    }

    /// Add an event.
    pub fn add_event(&mut self, event: Event) -> Result<(), RecordingError> {
        self.events.push(event);
        Ok(())
    }

    /// Add an interval.
    pub fn add_interval(&mut self, interval: Interval) -> Result<(), RecordingError> {
        self.intervals.push(interval);
        Ok(())
    }

    /// Add a table.
    pub fn add_table(&mut self, table: Table) -> Result<(), RecordingError> {
        self.tables.push(table);
        Ok(())
    }

    /// Add a tensor.
    pub fn add_tensor(&mut self, tensor: Tensor) -> Result<(), RecordingError> {
        self.tensors.push(tensor);
        Ok(())
    }

    /// Add coordinate frame.
    pub fn add_coordinate_frame(&mut self, frame: CoordinateFrame) -> Result<(), RecordingError> {
        self.coordinate_frames.push(frame);
        Ok(())
    }

    /// Add coordinate point.
    pub fn add_coordinate(&mut self, coordinate: CoordinatePoint) -> Result<(), RecordingError> {
        self.coordinates.push(coordinate);
        Ok(())
    }

    /// Add reference node.
    pub fn add_reference_node(
        &mut self,
        reference_node: ReferenceNode,
    ) -> Result<(), RecordingError> {
        self.reference_nodes.push(reference_node);
        Ok(())
    }

    /// Add reference edge.
    pub fn add_reference_edge(&mut self, edge: ReferenceEdge) -> Result<(), RecordingError> {
        self.reference_edges.push(edge);
        Ok(())
    }

    /// Add relationship.
    pub fn add_relationship(&mut self, relationship: Relationship) -> Result<(), RecordingError> {
        self.relationships.push(relationship);
        Ok(())
    }

    /// Add attachment.
    pub fn add_attachment(&mut self, attachment: Attachment) -> Result<(), RecordingError> {
        self.attachments.push(attachment);
        Ok(())
    }

    /// Add provenance activity.
    pub fn add_provenance(
        &mut self,
        provenance_activity: ProvenanceActivity,
    ) -> Result<(), RecordingError> {
        self.provenance.push(provenance_activity);
        Ok(())
    }

    /// Add loss receipt.
    pub fn add_loss_receipt(&mut self, loss_receipt: LossReceipt) -> Result<(), RecordingError> {
        self.loss_receipts.push(loss_receipt);
        Ok(())
    }

    /// Set extension properties.
    pub fn set_extensions(&mut self, extensions: PropertyBag) {
        self.extensions = extensions;
    }

    /// Finalize an immutable recording and validate integrity.
    pub fn freeze(self) -> Result<Recording, RecordingError> {
        let recording = Recording {
            identity: self.identity,
            clocks: self.clocks.into_iter().collect::<Vec<_>>().into(),
            signal_streams: self.signal_streams.into_iter().collect::<Vec<_>>().into(),
            events: self.events.into_iter().collect::<Vec<_>>().into(),
            intervals: self.intervals.into_iter().collect::<Vec<_>>().into(),
            tables: self.tables.into_iter().collect::<Vec<_>>().into(),
            tensors: self.tensors.into_iter().collect::<Vec<_>>().into(),
            coordinate_frames: self
                .coordinate_frames
                .into_iter()
                .collect::<Vec<_>>()
                .into(),
            coordinates: self.coordinates.into_iter().collect::<Vec<_>>().into(),
            reference_nodes: self.reference_nodes.into_iter().collect::<Vec<_>>().into(),
            reference_edges: self.reference_edges.into_iter().collect::<Vec<_>>().into(),
            relationships: self.relationships.into_iter().collect::<Vec<_>>().into(),
            attachments: self.attachments.into_iter().collect::<Vec<_>>().into(),
            provenance: self.provenance.into_iter().collect::<Vec<_>>().into(),
            loss_receipts: self.loss_receipts.into_iter().collect::<Vec<_>>().into(),
            extensions: self.extensions,
        };
        recording.verify()?;
        Ok(recording)
    }
}

/// Immutable recording container.
#[derive(Clone, Debug)]
pub struct Recording {
    identity: RecordingIdentity,
    clocks: Arc<[crate::semantic::Clock]>,
    signal_streams: Arc<[SignalStream]>,
    events: Arc<[Event]>,
    intervals: Arc<[Interval]>,
    tables: Arc<[Table]>,
    tensors: Arc<[Tensor]>,
    coordinate_frames: Arc<[CoordinateFrame]>,
    coordinates: Arc<[CoordinatePoint]>,
    reference_nodes: Arc<[ReferenceNode]>,
    reference_edges: Arc<[ReferenceEdge]>,
    relationships: Arc<[Relationship]>,
    attachments: Arc<[Attachment]>,
    provenance: Arc<[ProvenanceActivity]>,
    loss_receipts: Arc<[LossReceipt]>,
    extensions: PropertyBag,
}

impl Recording {
    /// Constructed only via [`RecordingBuilder`], immutable after freeze.
    pub fn identity(&self) -> &RecordingIdentity {
        &self.identity
    }

    /// All clocks.
    pub fn clocks(&self) -> &[Clock] {
        &self.clocks
    }

    /// All streams.
    pub fn signal_streams(&self) -> &[SignalStream] {
        &self.signal_streams
    }

    /// All events.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// All intervals.
    pub fn intervals(&self) -> &[Interval] {
        &self.intervals
    }

    /// All tables.
    pub fn tables(&self) -> &[Table] {
        &self.tables
    }

    /// All tensors.
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// All coordinate frames.
    pub fn coordinate_frames(&self) -> &[CoordinateFrame] {
        &self.coordinate_frames
    }

    /// All coordinates.
    pub fn coordinates(&self) -> &[CoordinatePoint] {
        &self.coordinates
    }

    /// All reference nodes.
    pub fn reference_nodes(&self) -> &[ReferenceNode] {
        &self.reference_nodes
    }

    /// All reference edges.
    pub fn reference_edges(&self) -> &[ReferenceEdge] {
        &self.reference_edges
    }

    /// All relationships.
    pub fn relationships(&self) -> &[Relationship] {
        &self.relationships
    }

    /// All attachments.
    pub fn attachments(&self) -> &[Attachment] {
        &self.attachments
    }

    /// All provenance activities.
    pub fn provenance(&self) -> &[ProvenanceActivity] {
        &self.provenance
    }

    /// All loss receipts.
    pub fn loss_receipts(&self) -> &[LossReceipt] {
        &self.loss_receipts
    }

    /// Extensions.
    pub fn extensions(&self) -> &PropertyBag {
        &self.extensions
    }

    /// Filter streams by modality.
    pub fn streams_by_modality(&self, modality: &ModalityId) -> Vec<&SignalStream> {
        self.signal_streams
            .iter()
            .filter(|stream| stream.modality() == modality)
            .collect()
    }

    fn insert_node_id(node_id: &str, ids: &mut BTreeSet<Arc<str>>) -> Result<(), RecordingError> {
        let id: Arc<str> = Arc::from(node_id);
        if !ids.insert(id.clone()) {
            return Err(RecordingError::DuplicateNodeId { node_id: id });
        }
        Ok(())
    }

    fn unique_properties(properties: &PropertyBag) -> Result<(), RecordingError> {
        let mut seen: BTreeSet<(Arc<str>, Arc<str>)> = BTreeSet::new();
        for property in properties.properties() {
            let name = property.name();
            let key = (Arc::from(name.namespace()), Arc::from(name.local()));
            if !seen.insert(key) {
                return Err(RecordingError::DuplicatePropertyName { name: name.clone() });
            }
        }
        Ok(())
    }

    fn has_node(node_id: &str, node_ids: &BTreeSet<Arc<str>>) -> bool {
        node_ids.contains(node_id)
    }

    /// Cross-check graph and value constraints.
    pub fn verify(&self) -> Result<(), RecordingError> {
        let mut clock_ids: BTreeSet<Arc<str>> = BTreeSet::new();
        let mut node_ids: BTreeSet<Arc<str>> = BTreeSet::new();
        let mut channel_ids: BTreeSet<Arc<str>> = BTreeSet::new();

        // Collect every addressable ID before resolving edges. Forward references
        // are legal and must not depend on storage-family ordering.
        for clock in self.clocks.iter() {
            if !clock_ids.insert(Arc::from(clock.id())) {
                return Err(RecordingError::DuplicateNodeId {
                    node_id: Arc::from(clock.id()),
                });
            }
            Self::insert_node_id(clock.id(), &mut node_ids)?;
        }
        for stream in self.signal_streams.iter() {
            Self::insert_node_id(stream.id(), &mut node_ids)?;
            for series in stream.series() {
                if !channel_ids.insert(Arc::from(series.channel().id())) {
                    return Err(RecordingError::DuplicateChannelId(Arc::from(
                        series.channel().id(),
                    )));
                }
                Self::insert_node_id(series.channel().id(), &mut node_ids)?;
            }
        }
        for event in self.events.iter() {
            Self::insert_node_id(event.id(), &mut node_ids)?;
        }
        for interval in self.intervals.iter() {
            Self::insert_node_id(interval.id(), &mut node_ids)?;
        }
        for table in self.tables.iter() {
            Self::insert_node_id(table.id(), &mut node_ids)?;
        }
        for tensor in self.tensors.iter() {
            Self::insert_node_id(tensor.id(), &mut node_ids)?;
        }
        for frame in self.coordinate_frames.iter() {
            Self::insert_node_id(frame.id(), &mut node_ids)?;
        }
        for point in self.coordinates.iter() {
            Self::insert_node_id(point.id(), &mut node_ids)?;
        }
        for node in self.reference_nodes.iter() {
            Self::insert_node_id(node.id(), &mut node_ids)?;
        }
        for edge in self.reference_edges.iter() {
            Self::insert_node_id(edge.id(), &mut node_ids)?;
        }
        for relation in self.relationships.iter() {
            Self::insert_node_id(relation.id(), &mut node_ids)?;
        }
        for attachment in self.attachments.iter() {
            Self::insert_node_id(attachment.id(), &mut node_ids)?;
        }
        for activity in self.provenance.iter() {
            Self::insert_node_id(activity.id(), &mut node_ids)?;
        }
        for receipt in self.loss_receipts.iter() {
            Self::insert_node_id(receipt.id(), &mut node_ids)?;
        }

        // Validate values and cross-references only after the ID universe is complete.
        for stream in self.signal_streams.iter() {
            for series in stream.series() {
                if series.channel().modality() != stream.modality() {
                    return Err(RecordingError::StreamChannelModalityMismatch {
                        stream_id: Arc::from(stream.id()),
                        channel_id: Arc::from(series.channel().id()),
                    });
                }
                if !clock_ids.contains(series.time_axis().clock_id()) {
                    return Err(RecordingError::UnknownClockId {
                        clock_id: Arc::from(series.time_axis().clock_id()),
                    });
                }
                if let Some(ticks) = series.time_axis().explicit_ticks() {
                    if ticks.len() != series.len() {
                        return Err(RecordingError::ExplicitTimestampCountMismatch {
                            stream_id: Arc::from(stream.id()),
                            channel_id: Arc::from(series.channel().id()),
                            explicit_ticks: ticks.len(),
                            sample_count: series.len(),
                        });
                    }
                    if ticks.windows(2).any(|pair| pair[1] < pair[0]) {
                        return Err(RecordingError::ExplicitTimestampOrderViolation {
                            stream_id: Arc::from(stream.id()),
                            channel_id: Arc::from(series.channel().id()),
                        });
                    }
                }
            }
        }
        for event in self.events.iter() {
            if !clock_ids.contains(event.clock_id()) {
                return Err(RecordingError::UnknownClockId {
                    clock_id: Arc::from(event.clock_id()),
                });
            }
            Self::unique_properties(event.properties())?;
        }
        for interval in self.intervals.iter() {
            if !clock_ids.contains(interval.clock_id()) {
                return Err(RecordingError::UnknownClockId {
                    clock_id: Arc::from(interval.clock_id()),
                });
            }
            if interval.start_tick() > interval.end_tick() {
                return Err(RecordingError::InvalidIntervalBounds {
                    interval_id: Arc::from(interval.id()),
                    start_tick: interval.start_tick(),
                    end_tick: interval.end_tick(),
                });
            }
        }
        for table in self.tables.iter() {
            let expected = table
                .columns()
                .first()
                .map_or(0, |column| column.values().len());
            let mut names: BTreeSet<(Arc<str>, Arc<str>)> = BTreeSet::new();
            for column in table.columns() {
                let key = (
                    Arc::from(column.name().namespace()),
                    Arc::from(column.name().local()),
                );
                if !names.insert(key) {
                    return Err(RecordingError::DuplicateTableColumnName {
                        table_id: Arc::from(table.id()),
                        name: column.name().clone(),
                    });
                }
                if column.values().len() != expected {
                    return Err(RecordingError::TableColumnLengthMismatch {
                        table_id: Arc::from(table.id()),
                        first_len: expected,
                        second_len: column.values().len(),
                    });
                }
                for value in column.values() {
                    if let Some(actual) = value.value_type() {
                        if actual != column.value_type() {
                            return Err(RecordingError::TableColumnTypeMismatch {
                                table_id: Arc::from(table.id()),
                                column: Arc::from(column.name().local()),
                                expected: column.value_type(),
                                actual,
                            });
                        }
                    }
                }
            }
        }
        for tensor in self.tensors.iter() {
            let expected = tensor
                .shape()
                .iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| RecordingError::TensorElementCountMismatch {
                    tensor_id: Arc::from(tensor.id()),
                    expected: u64::MAX,
                    actual: tensor.buffer().len(),
                })?;
            if expected != tensor.buffer().len() as u64 {
                return Err(RecordingError::TensorElementCountMismatch {
                    tensor_id: Arc::from(tensor.id()),
                    expected,
                    actual: tensor.buffer().len(),
                });
            }
        }
        for point in self.coordinates.iter() {
            if !Self::has_node(point.object_id(), &node_ids) {
                return Err(RecordingError::UnknownNodeId {
                    node_id: Arc::from(point.object_id()),
                });
            }
            let frame = self
                .coordinate_frames
                .iter()
                .find(|frame| frame.id() == point.frame_id())
                .ok_or_else(|| RecordingError::UnknownNodeId {
                    node_id: Arc::from(point.frame_id()),
                })?;
            if point.values().len() != frame.dimension() {
                return Err(RecordingError::CoordinateDimensionMismatch {
                    coordinate_id: Arc::from(point.id()),
                    frame_id: Arc::from(frame.id()),
                    expected: frame.dimension(),
                    actual: point.values().len(),
                });
            }
        }
        for edge in self.reference_edges.iter() {
            for endpoint in [edge.from(), edge.to()] {
                if !Self::has_node(endpoint, &node_ids) {
                    return Err(RecordingError::UnknownNodeId {
                        node_id: Arc::from(endpoint),
                    });
                }
            }
        }
        for relation in self.relationships.iter() {
            for endpoint in [relation.subject(), relation.object()] {
                if !Self::has_node(endpoint, &node_ids) {
                    return Err(RecordingError::UnknownNodeId {
                        node_id: Arc::from(endpoint),
                    });
                }
            }
        }
        for provenance in self.provenance.iter() {
            for input in provenance.inputs() {
                if !Self::has_node(input, &node_ids) {
                    return Err(RecordingError::UnknownNodeId {
                        node_id: Arc::from(input.as_ref()),
                    });
                }
            }
            for output in provenance.outputs() {
                if !Self::has_node(output, &node_ids) {
                    return Err(RecordingError::UnknownNodeId {
                        node_id: Arc::from(output.as_ref()),
                    });
                }
            }
        }

        Self::unique_properties(&self.extensions)?;
        Ok(())
    }
}
