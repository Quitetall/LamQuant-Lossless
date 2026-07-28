//! Adapter-independent owned ABIR value used at Node execution seams.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

use semantic_abir::{
    AbirDataset, ContentId, ElementType, OpenedDataset, PayloadAccess, PayloadAccessError,
    PayloadDescriptor, PayloadLease,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbirDatasetValueError {
    ConflictingPayload(ContentId),
    MissingPayload(ContentId),
    InvalidPayload(ContentId),
    RetainedExtentOverflow,
    RetainedExtentExceeded,
}

impl fmt::Display for AbirDatasetValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AbirDatasetValueError {}

#[derive(Clone, Debug, Default)]
pub struct NodePayloadStore {
    payloads: BTreeMap<ContentId, Vec<u8>>,
}

impl NodePayloadStore {
    pub fn get(&self, content_id: ContentId) -> Option<&[u8]> {
        self.payloads.get(&content_id).map(Vec::as_slice)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NodePayloadLease<'a>(&'a [u8]);

impl PayloadLease for NodePayloadLease<'_> {
    fn bytes(&self) -> &[u8] {
        self.0
    }
}

impl PayloadAccess for NodePayloadStore {
    type Lease<'a>
        = NodePayloadLease<'a>
    where
        Self: 'a;

    fn lease<'a>(
        &'a self,
        descriptor: &PayloadDescriptor,
    ) -> Result<Self::Lease<'a>, PayloadAccessError> {
        let bytes = self
            .payloads
            .get(&descriptor.content_id())
            .ok_or(PayloadAccessError::NotFound(descriptor.content_id()))?;
        if bytes.len() as u64 != descriptor.logical_bytes() {
            return Err(PayloadAccessError::LengthMismatch {
                expected: descriptor.logical_bytes(),
                actual: bytes.len(),
            });
        }
        Ok(NodePayloadLease(bytes))
    }
}

/// Validated ABIR semantics paired with a complete read-only payload closure.
///
/// Construction verifies logical identity, payload closure, and retained-size
/// bounds. Producers need no dependency on foreign-format adapters.
#[derive(Debug)]
pub struct AbirDatasetValue {
    opened: OpenedDataset<NodePayloadStore>,
}

impl AbirDatasetValue {
    pub fn try_new(
        dataset: AbirDataset,
        payloads: impl IntoIterator<Item = (ContentId, Vec<u8>)>,
        max_retained_bytes: u64,
    ) -> Result<Self, AbirDatasetValueError> {
        let mut store = NodePayloadStore::default();
        let mut retained_bytes = dataset.semantic_metadata_budget_bytes() as u64;
        if retained_bytes > max_retained_bytes {
            return Err(AbirDatasetValueError::RetainedExtentExceeded);
        }
        for (content_id, bytes) in payloads {
            if let Some(existing) = store.payloads.get(&content_id) {
                if existing != &bytes {
                    return Err(AbirDatasetValueError::ConflictingPayload(content_id));
                }
                continue;
            }
            retained_bytes = retained_bytes
                .checked_add(bytes.len() as u64)
                .and_then(|value| value.checked_add(64))
                .ok_or(AbirDatasetValueError::RetainedExtentOverflow)?;
            if retained_bytes > max_retained_bytes {
                return Err(AbirDatasetValueError::RetainedExtentExceeded);
            }
            store.payloads.insert(content_id, bytes);
        }
        for content_id in referenced_content_ids(&dataset) {
            if !store.payloads.contains_key(&content_id) {
                return Err(AbirDatasetValueError::MissingPayload(content_id));
            }
        }
        for atom in dataset.atoms() {
            if let Some(descriptor) = atom.payload() {
                let content_id = descriptor.content_id();
                let bytes = store
                    .get(content_id)
                    .ok_or(AbirDatasetValueError::MissingPayload(content_id))?;
                semantic_abir::verify_payload_content(descriptor, bytes)
                    .map_err(|_| AbirDatasetValueError::InvalidPayload(content_id))?;
            }
        }
        for capsule in dataset.source_capsules() {
            let content_id = capsule.content_id();
            let bytes = store
                .get(content_id)
                .ok_or(AbirDatasetValueError::MissingPayload(content_id))?;
            if semantic_abir::payload_content_id(ElementType::Bytes, bytes) != content_id {
                return Err(AbirDatasetValueError::InvalidPayload(content_id));
            }
        }
        Ok(Self {
            opened: OpenedDataset::new(dataset, store),
        })
    }

    pub fn dataset(&self) -> &AbirDataset {
        self.opened.dataset()
    }

    pub fn opened(&self) -> &OpenedDataset<NodePayloadStore> {
        &self.opened
    }

    pub fn payloads(&self) -> &NodePayloadStore {
        self.opened.access()
    }

    pub fn into_opened(self) -> OpenedDataset<NodePayloadStore> {
        self.opened
    }
}

fn referenced_content_ids(dataset: &AbirDataset) -> BTreeSet<ContentId> {
    let mut content_ids = dataset
        .payload_content_ids()
        .into_iter()
        .collect::<BTreeSet<_>>();
    content_ids.extend(
        dataset
            .clock_relations()
            .iter()
            .map(semantic_abir::ClockRelation::provenance),
    );
    content_ids.extend(dataset.proofs().iter().map(semantic_abir::Proof::payload));
    content_ids.extend(
        dataset
            .derived_artifacts()
            .iter()
            .map(semantic_abir::DerivedArtifact::content_id),
    );
    content_ids.extend(
        dataset
            .source_capsules()
            .iter()
            .map(semantic_abir::SourceCapsule::content_id),
    );
    for atom in dataset.atoms() {
        if let semantic_abir::Atom::BlobRef(blob) = atom {
            content_ids.insert(blob.integrity().digest());
        }
    }
    content_ids
}
