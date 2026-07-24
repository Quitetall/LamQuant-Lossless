#![forbid(unsafe_code)]

use abir_adapter::{
    Adapter, AdapterCapability, AdapterError, AdapterProfile, ExportPlan, FidelityReceipt,
    ForeignEntry, ForeignObject, ImportOutcome, InspectReport, MappingDisposition, MappingEntry,
    MappingReport, PayloadObject, PayloadResolver, ProfileId, ProfileStatus, SemanticCoverage,
    ValidationArtifact,
};
use hdf5_metno::types::{IntSize, TypeDescriptor};
use lamquant_core::source::{
    from_signal_bundle_with_interchange_bound_sources, DicomWaveformReader, EdfReader,
    SemanticSourceObject, SemanticTimedEvent, SignalBundle, SignalSourceReader, SourceMetadata,
};
use semantic_abir::{AbirDataset, ContentId, PayloadAccess, PayloadLease, ValidationLimits};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

mod dicom_full;
mod edf_full;
mod nwb_full;
mod xdf;
pub use dicom_full::DicomSemanticAdapter;
pub use edf_full::EdfAdapter;
pub use nwb_full::NwbAdapter;
pub use xdf::XdfAdapter;

#[derive(Clone, Copy)]
enum ParserKind {
    Edf,
    Dicom,
    Nwb,
}

struct StandardFileAdapter {
    profile: AdapterProfile,
    max_source_bytes: u64,
    parser: ParserKind,
}

impl StandardFileAdapter {
    fn new(profile: AdapterProfile, parser: ParserKind, max_source_bytes: u64) -> Self {
        Self {
            profile,
            max_source_bytes,
            parser,
        }
    }

    fn check<'a>(&self, source: &'a ForeignObject) -> Result<&'a ForeignEntry, AdapterError> {
        if source.profile != self.profile.id {
            return Err(AdapterError::ProfileMismatch {
                expected: self.profile.id.clone(),
                actual: source.profile.clone(),
            });
        }
        if source.entries.len() != 1 {
            return Err(AdapterError::InvalidSource(format!(
                "{} semantic profile requires exactly one file",
                self.profile.id.0
            )));
        }
        let entry = &source.entries[0];
        if !valid_relative_path(&entry.path) {
            return Err(AdapterError::InvalidPath(entry.path.clone()));
        }
        if u64::try_from(entry.bytes.len()).map_err(|_| AdapterError::SourceTooLarge)?
            > self.max_source_bytes
        {
            return Err(AdapterError::SourceTooLarge);
        }
        Ok(entry)
    }

    fn read_bundle(&self, entry: &ForeignEntry) -> Result<SignalBundle, AdapterError> {
        let temporary =
            tempfile::tempdir().map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let extension = match self.parser {
            ParserKind::Edf => "edf",
            ParserKind::Dicom => "dcm",
            ParserKind::Nwb => "nwb",
        };
        let path = temporary.path().join(format!("source.{extension}"));
        fs::write(&path, &entry.bytes)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let mut bundle = match self.parser {
            ParserKind::Edf => EdfReader::new(&path)
                .read_bundle()
                .map_err(|error| AdapterError::InvalidSource(error.to_string())),
            ParserKind::Dicom => DicomWaveformReader::new(&path)
                .read_bundle()
                .map_err(|error| AdapterError::InvalidSource(error.to_string())),
            ParserKind::Nwb => read_bounded_nwb_bundle(&path, self.max_source_bytes),
        }?;
        if matches!(self.parser, ParserKind::Edf) && bundle.metadata.format == "EDF+D" {
            return Err(AdapterError::UnsupportedMeaning(
                "EDF+D discontinuities require a piecewise ABIR time-axis mapping".to_owned(),
            ));
        }
        if bundle
            .signal
            .iter()
            .any(|channel| channel.len() != bundle.signal.first().map_or(0, Vec::len))
        {
            return Err(AdapterError::UnsupportedMeaning(
                "mixed-length NWB integer series require direct mixed-rate ABIR mapping".to_owned(),
            ));
        }
        // The adapter binds the complete foreign object as one exact source
        // capsule below. Reader-private reconstruction fragments would be
        // redundant payloads and must not become additional adapter exports.
        bundle.sidecar.clear();
        Ok(bundle)
    }

    fn modality(&self) -> semantic_abir::ConceptId {
        let value = match self.parser {
            ParserKind::Dicom => "abir:modality/ecg",
            ParserKind::Edf | ParserKind::Nwb => "abir:modality/unknown",
        };
        semantic_abir::ConceptId::new(value).expect("static modality concept is canonical")
    }

    fn matching_capsules<'a>(
        &self,
        dataset: &'a AbirDataset,
    ) -> Result<Vec<&'a semantic_abir::SourceCapsule>, AdapterError> {
        let namespace = binding_namespace(&self.profile.id, dataset)?;
        Ok(dataset
            .source_capsules()
            .iter()
            .filter(|capsule| capsule.source().namespace() == namespace)
            .collect())
    }
}

impl Adapter for StandardFileAdapter {
    fn profile(&self) -> &AdapterProfile {
        &self.profile
    }

    fn inspect(&self, source: &ForeignObject) -> Result<InspectReport, AdapterError> {
        let entry = self.check(source)?;
        let bundle = self.read_bundle(entry)?;
        Ok(InspectReport {
            profile: self.profile.id.clone(),
            entry_count: 1,
            logical_bytes: entry.bytes.len() as u64,
            risks: vec![
                "independent validator evidence is required before first-class release".to_owned(),
            ],
            required_resources: BTreeMap::from([
                ("max-source-bytes".to_owned(), self.max_source_bytes),
                ("channels".to_owned(), bundle.n_channels() as u64),
                (
                    "samples".to_owned(),
                    bundle.signal.first().map_or(0, Vec::len) as u64,
                ),
            ]),
        })
    }

    fn import(
        &self,
        source: &ForeignObject,
        limits: ValidationLimits,
    ) -> Result<ImportOutcome, AdapterError> {
        let entry = self.check(source)?;
        let bundle = self.read_bundle(entry)?;
        let source_id = payload_content_id(&entry.bytes);
        let capsule = SemanticSourceObject {
            value: entry.path.clone(),
            bytes: entry.bytes.clone(),
            media_type: entry.media_type.clone(),
        };
        let mapped = from_signal_bundle_with_interchange_bound_sources(
            bundle,
            self.modality(),
            format!("adapter.{}.binding.", self.profile.id.0),
            vec![capsule],
            vec![],
            limits,
        )
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let mut payloads = BTreeMap::new();
        for channel in &mapped.mapping.channels {
            let descriptor = mapped
                .opened
                .dataset()
                .atoms()
                .iter()
                .find(|atom| atom.id() == channel.atom_id)
                .and_then(|atom| atom.payload())
                .ok_or(AdapterError::MissingPayload(channel.content_id))?;
            let lease = mapped
                .opened
                .access()
                .lease(descriptor)
                .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
            payloads.insert(channel.content_id, lease.bytes().to_vec());
        }
        payloads.insert(source_id, entry.bytes.clone());
        let mut entries = vec![MappingEntry {
            source_path: entry.path.clone(),
            target: format!("source-capsule:{source_id}"),
            disposition: MappingDisposition::Exact,
            reason: None,
        }];
        entries.extend(mapped.mapping.channels.iter().map(|channel| MappingEntry {
            source_path: format!("{}/signal/{}", entry.path, channel.index),
            target: format!("atom:{}", channel.atom_id),
            disposition: MappingDisposition::Projected,
            reason: Some(
                "integer samples and rate are promoted; calibration and absolute timing remain capsule-only"
                    .to_owned(),
            ),
        }));
        Ok(ImportOutcome {
            dataset: mapped.opened.dataset().clone(),
            report: MappingReport {
                source_profile: self.profile.id.clone(),
                target_profile: ProfileId("abir.semantic.v1".to_owned()),
                semantic_coverage: SemanticCoverage::ProjectedSemantic,
                entries,
                preserved_unknowns: 1,
                sample_values_changed: false,
                timing_changed: true,
            },
            payloads: payloads
                .into_iter()
                .map(|(content_id, bytes)| PayloadObject { content_id, bytes })
                .collect(),
        })
    }

    fn plan_export(&self, dataset: &AbirDataset) -> Result<ExportPlan, AdapterError> {
        let capsules = self.matching_capsules(dataset)?;
        let unsupported = capsules.len() != 1;
        let mappings = capsules
            .iter()
            .map(|capsule| MappingEntry {
                source_path: capsule.source().value().to_owned(),
                target: capsule.source().value().to_owned(),
                disposition: MappingDisposition::Exact,
                reason: None,
            })
            .collect();
        let mut plan = ExportPlan {
            source_dataset: dataset.id().to_string(),
            target_profile: self.profile.id.clone(),
            mappings,
            requires_user_acceptance: false,
            unsupported,
            plan_id: String::new(),
        };
        plan.plan_id = plan_id(&plan);
        Ok(plan)
    }

    fn export(
        &self,
        dataset: &AbirDataset,
        plan: &ExportPlan,
        payloads: &dyn PayloadResolver,
    ) -> Result<(ForeignObject, FidelityReceipt), AdapterError> {
        let expected = self.plan_export(dataset)?;
        if expected != *plan || plan_id(plan) != plan.plan_id {
            return Err(AdapterError::ExportPlanMismatch);
        }
        if !plan.accepts_without_loss() {
            return Err(AdapterError::UnsupportedMeaning(format!(
                "dataset lacks one exact source capsule for adapter profile {}",
                self.profile.id.0
            )));
        }
        let capsule = self.matching_capsules(dataset)?[0];
        let bytes = payloads.resolve(capsule.content_id())?;
        if payload_content_id(&bytes) != capsule.content_id() {
            return Err(AdapterError::MissingPayload(capsule.content_id()));
        }
        Ok((
            ForeignObject {
                profile: self.profile.id.clone(),
                entries: vec![ForeignEntry {
                    path: capsule.source().value().to_owned(),
                    media_type: capsule.media_type().map(str::to_owned),
                    bytes,
                }],
            },
            FidelityReceipt {
                plan_id: plan.plan_id.clone(),
                exact_source_restoration: true,
                semantic_equivalence: true,
                output_content_ids: vec![capsule.content_id().to_string()],
            },
        ))
    }

    fn validate(&self, source: &ForeignObject) -> ValidationArtifact {
        let result = self
            .check(source)
            .and_then(|entry| self.read_bundle(entry).map(|_| ()));
        ValidationArtifact {
            profile: self.profile.id.clone(),
            internal_valid: result.is_ok(),
            independent_validator: self.profile.required_validator.clone(),
            independent_valid: None,
            diagnostics: result
                .err()
                .map(|error| error.to_string())
                .into_iter()
                .collect(),
        }
    }
}

pub struct DicomAdapter(StandardFileAdapter);

impl DicomAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        Self(StandardFileAdapter::new(
            profile(
                "dicom.ps3.2026c.ecg-i16",
                "DICOM",
                "PS3 2026c signed 16-bit ECG Waveform subset",
                &["application/dicom"],
                "pydicom",
            ),
            ParserKind::Dicom,
            max_source_bytes,
        ))
    }
}

pub struct NwbSubsetAdapter(StandardFileAdapter);

impl NwbSubsetAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        Self(StandardFileAdapter::new(
            profile(
                "nwb.2.10.0.single-integer-timeseries",
                "NWB",
                "2.10.0 single integer acquisition TimeSeries",
                &["application/x-nwb"],
                "pynwb.validate",
            ),
            ParserKind::Nwb,
            max_source_bytes,
        ))
    }
}

pub struct BidsAdapter {
    profile: AdapterProfile,
    max_source_bytes: u64,
    promote_events: bool,
}

pub struct BidsEventsAdapter(BidsAdapter);

struct BidsEvent {
    source_path: String,
    source_row: usize,
    source_content_id: ContentId,
    quarantined_columns: Vec<String>,
    semantic: SemanticTimedEvent,
}

impl BidsAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        Self {
            profile: profile(
                "bids.1.11.1.single-edf-eeg",
                "BIDS",
                "1.11.1 single EDF/BDF EEG recording",
                &["application/vnd.bids.dataset"],
                "bids-validator",
            ),
            max_source_bytes,
            promote_events: false,
        }
    }

    fn check<'a>(&self, source: &'a ForeignObject) -> Result<&'a ForeignEntry, AdapterError> {
        if source.profile != self.profile.id {
            return Err(AdapterError::ProfileMismatch {
                expected: self.profile.id.clone(),
                actual: source.profile.clone(),
            });
        }
        if source.entries.is_empty() {
            return Err(AdapterError::EmptySource);
        }
        let mut paths = BTreeSet::new();
        let mut total = 0_u64;
        for entry in &source.entries {
            if !valid_relative_path(&entry.path) {
                return Err(AdapterError::InvalidPath(entry.path.clone()));
            }
            if !paths.insert(&entry.path) {
                return Err(AdapterError::DuplicatePath(entry.path.clone()));
            }
            let entry_len =
                u64::try_from(entry.bytes.len()).map_err(|_| AdapterError::SourceTooLarge)?;
            total = total
                .checked_add(entry_len)
                .ok_or(AdapterError::SourceTooLarge)?;
        }
        if total > self.max_source_bytes {
            return Err(AdapterError::SourceTooLarge);
        }
        let description = source
            .entries
            .iter()
            .find(|entry| entry.path == "dataset_description.json")
            .ok_or_else(|| {
                AdapterError::InvalidSource("BIDS dataset_description.json is required".to_owned())
            })?;
        let metadata: serde_json::Value = serde_json::from_slice(&description.bytes)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        if metadata.get("BIDSVersion").and_then(|value| value.as_str()) != Some("1.11.1") {
            return Err(AdapterError::InvalidSource(
                "BIDSVersion must equal pinned profile 1.11.1".to_owned(),
            ));
        }
        let mut signals = source.entries.iter().filter(|entry| {
            let lower = entry.path.to_ascii_lowercase();
            lower.ends_with(".edf") || lower.ends_with(".bdf")
        });
        let signal = signals.next().ok_or_else(|| {
            AdapterError::UnsupportedMeaning(
                "current BIDS semantic slice requires one EDF/BDF recording".to_owned(),
            )
        })?;
        if signals.next().is_some() {
            return Err(AdapterError::UnsupportedMeaning(
                "multi-recording BIDS datasets require dataset-root composition".to_owned(),
            ));
        }
        Ok(signal)
    }

    fn capsules<'a>(
        &self,
        dataset: &'a AbirDataset,
    ) -> Result<Vec<&'a semantic_abir::SourceCapsule>, AdapterError> {
        let namespace = binding_namespace(&self.profile.id, dataset)?;
        Ok(dataset
            .source_capsules()
            .iter()
            .filter(|capsule| capsule.source().namespace() == namespace)
            .collect())
    }

    fn capsules_form_complete_tree(capsules: &[&semantic_abir::SourceCapsule]) -> bool {
        let mut paths = BTreeSet::new();
        let mut descriptions = 0_usize;
        let mut signals = 0_usize;
        for capsule in capsules {
            let path = capsule.source().value();
            if !valid_relative_path(path) || !paths.insert(path) {
                return false;
            }
            descriptions += usize::from(path == "dataset_description.json");
            let lower = path.to_ascii_lowercase();
            signals += usize::from(lower.ends_with(".edf") || lower.ends_with(".bdf"));
        }
        descriptions == 1 && signals == 1
    }

    fn events(
        &self,
        source: &ForeignObject,
        signal: &ForeignEntry,
    ) -> Result<Vec<BidsEvent>, AdapterError> {
        if !self.promote_events {
            return Ok(Vec::new());
        }
        let expected = bids_events_path(&signal.path)?;
        let mut candidates = source
            .entries
            .iter()
            .filter(|entry| entry.path.ends_with("_events.tsv"));
        let Some(entry) = candidates.next() else {
            return Ok(Vec::new());
        };
        if candidates.next().is_some() || entry.path != expected {
            return Err(AdapterError::UnsupportedMeaning(
                "bounded BIDS event mapping requires at most one events sidecar matching the recording"
                    .to_owned(),
            ));
        }
        parse_bids_events(entry)
    }
}

impl BidsEventsAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        let mut candidate = profile(
            "bids.1.11.1.single-edf-eeg-events",
            "BIDS",
            "1.11.1 single EDF/BDF EEG recording with bounded events",
            &["application/vnd.bids.dataset"],
            "bids-validator",
        );
        candidate.status = ProfileStatus::Forensic;
        Self(BidsAdapter {
            profile: candidate,
            max_source_bytes,
            promote_events: true,
        })
    }
}

macro_rules! delegate_adapter {
    ($adapter:ty) => {
        impl Adapter for $adapter {
            fn profile(&self) -> &AdapterProfile {
                self.0.profile()
            }

            fn inspect(&self, source: &ForeignObject) -> Result<InspectReport, AdapterError> {
                self.0.inspect(source)
            }

            fn import(
                &self,
                source: &ForeignObject,
                limits: ValidationLimits,
            ) -> Result<ImportOutcome, AdapterError> {
                self.0.import(source, limits)
            }

            fn plan_export(&self, dataset: &AbirDataset) -> Result<ExportPlan, AdapterError> {
                self.0.plan_export(dataset)
            }

            fn export(
                &self,
                dataset: &AbirDataset,
                plan: &ExportPlan,
                payloads: &dyn PayloadResolver,
            ) -> Result<(ForeignObject, FidelityReceipt), AdapterError> {
                self.0.export(dataset, plan, payloads)
            }

            fn validate(&self, source: &ForeignObject) -> ValidationArtifact {
                self.0.validate(source)
            }
        }
    };
}

delegate_adapter!(DicomAdapter);
delegate_adapter!(NwbSubsetAdapter);

impl Adapter for BidsAdapter {
    fn profile(&self) -> &AdapterProfile {
        &self.profile
    }

    fn inspect(&self, source: &ForeignObject) -> Result<InspectReport, AdapterError> {
        let signal = self.check(source)?;
        let events = self.events(source, signal)?;
        let parser =
            StandardFileAdapter::new(self.profile.clone(), ParserKind::Edf, self.max_source_bytes);
        let bundle = parser.read_bundle(signal)?;
        Ok(InspectReport {
            profile: self.profile.id.clone(),
            entry_count: source.entries.len(),
            logical_bytes: source
                .entries
                .iter()
                .map(|entry| entry.bytes.len() as u64)
                .sum(),
            risks: vec![
                if self.promote_events {
                    "BIDS event onset, duration, and trial_type are promoted; all other sidecar fields remain quarantined"
                } else {
                    "BIDS sidecars are preserved but only signal semantics are promoted"
                }
                .to_owned(),
                "independent bids-validator evidence is required for first-class status".to_owned(),
            ],
            required_resources: BTreeMap::from([
                ("max-source-bytes".to_owned(), self.max_source_bytes),
                ("channels".to_owned(), bundle.n_channels() as u64),
                ("events".to_owned(), events.len() as u64),
                (
                    "samples".to_owned(),
                    bundle.signal.first().map_or(0, Vec::len) as u64,
                ),
            ]),
        })
    }

    fn import(
        &self,
        source: &ForeignObject,
        limits: ValidationLimits,
    ) -> Result<ImportOutcome, AdapterError> {
        let signal = self.check(source)?;
        let events = self.events(source, signal)?;
        let semantic_events = events
            .iter()
            .map(|event| event.semantic.clone())
            .collect::<Vec<_>>();
        let parser =
            StandardFileAdapter::new(self.profile.clone(), ParserKind::Edf, self.max_source_bytes);
        let bundle = parser.read_bundle(signal)?;
        let capsules = source
            .entries
            .iter()
            .map(|entry| SemanticSourceObject {
                value: entry.path.clone(),
                bytes: entry.bytes.clone(),
                media_type: entry.media_type.clone(),
            })
            .collect::<Vec<_>>();
        let mapped = from_signal_bundle_with_interchange_bound_sources(
            bundle,
            semantic_abir::ConceptId::new("abir:modality/eeg")
                .expect("static modality concept is canonical"),
            format!("adapter.{}.binding.", self.profile.id.0),
            capsules,
            semantic_events,
            limits,
        )
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let mut payloads = BTreeMap::new();
        for channel in &mapped.mapping.channels {
            let descriptor = mapped
                .opened
                .dataset()
                .atoms()
                .iter()
                .find(|atom| atom.id() == channel.atom_id)
                .and_then(|atom| atom.payload())
                .ok_or(AdapterError::MissingPayload(channel.content_id))?;
            let lease = mapped
                .opened
                .access()
                .lease(descriptor)
                .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
            payloads.insert(channel.content_id, lease.bytes().to_vec());
        }
        for entry in &source.entries {
            payloads.insert(payload_content_id(&entry.bytes), entry.bytes.clone());
        }
        let mut entries = source
            .entries
            .iter()
            .map(|entry| MappingEntry {
                source_path: entry.path.clone(),
                target: format!("source-capsule:{}", payload_content_id(&entry.bytes)),
                disposition: if entry.path == signal.path {
                    MappingDisposition::Exact
                } else {
                    MappingDisposition::Quarantined
                },
                reason: (entry.path != signal.path)
                    .then(|| "preserved sidecar; semantic promotion pending".to_owned()),
            })
            .collect::<Vec<_>>();
        entries.extend(mapped.mapping.channels.iter().map(|channel| MappingEntry {
            source_path: format!("{}/signal/{}", signal.path, channel.index),
            target: format!("atom:{}", channel.atom_id),
            disposition: MappingDisposition::Projected,
            reason: Some(
                "integer samples and rate are promoted; EDF calibration and absolute timing remain capsule-only"
                    .to_owned(),
            ),
        }));
        if events.len() != mapped.mapping.events.len() {
            return Err(AdapterError::InvalidSource(
                "event bridge returned incomplete mapping evidence".to_owned(),
            ));
        }
        for (source_event, mapped_event) in events.iter().zip(&mapped.mapping.events) {
            entries.push(MappingEntry {
                source_path: format!(
                    "{}#row={};fields=onset,duration,trial_type",
                    source_event.source_path, source_event.source_row,
                ),
                target: format!("event:{}", mapped_event.event_id),
                disposition: MappingDisposition::Exact,
                reason: None,
            });
            if !source_event.quarantined_columns.is_empty() {
                entries.push(MappingEntry {
                    source_path: format!(
                        "{}#row={};fields={}",
                        source_event.source_path,
                        source_event.source_row,
                        source_event.quarantined_columns.join(","),
                    ),
                    target: format!("source-capsule:{}", source_event.source_content_id),
                    disposition: MappingDisposition::Quarantined,
                    reason: Some(
                        "event columns outside the bounded semantic core remain source-only"
                            .to_owned(),
                    ),
                });
            }
        }
        Ok(ImportOutcome {
            dataset: mapped.opened.dataset().clone(),
            report: MappingReport {
                source_profile: self.profile.id.clone(),
                target_profile: ProfileId("abir.semantic.v1".to_owned()),
                semantic_coverage: SemanticCoverage::ProjectedSemantic,
                entries,
                // Every non-signal object remains capsule-preserved even when
                // bounded row semantics are also promoted. This count is about
                // recoverable foreign objects, not whether some fields mapped.
                preserved_unknowns: source.entries.len().saturating_sub(1) as u64,
                sample_values_changed: false,
                timing_changed: true,
            },
            payloads: payloads
                .into_iter()
                .map(|(content_id, bytes)| PayloadObject { content_id, bytes })
                .collect(),
        })
    }

    fn plan_export(&self, dataset: &AbirDataset) -> Result<ExportPlan, AdapterError> {
        let capsules = self.capsules(dataset)?;
        // Source capsules can be assembled independently of adapter import.
        // Do not authorize exact BIDS restoration from a merely non-empty
        // matching set: the minimum pinned tree must still be present and its
        // paths must remain unambiguous.
        let unsupported = !Self::capsules_form_complete_tree(&capsules);
        let mappings = capsules
            .iter()
            .map(|capsule| MappingEntry {
                source_path: capsule.source().value().to_owned(),
                target: capsule.source().value().to_owned(),
                disposition: MappingDisposition::Exact,
                reason: None,
            })
            .collect();
        let mut plan = ExportPlan {
            source_dataset: dataset.id().to_string(),
            target_profile: self.profile.id.clone(),
            mappings,
            requires_user_acceptance: false,
            unsupported,
            plan_id: String::new(),
        };
        plan.plan_id = plan_id(&plan);
        Ok(plan)
    }

    fn export(
        &self,
        dataset: &AbirDataset,
        plan: &ExportPlan,
        payloads: &dyn PayloadResolver,
    ) -> Result<(ForeignObject, FidelityReceipt), AdapterError> {
        let expected = self.plan_export(dataset)?;
        if expected != *plan || plan_id(plan) != plan.plan_id {
            return Err(AdapterError::ExportPlanMismatch);
        }
        if !plan.accepts_without_loss() {
            return Err(AdapterError::UnsupportedMeaning(
                "dataset lacks a BIDS source tree".to_owned(),
            ));
        }
        let mut entries = Vec::new();
        let mut output_content_ids = Vec::new();
        for capsule in self.capsules(dataset)? {
            let bytes = payloads.resolve(capsule.content_id())?;
            if payload_content_id(&bytes) != capsule.content_id() {
                return Err(AdapterError::MissingPayload(capsule.content_id()));
            }
            output_content_ids.push(capsule.content_id().to_string());
            entries.push(ForeignEntry {
                path: capsule.source().value().to_owned(),
                media_type: capsule.media_type().map(str::to_owned),
                bytes,
            });
        }
        Ok((
            ForeignObject {
                profile: self.profile.id.clone(),
                entries,
            },
            FidelityReceipt {
                plan_id: plan.plan_id.clone(),
                exact_source_restoration: true,
                semantic_equivalence: true,
                output_content_ids,
            },
        ))
    }

    fn validate(&self, source: &ForeignObject) -> ValidationArtifact {
        let result = self.check(source).and_then(|signal| {
            self.events(source, signal)?;
            StandardFileAdapter::new(self.profile.clone(), ParserKind::Edf, self.max_source_bytes)
                .read_bundle(signal)
                .map(|_| ())
        });
        ValidationArtifact {
            profile: self.profile.id.clone(),
            internal_valid: result.is_ok(),
            independent_validator: self.profile.required_validator.clone(),
            independent_valid: None,
            diagnostics: result
                .err()
                .map(|error| error.to_string())
                .into_iter()
                .collect(),
        }
    }
}

impl Adapter for BidsEventsAdapter {
    fn profile(&self) -> &AdapterProfile {
        self.0.profile()
    }

    fn inspect(&self, source: &ForeignObject) -> Result<InspectReport, AdapterError> {
        self.0.inspect(source)
    }

    fn import(
        &self,
        source: &ForeignObject,
        limits: ValidationLimits,
    ) -> Result<ImportOutcome, AdapterError> {
        self.0.import(source, limits)
    }

    fn plan_export(&self, dataset: &AbirDataset) -> Result<ExportPlan, AdapterError> {
        self.0.plan_export(dataset)
    }

    fn export(
        &self,
        dataset: &AbirDataset,
        plan: &ExportPlan,
        payloads: &dyn PayloadResolver,
    ) -> Result<(ForeignObject, FidelityReceipt), AdapterError> {
        self.0.export(dataset, plan, payloads)
    }

    fn validate(&self, source: &ForeignObject) -> ValidationArtifact {
        self.0.validate(source)
    }
}

fn profile(
    id: &str,
    standard: &str,
    edition: &str,
    media_types: &[&str],
    validator: &str,
) -> AdapterProfile {
    AdapterProfile {
        id: ProfileId(id.to_owned()),
        standard: standard.to_owned(),
        edition: edition.to_owned(),
        media_types: media_types
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        status: ProfileStatus::Forensic,
        required_validator: validator.to_owned(),
        capabilities: BTreeSet::from([
            AdapterCapability::Inspect,
            AdapterCapability::Import,
            AdapterCapability::PlanExport,
            AdapterCapability::Export,
            AdapterCapability::Validate,
        ]),
    }
}

fn binding_namespace(profile: &ProfileId, dataset: &AbirDataset) -> Result<String, AdapterError> {
    let semantic = semantic_abir::interchange_content_id(dataset)
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    Ok(format!("adapter.{}.binding.{semantic}", profile.0))
}

fn valid_relative_path(path: &str) -> bool {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.chars().any(char::is_control)
    {
        return false;
    }
    let mut parts = path.split('/');
    let Some(first) = parts.next() else {
        return false;
    };
    if first.ends_with(':') || first.is_empty() || first == "." || first == ".." {
        return false;
    }
    parts.all(|part| !part.is_empty() && part != "." && part != "..")
}

fn bids_events_path(signal_path: &str) -> Result<String, AdapterError> {
    let (without_extension, extension) = signal_path.rsplit_once('.').ok_or_else(|| {
        AdapterError::InvalidSource("BIDS signal path has no extension".to_owned())
    })?;
    if !matches!(extension.to_ascii_lowercase().as_str(), "edf" | "bdf") {
        return Err(AdapterError::UnsupportedMeaning(
            "bounded BIDS event mapping requires an EDF/BDF signal".to_owned(),
        ));
    }
    let prefix = without_extension.strip_suffix("_eeg").ok_or_else(|| {
        AdapterError::UnsupportedMeaning(
            "bounded BIDS event mapping requires an _eeg signal suffix".to_owned(),
        )
    })?;
    Ok(format!("{prefix}_events.tsv"))
}

fn parse_bids_events(entry: &ForeignEntry) -> Result<Vec<BidsEvent>, AdapterError> {
    let text = std::str::from_utf8(&entry.bytes)
        .map_err(|_| AdapterError::InvalidSource("BIDS events TSV is not UTF-8".to_owned()))?;
    let mut lines = text.split_terminator('\n');
    let header = lines
        .next()
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .ok_or_else(|| AdapterError::InvalidSource("BIDS events TSV is empty".to_owned()))?;
    let columns = header.split('\t').collect::<Vec<_>>();
    let unique = columns.iter().copied().collect::<BTreeSet<_>>();
    if columns.len() != unique.len()
        || columns.iter().any(|column| {
            column.is_empty()
                || !column
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        })
    {
        return Err(AdapterError::InvalidSource(
            "BIDS events TSV has empty or duplicate columns".to_owned(),
        ));
    }
    let index = |name: &str| {
        columns
            .iter()
            .position(|column| *column == name)
            .ok_or_else(|| {
                AdapterError::UnsupportedMeaning(format!(
                    "bounded BIDS event mapping requires {name}"
                ))
            })
    };
    let onset = index("onset")?;
    let duration = index("duration")?;
    let trial_type = index("trial_type")?;
    let quarantined_columns = columns
        .iter()
        .filter(|column| !matches!(**column, "onset" | "duration" | "trial_type"))
        .map(|column| (*column).to_owned())
        .collect::<Vec<_>>();
    let source_content_id = payload_content_id(&entry.bytes);
    let mut events = Vec::new();
    for (line_index, raw_line) in lines.enumerate() {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() {
            return Err(AdapterError::InvalidSource(format!(
                "BIDS events TSV row {} is empty",
                line_index + 2
            )));
        }
        let values = line.split('\t').collect::<Vec<_>>();
        if values.len() != columns.len() {
            return Err(AdapterError::InvalidSource(format!(
                "BIDS events TSV row {} has the wrong column count",
                line_index + 2
            )));
        }
        let start = parse_exact_decimal(values[onset])?;
        let duration = parse_exact_decimal(values[duration])?;
        if duration.parts().0 < 0 {
            return Err(AdapterError::UnsupportedMeaning(format!(
                "BIDS events TSV row {} has negative duration",
                line_index + 2
            )));
        }
        let end = rational_add(start, duration)?;
        if values[trial_type] == "n/a" {
            return Err(AdapterError::UnsupportedMeaning(format!(
                "BIDS events TSV row {} has no trial_type for the bounded event-kind mapping",
                line_index + 2
            )));
        }
        let kind = semantic_abir::ConceptId::new(format!("bids:event/{}", values[trial_type]))
            .map_err(|_| {
                AdapterError::UnsupportedMeaning(format!(
                    "BIDS events TSV row {} trial_type is not a canonical concept token",
                    line_index + 2
                ))
            })?;
        events.push(BidsEvent {
            source_path: entry.path.clone(),
            source_row: line_index + 2,
            source_content_id,
            quarantined_columns: quarantined_columns.clone(),
            semantic: SemanticTimedEvent {
                kind,
                start,
                end,
                uncertainty: semantic_abir::Rational::new(0, 1).expect("zero is canonical"),
            },
        });
    }
    Ok(events)
}

fn parse_exact_decimal(value: &str) -> Result<semantic_abir::Rational, AdapterError> {
    if value.is_empty() || value.trim() != value {
        return Err(AdapterError::InvalidSource(
            "BIDS event time is not a canonical decimal".to_owned(),
        ));
    }
    let (mantissa, exponent) = match value.split_once(['e', 'E']) {
        Some((mantissa, exponent))
            if !mantissa.is_empty() && !exponent.is_empty() && !exponent.contains(['e', 'E']) =>
        {
            let exponent = exponent.parse::<i32>().map_err(|_| {
                AdapterError::InvalidSource("BIDS event exponent is invalid".to_owned())
            })?;
            (mantissa, exponent)
        }
        Some(_) => {
            return Err(AdapterError::InvalidSource(
                "BIDS event exponent is invalid".to_owned(),
            ));
        }
        None => (value, 0),
    };
    let negative = mantissa.starts_with('-');
    let unsigned = mantissa.strip_prefix(['-', '+']).unwrap_or(mantissa);
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        || unsigned.matches('.').count() > 1
    {
        return Err(AdapterError::InvalidSource(
            "BIDS event time is not a canonical decimal".to_owned(),
        ));
    }
    let digits = format!("{whole}{fraction}");
    let mut numerator = digits.parse::<i128>().map_err(|_| {
        AdapterError::UnsupportedMeaning("BIDS event time exceeds exact numeric limits".to_owned())
    })?;
    if negative {
        numerator = numerator.checked_neg().ok_or_else(|| {
            AdapterError::UnsupportedMeaning(
                "BIDS event time exceeds exact numeric limits".to_owned(),
            )
        })?;
    }
    let scale = i32::try_from(fraction.len())
        .map_err(|_| AdapterError::UnsupportedMeaning("BIDS event scale is too large".to_owned()))?
        .checked_sub(exponent)
        .ok_or_else(|| {
            AdapterError::UnsupportedMeaning("BIDS event scale is too large".to_owned())
        })?;
    let (numerator, denominator) = if scale >= 0 {
        let denominator = 10_i128.checked_pow(scale as u32).ok_or_else(|| {
            AdapterError::UnsupportedMeaning("BIDS event scale is too large".to_owned())
        })?;
        (numerator, denominator)
    } else {
        let factor = 10_i128.checked_pow(scale.unsigned_abs()).ok_or_else(|| {
            AdapterError::UnsupportedMeaning("BIDS event scale is too large".to_owned())
        })?;
        (
            numerator.checked_mul(factor).ok_or_else(|| {
                AdapterError::UnsupportedMeaning(
                    "BIDS event time exceeds exact numeric limits".to_owned(),
                )
            })?,
            1,
        )
    };
    semantic_abir::Rational::new(numerator, denominator)
        .map_err(|_| AdapterError::InvalidSource("BIDS event time is invalid".to_owned()))
}

fn rational_add(
    left: semantic_abir::Rational,
    right: semantic_abir::Rational,
) -> Result<semantic_abir::Rational, AdapterError> {
    let (left_numerator, left_denominator) = left.parts();
    let (right_numerator, right_denominator) = right.parts();
    let numerator = left_numerator
        .checked_mul(right_denominator)
        .and_then(|left| {
            right_numerator
                .checked_mul(left_denominator)
                .and_then(|right| left.checked_add(right))
        })
        .ok_or_else(|| {
            AdapterError::UnsupportedMeaning("BIDS event interval exceeds exact limits".to_owned())
        })?;
    let denominator = left_denominator
        .checked_mul(right_denominator)
        .ok_or_else(|| {
            AdapterError::UnsupportedMeaning("BIDS event interval exceeds exact limits".to_owned())
        })?;
    semantic_abir::Rational::new(numerator, denominator)
        .map_err(|_| AdapterError::InvalidSource("BIDS event interval is invalid".to_owned()))
}

fn read_bounded_nwb_bundle(
    path: &std::path::Path,
    max_expanded_bytes: u64,
) -> Result<SignalBundle, AdapterError> {
    let file = hdf5_metno::File::open(path)
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    let acquisition = file
        .group("acquisition")
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    let members = acquisition
        .member_names()
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    if members.len() != 1 {
        return Err(AdapterError::UnsupportedMeaning(
            "current NWB semantic profile requires one acquisition TimeSeries".to_owned(),
        ));
    }
    let series = acquisition
        .group(&members[0])
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    let starting_time = series
        .dataset("starting_time")
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    let start = starting_time
        .read_scalar::<f64>()
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    if !start.is_finite() || start != 0.0 {
        return Err(AdapterError::UnsupportedMeaning(
            "nonzero NWB starting_time requires exact ABIR time-origin promotion".to_owned(),
        ));
    }
    let rate = starting_time
        .attr("rate")
        .and_then(|attribute| attribute.read_scalar::<f64>())
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    if !rate.is_finite() || rate <= 0.0 {
        return Err(AdapterError::InvalidSource(
            "NWB TimeSeries rate must be finite and positive".to_owned(),
        ));
    }
    let data = series
        .dataset("data")
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    let shape = data.shape();
    if shape.is_empty() || shape.len() > 2 || shape.contains(&0) {
        return Err(AdapterError::UnsupportedMeaning(
            "bounded NWB profile requires a nonempty rank-1 or rank-2 acquisition data array"
                .to_owned(),
        ));
    }
    let elements = shape
        .iter()
        .try_fold(1_u64, |total, extent| total.checked_mul(*extent as u64));
    let expanded_bytes = elements
        .and_then(|count| count.checked_mul(8))
        .ok_or(AdapterError::SourceTooLarge)?;
    if expanded_bytes > max_expanded_bytes {
        return Err(AdapterError::SourceTooLarge);
    }

    let descriptor = data
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    let flat = match descriptor {
        TypeDescriptor::Integer(IntSize::U1) => read_nwb_values::<i8>(&data, i64::from)?,
        TypeDescriptor::Integer(IntSize::U2) => read_nwb_values::<i16>(&data, i64::from)?,
        TypeDescriptor::Integer(IntSize::U4) => read_nwb_values::<i32>(&data, i64::from)?,
        TypeDescriptor::Integer(IntSize::U8) => read_nwb_values::<i64>(&data, |value| value)?,
        TypeDescriptor::Unsigned(IntSize::U1) => read_nwb_values::<u8>(&data, i64::from)?,
        TypeDescriptor::Unsigned(IntSize::U2) => read_nwb_values::<u16>(&data, i64::from)?,
        TypeDescriptor::Unsigned(IntSize::U4) => read_nwb_values::<u32>(&data, i64::from)?,
        TypeDescriptor::Unsigned(IntSize::U8) => {
            let values = data
                .read_raw::<u64>()
                .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
            values
                .into_iter()
                .map(|value| {
                    i64::try_from(value).map_err(|_| {
                        AdapterError::UnsupportedMeaning(
                            "NWB u64 signal value exceeds the current ABIR integer range"
                                .to_owned(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => {
            return Err(AdapterError::UnsupportedMeaning(
                "bounded NWB profile requires integer acquisition data".to_owned(),
            ));
        }
    };
    let signal = if shape.len() == 1 {
        vec![flat]
    } else {
        let (samples, channels) = (shape[0], shape[1]);
        let mut signal = (0..channels)
            .map(|_| Vec::with_capacity(samples))
            .collect::<Vec<_>>();
        for row in flat.chunks_exact(channels) {
            for (channel, value) in signal.iter_mut().zip(row) {
                channel.push(*value);
            }
        }
        signal
    };
    let samples = signal.first().map_or(0, Vec::len);
    let channels = (0..signal.len())
        .map(|index| format!("acquisition/{}/data/{index}", members[0]))
        .collect::<Vec<_>>();
    Ok(SignalBundle {
        signal,
        sample_rate: rate,
        channels,
        phys_min: vec![0.0; shape.get(1).copied().unwrap_or(1)],
        phys_max: vec![0.0; shape.get(1).copied().unwrap_or(1)],
        duration_s: samples as f64 / rate,
        metadata: SourceMetadata {
            source_file: path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            format: "NWB".to_owned(),
            patient_id: String::new(),
            recording_info: format!("/acquisition/{}", members[0]),
            startdate: String::new(),
            phys_dim: String::new(),
        },
        sidecar: vec![],
    })
}

fn read_nwb_values<T>(
    data: &hdf5_metno::Dataset,
    widen: impl Fn(T) -> i64,
) -> Result<Vec<i64>, AdapterError>
where
    T: hdf5_metno::H5Type,
{
    data.read_raw::<T>()
        .map(|values| values.into_iter().map(widen).collect())
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

pub fn payload_content_id(bytes: &[u8]) -> ContentId {
    semantic_abir::payload_content_id(semantic_abir::ElementType::Bytes, bytes)
}

fn plan_id(plan: &ExportPlan) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"abir.adapter.export-plan.v2\0");
    hash_field(&mut hasher, plan.source_dataset.as_bytes());
    hash_field(&mut hasher, plan.target_profile.0.as_bytes());
    hasher.update(&[plan.requires_user_acceptance as u8, plan.unsupported as u8]);
    for mapping in &plan.mappings {
        hash_field(&mut hasher, mapping.source_path.as_bytes());
        hash_field(&mut hasher, mapping.target.as_bytes());
        hash_field(&mut hasher, disposition_label(mapping.disposition));
        match &mapping.reason {
            Some(reason) => {
                hasher.update(&[1]);
                hash_field(&mut hasher, reason.as_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn disposition_label(disposition: MappingDisposition) -> &'static [u8] {
    match disposition {
        MappingDisposition::Exact => b"exact",
        MappingDisposition::Projected => b"projected",
        MappingDisposition::Lossy => b"lossy",
        MappingDisposition::Quarantined => b"quarantined",
        MappingDisposition::Unsupported => b"unsupported",
        MappingDisposition::UserDecision => b"user-decision",
    }
}
