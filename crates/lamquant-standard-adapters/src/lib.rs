#![forbid(unsafe_code)]

use abir_adapter::{
    Adapter, AdapterCapability, AdapterError, AdapterProfile, ExportPlan, FidelityReceipt,
    ForeignEntry, ForeignObject, ImportOutcome, InspectReport, MappingDisposition, MappingEntry,
    MappingReport, PayloadObject, PayloadResolver, ProfileId, ProfileStatus, SemanticCoverage,
    ValidationArtifact,
};
use lamquant_abir_bridge::{from_legacy_with_source_capsules_and_limits, SourceCapsuleMapping};
use lamquant_core::source::{DicomWaveformReader, EdfReader, SignalSourceReader};
use semantic_abir::{AbirDataset, ContentId, PayloadAccess, PayloadLease, ValidationLimits};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::sync::Arc;

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

    fn read_legacy(&self, entry: &ForeignEntry) -> Result<legacy_abir::Abir, AdapterError> {
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
            ParserKind::Edf => EdfReader::new(&path).read_bundle(),
            ParserKind::Dicom => DicomWaveformReader::new(&path).read_bundle(),
            ParserKind::Nwb => lamquant_core::nwb::read_bundle(&path),
        }
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        if matches!(self.parser, ParserKind::Nwb) {
            bundle.sample_rate = nwb_sample_rate(&path)?;
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
        let sample_count = bundle.signal.first().map_or(0, Vec::len);
        let channels = bundle
            .signal
            .into_iter()
            .enumerate()
            .map(|(index, values)| legacy_abir::Channel {
                label: Arc::from(bundle.channels[index].as_str()),
                data: legacy_abir::Column::I64(Arc::from(values)),
                phys_min: bundle.phys_min[index],
                phys_max: bundle.phys_max[index],
            })
            .collect();
        Ok(legacy_abir::Abir::from_parts(
            channels,
            bundle.sample_rate,
            sample_count,
        ))
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
        let legacy = self.read_legacy(entry)?;
        Ok(InspectReport {
            profile: self.profile.id.clone(),
            entry_count: 1,
            logical_bytes: entry.bytes.len() as u64,
            risks: vec![
                "independent validator evidence is required before first-class release".to_owned(),
            ],
            required_resources: BTreeMap::from([
                ("max-source-bytes".to_owned(), self.max_source_bytes),
                ("channels".to_owned(), legacy.channels.len() as u64),
                ("samples".to_owned(), legacy.n_samples as u64),
            ]),
        })
    }

    fn import(
        &self,
        source: &ForeignObject,
        limits: ValidationLimits,
    ) -> Result<ImportOutcome, AdapterError> {
        let entry = self.check(source)?;
        let legacy = self.read_legacy(entry)?;
        let source_id = payload_content_id(&entry.bytes);
        let unbound = from_legacy_with_source_capsules_and_limits(&legacy, &[], limits)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let capsule = SourceCapsuleMapping {
            namespace: binding_namespace(&self.profile.id, &unbound.dataset)?,
            value: entry.path.clone(),
            content_id: source_id,
            media_type: entry.media_type.clone(),
        };
        let mapped = from_legacy_with_source_capsules_and_limits(&legacy, &[capsule], limits)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let mut payloads = BTreeMap::new();
        for channel in &mapped.mapping.channels {
            let descriptor = mapped
                .dataset
                .atoms()
                .iter()
                .find(|atom| atom.id() == channel.atom_id)
                .and_then(|atom| atom.payload())
                .ok_or(AdapterError::MissingPayload(channel.content_id))?;
            let lease = mapped
                .access
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
            disposition: MappingDisposition::Exact,
            reason: None,
        }));
        Ok(ImportOutcome {
            dataset: mapped.dataset,
            report: MappingReport {
                source_profile: self.profile.id.clone(),
                target_profile: ProfileId("abir.semantic.v1".to_owned()),
                semantic_coverage: SemanticCoverage::ExactSemantic,
                entries,
                preserved_unknowns: 1,
                sample_values_changed: false,
                timing_changed: false,
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
            return Err(AdapterError::UnsupportedMeaning(
                "dataset lacks one exact EDF source capsule".to_owned(),
            ));
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
            .and_then(|entry| self.read_legacy(entry).map(|_| ()));
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

pub struct EdfAdapter(StandardFileAdapter);

impl EdfAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        Self(StandardFileAdapter::new(
            profile(
                "edfplus.1.signal",
                "EDF/EDF+/BDF",
                "EDF+ 1 signal subset",
                &["application/edf", "application/bdf"],
                "pyedflib",
            ),
            ParserKind::Edf,
            max_source_bytes,
        ))
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

pub struct NwbAdapter(StandardFileAdapter);

impl NwbAdapter {
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

delegate_adapter!(EdfAdapter);
delegate_adapter!(DicomAdapter);
delegate_adapter!(NwbAdapter);

impl Adapter for BidsAdapter {
    fn profile(&self) -> &AdapterProfile {
        &self.profile
    }

    fn inspect(&self, source: &ForeignObject) -> Result<InspectReport, AdapterError> {
        let signal = self.check(source)?;
        let parser =
            StandardFileAdapter::new(self.profile.clone(), ParserKind::Edf, self.max_source_bytes);
        let legacy = parser.read_legacy(signal)?;
        Ok(InspectReport {
            profile: self.profile.id.clone(),
            entry_count: source.entries.len(),
            logical_bytes: source
                .entries
                .iter()
                .map(|entry| entry.bytes.len() as u64)
                .sum(),
            risks: vec![
                "BIDS sidecars are preserved but only signal semantics are promoted".to_owned(),
                "independent bids-validator evidence is required for first-class status".to_owned(),
            ],
            required_resources: BTreeMap::from([
                ("max-source-bytes".to_owned(), self.max_source_bytes),
                ("channels".to_owned(), legacy.channels.len() as u64),
                ("samples".to_owned(), legacy.n_samples as u64),
            ]),
        })
    }

    fn import(
        &self,
        source: &ForeignObject,
        limits: ValidationLimits,
    ) -> Result<ImportOutcome, AdapterError> {
        let signal = self.check(source)?;
        let parser =
            StandardFileAdapter::new(self.profile.clone(), ParserKind::Edf, self.max_source_bytes);
        let legacy = parser.read_legacy(signal)?;
        let unbound = from_legacy_with_source_capsules_and_limits(&legacy, &[], limits)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let namespace = binding_namespace(&self.profile.id, &unbound.dataset)?;
        let capsules = source
            .entries
            .iter()
            .map(|entry| SourceCapsuleMapping {
                namespace: namespace.clone(),
                value: entry.path.clone(),
                content_id: payload_content_id(&entry.bytes),
                media_type: entry.media_type.clone(),
            })
            .collect::<Vec<_>>();
        let mapped = from_legacy_with_source_capsules_and_limits(&legacy, &capsules, limits)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let mut payloads = BTreeMap::new();
        for channel in &mapped.mapping.channels {
            let descriptor = mapped
                .dataset
                .atoms()
                .iter()
                .find(|atom| atom.id() == channel.atom_id)
                .and_then(|atom| atom.payload())
                .ok_or(AdapterError::MissingPayload(channel.content_id))?;
            let lease = mapped
                .access
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
            disposition: MappingDisposition::Exact,
            reason: None,
        }));
        Ok(ImportOutcome {
            dataset: mapped.dataset,
            report: MappingReport {
                source_profile: self.profile.id.clone(),
                target_profile: ProfileId("abir.semantic.v1".to_owned()),
                semantic_coverage: SemanticCoverage::ExactSemantic,
                entries,
                preserved_unknowns: source.entries.len().saturating_sub(1) as u64,
                sample_values_changed: false,
                timing_changed: false,
            },
            payloads: payloads
                .into_iter()
                .map(|(content_id, bytes)| PayloadObject { content_id, bytes })
                .collect(),
        })
    }

    fn plan_export(&self, dataset: &AbirDataset) -> Result<ExportPlan, AdapterError> {
        let capsules = self.capsules(dataset)?;
        let unsupported = capsules.is_empty();
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
            StandardFileAdapter::new(self.profile.clone(), ParserKind::Edf, self.max_source_bytes)
                .read_legacy(signal)
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
        status: ProfileStatus::Semantic,
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

fn nwb_sample_rate(path: &std::path::Path) -> Result<f64, AdapterError> {
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
    let starting_time = acquisition
        .group(&members[0])
        .and_then(|series| series.dataset("starting_time"))
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    let rate = starting_time
        .attr("rate")
        .and_then(|attribute| attribute.read_scalar::<f64>())
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
    if !rate.is_finite() || rate <= 0.0 {
        return Err(AdapterError::InvalidSource(
            "NWB TimeSeries rate must be finite and positive".to_owned(),
        ));
    }
    Ok(rate)
}

pub fn payload_content_id(bytes: &[u8]) -> ContentId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"abir.adapter.payload.v1\0");
    hasher.update(bytes);
    ContentId::from_bytes(*hasher.finalize().as_bytes())
}

fn plan_id(plan: &ExportPlan) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"abir.adapter.export-plan.v1\0");
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
