use abir_adapter::{Adapter, ForeignEntry, ForeignObject, PayloadResolver, ProfileId};
use lamquant_core::source::{
    from_signal_bundle_with_semantics, SemanticSourceCapsule, SignalBundle, SourceMetadata,
};
use lamquant_standard_adapters::{BidsAdapter, BidsEventsAdapter};
use semantic_abir::{ContentId, ValidationLimits};
use std::collections::BTreeMap;

struct Payloads(BTreeMap<ContentId, Vec<u8>>);

impl PayloadResolver for Payloads {
    fn resolve(&self, content_id: ContentId) -> Result<Vec<u8>, abir_adapter::AdapterError> {
        self.0
            .get(&content_id)
            .cloned()
            .ok_or(abir_adapter::AdapterError::MissingPayload(content_id))
    }
}

fn bids_source() -> ForeignObject {
    ForeignObject {
        profile: ProfileId("bids.1.11.1.single-edf-eeg".to_owned()),
        entries: vec![
            ForeignEntry {
                path: "dataset_description.json".to_owned(),
                media_type: Some("application/json".to_owned()),
                bytes: include_bytes!("fixtures/bids-single-edf-eeg/dataset_description.json")
                    .to_vec(),
            },
            ForeignEntry {
                path: "README".to_owned(),
                media_type: Some("text/plain".to_owned()),
                bytes: include_bytes!("fixtures/bids-single-edf-eeg/README").to_vec(),
            },
            ForeignEntry {
                path: "participants.tsv".to_owned(),
                media_type: Some("text/tab-separated-values".to_owned()),
                bytes: include_bytes!("fixtures/bids-single-edf-eeg/participants.tsv").to_vec(),
            },
            ForeignEntry {
                path: "sub-01/eeg/sub-01_task-rest_eeg.edf".to_owned(),
                media_type: Some("application/edf".to_owned()),
                bytes: include_bytes!(
                    "fixtures/bids-single-edf-eeg/sub-01/eeg/sub-01_task-rest_eeg.edf"
                )
                .to_vec(),
            },
            ForeignEntry {
                path: "sub-01/eeg/sub-01_task-rest_eeg.json".to_owned(),
                media_type: Some("application/json".to_owned()),
                bytes: include_bytes!(
                    "fixtures/bids-single-edf-eeg/sub-01/eeg/sub-01_task-rest_eeg.json"
                )
                .to_vec(),
            },
            ForeignEntry {
                path: "sub-01/eeg/sub-01_task-rest_channels.tsv".to_owned(),
                media_type: Some("text/tab-separated-values".to_owned()),
                bytes: include_bytes!(
                    "fixtures/bids-single-edf-eeg/sub-01/eeg/sub-01_task-rest_channels.tsv"
                )
                .to_vec(),
            },
            ForeignEntry {
                path: "sub-01/eeg/sub-01_task-rest_events.tsv".to_owned(),
                media_type: Some("text/tab-separated-values".to_owned()),
                bytes: include_bytes!(
                    "fixtures/bids-single-edf-eeg/sub-01/eeg/sub-01_task-rest_events.tsv"
                )
                .to_vec(),
            },
        ],
    }
}

fn bids_events_source() -> ForeignObject {
    let mut source = bids_source();
    source.profile = ProfileId("bids.1.11.1.single-edf-eeg-events".to_owned());
    source
}

fn dataset_with_bids_capsule_paths(paths: &[&str]) -> semantic_abir::AbirDataset {
    let bundle = SignalBundle {
        signal: vec![vec![1, 2, 3, 4]],
        sample_rate: 250.0,
        channels: vec!["EEG".to_owned()],
        phys_min: vec![-1.0],
        phys_max: vec![1.0],
        duration_s: 4.0 / 250.0,
        metadata: SourceMetadata {
            source_file: "recording.edf".to_owned(),
            format: "EDF".to_owned(),
            patient_id: String::new(),
            recording_info: String::new(),
            startdate: String::new(),
            phys_dim: "uV".to_owned(),
        },
        sidecar: vec![],
    };
    let modality = semantic_abir::ConceptId::new("abir:modality/eeg").unwrap();
    let unbound = from_signal_bundle_with_semantics(
        bundle.clone(),
        modality.clone(),
        vec![],
        vec![],
        ValidationLimits::default(),
    )
    .unwrap();
    let semantic = semantic_abir::interchange_content_id(unbound.opened.dataset()).unwrap();
    let namespace = format!("adapter.bids.1.11.1.single-edf-eeg.binding.{semantic}");
    let capsules = paths
        .iter()
        .enumerate()
        .map(|(index, path)| SemanticSourceCapsule {
            namespace: namespace.clone(),
            value: (*path).to_owned(),
            bytes: vec![index as u8],
            media_type: None,
        })
        .collect();
    from_signal_bundle_with_semantics(
        bundle,
        modality,
        capsules,
        vec![],
        ValidationLimits::default(),
    )
    .unwrap()
    .opened
    .dataset()
    .clone()
}

#[test]
fn bids_tree_maps_signal_semantics_and_restores_every_source_byte() {
    let source = bids_events_source();
    let adapter = BidsEventsAdapter::new(2 * 1024 * 1024);
    let imported = adapter
        .import(&source, ValidationLimits::default())
        .expect("semantic BIDS import");
    assert_eq!(
        adapter.profile().status,
        abir_adapter::ProfileStatus::Forensic
    );
    assert_eq!(
        imported.report.semantic_coverage,
        abir_adapter::SemanticCoverage::ProjectedSemantic
    );
    assert!(imported.report.timing_changed);
    assert!(!imported.report.first_class_semantic());
    assert_eq!(
        imported.report.preserved_unknowns,
        source.entries.len() as u64 - 1
    );
    // The BIDS tree is bound exactly, and the EDF parser may additionally
    // retain format-native sidecars needed by other current-ABIR consumers.
    assert!(imported.dataset.source_capsules().len() >= source.entries.len());
    for entry in &source.entries {
        assert!(imported
            .dataset
            .source_capsules()
            .iter()
            .any(|capsule| capsule.source().value() == entry.path));
    }
    assert_eq!(imported.dataset.streams().len(), 1);
    assert_eq!(
        imported.dataset.streams()[0].modality().as_str(),
        "abir:modality/eeg"
    );
    assert_eq!(imported.dataset.events().len(), 1);
    let event = &imported.dataset.events()[0];
    assert_eq!(event.kind().as_str(), "bids:event/stimulus");
    assert_eq!(event.start().parts(), (1, 2));
    assert_eq!(event.end().parts(), (3, 5));
    assert_eq!(event.uncertainty().parts(), (0, 1));
    assert_eq!(imported.dataset.clocks().len(), 1);
    assert_eq!(
        imported.dataset.streams()[0].clock_id(),
        Some(event.clock_id())
    );
    assert!(imported.report.entries.iter().any(|entry| {
        entry
            .source_path
            .ends_with("_events.tsv#row=2;fields=onset,duration,trial_type")
            && entry.target.starts_with("event:")
            && entry.disposition == abir_adapter::MappingDisposition::Exact
    }));

    let payloads = Payloads(
        imported
            .payloads
            .iter()
            .map(|payload| (payload.content_id, payload.bytes.clone()))
            .collect(),
    );
    let plan = adapter.plan_export(&imported.dataset).unwrap();
    assert!(plan.accepts_without_loss());
    let (restored, receipt) = adapter.export(&imported.dataset, &plan, &payloads).unwrap();
    assert_eq!(restored, source);
    assert!(receipt.exact_source_restoration);
    assert!(receipt.semantic_equivalence);
}

#[test]
fn bids_rejects_missing_or_unpinned_description_and_multiple_recordings() {
    let adapter = BidsAdapter::new(2 * 1024 * 1024);
    let mut missing = bids_source();
    missing
        .entries
        .retain(|entry| entry.path != "dataset_description.json");
    assert!(adapter.inspect(&missing).is_err());

    let mut wrong_version = bids_source();
    wrong_version
        .entries
        .iter_mut()
        .find(|entry| entry.path == "dataset_description.json")
        .unwrap()
        .bytes = br#"{"Name":"ABIR conformance","BIDSVersion":"1.10.1"}"#.to_vec();
    assert!(adapter.inspect(&wrong_version).is_err());

    let mut multiple = bids_source();
    multiple.entries.push(ForeignEntry {
        path: "sub-02/eeg/sub-02_task-rest_eeg.edf".to_owned(),
        media_type: Some("application/edf".to_owned()),
        bytes: lamquant_common::ingest::synth_single_channel_edf(&[5, 6], 250.0),
    });
    assert!(adapter.inspect(&multiple).is_err());
}

#[test]
fn bids_rejects_ambiguous_paths_duplicates_and_oversized_trees() {
    let adapter = BidsAdapter::new(2 * 1024 * 1024);
    for bad_path in [
        "../dataset_description.json",
        "sub-01/../dataset_description.json",
        "sub-01//dataset_description.json",
        "C:/dataset_description.json",
        "sub-01\\dataset_description.json",
    ] {
        let mut source = bids_source();
        source.entries[0].path = bad_path.to_owned();
        assert!(adapter.inspect(&source).is_err(), "accepted {bad_path}");
    }

    let mut duplicate = bids_source();
    duplicate
        .entries
        .push(duplicate.entries.last().unwrap().clone());
    assert!(adapter.inspect(&duplicate).is_err());

    assert!(BidsAdapter::new(1).inspect(&bids_source()).is_err());
}

#[test]
fn bids_export_plan_rejects_capsule_sets_without_a_complete_tree() {
    let adapter = BidsAdapter::new(2 * 1024 * 1024);
    let sidecars_plan = adapter
        .plan_export(&dataset_with_bids_capsule_paths(&[
            "dataset_description.json",
            "participants.tsv",
        ]))
        .unwrap();
    assert!(sidecars_plan.unsupported);
    assert!(!sidecars_plan.accepts_without_loss());

    let invalid_path_plan = adapter
        .plan_export(&dataset_with_bids_capsule_paths(&[
            "../dataset_description.json",
            "sub-01/eeg/sub-01_task-rest_eeg.edf",
        ]))
        .unwrap();
    assert!(invalid_path_plan.unsupported);

    let duplicate_plan = adapter
        .plan_export(&dataset_with_bids_capsule_paths(&[
            "dataset_description.json",
            "sub-01/eeg/sub-01_task-rest_eeg.edf",
            "sub-01/eeg/sub-01_task-rest_eeg.edf",
        ]))
        .unwrap();
    assert!(duplicate_plan.unsupported);
}

#[test]
fn bids_import_honors_caller_validation_limits() {
    let limits = ValidationLimits {
        max_atoms: 0,
        ..ValidationLimits::default()
    };
    assert!(BidsAdapter::new(2 * 1024 * 1024)
        .import(&bids_source(), limits)
        .is_err());
}

#[test]
fn bids_events_fail_closed_on_malformed_or_unrepresentable_rows() {
    let adapter = BidsEventsAdapter::new(2 * 1024 * 1024);
    let events_path = "sub-01/eeg/sub-01_task-rest_events.tsv";
    for malformed in [
        "onset\tduration\ttrial_type\n0.5\tbad\tstimulus\n",
        "onset\tduration\ttrial_type\n0.5\t-0.1\tstimulus\n",
        "onset\tduration\ttrial_type\n0.5\t0.1\tnot representable\n",
        "onset\tduration\ttrial_type\n0.5\t0.1\tn/a\n",
        "onset\tduration\ttrial_type\n1e1000\t0.1\tstimulus\n",
        "onset\tduration\n0.5\t0.1\n",
        "onset\tduration\ttrial_type\n0.5\t0.1\tstimulus\textra\n",
    ] {
        let mut source = bids_events_source();
        source
            .entries
            .iter_mut()
            .find(|entry| entry.path == events_path)
            .unwrap()
            .bytes = malformed.as_bytes().to_vec();
        assert!(
            adapter
                .import(&source, ValidationLimits::default())
                .is_err(),
            "accepted malformed events row: {malformed:?}"
        );
    }
}

#[test]
fn bids_events_are_optional_but_ambiguous_sidecars_fail_closed() {
    let adapter = BidsEventsAdapter::new(2 * 1024 * 1024);
    let mut missing = bids_events_source();
    missing
        .entries
        .retain(|entry| !entry.path.ends_with("_events.tsv"));
    let imported = adapter
        .import(&missing, ValidationLimits::default())
        .expect("BIDS recordings do not require an events sidecar");
    assert!(imported.dataset.events().is_empty());
    assert!(imported.dataset.clocks().is_empty());

    let mut duplicate = bids_events_source();
    duplicate.entries.push(ForeignEntry {
        path: "sub-01/eeg/sub-01_task-other_events.tsv".to_owned(),
        media_type: Some("text/tab-separated-values".to_owned()),
        bytes: b"onset\tduration\ttrial_type\n1\t0\tother\n".to_vec(),
    });
    assert!(adapter
        .import(&duplicate, ValidationLimits::default())
        .is_err());
}

#[test]
fn bids_event_exponents_remain_exact_rationals() {
    let mut source = bids_events_source();
    source
        .entries
        .iter_mut()
        .find(|entry| entry.path.ends_with("_events.tsv"))
        .unwrap()
        .bytes = b"onset\tduration\ttrial_type\n1e-3\t2E-3\tstimulus\n".to_vec();
    let imported = BidsEventsAdapter::new(2 * 1024 * 1024)
        .import(&source, ValidationLimits::default())
        .unwrap();
    let event = &imported.dataset.events()[0];
    assert_eq!(event.start().parts(), (1, 1000));
    assert_eq!(event.end().parts(), (3, 1000));
}

#[test]
fn signal_only_profile_keeps_events_quarantined_and_compatible() {
    let mut source = bids_source();
    source
        .entries
        .iter_mut()
        .find(|entry| entry.path.ends_with("_events.tsv"))
        .unwrap()
        .bytes = b"not a TSV under the frozen signal-only profile".to_vec();
    let imported = BidsAdapter::new(2 * 1024 * 1024)
        .import(&source, ValidationLimits::default())
        .expect("the frozen signal-only profile must not reinterpret sidecars");
    assert!(imported.dataset.events().is_empty());
    assert!(imported.dataset.clocks().is_empty());
    assert_eq!(
        imported.report.preserved_unknowns,
        source.entries.len() as u64 - 1
    );
}

#[test]
fn additional_event_columns_are_explicitly_quarantined() {
    let mut source = bids_events_source();
    source
        .entries
        .iter_mut()
        .find(|entry| entry.path.ends_with("_events.tsv"))
        .unwrap()
        .bytes = b"onset\tduration\ttrial_type\tresponse_time\n0.5\t0.1\tstimulus\t0.42\n".to_vec();
    let imported = BidsEventsAdapter::new(2 * 1024 * 1024)
        .import(&source, ValidationLimits::default())
        .unwrap();
    assert!(imported.report.entries.iter().any(|entry| {
        entry.source_path.ends_with("#row=2;fields=response_time")
            && entry.disposition == abir_adapter::MappingDisposition::Quarantined
            && entry.target.starts_with("source-capsule:")
    }));
}
