use abir_adapter::{Adapter, ForeignEntry, ForeignObject, PayloadResolver, ProfileId};
use lamquant_standard_adapters::BidsAdapter;
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
        profile: ProfileId("bids.1.11.1".to_owned()),
        entries: vec![
            ForeignEntry {
                path: "dataset_description.json".to_owned(),
                media_type: Some("application/json".to_owned()),
                bytes: br#"{"Name":"ABIR conformance","BIDSVersion":"1.11.1"}"#.to_vec(),
            },
            ForeignEntry {
                path: "sub-01/eeg/sub-01_task-rest_eeg.edf".to_owned(),
                media_type: Some("application/edf".to_owned()),
                bytes: lamquant_common::ingest::synth_single_channel_edf(&[1, -2, 3, -4], 250.0),
            },
            ForeignEntry {
                path: "sub-01/eeg/sub-01_task-rest_events.tsv".to_owned(),
                media_type: Some("text/tab-separated-values".to_owned()),
                bytes: b"onset\tduration\n0\t1\n".to_vec(),
            },
        ],
    }
}

#[test]
fn bids_tree_maps_signal_semantics_and_restores_every_source_byte() {
    let source = bids_source();
    let adapter = BidsAdapter::new(2 * 1024 * 1024);
    let imported = adapter
        .import(&source, ValidationLimits::default())
        .expect("semantic BIDS import");
    assert!(imported.report.first_class_semantic());
    assert_eq!(imported.report.preserved_unknowns, 2);
    assert_eq!(
        imported.dataset.source_capsules().len(),
        source.entries.len()
    );
    assert_eq!(imported.dataset.streams().len(), 1);

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
    missing.entries.remove(0);
    assert!(adapter.inspect(&missing).is_err());

    let mut wrong_version = bids_source();
    wrong_version.entries[0].bytes =
        br#"{"Name":"ABIR conformance","BIDSVersion":"1.10.1"}"#.to_vec();
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
    duplicate.entries.push(duplicate.entries[2].clone());
    assert!(adapter.inspect(&duplicate).is_err());

    assert!(BidsAdapter::new(1).inspect(&bids_source()).is_err());
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
