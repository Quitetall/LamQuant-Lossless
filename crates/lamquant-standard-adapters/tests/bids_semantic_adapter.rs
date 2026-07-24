// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR 0143 BIDS 1.11.1 first-class adapter tests.
//!
//! The fixture is a small but complete dataset: a scalp recording, an
//! intracranial one, a physiological trace, an events table, an electrodes
//! table with a coordinate system, and a derivative. Every semantic the
//! profile owes is present in the tree rather than merely supported in code.

use abir_adapter::{
    Adapter, AdapterError, ForeignEntry, ForeignObject, PayloadResolver, ProfileId,
};
use lamquant_standard_adapters::BidsSemanticAdapter;
use semantic_abir::{ContentId, ValidationLimits};
use std::collections::BTreeMap;

struct Payloads(BTreeMap<ContentId, Vec<u8>>);

impl PayloadResolver for Payloads {
    fn resolve(&self, content_id: ContentId) -> Result<Vec<u8>, AdapterError> {
        self.0
            .get(&content_id)
            .cloned()
            .ok_or(AdapterError::MissingPayload(content_id))
    }
}

macro_rules! member {
    ($path:literal, $media:expr) => {
        ForeignEntry {
            path: $path.to_owned(),
            media_type: Some($media.to_owned()),
            bytes: include_bytes!(concat!("fixtures/bids-full/", $path)).to_vec(),
        }
    };
}

fn dataset() -> Vec<ForeignEntry> {
    vec![
        member!("dataset_description.json", "application/json"),
        member!("participants.tsv", "text/tab-separated-values"),
        member!("README", "text/plain"),
        member!("sub-01/eeg/sub-01_task-rest_eeg.edf", "application/edf"),
        member!("sub-01/eeg/sub-01_task-rest_eeg.json", "application/json"),
        member!(
            "sub-01/eeg/sub-01_task-rest_channels.tsv",
            "text/tab-separated-values"
        ),
        member!(
            "sub-01/eeg/sub-01_task-rest_events.tsv",
            "text/tab-separated-values"
        ),
        member!(
            "sub-01/eeg/sub-01_electrodes.tsv",
            "text/tab-separated-values"
        ),
        member!("sub-01/eeg/sub-01_coordsystem.json", "application/json"),
        member!(
            "sub-01/eeg/sub-01_task-rest_physio.tsv.gz",
            "application/gzip"
        ),
        member!("sub-01/ieeg/sub-01_task-rest_ieeg.edf", "application/edf"),
        member!(
            "derivatives/cleaned/sub-01/eeg/sub-01_task-rest_desc-clean_eeg.edf",
            "application/edf"
        ),
    ]
}

fn foreign(entries: Vec<ForeignEntry>) -> ForeignObject {
    ForeignObject {
        profile: ProfileId("bids.1.11.1".to_owned()),
        entries,
    }
}

#[test]
fn bids_reads_the_layout_as_the_semantic_it_is() {
    let adapter = BidsSemanticAdapter::new(1 << 24);
    let source = foreign(dataset());
    let outcome = adapter
        .import(&source, ValidationLimits::default())
        .expect("the fixture dataset imports");
    let abir = &outcome.dataset;

    // Three recordings, three different meanings: the same EDF bytes under
    // eeg/ and ieeg/ are scalp and intracranial respectively, and the physio
    // trace is neither.
    assert_eq!(abir.streams().len(), 3);
    let modalities: Vec<&str> = abir
        .streams()
        .iter()
        .map(|stream| stream.modality().as_str())
        .collect();
    for expected in [
        "abir:modality/eeg",
        "abir:modality/ieeg",
        "bids:modality/physio",
    ] {
        assert!(modalities.contains(&expected), "missing stream {expected}");
    }

    // Electrodes only index electrophysiology; a physiological trace is not an
    // electrode signal.
    let indexed = abir
        .streams()
        .iter()
        .filter(|stream| stream.channel_basis_id().is_some())
        .count();
    assert_eq!(indexed, 2);
    assert_eq!(abir.channel_bases().len(), 1);
    assert_eq!(abir.channel_bases()[0].channels().len(), 2);
    // A position is only meaningful against a stated system.
    assert_eq!(abir.coordinate_frames().len(), 1);

    assert_eq!(abir.events().len(), 1);

    let inspect = adapter.inspect(&source).expect("the dataset inspects");
    assert_eq!(inspect.required_resources["recordings"], 3);
    assert_eq!(inspect.required_resources["modalities"], 3);
    assert_eq!(inspect.required_resources["electrodes"], 2);
    assert_eq!(inspect.required_resources["events"], 1);
    assert_eq!(inspect.required_resources["derivatives"], 1);
}

#[test]
fn bids_derivatives_are_named_but_never_promoted_beside_raw_data() {
    let adapter = BidsSemanticAdapter::new(1 << 24);
    let outcome = adapter
        .import(&foreign(dataset()), ValidationLimits::default())
        .expect("the fixture dataset imports");
    // The derivative EDF is byte-identical to the raw one, so an adapter that
    // treated it as an observation would silently double the recording count.
    assert_eq!(outcome.dataset.streams().len(), 3);
    let quarantined = outcome
        .report
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.disposition,
                abir_adapter::MappingDisposition::Quarantined
            )
        })
        .count();
    assert_eq!(quarantined, 1);
    assert!(outcome.dataset.recordings()[0]
        .source_keys()
        .iter()
        .any(|key| key.namespace() == "bids.derivative"));
}

#[test]
fn bids_reverse_export_restores_every_member_byte_for_byte() {
    let adapter = BidsSemanticAdapter::new(1 << 24);
    let members = dataset();
    let outcome = adapter
        .import(&foreign(members.clone()), ValidationLimits::default())
        .expect("the fixture dataset imports");
    let payloads = Payloads(
        outcome
            .payloads
            .iter()
            .map(|payload| (payload.content_id, payload.bytes.clone()))
            .collect(),
    );
    let plan = adapter
        .plan_export(&outcome.dataset)
        .expect("export plans without loss");
    let (restored, receipt) = adapter
        .export(&outcome.dataset, &plan, &payloads)
        .expect("export succeeds");
    assert!(receipt.exact_source_restoration);
    // A BIDS dataset is a TREE: every member comes back, not one lucky file.
    assert_eq!(restored.entries.len(), members.len());
    let mut expected = members;
    expected.sort_by(|left, right| left.path.cmp(&right.path));
    for (restored_entry, original) in restored.entries.iter().zip(expected.iter()) {
        assert_eq!(restored_entry.path, original.path);
        assert_eq!(restored_entry.bytes, original.bytes);
    }
}

#[test]
fn bids_rejects_wrong_profile_duplicates_and_incomplete_datasets() {
    let adapter = BidsSemanticAdapter::new(1 << 24);

    let mut wrong_profile = foreign(dataset());
    wrong_profile.profile = ProfileId("edfplus.1".to_owned());
    assert!(matches!(
        adapter.import(&wrong_profile, ValidationLimits::default()),
        Err(AdapterError::ProfileMismatch { .. })
    ));

    let mut duplicated = foreign(dataset());
    duplicated.entries.push(duplicated.entries[0].clone());
    assert!(matches!(
        adapter.import(&duplicated, ValidationLimits::default()),
        Err(AdapterError::DuplicatePath(_))
    ));

    assert!(matches!(
        adapter.import(&foreign(Vec::new()), ValidationLimits::default()),
        Err(AdapterError::EmptySource)
    ));

    // No dataset_description.json means no declared BIDSVersion, and a dataset
    // that does not say which BIDS it is cannot be validated against one.
    let without_description: Vec<ForeignEntry> = dataset()
        .into_iter()
        .filter(|entry| !entry.path.ends_with("dataset_description.json"))
        .collect();
    assert!(adapter
        .import(&foreign(without_description), ValidationLimits::default())
        .is_err());

    // Nothing importable at all.
    let only_metadata: Vec<ForeignEntry> = dataset()
        .into_iter()
        .filter(|entry| entry.path.ends_with(".json") || entry.path.ends_with("README"))
        .collect();
    assert!(adapter
        .import(&foreign(only_metadata), ValidationLimits::default())
        .is_err());

    assert!(BidsSemanticAdapter::new(64)
        .import(&foreign(dataset()), ValidationLimits::default())
        .is_err());
}

#[test]
fn bids_declares_first_class_status_and_names_its_independent_validator() {
    let adapter = BidsSemanticAdapter::new(1 << 24);
    let profile = adapter.profile();
    assert_eq!(profile.id.0, "bids.1.11.1");
    assert_eq!(profile.edition, "1.11.1");
    assert_eq!(profile.required_validator, "bids-validator");
    assert!(matches!(
        profile.status,
        abir_adapter::ProfileStatus::Semantic
    ));
    let artifact = adapter.validate(&foreign(dataset()));
    assert!(artifact.internal_valid, "{:?}", artifact.diagnostics);
    assert_eq!(artifact.independent_valid, None);
}
