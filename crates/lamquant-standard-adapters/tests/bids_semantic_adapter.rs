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
use semantic_abir::{logical_content_id, Atom, ContentId, ElementType, ValidationLimits};
use std::collections::BTreeMap;
use std::io::Write;

mod support;

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
    let mut entries = vec![
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
        member!(
            "sub-01/eeg/sub-01_task-rest_physio.json",
            "application/json"
        ),
        member!("sub-01/ieeg/sub-01_task-rest_ieeg.edf", "application/edf"),
        member!(
            "derivatives/cleaned/sub-01/eeg/sub-01_task-rest_desc-clean_eeg.edf",
            "application/edf"
        ),
    ];
    replace_physio(&mut entries, "0.5\t1.5\n0.7\t1.4\n0.6\t1.6\n");
    entries
}

macro_rules! single_member {
    ($path:literal, $media:expr) => {
        ForeignEntry {
            path: $path.to_owned(),
            media_type: Some($media.to_owned()),
            bytes: include_bytes!(concat!("fixtures/bids-single-edf-eeg/", $path)).to_vec(),
        }
    };
}

fn single_edf_dataset() -> Vec<ForeignEntry> {
    vec![
        single_member!("dataset_description.json", "application/json"),
        single_member!("participants.tsv", "text/tab-separated-values"),
        single_member!("README", "text/plain"),
        single_member!("sub-01/eeg/sub-01_task-rest_eeg.edf", "application/edf"),
        single_member!("sub-01/eeg/sub-01_task-rest_eeg.json", "application/json"),
        single_member!(
            "sub-01/eeg/sub-01_task-rest_channels.tsv",
            "text/tab-separated-values"
        ),
        single_member!(
            "sub-01/eeg/sub-01_task-rest_events.tsv",
            "text/tab-separated-values"
        ),
    ]
}

fn replace_physio(entries: &mut [ForeignEntry], text: &str) {
    let physio = entries
        .iter_mut()
        .find(|entry| entry.path.ends_with("_physio.tsv.gz"))
        .unwrap();
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(text.as_bytes()).unwrap();
    physio.bytes = encoder.finish().unwrap();
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

    let physio = abir
        .streams()
        .iter()
        .find(|stream| stream.modality().as_str() == "bids:modality/physio")
        .unwrap();
    let block = abir
        .atoms()
        .iter()
        .find_map(|atom| {
            (atom.id() == physio.atoms()[0])
                .then_some(atom)
                .and_then(|atom| match atom {
                    semantic_abir::Atom::SignalBlock(block) => Some(block),
                    _ => None,
                })
        })
        .unwrap();
    let semantic_abir::TimeAxis::Regular(segment) = block.time_axis() else {
        panic!("BIDS physio sidecar declares a regular axis");
    };
    assert_eq!(segment.start().parts(), (-1, 2));
    assert_eq!(segment.rate().parts(), (250, 1));

    let inspect = adapter.inspect(&source).expect("the dataset inspects");
    assert_eq!(inspect.required_resources["recordings"], 3);
    assert_eq!(inspect.required_resources["modalities"], 3);
    assert_eq!(inspect.required_resources["electrodes"], 2);
    assert_eq!(inspect.required_resources["events"], 1);
    assert_eq!(inspect.required_resources["derivatives"], 1);

    for source_path in [
        "dataset_description.json#BIDSVersion",
        "dataset_description.json#Name",
    ] {
        assert!(
            outcome.report.entries.iter().any(|entry| {
                entry.source_path == source_path
                    && matches!(entry.disposition, abir_adapter::MappingDisposition::Exact)
            }),
            "missing exact mapping for {source_path}"
        );
    }
    let recording_keys = abir.recordings()[0].source_keys();
    assert!(recording_keys
        .iter()
        .any(|key| key.namespace() == "bids.version" && key.value() == "1.11.1"));
    assert!(recording_keys
        .iter()
        .any(|key| key.namespace() == "bids.dataset-name" && !key.value().is_empty()));
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
fn capsule_free_abir_exports_a_valid_bids_tree_and_reimports_signal_semantics() {
    let adapter = BidsSemanticAdapter::new(1 << 24);
    let imported = adapter
        .import(&foreign(single_edf_dataset()), ValidationLimits::default())
        .expect("single-recording BIDS dataset imports");
    let mut draft = imported.dataset.clone().into_draft();
    draft.clear_source_capsules();
    let capsule_free = draft
        .validate(ValidationLimits::default())
        .expect("source capsules are not semantic payloads");
    let resolver = Payloads(
        imported
            .payloads
            .iter()
            .map(|payload| (payload.content_id, payload.bytes.clone()))
            .collect(),
    );

    let plan = adapter
        .plan_export(&capsule_free)
        .expect("representable BIDS ABIR receives an export plan");
    assert!(plan.accepts_without_loss());
    let (written, receipt) = adapter
        .export(&capsule_free, &plan, &resolver)
        .expect("capsule-free BIDS semantic writeback succeeds");
    assert!(!receipt.exact_source_restoration);
    assert!(receipt.semantic_equivalence);
    assert!(adapter.validate(&written).internal_valid);
    support::dump_package33_output("bids", &written);

    let reimported = adapter
        .import(&written, ValidationLimits::default())
        .expect("written BIDS tree reimports");
    let (source_block, target_block) =
        match (&imported.dataset.atoms()[0], &reimported.dataset.atoms()[0]) {
            (Atom::SignalBlock(source), Atom::SignalBlock(target)) => (source, target),
            other => panic!("expected signal blocks, got {other:?}"),
        };
    assert_eq!(target_block.time_axis(), source_block.time_axis());
    let source_descriptor = imported.dataset.atoms()[0].payload().unwrap();
    let target_descriptor = reimported.dataset.atoms()[0].payload().unwrap();
    assert_eq!(target_descriptor.shape(), source_descriptor.shape());
    assert_eq!(target_descriptor.element(), source_descriptor.element());
    assert_eq!(
        reimported.dataset.events().len(),
        imported.dataset.events().len()
    );
    assert_eq!(
        resolver.resolve(source_descriptor.content_id()).unwrap(),
        reimported
            .payloads
            .iter()
            .find(|payload| payload.content_id == target_descriptor.content_id())
            .unwrap()
            .bytes
    );
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
fn bids_rejects_gzip_expansion_before_materializing_physio_samples() {
    let mut entries = dataset();
    let physio = entries
        .iter_mut()
        .find(|entry| entry.path.ends_with("_physio.tsv.gz"))
        .unwrap();
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    encoder.write_all(b"cardiac\n").unwrap();
    for _ in 0..2_000_000 {
        encoder.write_all(b"0\n").unwrap();
    }
    physio.bytes = encoder.finish().unwrap();
    let source = foreign(entries);
    let compressed_bytes = source
        .entries
        .iter()
        .map(|entry| entry.bytes.len() as u64)
        .sum::<u64>();
    let adapter = BidsSemanticAdapter::new(compressed_bytes + 1024);
    assert!(matches!(
        adapter.inspect(&source),
        Err(AdapterError::SourceTooLarge)
    ));
    assert!(matches!(
        adapter.import(&source, ValidationLimits::default()),
        Err(AdapterError::SourceTooLarge)
    ));
}

#[test]
fn bids_rejects_wide_tsv_headers_and_rows_before_collecting_fields() {
    let adapter = BidsSemanticAdapter::new(1 << 24);

    let mut wide_header = dataset();
    let events = wide_header
        .iter_mut()
        .find(|entry| entry.path.ends_with("_events.tsv"))
        .unwrap();
    events.bytes = std::iter::repeat("column")
        .take(16_385)
        .collect::<Vec<_>>()
        .join("\t")
        .into_bytes();
    assert!(matches!(
        adapter.inspect(&foreign(wide_header)),
        Err(AdapterError::SourceTooLarge)
    ));

    let mut wide_row = dataset();
    let events = wide_row
        .iter_mut()
        .find(|entry| entry.path.ends_with("_events.tsv"))
        .unwrap();
    events.bytes = format!(
        "onset\n{}",
        std::iter::repeat("0")
            .take(16_385)
            .collect::<Vec<_>>()
            .join("\t")
    )
    .into_bytes();
    assert!(matches!(
        adapter.inspect(&foreign(wide_row)),
        Err(AdapterError::SourceTooLarge)
    ));
}

#[test]
fn bids_caps_event_rows_across_all_tables() {
    let mut entries = dataset();
    let rows = std::iter::repeat("0")
        .take(262_144)
        .collect::<Vec<_>>()
        .join("\n");
    entries.push(ForeignEntry {
        path: "sub-01/eeg/sub-01_task-rest_run-02_events.tsv".to_owned(),
        media_type: Some("text/tab-separated-values".to_owned()),
        bytes: format!("onset\n{rows}").into_bytes(),
    });
    let adapter = BidsSemanticAdapter::new(1 << 24);
    assert!(matches!(
        adapter.inspect(&foreign(entries)),
        Err(AdapterError::SourceTooLarge)
    ));
}

#[test]
fn bids_physio_preserves_fractional_samples_and_rejects_nonfinite_values() {
    let adapter = BidsSemanticAdapter::new(1 << 24);
    let mut entries = dataset();
    let sidecar = entries
        .iter_mut()
        .find(|entry| entry.path.ends_with("_physio.json"))
        .unwrap();
    sidecar.bytes =
        br#"{"Columns":["cardiac"],"SamplingFrequency":2.5e2,"StartTime":-0.5}"#.to_vec();
    replace_physio(&mut entries, "0.5\n-1.25\n");
    let outcome = adapter
        .import(&foreign(entries), ValidationLimits::default())
        .unwrap();
    let stream = outcome
        .dataset
        .streams()
        .iter()
        .find(|stream| stream.modality().as_str() == "bids:modality/physio")
        .unwrap();
    let atom = outcome
        .dataset
        .atoms()
        .iter()
        .find(|atom| atom.id() == stream.atoms()[0])
        .unwrap();
    let descriptor = atom.payload().unwrap();
    assert_eq!(descriptor.element(), ElementType::F64);
    let payload = outcome
        .payloads
        .iter()
        .find(|payload| payload.content_id == descriptor.content_id())
        .unwrap();
    let expected = [0.5_f64, -1.25_f64]
        .into_iter()
        .flat_map(f64::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(payload.bytes, expected);

    for value in ["NaN", "inf", "-inf"] {
        let mut entries = dataset();
        replace_physio(&mut entries, &format!("{value}\t1\n"));
        assert!(matches!(
            adapter.import(&foreign(entries), ValidationLimits::default()),
            Err(AdapterError::InvalidSource(message)) if message.contains("not finite")
        ));
    }
}

#[test]
fn bids_physio_requires_sidecar_metadata_and_applies_inheritance() {
    let adapter = BidsSemanticAdapter::new(1 << 24);
    let mut inherited = dataset();
    let sidecar = inherited
        .iter_mut()
        .find(|entry| entry.path.ends_with("_physio.json"))
        .unwrap();
    sidecar.bytes = br#"{"StartTime":-0.5}"#.to_vec();
    inherited.push(ForeignEntry {
        path: "task-rest_physio.json".to_owned(),
        media_type: Some("application/json".to_owned()),
        bytes: br#"{"Columns":["cardiac","respiratory"],"SamplingFrequency":100}"#.to_vec(),
    });
    let outcome = adapter
        .import(&foreign(inherited), ValidationLimits::default())
        .expect("partial sidecars merge from root to leaf");
    let physio = outcome
        .dataset
        .streams()
        .iter()
        .find(|stream| stream.modality().as_str() == "bids:modality/physio")
        .unwrap();
    let block = outcome
        .dataset
        .atoms()
        .iter()
        .find_map(|atom| match atom {
            semantic_abir::Atom::SignalBlock(block) if atom.id() == physio.atoms()[0] => {
                Some(block)
            }
            _ => None,
        })
        .unwrap();
    let semantic_abir::TimeAxis::Regular(segment) = block.time_axis() else {
        panic!("merged physio metadata declares a regular axis");
    };
    assert_eq!(segment.rate().parts(), (100, 1));

    let missing = dataset()
        .into_iter()
        .filter(|entry| !entry.path.ends_with("_physio.json"))
        .collect();
    assert!(matches!(
        adapter.import(&foreign(missing), ValidationLimits::default()),
        Err(AdapterError::InvalidSource(message)) if message.contains("no applicable")
    ));
}

#[test]
fn bids_semantics_are_invariant_to_foreign_entry_order() {
    let adapter = BidsSemanticAdapter::new(1 << 24);
    let forward = adapter
        .import(&foreign(dataset()), ValidationLimits::default())
        .unwrap();
    let mut reversed = dataset();
    reversed.reverse();
    let reversed = adapter
        .import(&foreign(reversed), ValidationLimits::default())
        .unwrap();
    assert_eq!(
        logical_content_id(&forward.dataset).unwrap(),
        logical_content_id(&reversed.dataset).unwrap()
    );
}

#[test]
fn bids_rejects_multiple_applicable_physio_sidecars_at_one_level() {
    let adapter = BidsSemanticAdapter::new(1 << 24);
    let mut entries = dataset();
    entries.push(ForeignEntry {
        path: "sub-01/eeg/task-rest_physio.json".to_owned(),
        media_type: Some("application/json".to_owned()),
        bytes: br#"{"Columns":["cardiac","respiratory"]}"#.to_vec(),
    });
    assert!(matches!(
        adapter.import(&foreign(entries), ValidationLimits::default()),
        Err(AdapterError::InvalidSource(message))
            if message.contains("multiple applicable physio sidecars")
    ));
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
