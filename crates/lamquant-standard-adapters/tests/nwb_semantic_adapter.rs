// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR 0143 NWB 2.10.0 first-class adapter tests.
//!
//! The fixture is written by pynwb -- a separate implementation -- so the
//! adapter is read against a file it did not produce. It carries one series in
//! each container the profile must distinguish (acquisition, stimulus,
//! processing/behavior, processing/ecephys), a four-row electrodes table, and
//! two epoch intervals, and one ImageSeries whose bytes live in another file
//! entirely -- the external asset the profile must name without inventing.

use abir_adapter::{
    Adapter, AdapterError, ForeignEntry, ForeignObject, PayloadResolver, ProfileId,
};
use lamquant_standard_adapters::NwbAdapter;
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

fn fixture() -> Vec<u8> {
    include_bytes!("fixtures/multi_container_session.nwb").to_vec()
}

fn foreign(bytes: Vec<u8>) -> ForeignObject {
    ForeignObject {
        profile: ProfileId("nwb.2.10.0".to_owned()),
        entries: vec![ForeignEntry {
            path: "session.nwb".to_owned(),
            media_type: Some("application/x-nwb".to_owned()),
            bytes,
        }],
    }
}

#[test]
fn nwb_import_separates_containers_and_promotes_electrodes_and_intervals() {
    let adapter = NwbAdapter::new(1 << 26);
    let source = foreign(fixture());
    let outcome = adapter
        .import(&source, ValidationLimits::default())
        .expect("the pynwb fixture imports");
    let dataset = &outcome.dataset;

    // One stream per container: acquisition, stimulus, behavior and derived
    // are different claims, and collapsing them would erase that.
    assert_eq!(dataset.streams().len(), 4);
    let modalities: Vec<&str> = dataset
        .streams()
        .iter()
        .map(|stream| stream.modality().as_str())
        .collect();
    for expected in [
        "abir:modality/unknown",
        "nwb:modality/stimulus",
        "nwb:modality/behavior",
        "nwb:modality/derived",
    ] {
        assert!(modalities.contains(&expected), "missing stream {expected}");
    }

    // Only recorded acquisition is indexed by the electrode basis.
    let with_basis: Vec<_> = dataset
        .streams()
        .iter()
        .filter(|stream| stream.channel_basis_id().is_some())
        .collect();
    assert_eq!(with_basis.len(), 1);
    assert_eq!(with_basis[0].modality().as_str(), "abir:modality/unknown");
    assert_eq!(dataset.channel_bases().len(), 1);
    assert_eq!(dataset.channel_bases()[0].channels().len(), 4);

    // Both epochs became events on the one session clock.
    assert_eq!(dataset.events().len(), 2);
    assert_eq!(dataset.clocks().len(), 1);

    let inspect = adapter.inspect(&source).expect("the fixture inspects");
    assert_eq!(inspect.required_resources["series"], 4);
    assert_eq!(inspect.required_resources["electrodes"], 4);
    assert_eq!(inspect.required_resources["intervals"], 2);
    // The external asset is NAMED, never inlined: its bytes are in a file this
    // adapter was not handed, so producing content for it would be a
    // fabrication.
    assert_eq!(inspect.required_resources["external-assets"], 1);
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
    assert_eq!(
        quarantined, 1,
        "the external asset is quarantined, not dropped"
    );
}

#[test]
fn nwb_behavior_series_keeps_its_own_timestamps() {
    let adapter = NwbAdapter::new(1 << 26);
    let outcome = adapter
        .import(&foreign(fixture()), ValidationLimits::default())
        .expect("the pynwb fixture imports");
    // The behaviour series was written with timestamps rather than a rate, so
    // it must carry an explicit axis; inventing a rate would be a fabrication.
    let explicit = outcome
        .dataset
        .atoms()
        .iter()
        .filter_map(|atom| match atom {
            semantic_abir::Atom::SignalBlock(block) => Some(block.time_axis()),
            _ => None,
        })
        .filter(|axis| matches!(axis, semantic_abir::TimeAxis::Explicit { .. }))
        .count();
    assert_eq!(explicit, 1);
}

#[test]
fn nwb_reverse_export_restores_the_source_byte_for_byte() {
    let adapter = NwbAdapter::new(1 << 26);
    let bytes = fixture();
    let outcome = adapter
        .import(&foreign(bytes.clone()), ValidationLimits::default())
        .expect("the pynwb fixture imports");
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
    assert_eq!(restored.entries[0].bytes, bytes);
}

#[test]
fn nwb_rejects_wrong_profile_multiple_files_and_malformed_bytes() {
    let adapter = NwbAdapter::new(1 << 26);
    let valid = fixture();

    let mut wrong_profile = foreign(valid.clone());
    wrong_profile.profile = ProfileId("edfplus.1".to_owned());
    assert!(matches!(
        adapter.import(&wrong_profile, ValidationLimits::default()),
        Err(AdapterError::ProfileMismatch { .. })
    ));

    let mut two_files = foreign(valid.clone());
    two_files.entries.push(two_files.entries[0].clone());
    assert!(adapter
        .import(&two_files, ValidationLimits::default())
        .is_err());

    // Not HDF5 at all.
    assert!(adapter
        .import(
            &foreign(b"not an nwb file".to_vec()),
            ValidationLimits::default()
        )
        .is_err());

    // Truncated: the HDF5 layer must refuse rather than read garbage.
    assert!(adapter
        .import(
            &foreign(valid[..valid.len() / 3].to_vec()),
            ValidationLimits::default()
        )
        .is_err());

    // Beyond the declared byte budget.
    assert!(NwbAdapter::new(64)
        .import(&foreign(valid), ValidationLimits::default())
        .is_err());
}

#[test]
fn nwb_declares_first_class_status_and_names_its_independent_validator() {
    let adapter = NwbAdapter::new(1 << 26);
    let profile = adapter.profile();
    assert_eq!(profile.id.0, "nwb.2.10.0");
    assert_eq!(profile.edition, "2.10.0");
    assert_eq!(profile.required_validator, "pynwb.validate");
    assert!(matches!(
        profile.status,
        abir_adapter::ProfileStatus::Semantic
    ));
    let artifact = adapter.validate(&foreign(fixture()));
    assert!(artifact.internal_valid);
    assert_eq!(artifact.independent_valid, None);
}
