// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR 0143 DICOM PS3 first-class adapter tests.
//!
//! The fixture is a real 12-lead ECG waveform instance -- written by other
//! tooling, not by us -- extended with a referenced image and a referenced
//! structured report, so every semantic the profile owes is actually present
//! rather than merely supported in code.

use abir_adapter::{
    Adapter, AdapterError, ForeignEntry, ForeignObject, PayloadResolver, ProfileId,
};
use lamquant_standard_adapters::DicomSemanticAdapter;
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
    include_bytes!("fixtures/ecg_with_references.dcm").to_vec()
}

fn foreign(bytes: Vec<u8>) -> ForeignObject {
    ForeignObject {
        profile: ProfileId("dicom.ps3.2026c".to_owned()),
        entries: vec![ForeignEntry {
            path: "waveform.dcm".to_owned(),
            media_type: Some("application/dicom".to_owned()),
            bytes,
        }],
    }
}

#[test]
fn dicom_import_keeps_the_information_model_and_promotes_annotations() {
    let adapter = DicomSemanticAdapter::new(1 << 26);
    let source = foreign(fixture());
    let outcome = adapter
        .import(&source, ValidationLimits::default())
        .expect("the ECG fixture imports");
    let dataset = &outcome.dataset;

    // Patient, Study, Series and Equipment are separate records joined by
    // typed edges -- a waveform detached from them is clinically useless.
    assert_eq!(dataset.patients().len(), 1);
    assert_eq!(dataset.sessions().len(), 1);
    assert_eq!(dataset.acquisitions().len(), 1);
    assert_eq!(dataset.devices().len(), 1);
    assert!(dataset.source_relationships().len() >= 4);

    // Every annotation is a moment on the acquisition clock.
    assert_eq!(dataset.events().len(), 77);
    assert_eq!(dataset.clocks().len(), 1);

    // Two multiplex groups of twelve leads, each its own atom, indexed by one
    // channel basis so channel identity is semantic rather than positional.
    assert_eq!(dataset.channel_bases().len(), 1);
    assert_eq!(dataset.channel_bases()[0].channels().len(), 24);

    let inspect = adapter.inspect(&source).expect("the fixture inspects");
    assert_eq!(inspect.required_resources["channels"], 24);
    assert_eq!(inspect.required_resources["annotations"], 77);
    assert_eq!(inspect.required_resources["referenced-media"], 1);
    assert_eq!(inspect.required_resources["reports"], 1);
    assert_eq!(inspect.required_resources["private-tags"], 19);
}

#[test]
fn dicom_references_and_private_tags_are_named_but_never_invented() {
    let adapter = DicomSemanticAdapter::new(1 << 26);
    let outcome = adapter
        .import(&foreign(fixture()), ValidationLimits::default())
        .expect("the ECG fixture imports");
    // The referenced image and report live in other files; they are named and
    // quarantined, never inlined, because their bytes were never handed over.
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
        quarantined, 3,
        "one media, one report, and the private block"
    );
    let keys: Vec<String> = outcome.dataset.recordings()[0]
        .source_keys()
        .iter()
        .map(|key| key.namespace().to_owned())
        .collect();
    assert!(keys
        .iter()
        .any(|key| key.starts_with("dicom.referenced-media.")));
    assert!(keys.iter().any(|key| key.starts_with("dicom.report.")));
    // A vendor element is visible under its own group and element numbers.
    assert!(
        keys.iter()
            .filter(|key| key.starts_with("dicom.private."))
            .count()
            == 19
    );
}

#[test]
fn dicom_samples_stay_integers_with_the_stated_calibration() {
    let adapter = DicomSemanticAdapter::new(1 << 26);
    let outcome = adapter
        .import(&foreign(fixture()), ValidationLimits::default())
        .expect("the ECG fixture imports");
    let calibrated = outcome
        .dataset
        .atoms()
        .iter()
        .filter_map(|atom| match atom {
            semantic_abir::Atom::SignalBlock(block) => block.calibration(),
            _ => None,
        })
        .count();
    // Every lead states a sensitivity, so every lead carries a calibration and
    // no sample was rescaled to fake a physical unit.
    assert_eq!(calibrated, 24);
    assert!(!outcome.report.sample_values_changed);
}

#[test]
fn dicom_reverse_export_restores_the_source_byte_for_byte() {
    let adapter = DicomSemanticAdapter::new(1 << 26);
    let bytes = fixture();
    let outcome = adapter
        .import(&foreign(bytes.clone()), ValidationLimits::default())
        .expect("the ECG fixture imports");
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
fn dicom_rejects_wrong_profile_multiple_files_and_malformed_bytes() {
    let adapter = DicomSemanticAdapter::new(1 << 26);
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

    assert!(adapter
        .import(
            &foreign(b"not a dicom file".to_vec()),
            ValidationLimits::default()
        )
        .is_err());

    assert!(adapter
        .import(
            &foreign(valid[..valid.len() / 3].to_vec()),
            ValidationLimits::default()
        )
        .is_err());

    assert!(DicomSemanticAdapter::new(64)
        .import(&foreign(valid), ValidationLimits::default())
        .is_err());
}

#[test]
fn dicom_declares_first_class_status_and_names_its_independent_validator() {
    let adapter = DicomSemanticAdapter::new(1 << 26);
    let profile = adapter.profile();
    assert_eq!(profile.id.0, "dicom.ps3.2026c");
    assert_eq!(profile.edition, "PS3 2026c");
    assert_eq!(profile.required_validator, "pydicom");
    assert!(matches!(
        profile.status,
        abir_adapter::ProfileStatus::Semantic
    ));
    let artifact = adapter.validate(&foreign(fixture()));
    assert!(artifact.internal_valid);
    assert_eq!(artifact.independent_valid, None);
}
