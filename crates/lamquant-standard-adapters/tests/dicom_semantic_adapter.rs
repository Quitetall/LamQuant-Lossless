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
use semantic_abir::{Atom, ContentId, DatasetDraft, Recording, ValidationLimits};
use std::collections::BTreeMap;

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

fn plain_waveform_foreign() -> ForeignObject {
    foreign(
        include_bytes!("../../../lamquant-lossless/tests/fixtures/dicom/12lead_ecg.dcm").to_vec(),
    )
}

#[test]
fn capsule_free_abir_exports_dicom_and_reimports_waveform_semantics() {
    let adapter = DicomSemanticAdapter::new(1 << 26);
    let imported = adapter
        .import(&plain_waveform_foreign(), ValidationLimits::default())
        .expect("plain DICOM waveform imports");
    let mut draft = DatasetDraft::new(imported.dataset.id());
    let source_recording = &imported.dataset.recordings()[0];
    let mut writable_recording =
        Recording::new(source_recording.id(), source_recording.streams().to_vec());
    for key in source_recording.source_keys().iter().filter(|key| {
        !key.namespace().starts_with("dicom.private.")
            && !key.namespace().starts_with("dicom.referenced-media.")
            && !key.namespace().starts_with("dicom.report.")
    }) {
        writable_recording.add_source_key(key.clone());
    }
    draft.add_recording(writable_recording);
    for stream in imported.dataset.streams() {
        draft.add_stream(stream.clone());
    }
    for atom in imported.dataset.atoms() {
        draft.add_atom(atom.clone());
    }
    for clock in imported.dataset.clocks() {
        draft.add_clock(clock.clone());
    }
    for basis in imported.dataset.channel_bases() {
        draft.add_channel_basis(basis.clone());
    }
    for patient in imported.dataset.patients() {
        draft.add_patient(patient.clone());
    }
    for session in imported.dataset.sessions() {
        draft.add_session(session.clone());
    }
    for acquisition in imported.dataset.acquisitions() {
        draft.add_acquisition(acquisition.clone());
    }
    for device in imported.dataset.devices() {
        draft.add_device(device.clone());
    }
    for event in imported.dataset.events() {
        draft.add_event(event.clone());
    }
    for relationship in imported.dataset.source_relationships() {
        draft.add_source_relationship(*relationship);
    }
    let capsule_free = draft
        .validate(ValidationLimits::default())
        .expect("source capsule is not semantic payload");
    let resolver = Payloads(
        imported
            .payloads
            .iter()
            .map(|payload| (payload.content_id, payload.bytes.clone()))
            .collect(),
    );

    let plan = adapter.plan_export(&capsule_free).unwrap();
    assert!(plan.accepts_without_loss());
    let (written, receipt) = adapter
        .export(&capsule_free, &plan, &resolver)
        .expect("capsule-free DICOM semantic writeback succeeds");
    assert!(!receipt.exact_source_restoration);
    assert!(receipt.semantic_equivalence);
    assert!(adapter.validate(&written).internal_valid);
    support::dump_package33_output("dicom", &written);

    let reimported = adapter
        .import(&written, ValidationLimits::default())
        .expect("written DICOM reimports");
    assert_eq!(
        reimported.dataset.atoms().len(),
        imported.dataset.atoms().len()
    );
    for (source_atom, target_atom) in imported
        .dataset
        .atoms()
        .iter()
        .zip(reimported.dataset.atoms())
    {
        let (source_block, target_block) = match (source_atom, target_atom) {
            (Atom::SignalBlock(source), Atom::SignalBlock(target)) => (source, target),
            other => panic!("expected signal blocks, got {other:?}"),
        };
        assert_eq!(target_block.time_axis(), source_block.time_axis());
        assert_eq!(target_block.calibration(), source_block.calibration());
        let source_descriptor = source_atom.payload().unwrap();
        let target_descriptor = target_atom.payload().unwrap();
        assert_eq!(target_descriptor.shape(), source_descriptor.shape());
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

    for source_path in [
        "(0010,0020) PatientID",
        "(0010,0010) PatientName",
        "(0020,000D) StudyInstanceUID",
        "(0020,000E) SeriesInstanceUID",
        "(0008,0060) Modality",
        "(0008,0070) Manufacturer",
        "(0008,1090) ManufacturerModelName",
    ] {
        assert!(
            outcome.report.entries.iter().any(|entry| {
                entry.source_path == source_path
                    && matches!(entry.disposition, abir_adapter::MappingDisposition::Exact)
            }),
            "missing exact mapping for {source_path}"
        );
    }
    assert!(dataset.patients()[0]
        .source_keys()
        .iter()
        .any(|key| key.namespace() == "dicom.patient-name" && !key.value().is_empty()));
    assert!(dataset.devices()[0]
        .source_keys()
        .iter()
        .any(|key| key.namespace() == "dicom.manufacturer-model" && !key.value().is_empty()));
    assert!(
        outcome
            .report
            .entries
            .iter()
            .all(|entry| { entry.source_path != "(0018,1000) DeviceSerialNumber" }),
        "empty DeviceSerialNumber must not claim an exact semantic mapping"
    );
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
