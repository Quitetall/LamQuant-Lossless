use abir_adapter::{Adapter, ForeignEntry, ForeignObject, PayloadResolver, ProfileId};
use lamquant_standard_adapters::{payload_content_id, EdfAdapter};
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

#[test]
fn edf_import_maps_samples_and_restores_exact_source() {
    let bytes = lamquant_common::ingest::synth_single_channel_edf(&[1, -2, 3, -4], 250.0);
    let source = ForeignObject {
        profile: ProfileId("edfplus.1".to_owned()),
        entries: vec![ForeignEntry {
            path: "recording.edf".to_owned(),
            media_type: Some("application/edf".to_owned()),
            bytes: bytes.clone(),
        }],
    };
    let adapter = EdfAdapter::new(1024 * 1024);
    let imported = adapter
        .import(&source, ValidationLimits::default())
        .expect("semantic EDF import");
    assert!(imported.report.first_class_semantic());
    assert_eq!(imported.dataset.recordings().len(), 1);
    assert_eq!(imported.dataset.streams().len(), 1);
    assert_eq!(imported.dataset.atoms().len(), 1);
    assert_eq!(imported.dataset.source_capsules().len(), 1);

    let payloads = Payloads(
        imported
            .payloads
            .iter()
            .map(|payload| (payload.content_id, payload.bytes.clone()))
            .collect(),
    );
    assert_eq!(payloads.0.get(&payload_content_id(&bytes)), Some(&bytes));
    let plan = adapter.plan_export(&imported.dataset).unwrap();
    assert!(plan.accepts_without_loss());
    let (restored, receipt) = adapter.export(&imported.dataset, &plan, &payloads).unwrap();
    assert_eq!(restored, source);
    assert!(receipt.exact_source_restoration);
    assert!(receipt.semantic_equivalence);
}

#[test]
fn edf_rejects_wrong_profile_multiple_files_and_malformed_bytes() {
    let adapter = EdfAdapter::new(1024);
    let wrong = ForeignObject {
        profile: ProfileId("nwb.2.10.0".to_owned()),
        entries: vec![ForeignEntry {
            path: "bad.edf".to_owned(),
            media_type: None,
            bytes: b"not edf".to_vec(),
        }],
    };
    assert!(adapter.inspect(&wrong).is_err());

    let multiple = ForeignObject {
        profile: ProfileId("edfplus.1".to_owned()),
        entries: vec![
            ForeignEntry {
                path: "a.edf".to_owned(),
                media_type: None,
                bytes: vec![0; 256],
            },
            ForeignEntry {
                path: "b.edf".to_owned(),
                media_type: None,
                bytes: vec![0; 256],
            },
        ],
    };
    assert!(adapter
        .import(&multiple, ValidationLimits::default())
        .is_err());

    let malformed = ForeignObject {
        profile: ProfileId("edfplus.1".to_owned()),
        entries: vec![ForeignEntry {
            path: "bad.edf".to_owned(),
            media_type: None,
            bytes: vec![0; 256],
        }],
    };
    assert!(!adapter.validate(&malformed).internal_valid);
}

#[test]
fn edf_import_honors_caller_validation_limits() {
    let bytes = lamquant_common::ingest::synth_single_channel_edf(&[1, 2], 250.0);
    let source = ForeignObject {
        profile: ProfileId("edfplus.1".to_owned()),
        entries: vec![ForeignEntry {
            path: "recording.edf".to_owned(),
            media_type: Some("application/edf".to_owned()),
            bytes,
        }],
    };
    let limits = ValidationLimits {
        max_atoms: 0,
        ..ValidationLimits::default()
    };
    assert!(EdfAdapter::new(1024 * 1024)
        .import(&source, limits)
        .is_err());
}
