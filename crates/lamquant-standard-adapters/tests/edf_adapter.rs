use abir_adapter::{Adapter, ForeignEntry, ForeignObject, PayloadResolver, ProfileId};
use lamquant_core::source::{
    from_signal_bundle_with_overlays, SemanticSourceCapsule, SignalBundle, SourceMetadata,
};
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
        profile: ProfileId("edfplus.1.signal".to_owned()),
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
    assert_eq!(
        imported.report.semantic_coverage,
        abir_adapter::SemanticCoverage::ProjectedSemantic
    );
    assert!(imported.report.timing_changed);
    assert!(!imported.report.first_class_semantic());
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
    assert_eq!(
        plan.plan_id,
        "13e5ea79c81c7d16897f4312099acc342550b8a853c12cbfb4ce8eef9364e82b"
    );
    let (restored, receipt) = adapter.export(&imported.dataset, &plan, &payloads).unwrap();
    assert_eq!(restored, source);
    assert!(receipt.exact_source_restoration);
    assert!(receipt.semantic_equivalence);
}

#[test]
fn edf_rejects_wrong_profile_multiple_files_and_malformed_bytes() {
    let adapter = EdfAdapter::new(1024);
    let wrong = ForeignObject {
        profile: ProfileId("nwb.2.10.0.single-integer-timeseries".to_owned()),
        entries: vec![ForeignEntry {
            path: "bad.edf".to_owned(),
            media_type: None,
            bytes: b"not edf".to_vec(),
        }],
    };
    assert!(adapter.inspect(&wrong).is_err());

    let multiple = ForeignObject {
        profile: ProfileId("edfplus.1.signal".to_owned()),
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
        profile: ProfileId("edfplus.1.signal".to_owned()),
        entries: vec![ForeignEntry {
            path: "bad.edf".to_owned(),
            media_type: None,
            bytes: vec![0; 256],
        }],
    };
    assert!(!adapter.validate(&malformed).internal_valid);

    let mut discontinuous = lamquant_common::ingest::synth_single_channel_edf(&[1, 2, 3, 4], 250.0);
    discontinuous[192..197].copy_from_slice(b"EDF+D");
    let discontinuous = ForeignObject {
        profile: ProfileId("edfplus.1.signal".to_owned()),
        entries: vec![ForeignEntry {
            path: "discontinuous.edf".to_owned(),
            media_type: Some("application/edf".to_owned()),
            bytes: discontinuous,
        }],
    };
    assert!(adapter
        .import(&discontinuous, ValidationLimits::default())
        .is_err());
}

#[test]
fn edf_import_honors_caller_validation_limits() {
    let bytes = lamquant_common::ingest::synth_single_channel_edf(&[1, 2, 3, 4], 250.0);
    let source = ForeignObject {
        profile: ProfileId("edfplus.1.signal".to_owned()),
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

#[test]
fn stale_source_capsule_cannot_authorize_semantic_equivalence() {
    let bytes = lamquant_common::ingest::synth_single_channel_edf(&[1, 2, 3, 4], 250.0);
    let source = ForeignObject {
        profile: ProfileId("edfplus.1.signal".to_owned()),
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
    let capsule = &imported.dataset.source_capsules()[0];

    let changed = SignalBundle {
        signal: vec![vec![99, 100, 101, 102]],
        sample_rate: 250.0,
        channels: vec!["EEG".to_owned()],
        phys_min: vec![-1.0],
        phys_max: vec![1.0],
        duration_s: 4.0 / 250.0,
        metadata: SourceMetadata {
            source_file: "changed.edf".to_owned(),
            format: "EDF".to_owned(),
            patient_id: String::new(),
            recording_info: String::new(),
            startdate: String::new(),
            phys_dim: "uV".to_owned(),
        },
        sidecar: vec![],
    };
    let stale = SemanticSourceCapsule {
        namespace: capsule.source().namespace().to_owned(),
        value: capsule.source().value().to_owned(),
        bytes,
        media_type: capsule.media_type().map(str::to_owned),
    };
    let remapped =
        from_signal_bundle_with_overlays(changed, vec![stale], vec![], ValidationLimits::default())
            .expect("changed semantic dataset with retained stale capsule");
    let plan = adapter.plan_export(remapped.opened.dataset()).unwrap();
    assert!(plan.unsupported);
    assert!(!plan.accepts_without_loss());
}
