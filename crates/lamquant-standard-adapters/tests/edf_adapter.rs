use abir_adapter::{Adapter, ForeignEntry, ForeignObject, PayloadResolver, ProfileId};
use lamquant_core::source::{
    from_signal_bundle_with_overlays, SemanticSourceCapsule, SignalBundle, SourceMetadata,
};
use lamquant_standard_adapters::{payload_content_id, EdfAdapter};
use semantic_abir::{ContentId, ValidationLimits};
use std::collections::BTreeMap;

fn write_ascii(target: &mut [u8], value: &str) {
    assert!(value.len() <= target.len());
    target.fill(b' ');
    target[..value.len()].copy_from_slice(value.as_bytes());
}

fn edf_fixture(discontinuous: bool) -> Vec<u8> {
    let labels = ["EEG Fp1", "AUX", "EDF Annotations"];
    let units = ["uV", "mV", ""];
    let phys_min = ["-100", "-10", "-1"];
    let phys_max = ["100", "10", "1"];
    let digital_min = ["-32768", "-32768", "-32768"];
    let digital_max = ["32767", "32767", "32767"];
    let signal_count = labels.len();
    let header_len = 256 + signal_count * 256;
    let mut bytes = vec![b' '; header_len];
    write_ascii(&mut bytes[0..8], "0");
    write_ascii(&mut bytes[8..88], "patient");
    write_ascii(&mut bytes[88..168], "recording");
    write_ascii(&mut bytes[168..176], "22.07.26");
    write_ascii(&mut bytes[176..184], "13.00.00");
    write_ascii(&mut bytes[184..192], &header_len.to_string());
    write_ascii(
        &mut bytes[192..236],
        if discontinuous { "EDF+D" } else { "EDF+C" },
    );
    write_ascii(&mut bytes[236..244], "2");
    write_ascii(&mut bytes[244..252], "1");
    write_ascii(&mut bytes[252..256], &signal_count.to_string());
    let widths = [16_usize, 80, 8, 8, 8, 8, 8, 80, 8, 32];
    let fields: [&[&str]; 10] = [
        &labels,
        &["", "", ""],
        &units,
        &phys_min,
        &phys_max,
        &digital_min,
        &digital_max,
        &["", "", ""],
        &["4", "2", "64"],
        &["", "", ""],
    ];
    let mut offset = 256;
    for (width, values) in widths.into_iter().zip(fields) {
        for value in values {
            write_ascii(&mut bytes[offset..offset + width], value);
            offset += width;
        }
    }
    for record in 0..2 {
        let eeg = if record == 0 {
            [1_i16, 2, 3, 4]
        } else {
            [5, 6, 7, 8]
        };
        for value in eeg {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let aux = if record == 0 { [-3_i16, 4] } else { [-5, 6] };
        for value in aux {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let tal = if record == 0 {
            "+0\x14\x14\0+0.5\x151\x14\u{00e9}vent A\x14\0".as_bytes()
        } else if discontinuous {
            b"+2\x14\x14\0+2.25\x14event B\x14\0".as_slice()
        } else {
            b"+1\x14\x14\0+1.25\x14event B\x14\0".as_slice()
        };
        let mut annotation = vec![0_u8; 64 * 2];
        annotation[..tal.len()].copy_from_slice(tal);
        bytes.extend(annotation);
    }
    bytes
}

fn bdf_fixture() -> Vec<u8> {
    let signal_count = 1_usize;
    let header_len = 512_usize;
    let mut bytes = vec![b' '; header_len];
    bytes[0] = 0xff;
    bytes[1..8].copy_from_slice(b"BIOSEMI");
    write_ascii(&mut bytes[8..88], "patient");
    write_ascii(&mut bytes[88..168], "recording");
    write_ascii(&mut bytes[168..176], "22.07.26");
    write_ascii(&mut bytes[176..184], "13.00.00");
    write_ascii(&mut bytes[184..192], &header_len.to_string());
    write_ascii(&mut bytes[236..244], "1");
    write_ascii(&mut bytes[244..252], "1");
    write_ascii(&mut bytes[252..256], &signal_count.to_string());
    let values = [
        (16, "EEG Cz"),
        (80, ""),
        (8, "uV"),
        (8, "-100"),
        (8, "100"),
        (8, "-8388608"),
        (8, "8388607"),
        (80, ""),
        (8, "4"),
        (32, ""),
    ];
    let mut offset = 256;
    for (width, value) in values {
        write_ascii(&mut bytes[offset..offset + width], value);
        offset += width;
    }
    for value in [-8_388_608_i32, -1, 0, 8_388_607] {
        let raw = value as u32;
        bytes.extend_from_slice(&[raw as u8, (raw >> 8) as u8, (raw >> 16) as u8]);
    }
    bytes
}

struct Payloads(BTreeMap<ContentId, Vec<u8>>);

impl PayloadResolver for Payloads {
    fn resolve(&self, content_id: ContentId) -> Result<Vec<u8>, abir_adapter::AdapterError> {
        self.0
            .get(&content_id)
            .cloned()
            .ok_or(abir_adapter::AdapterError::MissingPayload(content_id))
    }
}

fn payloads(imported: &abir_adapter::ImportOutcome) -> Payloads {
    Payloads(
        imported
            .payloads
            .iter()
            .map(|payload| (payload.content_id, payload.bytes.clone()))
            .collect(),
    )
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
    assert_eq!(
        imported.report.semantic_coverage,
        abir_adapter::SemanticCoverage::ExactSemantic
    );
    assert!(!imported.report.timing_changed);
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
    assert!(!plan.plan_id.is_empty());
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

    let mut discontinuous = lamquant_common::ingest::synth_single_channel_edf(&[1, 2, 3, 4], 250.0);
    discontinuous[192..197].copy_from_slice(b"EDF+D");
    let discontinuous = ForeignObject {
        profile: ProfileId("edfplus.1".to_owned()),
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

#[test]
fn stale_source_capsule_cannot_authorize_semantic_equivalence() {
    let bytes = lamquant_common::ingest::synth_single_channel_edf(&[1, 2, 3, 4], 250.0);
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

#[test]
fn edfplus_discontinuous_annotations_off_rate_and_calibration_are_semantic() {
    let bytes = edf_fixture(true);
    let source = ForeignObject {
        profile: ProfileId("edfplus.1".to_owned()),
        entries: vec![ForeignEntry {
            path: "recording.edf".to_owned(),
            media_type: Some("application/edf".to_owned()),
            bytes: bytes.clone(),
        }],
    };
    let adapter = EdfAdapter::new(1 << 20);
    let imported = adapter
        .import(&source, ValidationLimits::default())
        .unwrap();
    assert_eq!(
        imported.report.semantic_coverage,
        abir_adapter::SemanticCoverage::ExactSemantic
    );
    assert_eq!(imported.dataset.atoms().len(), 3);
    assert_eq!(imported.dataset.events().len(), 2);
    assert_eq!(imported.payloads.len(), 4);
    assert!(imported.payloads.iter().any(|payload| payload
        .bytes
        .windows("\u{00e9}vent A".len())
        .any(|window| { window == "\u{00e9}vent A".as_bytes() })));
    let blocks = imported
        .dataset
        .atoms()
        .iter()
        .filter_map(|atom| match atom {
            semantic_abir::Atom::SignalBlock(block) => Some(block),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(blocks.len(), 2);
    let segments = match blocks[0].time_axis() {
        semantic_abir::TimeAxis::Piecewise(segments) => segments,
        other => panic!("expected EDF+D piecewise axis, got {other:?}"),
    };
    assert_eq!(segments.len(), 2);
    assert_eq!(
        segments[0].start(),
        semantic_abir::Rational::new(0, 1).unwrap()
    );
    assert_eq!(
        segments[1].start(),
        semantic_abir::Rational::new(2, 1).unwrap()
    );
    let calibration = blocks[0].calibration().unwrap();
    assert_eq!(
        calibration.scale(),
        semantic_abir::Rational::new(40, 13_107).unwrap()
    );
    assert_eq!(
        calibration.offset(),
        semantic_abir::Rational::new(20, 13_107).unwrap()
    );
    let plan = adapter.plan_export(&imported.dataset).unwrap();
    let (restored, receipt) = adapter
        .export(&imported.dataset, &plan, &payloads(&imported))
        .unwrap();
    assert_eq!(restored, source);
    assert!(receipt.exact_source_restoration);
    assert!(receipt.semantic_equivalence);
}

#[test]
fn bdf_signed_24_bit_samples_are_promoted_exactly() {
    let source = ForeignObject {
        profile: ProfileId("edfplus.1".to_owned()),
        entries: vec![ForeignEntry {
            path: "recording.bdf".to_owned(),
            media_type: Some("application/bdf".to_owned()),
            bytes: bdf_fixture(),
        }],
    };
    let imported = EdfAdapter::new(1 << 20)
        .import(&source, ValidationLimits::default())
        .unwrap();
    let descriptor = imported.dataset.atoms()[0].payload().unwrap();
    let payload = imported
        .payloads
        .iter()
        .find(|payload| payload.content_id == descriptor.content_id())
        .unwrap();
    let values = payload
        .bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values, [-8_388_608, -1, 0, 8_388_607]);
}
