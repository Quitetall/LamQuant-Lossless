use abir_adapter::{Adapter, ForeignEntry, ForeignObject, PayloadResolver, ProfileId};
use lamquant_standard_adapters::{DicomAdapter, NwbAdapter};
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

fn assert_semantic_round_trip(
    adapter: &dyn Adapter,
    source: ForeignObject,
    expected_modality: &str,
) {
    let imported = adapter
        .import(&source, ValidationLimits::default())
        .expect("semantic import");
    assert_eq!(
        imported.report.semantic_coverage,
        abir_adapter::SemanticCoverage::ProjectedSemantic
    );
    assert!(imported.report.timing_changed);
    assert!(!imported.report.first_class_semantic());
    assert!(!imported.dataset.atoms().is_empty());
    assert_eq!(
        imported.dataset.streams()[0].modality().as_str(),
        expected_modality
    );
    let payloads = Payloads(
        imported
            .payloads
            .iter()
            .map(|payload| (payload.content_id, payload.bytes.clone()))
            .collect(),
    );
    let plan = adapter.plan_export(&imported.dataset).unwrap();
    let (restored, receipt) = adapter.export(&imported.dataset, &plan, &payloads).unwrap();
    assert_eq!(restored, source);
    assert!(receipt.exact_source_restoration);
    assert!(receipt.semantic_equivalence);
}

#[test]
fn dicom_waveform_maps_semantics_and_restores_exact_source() {
    let bytes =
        include_bytes!("../../../lamquant-lossless/tests/fixtures/dicom/12lead_ecg.dcm").to_vec();
    assert_semantic_round_trip(
        &DicomAdapter::new(16 * 1024 * 1024),
        ForeignObject {
            profile: ProfileId("dicom.ps3.2026c.ecg-i16".to_owned()),
            entries: vec![ForeignEntry {
                path: "12lead_ecg.dcm".to_owned(),
                media_type: Some("application/dicom".to_owned()),
                bytes,
            }],
        },
        "abir:modality/ecg",
    );
}

#[test]
fn nwb_integer_timeseries_maps_semantics_and_restores_exact_source() {
    let bytes = include_bytes!("fixtures/single_integer_timeseries.nwb").to_vec();
    assert_semantic_round_trip(
        &NwbAdapter::new(16 * 1024 * 1024),
        ForeignObject {
            profile: ProfileId("nwb.2.10.0.single-integer-timeseries".to_owned()),
            entries: vec![ForeignEntry {
                path: "fixture.nwb".to_owned(),
                media_type: Some("application/x-nwb".to_owned()),
                bytes,
            }],
        },
        "abir:modality/unknown",
    );
}

#[test]
fn cross_profile_export_rejection_names_the_requested_profile() {
    let source = ForeignObject {
        profile: ProfileId("nwb.2.10.0.single-integer-timeseries".to_owned()),
        entries: vec![ForeignEntry {
            path: "fixture.nwb".to_owned(),
            media_type: Some("application/x-nwb".to_owned()),
            bytes: include_bytes!("fixtures/single_integer_timeseries.nwb").to_vec(),
        }],
    };
    let imported = NwbAdapter::new(16 * 1024 * 1024)
        .import(&source, ValidationLimits::default())
        .unwrap();
    let dicom = DicomAdapter::new(16 * 1024 * 1024);
    let plan = dicom.plan_export(&imported.dataset).unwrap();
    let error = dicom
        .export(&imported.dataset, &plan, &Payloads(BTreeMap::new()))
        .unwrap_err();
    match error {
        abir_adapter::AdapterError::UnsupportedMeaning(message) => assert_eq!(
            message,
            "dataset lacks one exact source capsule for adapter profile dicom.ps3.2026c.ecg-i16"
        ),
        other => panic!("unexpected export error: {other}"),
    }
}

#[test]
fn nwb_rejects_unpromoted_time_origin_and_preallocation_bomb() {
    let original = include_bytes!("fixtures/single_integer_timeseries.nwb");
    let temporary = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(temporary.path(), original).unwrap();
    {
        let file = hdf5_metno::File::open_rw(temporary.path()).unwrap();
        file.dataset("/acquisition/ElectricalSeries/starting_time")
            .unwrap()
            .write_scalar(&1.0_f64)
            .unwrap();
    }
    let shifted = ForeignObject {
        profile: ProfileId("nwb.2.10.0.single-integer-timeseries".to_owned()),
        entries: vec![ForeignEntry {
            path: "shifted.nwb".to_owned(),
            media_type: Some("application/x-nwb".to_owned()),
            bytes: std::fs::read(temporary.path()).unwrap(),
        }],
    };
    assert!(NwbAdapter::new(16 * 1024 * 1024)
        .import(&shifted, ValidationLimits::default())
        .is_err());

    let bomb = tempfile::NamedTempFile::new().unwrap();
    {
        let file = hdf5_metno::File::create(bomb.path()).unwrap();
        let acquisition = file.create_group("acquisition").unwrap();
        let series = acquisition.create_group("test_series").unwrap();
        series
            .new_dataset::<i16>()
            .shape((1_000_000, 2))
            .chunk((1024, 2))
            .create("data")
            .unwrap();
        let start = series.new_dataset::<f64>().create("starting_time").unwrap();
        start.write_scalar(&0.0_f64).unwrap();
        start
            .new_attr::<f64>()
            .create("rate")
            .unwrap()
            .write_scalar(&250.0_f64)
            .unwrap();
    }
    let bytes = std::fs::read(bomb.path()).unwrap();
    let limit = u64::try_from(bytes.len()).unwrap() + 1024;
    let expanded = ForeignObject {
        profile: ProfileId("nwb.2.10.0.single-integer-timeseries".to_owned()),
        entries: vec![ForeignEntry {
            path: "expanded.nwb".to_owned(),
            media_type: Some("application/x-nwb".to_owned()),
            bytes,
        }],
    };
    assert!(NwbAdapter::new(limit)
        .import(&expanded, ValidationLimits::default())
        .is_err());
}
