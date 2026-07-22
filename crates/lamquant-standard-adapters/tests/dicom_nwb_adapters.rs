use abir_adapter::{Adapter, ForeignEntry, ForeignObject, PayloadResolver, ProfileId};
use lamquant_standard_adapters::{DicomAdapter, NwbAdapter};
use semantic_abir::{ContentId, ValidationLimits};
use std::collections::BTreeMap;
use std::process::Command;

struct Payloads(BTreeMap<ContentId, Vec<u8>>);

impl PayloadResolver for Payloads {
    fn resolve(&self, content_id: ContentId) -> Result<Vec<u8>, abir_adapter::AdapterError> {
        self.0
            .get(&content_id)
            .cloned()
            .ok_or(abir_adapter::AdapterError::MissingPayload(content_id))
    }
}

fn assert_semantic_round_trip(adapter: &dyn Adapter, source: ForeignObject) {
    let imported = adapter
        .import(&source, ValidationLimits::default())
        .expect("semantic import");
    assert!(imported.report.first_class_semantic());
    assert!(!imported.dataset.atoms().is_empty());
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
    );
}

#[test]
fn nwb_integer_timeseries_maps_semantics_and_restores_exact_source() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("fixture.nwb");
    let status = Command::new("python3")
        .arg("-c")
        .arg(
            "import h5py,numpy as np,sys; f=h5py.File(sys.argv[1],'w'); \
             f.attrs['nwb_version']='2.10.0'; a=f.create_group('acquisition'); \
             e=a.create_group('ElectricalSeries'); \
             e.create_dataset('data',data=np.array([[1,10],[-2,20],[3,30],[-4,40]],dtype='<i2')); \
             t=e.create_dataset('starting_time',data=0.0); t.attrs['rate']=200.0; \
             f.close()",
        )
        .arg(&path)
        .status()
        .expect("python3 and h5py are required conformance tools");
    assert!(status.success(), "h5py NWB fixture generation failed");
    assert_semantic_round_trip(
        &NwbAdapter::new(16 * 1024 * 1024),
        ForeignObject {
            profile: ProfileId("nwb.2.10.0.single-integer-timeseries".to_owned()),
            entries: vec![ForeignEntry {
                path: "fixture.nwb".to_owned(),
                media_type: Some("application/x-nwb".to_owned()),
                bytes: std::fs::read(path).unwrap(),
            }],
        },
    );
}
