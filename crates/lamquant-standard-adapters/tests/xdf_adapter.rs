// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR 0143 XDF 1.0 first-class adapter tests.
//!
//! The fixtures are framed by hand from the XDF 1.0 specification rather than
//! produced by a writer of ours, so a bug shared between our reader and our
//! writer cannot make these pass.

use abir_adapter::{
    Adapter, AdapterError, ForeignEntry, ForeignObject, PayloadResolver, ProfileId,
};
use lamquant_standard_adapters::XdfAdapter;
use semantic_abir::{ContentId, ValidationLimits};
use std::collections::BTreeMap;

const BOUNDARY_UUID: [u8; 16] = [
    0x43, 0xA5, 0x46, 0xDC, 0xCB, 0xF5, 0x41, 0x0F, 0xB3, 0x0E, 0xD5, 0x46, 0x73, 0x83, 0xCB, 0xE4,
];

struct Payloads(BTreeMap<ContentId, Vec<u8>>);

impl PayloadResolver for Payloads {
    fn resolve(&self, content_id: ContentId) -> Result<Vec<u8>, AdapterError> {
        self.0
            .get(&content_id)
            .cloned()
            .ok_or(AdapterError::MissingPayload(content_id))
    }
}

/// Frame one chunk: `NumLengthBytes | Length | Tag | Content`, where `Length`
/// counts the tag as well.
fn chunk(tag: u16, content: &[u8]) -> Vec<u8> {
    let length = (content.len() + 2) as u64;
    let mut bytes = Vec::new();
    if length < 256 {
        bytes.push(1);
        bytes.push(length as u8);
    } else {
        bytes.push(4);
        bytes.extend_from_slice(&(length as u32).to_le_bytes());
    }
    bytes.extend_from_slice(&tag.to_le_bytes());
    bytes.extend_from_slice(content);
    bytes
}

fn stream_header(id: u32, xml: &str) -> Vec<u8> {
    let mut content = id.to_le_bytes().to_vec();
    content.extend_from_slice(xml.as_bytes());
    chunk(2, &content)
}

fn eeg_header_xml() -> String {
    concat!(
        "<?xml version=\"1.0\"?><info><name>BioSemi</name><type>EEG</type>",
        "<channel_count>2</channel_count><nominal_srate>500</nominal_srate>",
        "<channel_format>int16</channel_format><desc><channels>",
        "<channel><label>Fp1</label></channel><channel><label>Fp2</label></channel>",
        "</channels></desc></info>"
    )
    .to_owned()
}

fn marker_header_xml() -> String {
    concat!(
        "<?xml version=\"1.0\"?><info><name>Markers</name><type>Markers</type>",
        "<channel_count>1</channel_count><nominal_srate>0</nominal_srate>",
        "<channel_format>string</channel_format></info>"
    )
    .to_owned()
}

/// Two streams: a regular int16 EEG stream and an irregular string marker
/// stream, plus clock offsets for both and a boundary between them. That is
/// every semantic the profile requires, in one file.
fn xdf_fixture() -> Vec<u8> {
    let mut file = b"XDF:".to_vec();
    file.extend_from_slice(&chunk(
        1,
        b"<?xml version=\"1.0\"?><info><version>1.0</version></info>",
    ));
    file.extend_from_slice(&stream_header(1, &eeg_header_xml()));
    file.extend_from_slice(&stream_header(2, &marker_header_xml()));

    // EEG: four deduced-timestamp samples over two channels.
    let mut samples = 1_u32.to_le_bytes().to_vec();
    samples.push(1);
    samples.push(4);
    for index in 0..4_i16 {
        samples.push(0);
        samples.extend_from_slice(&(index * 10).to_le_bytes());
        samples.extend_from_slice(&(-index * 10).to_le_bytes());
    }
    file.extend_from_slice(&chunk(3, &samples));

    // Markers: two explicitly stamped string samples.
    let mut markers = 2_u32.to_le_bytes().to_vec();
    markers.push(1);
    markers.push(2);
    for (stamp, text) in [(0.5_f64, "start"), (1.25_f64, "stop")] {
        markers.push(8);
        markers.extend_from_slice(&stamp.to_le_bytes());
        markers.push(1);
        markers.push(text.len() as u8);
        markers.extend_from_slice(text.as_bytes());
    }
    file.extend_from_slice(&chunk(3, &markers));

    // A boundary, then clock offsets for each stream.
    file.extend_from_slice(&chunk(5, &BOUNDARY_UUID));
    for (id, collection, offset) in [
        (1_u32, 10.0_f64, -0.001_f64),
        (1, 20.0, -0.002),
        (2, 10.0, 0.003),
    ] {
        let mut content = id.to_le_bytes().to_vec();
        content.extend_from_slice(&collection.to_le_bytes());
        content.extend_from_slice(&offset.to_le_bytes());
        file.extend_from_slice(&chunk(4, &content));
    }

    for id in [1_u32, 2] {
        let mut content = id.to_le_bytes().to_vec();
        content.extend_from_slice(
            b"<?xml version=\"1.0\"?><info><first_timestamp>0</first_timestamp></info>",
        );
        file.extend_from_slice(&chunk(6, &content));
    }
    file
}

fn foreign(bytes: Vec<u8>) -> ForeignObject {
    ForeignObject {
        profile: ProfileId("xdf.1.0".to_owned()),
        entries: vec![ForeignEntry {
            path: "session.xdf".to_owned(),
            media_type: Some("application/x-xdf".to_owned()),
            bytes,
        }],
    }
}

#[test]
fn xdf_import_maps_every_stream_clock_and_boundary() {
    let adapter = XdfAdapter::new(1 << 20);
    let source = foreign(xdf_fixture());
    let outcome = adapter
        .import(&source, ValidationLimits::default())
        .expect("fixture imports");
    let dataset = &outcome.dataset;

    // Two streams, each with its OWN clock, plus the recording host clock.
    assert_eq!(dataset.streams().len(), 2);
    assert_eq!(dataset.clocks().len(), 3);
    // One relation per stream that reported offsets.
    assert_eq!(dataset.clock_relations().len(), 2);
    // Every relation points at the same host clock, which is what makes the
    // two streams comparable at all.
    let host: Vec<_> = dataset
        .clock_relations()
        .iter()
        .map(|relation| relation.to_clock_id())
        .collect();
    assert_eq!(host[0], host[1]);
    // The boundary became a real event.
    assert_eq!(dataset.events().len(), 1);

    // The EEG stream is regular; the marker stream is not, and neither had a
    // rate invented for it.
    let signal_atoms = dataset
        .atoms()
        .iter()
        .filter(|atom| matches!(atom, semantic_abir::Atom::SignalBlock(_)))
        .count();
    assert_eq!(signal_atoms, 1, "only the EEG stream is a signal block");

    let inspect = adapter.inspect(&source).expect("fixture inspects");
    assert_eq!(inspect.required_resources["streams"], 2);
    assert_eq!(inspect.required_resources["boundaries"], 1);
    assert_eq!(inspect.required_resources["clock-offsets"], 3);
}

/// A NUMERIC stream with explicit per-sample timestamps. The string marker
/// stream in the main fixture becomes a blob, so it never exercised the
/// explicit time axis; this one does, and ABIR requires that axis to name a
/// companion payload belonging to a real atom.
fn irregular_numeric_fixture() -> Vec<u8> {
    let xml = concat!(
        "<?xml version=\"1.0\"?><info><name>Irregular</name><type>EEG</type>",
        "<channel_count>1</channel_count><nominal_srate>0</nominal_srate>",
        "<channel_format>int16</channel_format></info>"
    );
    let mut file = b"XDF:".to_vec();
    file.extend_from_slice(&chunk(
        1,
        b"<?xml version=\"1.0\"?><info><version>1.0</version></info>",
    ));
    file.extend_from_slice(&stream_header(1, xml));
    let mut samples = 1_u32.to_le_bytes().to_vec();
    samples.push(1);
    samples.push(3);
    for (index, stamp) in [0.0_f64, 0.75, 2.5].into_iter().enumerate() {
        samples.push(8);
        samples.extend_from_slice(&stamp.to_le_bytes());
        samples.extend_from_slice(&(index as i16).to_le_bytes());
    }
    file.extend_from_slice(&chunk(3, &samples));
    file
}

#[test]
fn xdf_irregular_numeric_stream_carries_its_own_timestamps() {
    let adapter = XdfAdapter::new(1 << 20);
    let outcome = adapter
        .import(
            &foreign(irregular_numeric_fixture()),
            ValidationLimits::default(),
        )
        .expect("an irregular numeric stream imports");
    let block = outcome
        .dataset
        .atoms()
        .iter()
        .find_map(|atom| match atom {
            semantic_abir::Atom::SignalBlock(block) => Some(block),
            _ => None,
        })
        .expect("the stream is a signal block");
    // No rate was invented for a stream that declared none.
    let semantic_abir::TimeAxis::Explicit { timestamps, count } = block.time_axis() else {
        panic!("an irregular stream must not be given a regular axis");
    };
    assert_eq!(*count, 3);
    // The companion payload resolves to a real atom, which is what ABIR
    // validation refuses to let dangle.
    assert!(outcome
        .dataset
        .atoms()
        .iter()
        .filter_map(semantic_abir::Atom::payload)
        .any(|payload| payload.content_id() == *timestamps));
}

#[test]
fn xdf_reverse_export_restores_the_source_byte_for_byte() {
    let adapter = XdfAdapter::new(1 << 20);
    let bytes = xdf_fixture();
    let source = foreign(bytes.clone());
    let outcome = adapter
        .import(&source, ValidationLimits::default())
        .expect("fixture imports");
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
    assert!(receipt.semantic_equivalence);
    assert_eq!(restored.entries[0].bytes, bytes);
}

#[test]
fn xdf_rejects_wrong_profile_multiple_files_and_malformed_bytes() {
    let adapter = XdfAdapter::new(1 << 20);
    let valid = xdf_fixture();

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

    // No magic.
    let mut no_magic = valid.clone();
    no_magic[0..4].copy_from_slice(b"XXXX");
    assert!(adapter
        .import(&foreign(no_magic), ValidationLimits::default())
        .is_err());

    // Truncated mid-chunk: the reader must refuse rather than read past.
    let truncated = valid[..valid.len() / 2].to_vec();
    assert!(adapter
        .import(&foreign(truncated), ValidationLimits::default())
        .is_err());

    // Samples for a stream whose header never appeared.
    let mut orphan = b"XDF:".to_vec();
    orphan.extend_from_slice(&chunk(
        1,
        b"<?xml version=\"1.0\"?><info><version>1.0</version></info>",
    ));
    let mut samples = 9_u32.to_le_bytes().to_vec();
    samples.push(1);
    samples.push(1);
    samples.push(0);
    samples.extend_from_slice(&0_i16.to_le_bytes());
    orphan.extend_from_slice(&chunk(3, &samples));
    assert!(adapter
        .import(&foreign(orphan), ValidationLimits::default())
        .is_err());

    // A boundary chunk carrying the wrong UUID is not a boundary.
    let mut wrong_boundary = b"XDF:".to_vec();
    wrong_boundary.extend_from_slice(&chunk(
        1,
        b"<?xml version=\"1.0\"?><info><version>1.0</version></info>",
    ));
    wrong_boundary.extend_from_slice(&stream_header(1, &eeg_header_xml()));
    wrong_boundary.extend_from_slice(&chunk(5, &[0_u8; 16]));
    assert!(adapter
        .import(&foreign(wrong_boundary), ValidationLimits::default())
        .is_err());
}

#[test]
fn xdf_declares_first_class_status_and_names_its_independent_validator() {
    let adapter = XdfAdapter::new(1 << 20);
    let profile = adapter.profile();
    assert_eq!(profile.id.0, "xdf.1.0");
    assert_eq!(profile.edition, "1.0");
    assert_eq!(profile.required_validator, "pyxdf");
    assert!(matches!(
        profile.status,
        abir_adapter::ProfileStatus::Semantic
    ));

    let artifact = adapter.validate(&foreign(xdf_fixture()));
    assert!(artifact.internal_valid);
    // The adapter never claims the independent verdict on its own behalf.
    assert_eq!(artifact.independent_valid, None);
    assert_eq!(artifact.independent_validator, "pyxdf");
}
