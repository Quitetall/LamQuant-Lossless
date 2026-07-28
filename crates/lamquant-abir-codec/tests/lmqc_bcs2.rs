use lamquant_abir_codec::{
    bcs2_to_lmqc, lmqc_to_bcs2, open_lmqc_bcs2, LmqcBundleError, LmqcBundleInput, LmqcPayloadKind,
};
use lamquant_lml_mcu::{
    crc32::crc32,
    lmqc::{encode_lmqc, LmqcError, PAYLOAD_FP16_LATENT, PAYLOAD_FSQ_TOKENS},
};
use semantic_abir::{ContentId, Rational, SourceKey};
use semantic_abir_bcs::{
    encode_codec_bundle, raw_content_id, Bcs2Error, CodecBundleError, CodecBundleInput,
    CodecBundleView, CodecFidelity, CodecFidelityKind, CodecImplementation, CodecParameterValue,
    CodecProfile, ModelProvenance, PccpStatus, ResourceBounds, CAP_LMQC_LEGACY_V1,
};
use sha2::{Digest, Sha256};

const FROZEN_FIXTURES: [(&str, &str, LmqcPayloadKind, bool, bool); 4] = [
    (
        include_str!("fixtures/lmqc/fp16_none.hex"),
        "c8ba35763f3aa7709cea36750ecaaf4aada5b3e8ebfd911e0ee00f9fdea086d8",
        LmqcPayloadKind::Fp16Latent,
        false,
        false,
    ),
    (
        include_str!("fixtures/lmqc/fsq_coords.hex"),
        "92e319b6da97839a3f19c20fd798c805e2f373562a1e176fd26cc6d02c759978",
        LmqcPayloadKind::FsqTokens,
        true,
        false,
    ),
    (
        include_str!("fixtures/lmqc/fp16_names.hex"),
        "da096f8477567eb942d0a5f788fc65473b0b67ce19bedad2c7f5dd6306081532",
        LmqcPayloadKind::Fp16Latent,
        false,
        true,
    ),
    (
        include_str!("fixtures/lmqc/fsq_coords_names.hex"),
        "2c6bf21740b852d20c49c6c95681152f79715fc6670b0e7625d071925d372009",
        LmqcPayloadKind::FsqTokens,
        true,
        true,
    ),
];

fn producer() -> CodecImplementation {
    CodecImplementation {
        build_id: "legacy-lmqc-fixture".into(),
        implementation_id: ContentId::from_bytes([0x31; 32]),
        kernel_id: "org.quitetall.lamquant.lmqc.fp16-latent-v1".into(),
    }
}

fn model() -> ModelProvenance {
    ModelProvenance {
        checkpoint_content_id: ContentId::from_bytes([0x41; 32]),
        checkpoint_sha256: [0x42; 32],
        pccp_change_id: "PCCP-LMQC-FIXTURE".into(),
        pccp_evidence_id: ContentId::from_bytes([0x43; 32]),
        pccp_status: PccpStatus::Candidate,
    }
}

fn import_policy() -> LmqcBundleInput {
    LmqcBundleInput {
        coordinate_uncertainty: Rational::new(1, 1_000).unwrap(),
        fidelity: CodecFidelity {
            bound: Some(CodecParameterValue::Rational {
                denominator: "1000".into(),
                numerator: "75".into(),
            }),
            contract_id: ContentId::from_bytes([0x51; 32]),
            kind: CodecFidelityKind::Bounded,
            metric: Some("prd".into()),
        },
        implementation: producer(),
        model_provenance: model(),
    }
}

fn reseal_lmqc(bytes: &mut [u8]) {
    let crc_offset = bytes.len() - 4;
    let checksum = crc32(&bytes[..crc_offset]);
    bytes[crc_offset..].copy_from_slice(&checksum.to_le_bytes());
}

fn frozen_fixture(hex: &str, expected_sha256: &str) -> Vec<u8> {
    let compact = hex
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect::<Vec<_>>();
    assert_eq!(compact.len() % 2, 0, "fixture hex must contain byte pairs");
    let bytes = compact
        .chunks_exact(2)
        .map(|pair| {
            let text = core::str::from_utf8(pair).expect("fixture hex is ASCII");
            u8::from_str_radix(text, 16).expect("fixture contains valid hex")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        expected_sha256,
        "frozen fixture content changed"
    );
    bytes
}

#[test]
fn frozen_lmqc_fixtures_restore_byte_exact_across_payload_and_montage_forms() {
    for (hex, sha256, payload_kind, has_coords, has_names) in FROZEN_FIXTURES {
        let legacy = frozen_fixture(hex, sha256);
        let bcs2 =
            lmqc_to_bcs2(&legacy, import_policy(), ResourceBounds::default()).expect("import");
        let opened = open_lmqc_bcs2(&bcs2, ResourceBounds::default()).expect("open");
        assert_eq!(opened.container().payload_kind, payload_kind);
        assert_eq!(opened.container().coords.is_some(), has_coords);
        assert_eq!(opened.container().channels.is_some(), has_names);
        assert_eq!(
            bcs2_to_lmqc(&bcs2, ResourceBounds::default()).expect("restore"),
            legacy
        );
    }
}

#[test]
fn legacy_lmqc_converts_to_semantic_bundle_and_restores_byte_exact() {
    let names = vec!["Fp1-Ref".to_string(), "Fp2-Ref".to_string()];
    let coordinates = [0.081_f32, -0.0725, 0.0341, f32::NAN, f32::NAN, f32::NAN];
    let neural_payload = [0x11, 0x22, 0x33, 0x44, 0x55];
    let legacy = encode_lmqc(
        2,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        Some(&coordinates),
        Some(&names),
        &neural_payload,
    )
    .expect("valid legacy fixture");

    let bcs2 =
        lmqc_to_bcs2(&legacy, import_policy(), ResourceBounds::default()).expect("legacy import");
    assert!(matches!(
        CodecBundleView::open(&bcs2, ResourceBounds::default()),
        Err(CodecBundleError::Bcs2(Bcs2Error::UnsupportedCapabilities(
            CAP_LMQC_LEGACY_V1
        )))
    ));
    let opened = open_lmqc_bcs2(&bcs2, ResourceBounds::default()).expect("validated BCS2 import");

    assert_eq!(
        opened.bundle().catalog().profile(),
        CodecProfile::LmqProgressive
    );
    assert_eq!(opened.neural_payload(), neural_payload);
    assert!(
        opened.bundle().packet(1).unwrap().len() < legacy.len(),
        "source re-emit metadata must not duplicate the neural payload"
    );
    assert_eq!(opened.container().n_channels, 2);
    assert_eq!(opened.container().latent_c, 32);
    assert_eq!(opened.container().latent_t, 79);
    assert_eq!(opened.container().sample_rate, 250);
    assert_eq!(opened.container().window_samples, 2_500);

    let dataset = opened.dataset();
    assert_eq!(dataset.channel_bases().len(), 1);
    let channels = dataset.channel_bases()[0].channels();
    assert_eq!(channels.len(), 2);
    assert_eq!(
        channels[0].source_keys(),
        &[SourceKey::new("lmqc.channel-label", "Fp1-Ref").unwrap()]
    );
    assert_eq!(
        channels[1].source_keys(),
        &[SourceKey::new("lmqc.channel-label", "Fp2-Ref").unwrap()]
    );
    assert!(channels[0].coordinate_frame_id().is_some());
    assert!(channels[1].coordinate_frame_id().is_none());
    assert_eq!(dataset.coordinate_frames().len(), 2);
    let electrode_frame = dataset
        .coordinate_frames()
        .iter()
        .find(|frame| Some(frame.id()) == channels[0].coordinate_frame_id())
        .unwrap();
    let montage_root_id = electrode_frame
        .parent_id()
        .expect("located electrodes share a declared montage frame");
    let montage_root = dataset
        .coordinate_frames()
        .iter()
        .find(|frame| frame.id() == montage_root_id)
        .expect("electrode parent is present");
    assert!(montage_root.parent_id().is_none());
    assert!(montage_root.transform().is_none());
    assert_eq!(
        electrode_frame.uncertainty(),
        Rational::new(1, 1_000).unwrap()
    );
    assert_eq!(dataset.source_capsules().len(), 1);
    assert_eq!(
        dataset.source_capsules()[0].content_id(),
        raw_content_id(&legacy)
    );
    assert_eq!(dataset.clocks().len(), 1);
    assert_eq!(dataset.clocks()[0].rate(), Rational::new(250, 1).unwrap());
    assert_eq!(
        dataset.streams()[0].clock_id(),
        Some(dataset.clocks()[0].id())
    );
    assert_eq!(
        bcs2_to_lmqc(&bcs2, ResourceBounds::default()).unwrap(),
        legacy
    );
}

#[test]
fn unknown_flags_and_reserved_bytes_fail_preflight_but_high_channel_counts_import() {
    let canonical = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        None,
        &[0x11],
    )
    .unwrap();

    let mut unknown_flags = canonical.clone();
    unknown_flags[5] |= 0x80;
    reseal_lmqc(&mut unknown_flags);
    assert!(matches!(
        lmqc_to_bcs2(&unknown_flags, import_policy(), ResourceBounds::default()),
        Err(LmqcBundleError::InvalidLegacy("unknown LMQC header flags"))
    ));

    let mut reserved = canonical;
    reserved[7] = 1;
    reseal_lmqc(&mut reserved);
    assert!(matches!(
        lmqc_to_bcs2(&reserved, import_policy(), ResourceBounds::default()),
        Err(LmqcBundleError::InvalidLegacy("nonzero LMQC reserved byte"))
    ));

    let high_channel_count = encode_lmqc(
        1_025,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        None,
        &[0x11],
    )
    .unwrap();
    let migrated = lmqc_to_bcs2(
        &high_channel_count,
        import_policy(),
        ResourceBounds::default(),
    )
    .expect("LMQC wire range is not narrowed by converter policy");
    assert_eq!(
        open_lmqc_bcs2(&migrated, ResourceBounds::default())
            .unwrap()
            .container()
            .n_channels,
        1_025
    );
}

#[test]
fn import_and_inverse_share_source_size_bound() {
    let legacy = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FSQ_TOKENS,
        None,
        None,
        &[0x5a; 256],
    )
    .unwrap();
    let bounds = ResourceBounds {
        max_frame_bytes: u32::try_from(legacy.len() - 1).unwrap(),
        ..ResourceBounds::default()
    };
    assert!(matches!(
        lmqc_to_bcs2(&legacy, import_policy(), bounds),
        Err(LmqcBundleError::InvalidLegacy(
            "source container exceeds resource bound"
        ))
    ));

    let bundle = lmqc_to_bcs2(&legacy, import_policy(), ResourceBounds::default()).unwrap();
    let inverse_bounds = ResourceBounds {
        max_frame_bytes: u32::try_from(legacy.len() - 1).unwrap(),
        ..ResourceBounds::default()
    };
    let inverse = bcs2_to_lmqc(&bundle, inverse_bounds);
    assert!(
        inverse.is_err(),
        "inverse must refuse a source above its caller resource bound"
    );
}

#[test]
fn inverse_rejects_missing_packet_capability_and_corrupt_reemit_metadata() {
    let legacy = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        None,
        &[0x11, 0x22],
    )
    .unwrap();
    let bundle = lmqc_to_bcs2(&legacy, import_policy(), ResourceBounds::default()).unwrap();
    let opened = open_lmqc_bcs2(&bundle, ResourceBounds::default()).unwrap();
    let catalog = opened.bundle().catalog();
    let packets = [
        opened.bundle().packet(0).unwrap(),
        opened.bundle().packet(1).unwrap(),
    ];
    let missing_capability = encode_codec_bundle(
        CodecBundleInput {
            required_capabilities: 0,
            canonical_semantics: opened.bundle().canonical_semantics(),
            fidelity: catalog.fidelity().clone(),
            implementation: catalog.implementation().clone(),
            model_provenance: catalog.model_provenance().cloned(),
            packets: &packets,
            parameters: catalog.parameters().to_vec(),
            profile: CodecProfile::LmqProgressive,
        },
        ResourceBounds::default(),
    )
    .unwrap();
    assert!(matches!(
        open_lmqc_bcs2(&missing_capability, ResourceBounds::default()),
        Err(LmqcBundleError::SemanticMismatch)
    ));

    let mut corrupt_metadata = packets[1].to_vec();
    *corrupt_metadata.last_mut().unwrap() ^= 1;
    let corrupt_packets = [packets[0], corrupt_metadata.as_slice()];
    let corrupt_bundle = encode_codec_bundle(
        CodecBundleInput {
            required_capabilities: CAP_LMQC_LEGACY_V1,
            canonical_semantics: opened.bundle().canonical_semantics(),
            fidelity: catalog.fidelity().clone(),
            implementation: catalog.implementation().clone(),
            model_provenance: catalog.model_provenance().cloned(),
            packets: &corrupt_packets,
            parameters: catalog.parameters().to_vec(),
            profile: CodecProfile::LmqProgressive,
        },
        ResourceBounds::default(),
    )
    .unwrap();
    assert!(matches!(
        open_lmqc_bcs2(&corrupt_bundle, ResourceBounds::default()),
        Err(LmqcBundleError::Legacy(LmqcError::CrcMismatch))
    ));
}

#[test]
fn optional_montage_forms_and_both_payload_kinds_restore_exactly() {
    let names = vec!["Fp1".to_string(), "Fp2".to_string()];
    let coordinates = [0.08, 0.0, 0.03, -0.08, 0.0, 0.03];
    let cases = [
        (PAYLOAD_FP16_LATENT, None, None),
        (PAYLOAD_FSQ_TOKENS, Some(coordinates.as_slice()), None),
        (PAYLOAD_FP16_LATENT, None, Some(names.as_slice())),
        (
            PAYLOAD_FSQ_TOKENS,
            Some(coordinates.as_slice()),
            Some(names.as_slice()),
        ),
    ];
    for (payload_kind, coords, labels) in cases {
        let legacy = encode_lmqc(
            2,
            32,
            79,
            250,
            2_500,
            payload_kind,
            coords,
            labels,
            &[0x5a; 128],
        )
        .unwrap();
        let first = lmqc_to_bcs2(&legacy, import_policy(), ResourceBounds::default()).unwrap();
        let second = lmqc_to_bcs2(&legacy, import_policy(), ResourceBounds::default()).unwrap();
        assert_eq!(first, second, "migration must be deterministic");
        assert_eq!(
            bcs2_to_lmqc(&first, ResourceBounds::default()).unwrap(),
            legacy
        );
    }
}

#[test]
fn corrupt_montage_and_unknown_payload_kind_fail_closed() {
    let partial_nan = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        Some(&[0.08, f32::NAN, 0.03]),
        None,
        &[0x11],
    )
    .unwrap();
    assert!(matches!(
        lmqc_to_bcs2(&partial_nan, import_policy(), ResourceBounds::default()),
        Err(LmqcBundleError::Montage(_))
    ));

    let unknown_payload = encode_lmqc(1, 32, 79, 250, 2_500, 0xff, None, None, &[0x11]).unwrap();
    assert!(matches!(
        lmqc_to_bcs2(&unknown_payload, import_policy(), ResourceBounds::default()),
        Err(LmqcBundleError::InvalidLegacy("unknown payload kind"))
    ));
}

#[test]
fn import_rejects_zero_semantic_extent() {
    let legacy = encode_lmqc(
        0,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        None,
        &[0x11],
    )
    .unwrap();
    assert!(matches!(
        lmqc_to_bcs2(&legacy, import_policy(), ResourceBounds::default()),
        Err(LmqcBundleError::InvalidLegacy("zero channels"))
    ));
}

#[test]
fn import_rejects_false_exact_fidelity_claim() {
    let legacy = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        None,
        &[0x11],
    )
    .unwrap();
    let mut policy = import_policy();
    policy.fidelity = CodecFidelity {
        bound: None,
        contract_id: ContentId::from_bytes([0x61; 32]),
        kind: CodecFidelityKind::Exact,
        metric: None,
    };
    assert!(matches!(
        lmqc_to_bcs2(&legacy, policy, ResourceBounds::default()),
        Err(LmqcBundleError::InvalidLegacy(
            "lossy LMQC cannot claim exact fidelity"
        ))
    ));
}

#[test]
fn import_rejects_negative_coordinate_uncertainty() {
    let legacy = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        None,
        &[0x11],
    )
    .unwrap();
    let mut policy = import_policy();
    policy.coordinate_uncertainty = Rational::new(-1, 1_000).unwrap();
    assert!(matches!(
        lmqc_to_bcs2(&legacy, policy, ResourceBounds::default()),
        Err(LmqcBundleError::InvalidLegacy(
            "negative coordinate uncertainty"
        ))
    ));
}

#[test]
fn metadata_budget_is_enforced_before_large_channel_label_projection() {
    let names = vec!["x".repeat(64 * 1024)];
    let legacy = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        Some(&names),
        &[0x11],
    )
    .unwrap();
    let bounds = ResourceBounds {
        max_catalog_bytes: 1_024,
        ..ResourceBounds::default()
    };
    assert!(matches!(
        lmqc_to_bcs2(&legacy, import_policy(), bounds),
        Err(LmqcBundleError::InvalidLegacy(
            "legacy metadata exceeds catalog bound"
        ))
    ));
}

#[test]
fn escaped_label_expansion_is_bounded_before_allocation() {
    let names = vec!["\0".repeat(64 * 1024)];
    let legacy = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        Some(&names),
        &[0x11],
    )
    .unwrap();
    let bounds = ResourceBounds {
        max_catalog_bytes: 100 * 1024,
        ..ResourceBounds::default()
    };
    assert!(matches!(
        lmqc_to_bcs2(&legacy, import_policy(), bounds),
        Err(LmqcBundleError::InvalidLegacy(
            "projected channel labels exceed catalog bound"
        ))
    ));
}

#[test]
fn control_character_labels_use_reversible_source_key_and_restore_exactly() {
    let label = "Fp1\tRef\0".to_string();
    let legacy = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        Some(core::slice::from_ref(&label)),
        &[0x11],
    )
    .unwrap();
    let bcs2 =
        lmqc_to_bcs2(&legacy, import_policy(), ResourceBounds::default()).expect("legacy import");
    let opened = open_lmqc_bcs2(&bcs2, ResourceBounds::default()).unwrap();
    let key = &opened.dataset().channel_bases()[0].channels()[0].source_keys()[0];
    assert_eq!(key.namespace(), "lmqc.channel-label.utf8-hex");
    assert_eq!(key.value(), "4670310952656600");
    assert_eq!(
        bcs2_to_lmqc(&bcs2, ResourceBounds::default()).unwrap(),
        legacy
    );
}

#[test]
fn inverse_rejects_valid_bundle_whose_semantics_describe_another_source() {
    let original_payload = [0x11, 0x22];
    let original = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        Some(&["Fp1".to_string()]),
        &original_payload,
    )
    .unwrap();
    let other = encode_lmqc(
        1,
        32,
        79,
        250,
        2_500,
        PAYLOAD_FP16_LATENT,
        None,
        Some(&["Cz".to_string()]),
        &[0x33, 0x44],
    )
    .unwrap();
    let other_bundle = lmqc_to_bcs2(&other, import_policy(), ResourceBounds::default()).unwrap();
    let opened_other = open_lmqc_bcs2(&other_bundle, ResourceBounds::default()).unwrap();
    let original_bundle =
        lmqc_to_bcs2(&original, import_policy(), ResourceBounds::default()).unwrap();
    let opened_original = open_lmqc_bcs2(&original_bundle, ResourceBounds::default()).unwrap();
    let catalog = opened_other.bundle().catalog();
    let packets = [
        opened_original.bundle().packet(0).unwrap(),
        opened_original.bundle().packet(1).unwrap(),
    ];
    let mismatched = encode_codec_bundle(
        CodecBundleInput {
            required_capabilities: CAP_LMQC_LEGACY_V1,
            canonical_semantics: opened_other.bundle().canonical_semantics(),
            fidelity: catalog.fidelity().clone(),
            implementation: catalog.implementation().clone(),
            model_provenance: catalog.model_provenance().cloned(),
            packets: &packets,
            parameters: catalog.parameters().to_vec(),
            profile: CodecProfile::LmqProgressive,
        },
        ResourceBounds::default(),
    )
    .unwrap();

    assert!(matches!(
        open_lmqc_bcs2(&mismatched, ResourceBounds::default()),
        Err(LmqcBundleError::SemanticMismatch)
    ));
}
