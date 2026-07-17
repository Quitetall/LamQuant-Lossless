use lamquant_lml_optimum_v2::bgf1_learned::{
    Bgf1ChannelIdentity, Bgf1LearnedCodec, Bgf1LearnedMode,
};
use lamquant_lml_optimum_v2::model_pack::{ModelPack, Tensor, TensorDtype};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::process::Command;

fn learned_model_pack() -> Vec<u8> {
    let descriptor = [239_i32, 1, 1, 7_424]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    let mut predict_second = vec![32_u8; 512];
    predict_second[256..].fill(16);
    let mut update_first = vec![(-8_i8) as u8; 512];
    update_first[256..].fill(4);
    let mut temporal = vec![0_u8; 256 * 16];
    for row in temporal.chunks_exact_mut(16) {
        row[0] = 32;
    }
    let tensors = vec![
        Tensor {
            name: "bgf1.descriptor".into(),
            dtype: TensorDtype::I32,
            shape: vec![4],
            scale_numerator: 1,
            scale_shift: 0,
            data: descriptor,
        },
        Tensor {
            name: "coupling.predict_second".into(),
            dtype: TensorDtype::I8,
            shape: vec![2, 256],
            scale_numerator: 1,
            scale_shift: 6,
            data: predict_second,
        },
        Tensor {
            name: "coupling.update_first".into(),
            dtype: TensorDtype::I8,
            shape: vec![2, 256],
            scale_numerator: 1,
            scale_shift: 6,
            data: update_first,
        },
        Tensor {
            name: "entropy.exponent_logits".into(),
            dtype: TensorDtype::I8,
            shape: vec![16, 16],
            scale_numerator: 1,
            scale_shift: 0,
            data: vec![0; 256],
        },
        Tensor {
            name: "entropy.mantissa_logits".into(),
            dtype: TensorDtype::I8,
            shape: vec![16, 16],
            scale_numerator: 1,
            scale_shift: 0,
            data: vec![0; 256],
        },
        Tensor {
            name: "entropy.sign_logits".into(),
            dtype: TensorDtype::I8,
            shape: vec![256],
            scale_numerator: 1,
            scale_shift: 0,
            data: vec![0; 256],
        },
        Tensor {
            name: "entropy.token_magnitude_bias".into(),
            dtype: TensorDtype::I8,
            shape: vec![256],
            scale_numerator: 1,
            scale_shift: 0,
            data: vec![0; 256],
        },
        Tensor {
            name: "prior.graph".into(),
            dtype: TensorDtype::I8,
            shape: vec![256, 4],
            scale_numerator: 1,
            scale_shift: 6,
            data: vec![0; 1_024],
        },
        Tensor {
            name: "prior.scale_bias".into(),
            dtype: TensorDtype::I8,
            shape: vec![256],
            scale_numerator: 1,
            scale_shift: 4,
            data: vec![0; 256],
        },
        Tensor {
            name: "prior.temporal".into(),
            dtype: TensorDtype::I8,
            shape: vec![256, 16],
            scale_numerator: 1,
            scale_shift: 6,
            data: temporal,
        },
    ];
    ModelPack::encode(&tensors).expect("synthetic learned model")
}

fn token_weighted_model_pack() -> Vec<u8> {
    let mut tensors = ModelPack::decode(&learned_model_pack()).unwrap().tensors;
    for tensor in &mut tensors {
        let signed = match tensor.name.as_str() {
            "coupling.predict_second" => (0..2)
                .flat_map(|layer| (0..256).map(move |token| ((layer * 17 + token * 5) % 41) - 20))
                .collect::<Vec<_>>(),
            "coupling.update_first" => (0..2)
                .flat_map(|layer| {
                    (0..256).map(move |token| ((layer * 13 + 37 - (token * 3) % 37) % 37) - 18)
                })
                .collect(),
            "entropy.exponent_logits" => (0..16)
                .flat_map(|scale| {
                    (0..16).map(move |position| ((scale * 7 + position * 3) % 31) - 15)
                })
                .collect(),
            "entropy.mantissa_logits" => (0..16)
                .flat_map(|scale| {
                    (0..16).map(move |position| ((scale * 5 + 29 - (position * 7) % 29) % 29) - 14)
                })
                .collect(),
            "entropy.sign_logits" => (0..256).map(|token| ((token * 11) % 41) - 20).collect(),
            "entropy.token_magnitude_bias" => {
                (0..256).map(|token| ((token * 5) % 33) - 16).collect()
            }
            "prior.graph" => (0..256)
                .flat_map(|token| (0..4).map(move |slot| ((token + slot * 7) % 9) - 4))
                .collect(),
            "prior.scale_bias" => (0..256).map(|token| ((token * 11) % 65) - 32).collect(),
            "prior.temporal" => (0..256)
                .flat_map(|token| (0..16).map(move |lag| ((token * 3 + lag * 5) % 13) - 6))
                .collect(),
            _ => continue,
        };
        tensor.data = signed
            .into_iter()
            .map(|value| (value as i8) as u8)
            .collect();
    }
    ModelPack::encode(&tensors).unwrap()
}

fn fixture() -> (Vec<Bgf1ChannelIdentity>, Vec<Vec<i64>>) {
    (
        vec![
            Bgf1ChannelIdentity::new(0, "EEG C4"),
            Bgf1ChannelIdentity::new(1, "ECG EKG"),
            Bgf1ChannelIdentity::new(2, "EEG C3"),
            Bgf1ChannelIdentity::new(3, "1"),
        ],
        vec![
            vec![10, 11, 13, 12, 8, 7],
            vec![100, 101, 99, 102, 100, 98],
            vec![-5, -3, -4, 0, 2, 1],
            vec![0, 1, 0, 1, 0, 1],
        ],
    )
}

fn hex_bytes(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => panic!("invalid hex digit"),
            };
            (digit(pair[0]) << 4) | digit(pair[1])
        })
        .collect()
}

fn crc32c(packed: &[u8]) -> u32 {
    let mut state = 0xffff_ffffu32;
    for &byte in packed {
        state ^= u32::from(byte);
        for _ in 0..8 {
            state = (state >> 1) ^ (0x82f6_3b78 & 0u32.wrapping_sub(state & 1));
        }
    }
    state ^ 0xffff_ffff
}

fn recrc(packet: &mut [u8]) {
    packet[83..87].fill(0);
    let checksum = crc32c(packet);
    packet[83..87].copy_from_slice(&checksum.to_le_bytes());
}

#[test]
fn learned_codec_accepts_the_python_lqw1_profile() {
    let pack = learned_model_pack();
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&pack)),
        [
            0xf4, 0xaf, 0x21, 0x9c, 0xe5, 0xc2, 0x7a, 0x96, 0xad, 0x1a, 0xb3, 0x56, 0x7d, 0xad,
            0x50, 0xc6, 0x4c, 0x99, 0xf9, 0x85, 0x2c, 0x08, 0x1f, 0x12, 0x80, 0x91, 0x12, 0x23,
            0xab, 0xd4, 0xe0, 0x91,
        ]
    );
    Bgf1LearnedCodec::from_lqw1(&pack).expect("strict BGF1 model");
}

#[test]
fn learned_modes_are_byte_identical_to_the_python_carrier() {
    let codec = Bgf1LearnedCodec::from_lqw1(&learned_model_pack()).unwrap();
    let (identities, signal) = fixture();
    let cases = [
        (
            Bgf1LearnedMode::NoFlow,
            "4c4d4f310300034247463101011000040001000600000000e80300ef000000f4af219ce5c27a96ad1ab3567dad50c64c99f9852c081f1280911223abd4e091200000001000000018000000170000000cb788b77885eeb50200064545472043330000064545472043340300013101000745434720454b4701ffffff00ffffffffffffffffffffff0201000000000000060000000000000017000000a0000000bccf2d0c89e84013c3a866f61bf4862680b0bf473b972d",
        ),
        (
            Bgf1LearnedMode::Flow,
            "4c4d4f310300034247463101011000040001000600000000e80300ef000000f4af219ce5c27a96ad1ab3567dad50c64c99f9852c081f1280911223abd4e0912000000010000000180000001a0000000cb788b7a63b79fd0200064545472043330000064545472043340300013101000745434720454b4701ffffff00ffffffffffffffffffffff030100000000000006000000000000001a000000ba000000cfe73e07bf64f10e80906277368f21bf26b662a3b735dccbfe82",
        ),
    ];
    for (mode, expected) in cases {
        assert_eq!(
            codec
                .encode_window(&signal, &identities, 256_000, 16, mode)
                .unwrap(),
            hex_bytes(expected),
        );
    }
}

#[test]
fn learned_codec_decodes_python_packets_and_reencodes_canonically() {
    let codec = Bgf1LearnedCodec::from_lqw1(&learned_model_pack()).unwrap();
    let (identities, signal) = fixture();
    let cases = [
        (
            Bgf1LearnedMode::NoFlow,
            160,
            "4c4d4f310300034247463101011000040001000600000000e80300ef000000f4af219ce5c27a96ad1ab3567dad50c64c99f9852c081f1280911223abd4e091200000001000000018000000170000000cb788b77885eeb50200064545472043330000064545472043340300013101000745434720454b4701ffffff00ffffffffffffffffffffff0201000000000000060000000000000017000000a0000000bccf2d0c89e84013c3a866f61bf4862680b0bf473b972d",
        ),
        (
            Bgf1LearnedMode::Flow,
            186,
            "4c4d4f310300034247463101011000040001000600000000e80300ef000000f4af219ce5c27a96ad1ab3567dad50c64c99f9852c081f1280911223abd4e0912000000010000000180000001a0000000cb788b7a63b79fd0200064545472043330000064545472043340300013101000745434720454b4701ffffff00ffffffffffffffffffffff030100000000000006000000000000001a000000ba000000cfe73e07bf64f10e80906277368f21bf26b662a3b735dccbfe82",
        ),
    ];
    for (mode, event_count, packet) in cases {
        let packet = hex_bytes(packet);
        let decoded = codec.decode_window(&packet).unwrap();
        assert_eq!(decoded.samples, signal);
        assert_eq!(decoded.identities, identities);
        assert_eq!(decoded.sample_rate_mhz, 256_000);
        assert_eq!(decoded.bit_depth, 16);
        assert_eq!(decoded.mode, mode);
        assert_eq!(decoded.event_count, event_count);
        assert_eq!(
            codec
                .encode_window(
                    &decoded.samples,
                    &decoded.identities,
                    decoded.sample_rate_mhz,
                    decoded.bit_depth,
                    decoded.mode,
                )
                .unwrap(),
            packet,
        );
    }
}

#[test]
fn learned_carrier_is_invariant_when_identity_label_and_samples_move_together() {
    let codec = Bgf1LearnedCodec::from_lqw1(&learned_model_pack()).unwrap();
    let (identities, signal) = fixture();
    let permutation = [2, 0, 3, 1];
    let permuted_identities = permutation
        .iter()
        .map(|&index| identities[index].clone())
        .collect::<Vec<_>>();
    let permuted_signal = permutation
        .iter()
        .map(|&index| signal[index].clone())
        .collect::<Vec<_>>();

    for mode in [Bgf1LearnedMode::NoFlow, Bgf1LearnedMode::Flow] {
        assert_eq!(
            codec
                .encode_window(&signal, &identities, 256_000, 16, mode)
                .unwrap(),
            codec
                .encode_window(&permuted_signal, &permuted_identities, 256_000, 16, mode,)
                .unwrap(),
        );
    }
}

#[test]
fn spaced_bipolar_labels_match_python_with_token_weighted_parameters() {
    let pack = token_weighted_model_pack();
    let expected_pack_sha256: [u8; 32] =
        hex_bytes("8aabfe062434ab54b502362974c3ff1b45369b23639eb69eb5b2d81295c8a8a2")
            .try_into()
            .unwrap();
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&pack)),
        expected_pack_sha256,
    );
    let codec = Bgf1LearnedCodec::from_lqw1(&pack).unwrap();
    let identities = vec![
        Bgf1ChannelIdentity::new(0, "EEG C3 - C4"),
        Bgf1ChannelIdentity::new(1, "EEG C4 - P4"),
        Bgf1ChannelIdentity::new(2, "EEG P3 - O1"),
    ];
    let signal = vec![
        vec![
            -1913, -1937, -1947, -1943, -1925, -1893, -1847, -1787, -1713, -1625, -1523, -1407,
        ],
        vec![
            -1884, -1908, -1918, -1914, -1896, -1864, -1818, -1758, -1684, -1596, -1494, -1378,
        ],
        vec![
            -1855, -1879, -1889, -1885, -1867, -1835, -1789, -1729, -1655, -1567, -1465, -1349,
        ],
    ];
    let cases = [
        (
            Bgf1LearnedMode::NoFlow,
            "4c4d4f310300034247463101011000030001000c00000000d00700ef0000008aabfe062434ab54b502362974c3ff1b45369b23639eb69eb5b2d81295c8a8a22a0000000c0000001800000068000000bf4cf776b6e7bfd000000b454547204333202d20433401000b454547204334202d20503402000b454547205033202d204f310102ffff0002ffff0001ffff02010000000000000c000000000000006800000040030000a06fc501babab69caec7b009f766ce4793eaa9233defde4317703b93178868c9e0fe509d717a862d33df5a189de4646d59fd0662fae23e782274c33b701b89230b92c4142a16fe1206ad87b49ee0062d52e3dc6109db3dacb0b0dc5b91b7a23aa72576535c1303da",
        ),
        (
            Bgf1LearnedMode::Flow,
            "4c4d4f310300034247463101011000030001000c00000000d00700ef0000008aabfe062434ab54b502362974c3ff1b45369b23639eb69eb5b2d81295c8a8a22a0000000c000000180000006b000000bf4cf776f204e32800000b454547204333202d20433401000b454547204334202d20503402000b454547205033202d204f310102ffff0002ffff0001ffff03010000000000000c000000000000006b0000005e0300009af96033279f8998c62f983214aedd1b453759bfdcbf78810ef1ef1d5b510050adddb9a46bcd257256a0c9aacc1b7f063de2baaa0d6cf6d17713b51bb9f8a6eaf832f31f1240b6226d65b39ded3b0b2588f26c2619e1dce624737609b38dc362b388261563ef3bc97110c0",
        ),
    ];
    for (mode, expected) in cases {
        let expected = hex_bytes(expected);
        assert_eq!(
            codec
                .encode_window(&signal, &identities, 512_000, 16, mode)
                .unwrap(),
            expected,
        );
        assert_eq!(codec.decode_window(&expected).unwrap().samples, signal);
    }

    let dotted_identities = vec![
        Bgf1ChannelIdentity::new(0, "EEG C3.-C4"),
        Bgf1ChannelIdentity::new(1, "EEG C4 - P4"),
        Bgf1ChannelIdentity::new(2, "EEG P3 - O1"),
    ];
    let dotted_cases = [
        (
            Bgf1LearnedMode::NoFlow,
            "4c4d4f310300034247463101011000030001000c00000000d00700ef0000008aabfe062434ab54b502362974c3ff1b45369b23639eb69eb5b2d81295c8a8a2290000000c0000001800000068000000bf4cf77648e1452800000a4545472043332e2d433401000b454547204334202d20503402000b454547205033202d204f310102ffff0002ffff0001ffff02010000000000000c0000000000000068000000460300006c6eb119c7eaae36a2cd30c63026dc76f39aab1ace20d84e3d97cd4e423b8167df61434f97b3ab53097f64a9c6265af4346721e73f3f519085c6e3cc4f24d3f15bb6f07bcb2801652bcc5ebc8cec43ae7b08d483fd344822beba36758f654c4d849ebe4e5c1303da",
        ),
        (
            Bgf1LearnedMode::Flow,
            "4c4d4f310300034247463101011000030001000c00000000d00700ef0000008aabfe062434ab54b502362974c3ff1b45369b23639eb69eb5b2d81295c8a8a2290000000c000000180000006a000000bf4cf7764814951d00000a4545472043332e2d433401000b454547204334202d20503402000b454547205033202d204f310102ffff0002ffff0001ffff03010000000000000c000000000000006a0000004c0300004975df18d49d200f59cf54f479ba51b8d2885227ca2ad96f41964e944915b57e42b575b20a78b007eb6a9475e941d1734e1c7cdb61934d62009257a49ed4394824ceb7f96686935802dc0e33903732df813ca0a976607998a5a3184ceb22fd2c9f2dd4a14a652a694594",
        ),
    ];
    for (mode, expected) in dotted_cases {
        let expected = hex_bytes(expected);
        assert_eq!(
            codec
                .encode_window(&signal, &dotted_identities, 512_000, 16, mode)
                .unwrap(),
            expected,
        );
        assert_eq!(codec.decode_window(&expected).unwrap().samples, signal);
    }
}

#[test]
fn learned_worker_protocol_forces_each_arm_and_round_trips_lqraw() {
    let root = std::env::temp_dir().join(format!("optimum_v2_bgf1_learned_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let model_path = root.join("model.lqw1");
    let raw_path = root.join("input.lqraw");
    let meta_path = root.join("meta.json");
    fs::write(&model_path, learned_model_pack()).unwrap();
    fs::write(
        &meta_path,
        br#"{"channel_labels":["EEG C4","ECG EKG","EEG C3","1"]}"#,
    )
    .unwrap();
    let (_, signal) = fixture();
    let mut raw = Vec::new();
    raw.extend_from_slice(b"LQR1");
    raw.extend_from_slice(&[1, 4, 16, 0]);
    raw.extend_from_slice(&256_000u32.to_le_bytes());
    raw.extend_from_slice(&4u32.to_le_bytes());
    raw.extend_from_slice(&6u32.to_le_bytes());
    for channel in &signal {
        for &sample in channel {
            raw.extend_from_slice(&(sample as i32).to_le_bytes());
        }
    }
    fs::write(&raw_path, &raw).unwrap();

    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let descriptor = root.join("descriptor.json");
    assert!(Command::new(binary)
        .args(["describe", descriptor.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    let descriptor: serde_json::Value =
        serde_json::from_slice(&fs::read(descriptor).unwrap()).unwrap();
    assert_eq!(
        descriptor["learned_worker"],
        serde_json::json!({
            "encode": "learned-encode MODE MODEL INPUT META_JSON OUTPUT",
            "decode": "learned-decode MODEL INPUT OUTPUT",
            "modes": [2, 3],
            "model_id": 239,
        })
    );
    for mode in ["2", "3"] {
        let packet = root.join(format!("mode-{mode}.lmo"));
        let recon = root.join(format!("mode-{mode}.lqraw"));
        assert!(Command::new(binary)
            .args([
                "learned-encode",
                mode,
                model_path.to_str().unwrap(),
                raw_path.to_str().unwrap(),
                meta_path.to_str().unwrap(),
                packet.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success());
        assert!(Command::new(binary)
            .args([
                "learned-decode",
                model_path.to_str().unwrap(),
                packet.to_str().unwrap(),
                recon.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success());
        assert_eq!(fs::read(recon).unwrap(), raw);
    }
    assert!(!Command::new(binary)
        .args([
            "learned-encode",
            "keep-best",
            model_path.to_str().unwrap(),
            raw_path.to_str().unwrap(),
            meta_path.to_str().unwrap(),
            root.join("invalid-mode.lmo").to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());
    let missing_labels = root.join("missing-labels.json");
    fs::write(&missing_labels, b"{}").unwrap();
    assert!(!Command::new(binary)
        .args([
            "learned-encode",
            "2",
            model_path.to_str().unwrap(),
            raw_path.to_str().unwrap(),
            missing_labels.to_str().unwrap(),
            root.join("missing-labels.lmo").to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn learned_encode_rejects_governed_raw_paths_in_every_operand_before_open() {
    let root = std::env::temp_dir().join(format!(
        "optimum_v2_bgf1_governed_raw_guard_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let model = root.join("synthetic-model.lqw1");
    let input = root.join("synthetic-input.lqraw");
    let metadata = root.join("synthetic-meta.json");
    let packet = root.join("must-not-exist.lmo");
    fs::write(&model, b"not a model").unwrap();
    fs::write(&input, b"not an LQR1").unwrap();
    fs::write(&metadata, b"not metadata").unwrap();

    for (operand, index) in [("MODEL", 2), ("INPUT", 3), ("META_JSON", 4), ("OUTPUT", 5)] {
        let governed_path = Path::new("/mnt/4tb/LamQuant/outputs/optimum-v2-development-v2-2k/raw")
            .join(format!(
                "must-not-be-opened-{}-{operand}.lqraw",
                std::process::id()
            ));
        let mut args = vec![
            "learned-encode".to_owned(),
            "2".to_owned(),
            model.to_string_lossy().into_owned(),
            input.to_string_lossy().into_owned(),
            metadata.to_string_lossy().into_owned(),
            packet.to_string_lossy().into_owned(),
        ];
        args[index] = governed_path.to_string_lossy().into_owned();
        let output = Command::new(env!("CARGO_BIN_EXE_optimum-v2-codec"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{operand} unexpectedly succeeded");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("governed construction raw root"),
            "unexpected {operand} stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!packet.exists());
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn learned_decoder_fails_closed_on_model_structure_and_canonicality_faults() {
    let pack = learned_model_pack();
    let codec = Bgf1LearnedCodec::from_lqw1(&pack).unwrap();
    let (_, signal) = fixture();
    let identities = fixture().0;
    let packet = codec
        .encode_window(&signal, &identities, 256_000, 16, Bgf1LearnedMode::Flow)
        .unwrap();

    for end in 0..packet.len() {
        assert!(codec.decode_window(&packet[..end]).is_err(), "prefix {end}");
    }
    let mut trailing = packet.clone();
    trailing.push(0);
    assert!(codec.decode_window(&trailing).is_err());

    let identity_length = u32::from_le_bytes(packet[63..67].try_into().unwrap()) as usize;
    let graph_start = 87 + identity_length;
    let mut graph_fault = packet.clone();
    graph_fault[graph_start] ^= 1;
    recrc(&mut graph_fault);
    assert!(codec.decode_window(&graph_fault).is_err());

    let graph_length = u32::from_le_bytes(packet[67..71].try_into().unwrap()) as usize;
    let directory_start = graph_start + graph_length;
    let mut event_fault = packet.clone();
    event_fault[directory_start + 20..directory_start + 24].copy_from_slice(&0u32.to_le_bytes());
    recrc(&mut event_fault);
    assert!(codec.decode_window(&event_fault).is_err());

    let mut flags_fault = packet.clone();
    flags_fault[12] = 0;
    recrc(&mut flags_fault);
    assert!(codec.decode_window(&flags_fault).is_err());

    let mut wrong_tensors = ModelPack::decode(&pack).unwrap().tensors;
    for tensor in &mut wrong_tensors {
        if tensor.dtype == TensorDtype::I8 {
            tensor.data.fill(0);
        }
    }
    let wrong_model = ModelPack::encode(&wrong_tensors).unwrap();
    let wrong_codec = Bgf1LearnedCodec::from_lqw1(&wrong_model).unwrap();
    assert!(wrong_codec.decode_window(&packet).is_err());
}
