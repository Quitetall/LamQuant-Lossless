use lamquant_lml_optimum_v2::bgf1_model_pack::{
    Bgf1ModelPack, BGF1_EXPECTED_PACK_BYTES, BGF1_LEARNED_PARAMETER_COUNT, BGF1_MODEL_ID,
};
use lamquant_lml_optimum_v2::model_pack::{ModelPack, ModelTensor, TensorDtype, LQW_MAGIC};
use sha2::{Digest, Sha256};

fn tensors() -> Vec<ModelTensor> {
    vec![
        ModelTensor {
            name: "z.weight".into(),
            dtype: TensorDtype::I8,
            shape: vec![2, 2],
            scale_numerator: 1,
            scale_shift: 7,
            data: vec![1, 2, 3, 4],
        },
        ModelTensor {
            name: "a.bias".into(),
            dtype: TensorDtype::I16,
            shape: vec![2],
            scale_numerator: 3,
            scale_shift: 9,
            data: vec![5, 0, 6, 0],
        },
    ]
}

fn bgf1_zero_tensors() -> Vec<ModelTensor> {
    let descriptor = [239_i32, 1, 1, 7_424]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    vec![
        ModelTensor {
            name: "bgf1.descriptor".into(),
            dtype: TensorDtype::I32,
            shape: vec![4],
            scale_numerator: 1,
            scale_shift: 0,
            data: descriptor,
        },
        ModelTensor {
            name: "coupling.predict_second".into(),
            dtype: TensorDtype::I8,
            shape: vec![2, 256],
            scale_numerator: 1,
            scale_shift: 6,
            data: vec![0; 512],
        },
        ModelTensor {
            name: "coupling.update_first".into(),
            dtype: TensorDtype::I8,
            shape: vec![2, 256],
            scale_numerator: 1,
            scale_shift: 6,
            data: vec![0; 512],
        },
        ModelTensor {
            name: "entropy.exponent_logits".into(),
            dtype: TensorDtype::I8,
            shape: vec![16, 16],
            scale_numerator: 1,
            scale_shift: 0,
            data: vec![0; 256],
        },
        ModelTensor {
            name: "entropy.mantissa_logits".into(),
            dtype: TensorDtype::I8,
            shape: vec![16, 16],
            scale_numerator: 1,
            scale_shift: 0,
            data: vec![0; 256],
        },
        ModelTensor {
            name: "entropy.sign_logits".into(),
            dtype: TensorDtype::I8,
            shape: vec![256],
            scale_numerator: 1,
            scale_shift: 0,
            data: vec![0; 256],
        },
        ModelTensor {
            name: "entropy.token_magnitude_bias".into(),
            dtype: TensorDtype::I8,
            shape: vec![256],
            scale_numerator: 1,
            scale_shift: 0,
            data: vec![0; 256],
        },
        ModelTensor {
            name: "prior.graph".into(),
            dtype: TensorDtype::I8,
            shape: vec![256, 4],
            scale_numerator: 1,
            scale_shift: 6,
            data: vec![0; 1_024],
        },
        ModelTensor {
            name: "prior.scale_bias".into(),
            dtype: TensorDtype::I8,
            shape: vec![256],
            scale_numerator: 1,
            scale_shift: 4,
            data: vec![0; 256],
        },
        ModelTensor {
            name: "prior.temporal".into(),
            dtype: TensorDtype::I8,
            shape: vec![256, 16],
            scale_numerator: 1,
            scale_shift: 6,
            data: vec![0; 4_096],
        },
    ]
}

fn bgf1_sentinel_tensors() -> Vec<ModelTensor> {
    let mut tensors = bgf1_zero_tensors();
    for tensor in &mut tensors {
        match tensor.name.as_str() {
            "coupling.predict_second" => {
                tensor.data[0] = (-128_i8) as u8;
                tensor.data[511] = 127;
            }
            "coupling.update_first" => {
                tensor.data[1] = 11;
                tensor.data[510] = (-12_i8) as u8;
            }
            "entropy.exponent_logits" => {
                tensor.data[0] = (-124_i8) as u8;
                tensor.data[255] = 123;
            }
            "entropy.mantissa_logits" => {
                tensor.data[1] = (-123_i8) as u8;
                tensor.data[254] = 122;
            }
            "entropy.sign_logits" => {
                tensor.data[0] = (-121_i8) as u8;
                tensor.data[255] = 120;
            }
            "entropy.token_magnitude_bias" => {
                tensor.data[0] = (-122_i8) as u8;
                tensor.data[255] = 121;
            }
            "prior.graph" => {
                tensor.data[0] = (-126_i8) as u8;
                tensor.data[1_023] = 125;
            }
            "prior.scale_bias" => {
                tensor.data[0] = (-125_i8) as u8;
                tensor.data[255] = 124;
            }
            "prior.temporal" => {
                tensor.data[0] = (-127_i8) as u8;
                tensor.data[18] = 17;
                tensor.data[4_095] = 126;
            }
            "bgf1.descriptor" => {}
            name => panic!("unexpected BGF1 tensor {name}"),
        }
    }
    tensors
}

#[test]
fn lqw1_round_trip_preserves_canonical_directory() {
    let encoded = ModelPack::encode(&tensors()).unwrap();
    assert_eq!(&encoded[..4], LQW_MAGIC);
    let decoded = ModelPack::decode(&encoded).unwrap();
    assert_eq!(decoded.tensors[0].name, "a.bias");
    assert_eq!(decoded.tensors[1].name, "z.weight");
    assert_eq!(ModelPack::encode(&decoded.tensors).unwrap(), encoded);
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&encoded)),
        [
            0xd5, 0x29, 0xb3, 0x1f, 0x3b, 0xb7, 0xdc, 0xf7, 0xd3, 0xab, 0xf4, 0x0b, 0xef, 0x16,
            0x4e, 0xc5, 0xec, 0x24, 0xd0, 0x0a, 0xb5, 0xc6, 0x5c, 0x11, 0xcd, 0x35, 0xe1, 0xc6,
            0xbb, 0xa9, 0x0b, 0x45,
        ]
    );
}

#[test]
fn lqw1_rejects_duplicate_shape_and_digest_faults() {
    let mut duplicate = tensors();
    duplicate[1].name = duplicate[0].name.clone();
    assert!(ModelPack::encode(&duplicate).is_err());

    let mut bad_shape = tensors();
    bad_shape[0].shape = vec![3, 2];
    assert!(ModelPack::encode(&bad_shape).is_err());

    let mut noncanonical_scale = tensors();
    noncanonical_scale[0].scale_numerator = 2;
    noncanonical_scale[0].scale_shift = 2;
    assert!(ModelPack::encode(&noncanonical_scale).is_err());

    let mut corrupt = ModelPack::encode(&tensors()).unwrap();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(ModelPack::decode(&corrupt)
        .unwrap_err()
        .to_string()
        .contains("SHA-256"));
}

#[test]
fn lqw1_bgf1_zero_pack_matches_the_python_wire_golden() {
    let encoded = ModelPack::encode(&bgf1_zero_tensors()).unwrap();
    assert_eq!(encoded.len(), BGF1_EXPECTED_PACK_BYTES);
    assert_eq!(u32::from_le_bytes(encoded[8..12].try_into().unwrap()), 457);
    assert_eq!(
        u32::from_le_bytes(encoded[12..16].try_into().unwrap()),
        7_440
    );
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&encoded)),
        [
            0xaf, 0x53, 0x5b, 0xa3, 0x74, 0x34, 0x5c, 0x34, 0x57, 0xc3, 0x23, 0xd6, 0x79, 0x29,
            0x19, 0x9e, 0xa2, 0x4e, 0x9d, 0x7c, 0x05, 0xec, 0xf9, 0xdd, 0xab, 0x10, 0x7b, 0xb1,
            0xfd, 0xb1, 0x15, 0x32,
        ]
    );
    let decoded = Bgf1ModelPack::decode(&encoded).unwrap();
    assert_eq!(decoded.model_id, BGF1_MODEL_ID);
    assert_eq!(
        decoded.learned_parameter_count,
        BGF1_LEARNED_PARAMETER_COUNT
    );
    assert_eq!(decoded.tensors.len(), 10);
    assert_eq!(decoded.tensors[0].name, "bgf1.descriptor");
    assert_eq!(ModelPack::encode(&decoded.tensors).unwrap(), encoded);
}

#[test]
fn lqw1_bgf1_profile_rejects_rehashed_descriptor_and_tensor_substitution() {
    let mut wrong_descriptor = bgf1_zero_tensors();
    wrong_descriptor[0].data[..4].copy_from_slice(&(BGF1_MODEL_ID - 1).to_le_bytes());
    let wrong_descriptor = ModelPack::encode(&wrong_descriptor).unwrap();
    assert!(ModelPack::decode(&wrong_descriptor).is_ok());
    assert!(Bgf1ModelPack::decode(&wrong_descriptor).is_err());

    let mut wrong_name = bgf1_zero_tensors();
    wrong_name[8].name = "prior.scale_biaz".into();
    let wrong_name = ModelPack::encode(&wrong_name).unwrap();
    assert!(Bgf1ModelPack::decode(&wrong_name).is_err());

    let mut wrong_scale = bgf1_zero_tensors();
    wrong_scale[8].scale_shift = 6;
    let wrong_scale = ModelPack::encode(&wrong_scale).unwrap();
    assert!(Bgf1ModelPack::decode(&wrong_scale).is_err());
}

#[test]
fn lqw1_bgf1_asymmetric_pack_matches_python_signed_layout_golden() {
    let encoded = ModelPack::encode(&bgf1_sentinel_tensors()).unwrap();
    assert_eq!(encoded.len(), BGF1_EXPECTED_PACK_BYTES);
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&encoded)),
        [
            0xeb, 0xb9, 0x24, 0x7f, 0x68, 0x67, 0x81, 0x0b, 0x6a, 0xdb, 0xc9, 0x0e, 0x50, 0x29,
            0x31, 0x09, 0x51, 0xf2, 0x3e, 0xa7, 0x3f, 0x77, 0x95, 0xf4, 0x6a, 0x16, 0x4e, 0x05,
            0x5b, 0xf1, 0x1f, 0x51,
        ]
    );
    assert!(Bgf1ModelPack::decode(&encoded).is_ok());
}
