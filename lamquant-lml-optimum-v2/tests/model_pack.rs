use lamquant_lml_optimum_v2::model_pack::{ModelPack, Tensor, TensorDtype, LQW_MAGIC};
use sha2::{Digest, Sha256};

fn tensors() -> Vec<Tensor> {
    vec![
        Tensor {
            name: "z.weight".into(),
            dtype: TensorDtype::I8,
            shape: vec![2, 2],
            scale_numerator: 1,
            scale_shift: 7,
            data: vec![1, 2, 3, 4],
        },
        Tensor {
            name: "a.bias".into(),
            dtype: TensorDtype::I16,
            shape: vec![2],
            scale_numerator: 3,
            scale_shift: 9,
            data: vec![5, 0, 6, 0],
        },
    ]
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
