use lamquant_model_pack::{
    ModelPack, ModelPackView, ModelTensor, PackErrorKind, TensorDtype, LQW_MAGIC,
};
use sha2::{Digest, Sha256};

fn tensors() -> Vec<ModelTensor> {
    vec![
        ModelTensor {
            name: "alpha".into(),
            dtype: TensorDtype::I8,
            shape: vec![2],
            scale_numerator: 1,
            scale_shift: 0,
            data: vec![0x7f, 0x80],
        },
        ModelTensor {
            name: "beta".into(),
            dtype: TensorDtype::I16,
            shape: vec![1, 2],
            scale_numerator: 3,
            scale_shift: 2,
            data: vec![1, 0, 0xff, 0xff],
        },
    ]
}

#[test]
fn canonical_pack_round_trips_through_borrowed_view() {
    let encoded = ModelPack::encode(&tensors()).unwrap();
    assert_eq!(&encoded[..4], LQW_MAGIC);

    let view = ModelPackView::decode(&encoded).unwrap();
    assert_eq!(view.tensor_count(), 2);
    assert_eq!(
        view.sha256(),
        [
            0x5c, 0x0e, 0xac, 0xca, 0xa8, 0xf1, 0x22, 0xd2, 0x25, 0xa0, 0xeb, 0xe2, 0x9e, 0x78,
            0xf2, 0x60, 0x63, 0x09, 0xb0, 0xc0, 0xd4, 0xe5, 0x13, 0x28, 0x3d, 0x82, 0xa4, 0x30,
            0x51, 0x92, 0xd7, 0x41,
        ]
    );

    let alpha = view.get("alpha").unwrap().unwrap();
    assert_eq!(alpha.shape().collect::<Vec<_>>(), [2]);
    assert_eq!(alpha.data(), [0x7f, 0x80]);
    let encoded_start = encoded.as_ptr() as usize;
    let encoded_end = encoded_start + encoded.len();
    let data_start = alpha.data().as_ptr() as usize;
    assert!((encoded_start..encoded_end).contains(&data_start));

    let owned = view.to_owned();
    assert_eq!(owned.tensors, tensors());
    assert_eq!(ModelPack::encode(&owned.tensors).unwrap(), encoded);
}

#[test]
fn corruption_and_noncanonical_directory_fail_closed() {
    let encoded = ModelPack::encode(&tensors()).unwrap();

    let mut corrupt = encoded.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert_eq!(
        ModelPackView::decode(&corrupt).unwrap_err().kind(),
        PackErrorKind::Integrity
    );

    let mut noncanonical = encoded;
    noncanonical[48 + 9] = 1;
    noncanonical[16..48].fill(0);
    let digest: [u8; 32] = Sha256::digest(&noncanonical).into();
    noncanonical[16..48].copy_from_slice(&digest);
    assert_eq!(
        ModelPackView::decode(&noncanonical).unwrap_err().kind(),
        PackErrorKind::InvalidPacket
    );
}
