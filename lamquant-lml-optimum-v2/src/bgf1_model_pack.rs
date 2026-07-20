//! Strict BGF1 v1 profile over the generic canonical LQW1 container.

use crate::model_pack::{ModelPack, Tensor, TensorDtype};
use crate::OptimumV2Error;

pub const BGF1_MODEL_ID: i32 = 239;
pub const BGF1_ARCHITECTURE_VERSION: i32 = 1;
pub const BGF1_PACK_PROFILE_VERSION: i32 = 1;
pub const BGF1_LEARNED_PARAMETER_COUNT: usize = 7_424;
pub const BGF1_MAX_LEARNED_PARAMETERS: usize = 8_192;
pub const BGF1_EXPECTED_PACK_BYTES: usize = 7_945;
pub const BGF1_MAX_PACK_BYTES: usize = 12_288;

#[derive(Clone, Copy)]
struct TensorSpec {
    name: &'static str,
    dtype: TensorDtype,
    shape: &'static [u32],
    scale_shift: u8,
}

const PROFILE: [TensorSpec; 10] = [
    TensorSpec {
        name: "bgf1.descriptor",
        dtype: TensorDtype::I32,
        shape: &[4],
        scale_shift: 0,
    },
    TensorSpec {
        name: "coupling.predict_second",
        dtype: TensorDtype::I8,
        shape: &[2, 256],
        scale_shift: 6,
    },
    TensorSpec {
        name: "coupling.update_first",
        dtype: TensorDtype::I8,
        shape: &[2, 256],
        scale_shift: 6,
    },
    TensorSpec {
        name: "entropy.exponent_logits",
        dtype: TensorDtype::I8,
        shape: &[16, 16],
        scale_shift: 0,
    },
    TensorSpec {
        name: "entropy.mantissa_logits",
        dtype: TensorDtype::I8,
        shape: &[16, 16],
        scale_shift: 0,
    },
    TensorSpec {
        name: "entropy.sign_logits",
        dtype: TensorDtype::I8,
        shape: &[256],
        scale_shift: 0,
    },
    TensorSpec {
        name: "entropy.token_magnitude_bias",
        dtype: TensorDtype::I8,
        shape: &[256],
        scale_shift: 0,
    },
    TensorSpec {
        name: "prior.graph",
        dtype: TensorDtype::I8,
        shape: &[256, 4],
        scale_shift: 6,
    },
    TensorSpec {
        name: "prior.scale_bias",
        dtype: TensorDtype::I8,
        shape: &[256],
        scale_shift: 4,
    },
    TensorSpec {
        name: "prior.temporal",
        dtype: TensorDtype::I8,
        shape: &[256, 16],
        scale_shift: 6,
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bgf1ModelPack {
    pub model_id: i32,
    pub learned_parameter_count: usize,
    pub tensors: Vec<Tensor>,
    pub lqw1_sha256: [u8; 32],
}

impl Bgf1ModelPack {
    pub fn decode(bytes: &[u8]) -> Result<Self, OptimumV2Error> {
        if bytes.len() != BGF1_EXPECTED_PACK_BYTES || bytes.len() > BGF1_MAX_PACK_BYTES {
            return Err(OptimumV2Error::InvalidPacket(
                "BGF1 LQW1 pack has the wrong profile length".into(),
            ));
        }
        let pack = ModelPack::decode(bytes)?;
        if pack.tensors.len() != PROFILE.len() {
            return Err(OptimumV2Error::InvalidPacket(
                "BGF1 LQW1 tensor profile is invalid".into(),
            ));
        }
        for (tensor, spec) in pack.tensors.iter().zip(PROFILE) {
            if tensor.name != spec.name
                || tensor.dtype != spec.dtype
                || tensor.shape != spec.shape
                || tensor.scale_numerator != 1
                || tensor.scale_shift != spec.scale_shift
            {
                return Err(OptimumV2Error::InvalidPacket(
                    "BGF1 LQW1 tensor metadata is invalid".into(),
                ));
            }
        }
        let learned_parameter_count = pack
            .tensors
            .iter()
            .filter(|tensor| tensor.dtype == TensorDtype::I8)
            .try_fold(0usize, |count, tensor| count.checked_add(tensor.data.len()))
            .ok_or_else(|| OptimumV2Error::InvalidPacket("BGF1 parameter count overflow".into()))?;
        if learned_parameter_count != BGF1_LEARNED_PARAMETER_COUNT
            || learned_parameter_count > BGF1_MAX_LEARNED_PARAMETERS
        {
            return Err(OptimumV2Error::InvalidPacket(
                "BGF1 learned parameter budget is invalid".into(),
            ));
        }
        let descriptor = &pack.tensors[0].data;
        let fields = [
            i32::from_le_bytes(descriptor[0..4].try_into().unwrap()),
            i32::from_le_bytes(descriptor[4..8].try_into().unwrap()),
            i32::from_le_bytes(descriptor[8..12].try_into().unwrap()),
            i32::from_le_bytes(descriptor[12..16].try_into().unwrap()),
        ];
        if fields
            != [
                BGF1_MODEL_ID,
                BGF1_ARCHITECTURE_VERSION,
                BGF1_PACK_PROFILE_VERSION,
                BGF1_LEARNED_PARAMETER_COUNT as i32,
            ]
        {
            return Err(OptimumV2Error::InvalidPacket(
                "BGF1 LQW1 descriptor is invalid".into(),
            ));
        }
        let canonical = ModelPack::encode(&pack.tensors).map_err(|error| {
            OptimumV2Error::Integrity(format!("BGF1 LQW1 cannot be re-encoded: {error}"))
        })?;
        if canonical != bytes {
            return Err(OptimumV2Error::Integrity(
                "BGF1 LQW1 pack is not byte-canonical".into(),
            ));
        }
        Ok(Self {
            model_id: fields[0],
            learned_parameter_count,
            tensors: pack.tensors,
            lqw1_sha256: pack.sha256,
        })
    }
}
