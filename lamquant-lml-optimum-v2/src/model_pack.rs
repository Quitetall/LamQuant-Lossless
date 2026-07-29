//! Owned compatibility facade over the shared canonical LQW1 model-pack primitive.

use crate::OptimumV2Error;

pub use lamquant_model_pack::{
    ModelTensor, TensorDtype, LQW_MAGIC, LQW_VERSION, MAX_PACK_BYTES, MAX_RANK, MAX_TENSORS,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelPack {
    pub tensors: Vec<ModelTensor>,
    pub sha256: [u8; 32],
}

impl ModelPack {
    pub fn encode(tensors: &[ModelTensor]) -> Result<Vec<u8>, OptimumV2Error> {
        lamquant_model_pack::ModelPack::encode(tensors).map_err(Into::into)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, OptimumV2Error> {
        let pack = lamquant_model_pack::ModelPack::decode(bytes)?;
        Ok(Self {
            tensors: pack.tensors,
            sha256: pack.sha256,
        })
    }
}
