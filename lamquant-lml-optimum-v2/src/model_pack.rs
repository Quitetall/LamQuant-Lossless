//! Canonical frozen integer model pack for Optimum-v2 learned modes.

use sha2::{Digest, Sha256};

use crate::OptimumV2Error;

pub const LQW_MAGIC: &[u8; 4] = b"LQW1";
pub const LQW_VERSION: u8 = 1;
const HEADER_LEN: usize = 48;
const DIGEST_OFFSET: usize = 16;
const FIXED_ENTRY_LEN: usize = 20;
const MAX_TENSORS: usize = 4096;
const MAX_RANK: usize = 8;
const MAX_PACK_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorDtype {
    I8 = 1,
    I16 = 2,
    I32 = 3,
}

impl TensorDtype {
    fn width(self) -> usize {
        match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
        }
    }

    fn parse(value: u8) -> Result<Self, OptimumV2Error> {
        match value {
            1 => Ok(Self::I8),
            2 => Ok(Self::I16),
            3 => Ok(Self::I32),
            _ => Err(OptimumV2Error::InvalidPacket(
                "LQW1 tensor dtype is unsupported".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTensor {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<u32>,
    pub scale_numerator: i32,
    pub scale_shift: u8,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPack {
    pub tensors: Vec<ModelTensor>,
    pub sha256: [u8; 32],
}

impl ModelPack {
    pub fn encode(tensors: &[ModelTensor]) -> Result<Vec<u8>, OptimumV2Error> {
        if tensors.is_empty() || tensors.len() > MAX_TENSORS {
            return Err(OptimumV2Error::InvalidInput(
                "LQW1 tensor count is outside bounds".into(),
            ));
        }
        let mut tensors = tensors.to_vec();
        tensors.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
        for pair in tensors.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(OptimumV2Error::InvalidInput(
                    "LQW1 tensor names must be unique".into(),
                ));
            }
        }

        let mut directory = Vec::new();
        let mut payload = Vec::new();
        for tensor in &tensors {
            validate_tensor(tensor, OptimumV2Error::InvalidInput)?;
            let name = tensor.name.as_bytes();
            let offset = u32::try_from(payload.len()).map_err(|_| {
                OptimumV2Error::InvalidInput("LQW1 payload offset exceeds u32".into())
            })?;
            let length = u32::try_from(tensor.data.len()).map_err(|_| {
                OptimumV2Error::InvalidInput("LQW1 tensor length exceeds u32".into())
            })?;
            directory.extend_from_slice(
                &u16::try_from(name.len())
                    .map_err(|_| {
                        OptimumV2Error::InvalidInput("LQW1 tensor name is too long".into())
                    })?
                    .to_le_bytes(),
            );
            directory.push(tensor.dtype as u8);
            directory.push(tensor.shape.len() as u8);
            directory.extend_from_slice(&tensor.scale_numerator.to_le_bytes());
            directory.push(tensor.scale_shift);
            directory.extend_from_slice(&[0u8; 3]);
            directory.extend_from_slice(&offset.to_le_bytes());
            directory.extend_from_slice(&length.to_le_bytes());
            directory.extend_from_slice(name);
            for &dimension in &tensor.shape {
                directory.extend_from_slice(&dimension.to_le_bytes());
            }
            payload.extend_from_slice(&tensor.data);
        }
        let total = HEADER_LEN
            .checked_add(directory.len())
            .and_then(|value| value.checked_add(payload.len()))
            .ok_or_else(|| OptimumV2Error::InvalidInput("LQW1 pack length overflow".into()))?;
        if total > MAX_PACK_BYTES {
            return Err(OptimumV2Error::InvalidInput(
                "LQW1 pack exceeds 64 MiB".into(),
            ));
        }
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(LQW_MAGIC);
        out.push(LQW_VERSION);
        out.push(0);
        out.extend_from_slice(&(tensors.len() as u16).to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(directory.len())
                .map_err(|_| OptimumV2Error::InvalidInput("LQW1 directory exceeds u32".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &u32::try_from(payload.len())
                .map_err(|_| OptimumV2Error::InvalidInput("LQW1 payload exceeds u32".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(&[0u8; 32]);
        out.extend_from_slice(&directory);
        out.extend_from_slice(&payload);
        let digest: [u8; 32] = Sha256::digest(&out).into();
        out[DIGEST_OFFSET..DIGEST_OFFSET + 32].copy_from_slice(&digest);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, OptimumV2Error> {
        if bytes.len() < HEADER_LEN || bytes.len() > MAX_PACK_BYTES {
            return Err(OptimumV2Error::InvalidPacket(
                "LQW1 length is outside bounds".into(),
            ));
        }
        if &bytes[..4] != LQW_MAGIC || bytes[4] != LQW_VERSION || bytes[5] != 0 {
            return Err(OptimumV2Error::InvalidPacket(
                "LQW1 magic, version, or flags are invalid".into(),
            ));
        }
        let tensor_count = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
        if tensor_count == 0 || tensor_count > MAX_TENSORS {
            return Err(OptimumV2Error::InvalidPacket(
                "LQW1 tensor count is outside bounds".into(),
            ));
        }
        let directory_len = read_u32(bytes, 8)? as usize;
        let payload_len = read_u32(bytes, 12)? as usize;
        let expected_len = HEADER_LEN
            .checked_add(directory_len)
            .and_then(|value| value.checked_add(payload_len))
            .ok_or_else(|| OptimumV2Error::InvalidPacket("LQW1 length overflow".into()))?;
        if expected_len != bytes.len() {
            return Err(OptimumV2Error::InvalidPacket(
                "LQW1 section lengths do not match pack".into(),
            ));
        }
        let expected_digest: [u8; 32] =
            bytes[DIGEST_OFFSET..DIGEST_OFFSET + 32].try_into().unwrap();
        let mut hasher = Sha256::new();
        hasher.update(&bytes[..DIGEST_OFFSET]);
        hasher.update([0u8; 32]);
        hasher.update(&bytes[DIGEST_OFFSET + 32..]);
        let actual_digest: [u8; 32] = hasher.finalize().into();
        if actual_digest != expected_digest {
            return Err(OptimumV2Error::Integrity("LQW1 SHA-256 mismatch".into()));
        }

        let directory_end = HEADER_LEN
            .checked_add(directory_len)
            .ok_or_else(|| OptimumV2Error::InvalidPacket("LQW1 directory overflow".into()))?;
        let mut cursor = HEADER_LEN;
        let mut previous_name: Option<Vec<u8>> = None;
        let mut expected_offset = 0usize;
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let fixed_end = cursor.checked_add(FIXED_ENTRY_LEN).ok_or_else(|| {
                OptimumV2Error::InvalidPacket("LQW1 directory cursor overflow".into())
            })?;
            if fixed_end > directory_end {
                return Err(OptimumV2Error::InvalidPacket(
                    "LQW1 tensor directory is truncated".into(),
                ));
            }
            let name_len = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
            let dtype = TensorDtype::parse(bytes[cursor + 2])?;
            let rank = bytes[cursor + 3] as usize;
            let scale_numerator =
                i32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap());
            let scale_shift = bytes[cursor + 8];
            if bytes[cursor + 9..cursor + 12] != [0u8; 3] {
                return Err(OptimumV2Error::InvalidPacket(
                    "LQW1 tensor reserved bytes are nonzero".into(),
                ));
            }
            let offset = read_u32(bytes, cursor + 12)? as usize;
            let length = read_u32(bytes, cursor + 16)? as usize;
            cursor = fixed_end;
            let variable_len = name_len
                .checked_add(rank.checked_mul(4).ok_or_else(|| {
                    OptimumV2Error::InvalidPacket("LQW1 rank length overflow".into())
                })?)
                .ok_or_else(|| OptimumV2Error::InvalidPacket("LQW1 entry overflow".into()))?;
            let variable_end = cursor.checked_add(variable_len).ok_or_else(|| {
                OptimumV2Error::InvalidPacket("LQW1 variable entry overflow".into())
            })?;
            if rank == 0 || rank > MAX_RANK || variable_end > directory_end {
                return Err(OptimumV2Error::InvalidPacket(
                    "LQW1 tensor name or shape is invalid".into(),
                ));
            }
            let name_end = cursor
                .checked_add(name_len)
                .ok_or_else(|| OptimumV2Error::InvalidPacket("LQW1 tensor name overflow".into()))?;
            let name_bytes = &bytes[cursor..name_end];
            if name_bytes.is_empty()
                || previous_name
                    .as_deref()
                    .is_some_and(|previous| previous >= name_bytes)
            {
                return Err(OptimumV2Error::InvalidPacket(
                    "LQW1 tensor names are not canonical".into(),
                ));
            }
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| OptimumV2Error::InvalidPacket("LQW1 tensor name is not UTF-8".into()))?
                .to_owned();
            cursor = name_end;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(read_u32(bytes, cursor)?);
                cursor = cursor.checked_add(4).ok_or_else(|| {
                    OptimumV2Error::InvalidPacket("LQW1 shape cursor overflow".into())
                })?;
            }
            let payload_end = offset.checked_add(length).ok_or_else(|| {
                OptimumV2Error::InvalidPacket("LQW1 tensor payload range overflow".into())
            })?;
            if offset != expected_offset || payload_end > payload_len {
                return Err(OptimumV2Error::InvalidPacket(
                    "LQW1 tensor payloads overlap or have gaps".into(),
                ));
            }
            let payload_start = directory_end.checked_add(offset).ok_or_else(|| {
                OptimumV2Error::InvalidPacket("LQW1 payload start overflow".into())
            })?;
            let payload_absolute_end = payload_start
                .checked_add(length)
                .ok_or_else(|| OptimumV2Error::InvalidPacket("LQW1 payload end overflow".into()))?;
            let tensor = ModelTensor {
                name,
                dtype,
                shape,
                scale_numerator,
                scale_shift,
                data: bytes[payload_start..payload_absolute_end].to_vec(),
            };
            validate_tensor(&tensor, OptimumV2Error::InvalidPacket)?;
            previous_name = Some(name_bytes.to_vec());
            expected_offset = expected_offset.checked_add(length).ok_or_else(|| {
                OptimumV2Error::InvalidPacket("LQW1 expected offset overflow".into())
            })?;
            tensors.push(tensor);
        }
        if cursor != directory_end || expected_offset != payload_len {
            return Err(OptimumV2Error::InvalidPacket(
                "LQW1 directory or payload has trailing bytes".into(),
            ));
        }
        Ok(Self {
            tensors,
            sha256: expected_digest,
        })
    }
}

fn validate_tensor(
    tensor: &ModelTensor,
    error: fn(String) -> OptimumV2Error,
) -> Result<(), OptimumV2Error> {
    if tensor.name.is_empty()
        || !tensor
            .name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        || tensor.shape.is_empty()
        || tensor.shape.len() > MAX_RANK
        || tensor.scale_shift > 31
        || tensor.scale_numerator == 0
        || (tensor.scale_shift > 0 && tensor.scale_numerator % 2 == 0)
    {
        return Err(error("LQW1 tensor metadata is invalid".into()));
    }
    let elements = tensor.shape.iter().try_fold(1usize, |product, &dimension| {
        if dimension == 0 {
            None
        } else {
            product.checked_mul(dimension as usize)
        }
    });
    let expected = elements.and_then(|count| count.checked_mul(tensor.dtype.width()));
    if expected != Some(tensor.data.len()) {
        return Err(error("LQW1 tensor shape does not match data length".into()));
    }
    Ok(())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, OptimumV2Error> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| OptimumV2Error::InvalidPacket("LQW1 offset overflow".into()))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| OptimumV2Error::InvalidPacket("LQW1 u32 is truncated".into()))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}
