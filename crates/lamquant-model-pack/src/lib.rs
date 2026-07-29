#![no_std]

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use sha2::{Digest, Sha256};

pub const LQW_MAGIC: &[u8; 4] = b"LQW1";
pub const LQW_VERSION: u8 = 1;
pub const MAX_PACK_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_TENSORS: usize = 4096;
pub const MAX_RANK: usize = 8;

const HEADER_LEN: usize = 48;
const DIGEST_OFFSET: usize = 16;
const FIXED_ENTRY_LEN: usize = 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PackErrorKind {
    InvalidInput,
    InvalidPacket,
    Integrity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackError {
    kind: PackErrorKind,
    message: &'static str,
}

impl PackError {
    const fn input(message: &'static str) -> Self {
        Self {
            kind: PackErrorKind::InvalidInput,
            message,
        }
    }

    const fn packet(message: &'static str) -> Self {
        Self {
            kind: PackErrorKind::InvalidPacket,
            message,
        }
    }

    const fn integrity(message: &'static str) -> Self {
        Self {
            kind: PackErrorKind::Integrity,
            message,
        }
    }

    pub const fn kind(self) -> PackErrorKind {
        self.kind
    }

    pub const fn message(self) -> &'static str {
        self.message
    }
}

impl fmt::Display for PackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TensorDtype {
    I8 = 1,
    I16 = 2,
    I32 = 3,
}

impl TensorDtype {
    pub const fn width(self) -> usize {
        match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
        }
    }

    fn parse(value: u8) -> Result<Self, PackError> {
        match value {
            1 => Ok(Self::I8),
            2 => Ok(Self::I16),
            3 => Ok(Self::I32),
            _ => Err(PackError::packet("LQW1 tensor dtype is unsupported")),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelTensor {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<u32>,
    pub scale_numerator: i32,
    pub scale_shift: u8,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelPack {
    pub tensors: Vec<ModelTensor>,
    pub sha256: [u8; 32],
}

impl ModelPack {
    pub fn encode(tensors: &[ModelTensor]) -> Result<Vec<u8>, PackError> {
        if tensors.is_empty() || tensors.len() > MAX_TENSORS {
            return Err(PackError::input("LQW1 tensor count is outside bounds"));
        }
        let mut tensors = tensors.iter().collect::<Vec<_>>();
        tensors.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
        if tensors.windows(2).any(|pair| pair[0].name == pair[1].name) {
            return Err(PackError::input("LQW1 tensor names must be unique"));
        }

        let mut directory_len = 0_usize;
        let mut payload_len = 0_usize;
        for tensor in &tensors {
            validate_owned_tensor(tensor)?;
            u16::try_from(tensor.name.len())
                .map_err(|_| PackError::input("LQW1 tensor name is too long"))?;
            u8::try_from(tensor.shape.len())
                .map_err(|_| PackError::input("LQW1 tensor rank exceeds u8"))?;
            u32::try_from(tensor.data.len())
                .map_err(|_| PackError::input("LQW1 tensor length exceeds u32"))?;
            directory_len = directory_len
                .checked_add(FIXED_ENTRY_LEN)
                .and_then(|value| value.checked_add(tensor.name.len()))
                .and_then(|value| value.checked_add(tensor.shape.len().checked_mul(4)?))
                .ok_or_else(|| PackError::input("LQW1 directory length overflow"))?;
            payload_len = payload_len
                .checked_add(tensor.data.len())
                .ok_or_else(|| PackError::input("LQW1 payload length overflow"))?;
        }

        let total = HEADER_LEN
            .checked_add(directory_len)
            .and_then(|value| value.checked_add(payload_len))
            .ok_or_else(|| PackError::input("LQW1 pack length overflow"))?;
        if total > MAX_PACK_BYTES {
            return Err(PackError::input("LQW1 pack exceeds 64 MiB"));
        }
        let tensor_count = u16::try_from(tensors.len())
            .map_err(|_| PackError::input("LQW1 tensor count exceeds u16"))?;
        let directory_len_wire = u32::try_from(directory_len)
            .map_err(|_| PackError::input("LQW1 directory exceeds u32"))?;
        let payload_len_wire =
            u32::try_from(payload_len).map_err(|_| PackError::input("LQW1 payload exceeds u32"))?;

        let mut output = Vec::with_capacity(total);
        output.extend_from_slice(LQW_MAGIC);
        output.push(LQW_VERSION);
        output.push(0);
        output.extend_from_slice(&tensor_count.to_le_bytes());
        output.extend_from_slice(&directory_len_wire.to_le_bytes());
        output.extend_from_slice(&payload_len_wire.to_le_bytes());
        output.extend_from_slice(&[0_u8; 32]);
        let mut payload_offset = 0_usize;
        for tensor in &tensors {
            let name = tensor.name.as_bytes();
            let name_len = u16::try_from(name.len())
                .map_err(|_| PackError::input("LQW1 tensor name is too long"))?;
            let rank = u8::try_from(tensor.shape.len())
                .map_err(|_| PackError::input("LQW1 tensor rank exceeds u8"))?;
            let offset = u32::try_from(payload_offset)
                .map_err(|_| PackError::input("LQW1 payload offset exceeds u32"))?;
            let length = u32::try_from(tensor.data.len())
                .map_err(|_| PackError::input("LQW1 tensor length exceeds u32"))?;
            output.extend_from_slice(&name_len.to_le_bytes());
            output.push(tensor.dtype as u8);
            output.push(rank);
            output.extend_from_slice(&tensor.scale_numerator.to_le_bytes());
            output.push(tensor.scale_shift);
            output.extend_from_slice(&[0_u8; 3]);
            output.extend_from_slice(&offset.to_le_bytes());
            output.extend_from_slice(&length.to_le_bytes());
            output.extend_from_slice(name);
            for dimension in &tensor.shape {
                output.extend_from_slice(&dimension.to_le_bytes());
            }
            payload_offset = payload_offset
                .checked_add(tensor.data.len())
                .ok_or_else(|| PackError::input("LQW1 payload offset overflow"))?;
        }
        debug_assert_eq!(output.len(), HEADER_LEN + directory_len);
        for tensor in tensors {
            output.extend_from_slice(&tensor.data);
        }
        debug_assert_eq!(output.len(), total);
        let digest: [u8; 32] = Sha256::digest(&output).into();
        output[DIGEST_OFFSET..DIGEST_OFFSET + 32].copy_from_slice(&digest);
        Ok(output)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, PackError> {
        Ok(ModelPackView::decode(bytes)?.to_owned())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ModelPackView<'a> {
    bytes: &'a [u8],
    tensor_count: u16,
    directory_end: usize,
    sha256: [u8; 32],
}

impl<'a> ModelPackView<'a> {
    pub fn decode(bytes: &'a [u8]) -> Result<Self, PackError> {
        if bytes.len() < HEADER_LEN || bytes.len() > MAX_PACK_BYTES {
            return Err(PackError::packet("LQW1 length is outside bounds"));
        }
        if bytes.get(..4) != Some(LQW_MAGIC) || bytes[4] != LQW_VERSION || bytes[5] != 0 {
            return Err(PackError::packet(
                "LQW1 magic, version, or flags are invalid",
            ));
        }
        let tensor_count = u16::from_le_bytes([bytes[6], bytes[7]]);
        if tensor_count == 0 || usize::from(tensor_count) > MAX_TENSORS {
            return Err(PackError::packet("LQW1 tensor count is outside bounds"));
        }
        let directory_len = usize::try_from(read_u32(bytes, 8)?)
            .map_err(|_| PackError::packet("LQW1 directory length exceeds usize"))?;
        let payload_len = usize::try_from(read_u32(bytes, 12)?)
            .map_err(|_| PackError::packet("LQW1 payload length exceeds usize"))?;
        let expected_len = HEADER_LEN
            .checked_add(directory_len)
            .and_then(|value| value.checked_add(payload_len))
            .ok_or_else(|| PackError::packet("LQW1 length overflow"))?;
        if expected_len != bytes.len() {
            return Err(PackError::packet("LQW1 section lengths do not match pack"));
        }

        let expected_digest: [u8; 32] = bytes[DIGEST_OFFSET..DIGEST_OFFSET + 32]
            .try_into()
            .map_err(|_| PackError::packet("LQW1 digest is truncated"))?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes[..DIGEST_OFFSET]);
        hasher.update([0_u8; 32]);
        hasher.update(&bytes[DIGEST_OFFSET + 32..]);
        let actual_digest: [u8; 32] = hasher.finalize().into();
        if actual_digest != expected_digest {
            return Err(PackError::integrity("LQW1 SHA-256 mismatch"));
        }

        let directory_end = HEADER_LEN
            .checked_add(directory_len)
            .ok_or_else(|| PackError::packet("LQW1 directory overflow"))?;
        validate_directory(bytes, usize::from(tensor_count), directory_end, payload_len)?;
        Ok(Self {
            bytes,
            tensor_count,
            directory_end,
            sha256: expected_digest,
        })
    }

    pub const fn tensor_count(self) -> u16 {
        self.tensor_count
    }

    pub const fn sha256(self) -> [u8; 32] {
        self.sha256
    }

    pub fn tensors(self) -> TensorIter<'a> {
        TensorIter {
            bytes: self.bytes,
            cursor: HEADER_LEN,
            directory_end: self.directory_end,
            remaining: self.tensor_count,
        }
    }

    pub fn get(self, name: &str) -> Result<Option<TensorView<'a>>, PackError> {
        if !valid_name(name.as_bytes()) {
            return Err(PackError::input("LQW1 lookup name is invalid"));
        }
        for tensor in self.tensors() {
            match tensor.name().as_bytes().cmp(name.as_bytes()) {
                core::cmp::Ordering::Less => {}
                core::cmp::Ordering::Equal => return Ok(Some(tensor)),
                core::cmp::Ordering::Greater => return Ok(None),
            }
        }
        Ok(None)
    }

    pub fn to_owned(self) -> ModelPack {
        ModelPack {
            tensors: self.tensors().map(TensorView::to_owned).collect(),
            sha256: self.sha256,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    name: &'a str,
    dtype: TensorDtype,
    shape_bytes: &'a [u8],
    scale_numerator: i32,
    scale_shift: u8,
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub const fn name(self) -> &'a str {
        self.name
    }

    pub const fn dtype(self) -> TensorDtype {
        self.dtype
    }

    pub fn shape(self) -> ShapeIter<'a> {
        ShapeIter {
            bytes: self.shape_bytes,
        }
    }

    pub const fn scale_numerator(self) -> i32 {
        self.scale_numerator
    }

    pub const fn scale_shift(self) -> u8 {
        self.scale_shift
    }

    pub const fn data(self) -> &'a [u8] {
        self.data
    }

    fn to_owned(self) -> ModelTensor {
        ModelTensor {
            name: self.name.to_string(),
            dtype: self.dtype,
            shape: self.shape().collect(),
            scale_numerator: self.scale_numerator,
            scale_shift: self.scale_shift,
            data: self.data.to_vec(),
        }
    }
}

pub struct ShapeIter<'a> {
    bytes: &'a [u8],
}

impl Iterator for ShapeIter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.bytes.get(..4)?;
        self.bytes = &self.bytes[4..];
        Some(u32::from_le_bytes(value.try_into().ok()?))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.bytes.len() / 4;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ShapeIter<'_> {}

pub struct TensorIter<'a> {
    bytes: &'a [u8],
    cursor: usize,
    directory_end: usize,
    remaining: u16,
}

impl<'a> Iterator for TensorIter<'a> {
    type Item = TensorView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let cursor = self.cursor;
        let name_len = usize::from(u16::from_le_bytes(
            self.bytes.get(cursor..cursor + 2)?.try_into().ok()?,
        ));
        let dtype = TensorDtype::parse(*self.bytes.get(cursor + 2)?).ok()?;
        let rank = usize::from(*self.bytes.get(cursor + 3)?);
        let scale_numerator =
            i32::from_le_bytes(self.bytes.get(cursor + 4..cursor + 8)?.try_into().ok()?);
        let scale_shift = *self.bytes.get(cursor + 8)?;
        let offset = usize::try_from(read_u32(self.bytes, cursor + 12).ok()?).ok()?;
        let length = usize::try_from(read_u32(self.bytes, cursor + 16).ok()?).ok()?;
        let name_start = cursor + FIXED_ENTRY_LEN;
        let name_end = name_start.checked_add(name_len)?;
        let shape_end = name_end.checked_add(rank.checked_mul(4)?)?;
        let name = core::str::from_utf8(self.bytes.get(name_start..name_end)?).ok()?;
        let shape_bytes = self.bytes.get(name_end..shape_end)?;
        let payload_start = self.directory_end.checked_add(offset)?;
        let payload_end = payload_start.checked_add(length)?;
        let data = self.bytes.get(payload_start..payload_end)?;
        self.cursor = shape_end;
        self.remaining -= 1;
        Some(TensorView {
            name,
            dtype,
            shape_bytes,
            scale_numerator,
            scale_shift,
            data,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::from(self.remaining);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for TensorIter<'_> {}

fn validate_directory(
    bytes: &[u8],
    tensor_count: usize,
    directory_end: usize,
    payload_len: usize,
) -> Result<(), PackError> {
    let mut cursor = HEADER_LEN;
    let mut previous_name: Option<&[u8]> = None;
    let mut expected_offset = 0_usize;
    for _ in 0..tensor_count {
        let fixed_end = cursor
            .checked_add(FIXED_ENTRY_LEN)
            .ok_or_else(|| PackError::packet("LQW1 directory cursor overflow"))?;
        if fixed_end > directory_end {
            return Err(PackError::packet("LQW1 tensor directory is truncated"));
        }
        let name_len = usize::from(u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]));
        let dtype = TensorDtype::parse(bytes[cursor + 2])?;
        let rank = usize::from(bytes[cursor + 3]);
        let scale_numerator = i32::from_le_bytes(
            bytes[cursor + 4..cursor + 8]
                .try_into()
                .map_err(|_| PackError::packet("LQW1 scale numerator is truncated"))?,
        );
        let scale_shift = bytes[cursor + 8];
        if bytes[cursor + 9..cursor + 12] != [0_u8; 3] {
            return Err(PackError::packet("LQW1 tensor reserved bytes are nonzero"));
        }
        let offset = usize::try_from(read_u32(bytes, cursor + 12)?)
            .map_err(|_| PackError::packet("LQW1 tensor offset exceeds usize"))?;
        let length = usize::try_from(read_u32(bytes, cursor + 16)?)
            .map_err(|_| PackError::packet("LQW1 tensor length exceeds usize"))?;
        cursor = fixed_end;
        let shape_bytes_len = rank
            .checked_mul(4)
            .ok_or_else(|| PackError::packet("LQW1 rank length overflow"))?;
        let variable_len = name_len
            .checked_add(shape_bytes_len)
            .ok_or_else(|| PackError::packet("LQW1 entry overflow"))?;
        let variable_end = cursor
            .checked_add(variable_len)
            .ok_or_else(|| PackError::packet("LQW1 variable entry overflow"))?;
        if rank == 0 || rank > MAX_RANK || variable_end > directory_end {
            return Err(PackError::packet("LQW1 tensor name or shape is invalid"));
        }
        let name_end = cursor
            .checked_add(name_len)
            .ok_or_else(|| PackError::packet("LQW1 tensor name overflow"))?;
        let name = &bytes[cursor..name_end];
        if !valid_name(name) || previous_name.is_some_and(|previous| previous >= name) {
            return Err(PackError::packet("LQW1 tensor names are not canonical"));
        }
        cursor = name_end;
        let mut elements = 1_usize;
        for _ in 0..rank {
            let dimension = usize::try_from(read_u32(bytes, cursor)?)
                .map_err(|_| PackError::packet("LQW1 dimension exceeds usize"))?;
            if dimension == 0 {
                return Err(PackError::packet("LQW1 tensor dimension must be nonzero"));
            }
            elements = elements
                .checked_mul(dimension)
                .ok_or_else(|| PackError::packet("LQW1 tensor shape product overflow"))?;
            cursor = cursor
                .checked_add(4)
                .ok_or_else(|| PackError::packet("LQW1 shape cursor overflow"))?;
        }
        if scale_numerator == 0 || scale_shift > 31 || (scale_shift > 0 && scale_numerator % 2 == 0)
        {
            return Err(PackError::packet("LQW1 tensor scale is invalid"));
        }
        let expected_length = elements
            .checked_mul(dtype.width())
            .ok_or_else(|| PackError::packet("LQW1 tensor byte length overflow"))?;
        if expected_length != length {
            return Err(PackError::packet(
                "LQW1 tensor shape does not match data length",
            ));
        }
        let payload_end = offset
            .checked_add(length)
            .ok_or_else(|| PackError::packet("LQW1 tensor payload range overflow"))?;
        if offset != expected_offset || payload_end > payload_len {
            return Err(PackError::packet(
                "LQW1 tensor payloads overlap or have gaps",
            ));
        }
        previous_name = Some(name);
        expected_offset = payload_end;
    }
    if cursor != directory_end || expected_offset != payload_len {
        return Err(PackError::packet(
            "LQW1 directory or payload has trailing bytes",
        ));
    }
    Ok(())
}

fn validate_owned_tensor(tensor: &ModelTensor) -> Result<(), PackError> {
    if !valid_name(tensor.name.as_bytes())
        || tensor.shape.is_empty()
        || tensor.shape.len() > MAX_RANK
        || tensor.scale_shift > 31
        || tensor.scale_numerator == 0
        || (tensor.scale_shift > 0 && tensor.scale_numerator % 2 == 0)
    {
        return Err(PackError::input("LQW1 tensor metadata is invalid"));
    }
    let elements = tensor.shape.iter().try_fold(1_usize, |product, dimension| {
        if *dimension == 0 {
            None
        } else {
            usize::try_from(*dimension)
                .ok()
                .and_then(|dimension| product.checked_mul(dimension))
        }
    });
    let expected = elements.and_then(|count| count.checked_mul(tensor.dtype.width()));
    if expected != Some(tensor.data.len()) {
        return Err(PackError::input(
            "LQW1 tensor shape does not match data length",
        ));
    }
    Ok(())
}

fn valid_name(name: &[u8]) -> bool {
    !name.is_empty()
        && name
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(*byte, b'_' | b'-' | b'.'))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, PackError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| PackError::packet("LQW1 offset overflow"))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| PackError::packet("LQW1 u32 is truncated"))?;
    Ok(u32::from_le_bytes(value.try_into().map_err(|_| {
        PackError::packet("LQW1 u32 has wrong width")
    })?))
}
