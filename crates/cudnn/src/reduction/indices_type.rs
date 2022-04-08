use crate::sys;

/// Indicates the data type of the indices computed by a reduction operation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndicesType {
    U8,
    U16,
    U32,
    U64,
}

impl From<IndicesType> for sys::cudnnIndicesType_t {
    fn from(mode: IndicesType) -> Self {
        match mode {
            IndicesType::U8 => Self::CUDNN_8BIT_INDICES,
            IndicesType::U16 => Self::CUDNN_16BIT_INDICES,
            IndicesType::U32 => Self::CUDNN_32BIT_INDICES,
            IndicesType::U64 => Self::CUDNN_64BIT_INDICES,
        }
    }
}
