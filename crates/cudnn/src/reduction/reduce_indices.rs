use crate::sys;

/// Indicates whether a reduction operation should compute indices or not.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceIndices {
    /// Do not compute indices.
    None,
    /// Compute indices. The resulting indices are relative to the dimensions being reduced, and
    /// flattened.
    Flattened,
}

impl From<ReduceIndices> for sys::cudnnReduceTensorIndices_t {
    fn from(mode: ReduceIndices) -> Self {
        match mode {
            ReduceIndices::None => Self::CUDNN_REDUCE_TENSOR_NO_INDICES,
            ReduceIndices::Flattened => Self::CUDNN_REDUCE_TENSOR_FLATTENED_INDICES,
        }
    }
}
