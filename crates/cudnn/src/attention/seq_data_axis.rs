use crate::sys;

/// Describes and indexes active dimensions in the `SeqDataDescriptor` `dim` field. This enum is
/// also used in the `axis` argument of the `SeqDataDescriptor` constructor to  define the layout
/// of the sequence data buffer in memory.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSeqDataAxis_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SeqDataAxis {
    /// Identifies the time (sequence length) dimension or specifies the time in the data layout.
    TimeDim,
    /// Identifies the batch dimension or specifies the batch in the data layout.
    BatchDim,
    /// Identifies the beam dimension or specifies the beam in the data layout.
    BeamDim,
    /// Identifies the vect (vector) dimension or specifies the vector in the data layout.
    VectDim,
}

impl From<SeqDataAxis> for sys::cudnnSeqDataAxis_t {
    fn from(axis: SeqDataAxis) -> Self {
        match axis {
            SeqDataAxis::TimeDim => sys::cudnnSeqDataAxis_t::CUDNN_SEQDATA_TIME_DIM,
            SeqDataAxis::BatchDim => sys::cudnnSeqDataAxis_t::CUDNN_SEQDATA_BATCH_DIM,
            SeqDataAxis::BeamDim => sys::cudnnSeqDataAxis_t::CUDNN_SEQDATA_BEAM_DIM,
            SeqDataAxis::VectDim => sys::cudnnSeqDataAxis_t::CUDNN_SEQDATA_VECT_DIM,
        }
    }
}

impl<T> std::ops::Index<SeqDataAxis> for [T; 4] {
    type Output = T;

    fn index(&self, index: SeqDataAxis) -> &Self::Output {
        let raw: sys::cudnnSeqDataAxis_t = index.into();
        self.index(raw as usize)
    }
}

impl<T> std::ops::IndexMut<SeqDataAxis> for [T; 4] {
    fn index_mut(&mut self, index: SeqDataAxis) -> &mut Self::Output {
        let raw: sys::cudnnSeqDataAxis_t = index.into();
        self.index_mut(raw as usize)
    }
}
