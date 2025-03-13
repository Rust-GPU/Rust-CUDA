use crate::sys;

#[non_exhaustive]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ReductionMode {
    Add,
    Amax,
    Avg,
    Max,
    Min,
    Mul,
    MulNoZeros,
    Norm1,
    Norm2,
}

impl From<ReductionMode> for sys::cudnnReduceTensorOp_t {
    fn from(mode: ReductionMode) -> Self {
        match mode {
            ReductionMode::Add => sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
            ReductionMode::Amax => sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_AMAX,
            ReductionMode::Avg => sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_AVG,
            ReductionMode::Max => sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MAX,
            ReductionMode::Min => sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MIN,
            ReductionMode::Mul => sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MUL,
            ReductionMode::MulNoZeros => {
                sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS
            }
            ReductionMode::Norm1 => sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_NORM1,
            ReductionMode::Norm2 => sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_NORM2,
        }
    }
}
