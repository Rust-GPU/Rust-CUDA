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

impl From<ReductionMode> for cudnn_sys::cudnnReduceTensorOp_t {
    fn from(mode: ReductionMode) -> Self {
        match mode {
            ReductionMode::Add => cudnn_sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
            ReductionMode::Amax => cudnn_sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_AMAX,
            ReductionMode::Avg => cudnn_sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_AVG,
            ReductionMode::Max => cudnn_sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MAX,
            ReductionMode::Min => cudnn_sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MIN,
            ReductionMode::Mul => cudnn_sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MUL,
            ReductionMode::MulNoZeros => {
                cudnn_sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS
            }
            ReductionMode::Norm1 => cudnn_sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_NORM1,
            ReductionMode::Norm2 => cudnn_sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_NORM2,
        }
    }
}
