use crate::sys;

/// Tensor reduction operation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Add,
    Mul,
    Min,
    Max,
    Amax,
    Avg,
    Norm1,
    Norm2,
    MulNoZeros,
}

impl From<ReduceOp> for sys::cudnnReduceTensorOp_t {
    fn from(op: ReduceOp) -> Self {
        match op {
            ReduceOp::Add => Self::CUDNN_REDUCE_TENSOR_ADD,
            ReduceOp::Mul => Self::CUDNN_REDUCE_TENSOR_MUL,
            ReduceOp::Min => Self::CUDNN_REDUCE_TENSOR_MIN,
            ReduceOp::Max => Self::CUDNN_REDUCE_TENSOR_MAX,
            ReduceOp::Amax => Self::CUDNN_REDUCE_TENSOR_AMAX,
            ReduceOp::Avg => Self::CUDNN_REDUCE_TENSOR_AVG,
            ReduceOp::Norm1 => Self::CUDNN_REDUCE_TENSOR_NORM1,
            ReduceOp::Norm2 => Self::CUDNN_REDUCE_TENSOR_NORM2,
            ReduceOp::MulNoZeros => Self::CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS,
        }
    }
}
