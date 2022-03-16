use crate::sys;

/// A unary tensor core operation.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensorOp_t)
/// may offer additional information about the APi behavior.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Sqrt,
    Not,
}

impl From<UnaryOp> for sys::cudnnOpTensorOp_t {
    fn from(op: UnaryOp) -> Self {
        match op {
            UnaryOp::Sqrt => Self::CUDNN_OP_TENSOR_SQRT,
            UnaryOp::Not => Self::CUDNN_OP_TENSOR_NOT,
        }
    }
}

/// A binary tensor core operation.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensorOp_t)
/// may offer additional information about the APi behavior.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Mul,
    Min,
    Max,
}

impl From<BinaryOp> for sys::cudnnOpTensorOp_t {
    fn from(op: BinaryOp) -> Self {
        match op {
            BinaryOp::Add => Self::CUDNN_OP_TENSOR_ADD,
            BinaryOp::Mul => Self::CUDNN_OP_TENSOR_MUL,
            BinaryOp::Min => Self::CUDNN_OP_TENSOR_MIN,
            BinaryOp::Max => Self::CUDNN_OP_TENSOR_MAX,
        }
    }
}
