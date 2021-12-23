use crate::sys;

/// Enum indicating a Tensor Core operation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpTensorOp {
    /// Addition.
    Add,
    /// Multiplication.
    Mul,
    /// Minimum comparison.
    Min,
    /// Maximum comparison.
    Max,
    /// Square root.
    Sqrt,
    /// Negation.
    Not,
}

impl From<OpTensorOp> for sys::cudnnOpTensorOp_t {
    fn from(op: OpTensorOp) -> sys::cudnnOpTensorOp_t {
        match op {
            OpTensorOp::Add => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
            OpTensorOp::Mul => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL,
            OpTensorOp::Min => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MIN,
            OpTensorOp::Max => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MAX,
            OpTensorOp::Sqrt => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_SQRT,
            OpTensorOp::Not => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_NOT,
        }
    }
}

impl From<sys::cudnnOpTensorOp_t> for OpTensorOp {
    fn from(raw: sys::cudnnOpTensorOp_t) -> Self {
        match raw {
            sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD => OpTensorOp::Add,
            sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL => OpTensorOp::Mul,
            sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MIN => OpTensorOp::Min,
            sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MAX => OpTensorOp::Max,
            sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_SQRT => OpTensorOp::Sqrt,
            sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_NOT => OpTensorOp::Not,
        }
    }
}
