use crate::sys;

/// Addition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Add;

/// Multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mul;

/// Minimum comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Min;

/// Maximum comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Max;

/// Square root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sqrt;

/// Negation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Not;

/// A tensor core operation.
pub trait OpTensorOp {
    fn into_raw() -> sys::cudnnOpTensorOp_t;
}

impl OpTensorOp for Add {
    fn into_raw() -> sys::cudnnOpTensorOp_t {
        sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD
    }
}

impl OpTensorOp for Mul {
    fn into_raw() -> sys::cudnnOpTensorOp_t {
        sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL
    }
}

impl OpTensorOp for Min {
    fn into_raw() -> sys::cudnnOpTensorOp_t {
        sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MIN
    }
}

impl OpTensorOp for Max {
    fn into_raw() -> sys::cudnnOpTensorOp_t {
        sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MAX
    }
}

impl OpTensorOp for Sqrt {
    fn into_raw() -> sys::cudnnOpTensorOp_t {
        sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_SQRT
    }
}

impl OpTensorOp for Not {
    fn into_raw() -> sys::cudnnOpTensorOp_t {
        sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_NOT
    }
}

/// A unary tensor core operation.
pub trait UnaryOp: OpTensorOp {}

/// A binary tensor core operation.
pub trait BinaryOp: OpTensorOp {}

impl BinaryOp for Add {}
impl BinaryOp for Mul {}
impl BinaryOp for Min {}
impl BinaryOp for Max {}
impl UnaryOp for Sqrt {}
impl UnaryOp for Not {}
