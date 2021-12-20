use crate::{
    data_type::*,
    error::{CudnnError, IntoResult},
    nan_propagation::NanPropagation,
    sys,
};
use std::{marker::PhantomData, mem::MaybeUninit};

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

impl OpTensorOp {
    /// Returns the corresponding raw cuDNNN type.
    pub fn into_raw(self) -> sys::cudnnOpTensorOp_t {
        match self {
            OpTensorOp::Add => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
            OpTensorOp::Mul => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL,
            OpTensorOp::Min => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MIN,
            OpTensorOp::Max => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MAX,
            OpTensorOp::Sqrt => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_SQRT,
            OpTensorOp::Not => sys::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_NOT,
        }
    }
}

/// The description of a Tensor Core operation.
///
///
/// As specified in the cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters),
/// admissible types for scaling parameters are `f32` and `f64` for `f32` and `f64` tensors
/// respectively.
pub struct OpTensorDescriptor<T: DataType> {
    pub(crate) raw: sys::cudnnOpTensorDescriptor_t,
    comp_type: PhantomData<T>,
}

impl<T> OpTensorDescriptor<T>
where
    T: DataType,
{
    /// Creates a tensor point-wise math descriptor.
    ///
    /// # Arguments
    ///
    /// * `op` - the tensor point-wise math operation to be performed.
    ///
    /// * `nan_opt` - a NaN propagation policy.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{OpTensorOp, NanPropagation, OpTensorDescriptor};
    ///
    /// let op = OpTensorOp::Add;
    /// let nan_policy = NanPropagation::PropagateNaN;
    ///
    /// // We are stating that the computation must be done in f32.
    /// let desc = OpTensorDescriptor::<f32>::new(op, nan_policy)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(op: OpTensorOp, nan_opt: NanPropagation) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreateOpTensorDescriptor(raw.as_mut_ptr()).into_result()?;
            let mut raw = raw.assume_init();

            sys::cudnnSetOpTensorDescriptor(raw, op.into_raw(), T::into_raw(), nan_opt.into_raw())
                .into_result()?;

            Ok(Self {
                raw,
                comp_type: PhantomData,
            })
        }
    }
}

impl<T: DataType> Drop for OpTensorDescriptor<T> {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyOpTensorDescriptor(self.raw);
        }
    }
}

/// Supported data type configurations for point-wise tensor core operations.
///
/// | CompType | Atype    | Btype    | Ctype    |
/// |----------|----------|----------|----------|
/// | f32      | f32      | f32      | f32      |
/// | f32      | i8       | i8       | f32      |
/// | f64      | f64      | f64      | f64      |
/// | f32      | i8       | i8       | i8       |
/// | f32      | f32      | f32      | i8       |
pub trait SupportedOp<Atype, Btype, Ctype>
where
    Self: DataType,
    Atype: DataType,
    Btype: DataType,
    Ctype: DataType,
{
}

macro_rules! impl_supported_types {
    ($comp_type:ty, $a_type:ty, $b_type:ty, $c_type:ty) => {
        impl SupportedOp<$a_type, $b_type, $c_type> for $comp_type {}
    };
}

impl_supported_types!(f32, f32, f32, f32);
impl_supported_types!(f32, i8, i8, f32);
impl_supported_types!(f64, f64, f64, f64);
impl_supported_types!(f32, i8, i8, i8);
impl_supported_types!(f32, f32, f32, i8);
