use crate::{
    data_type::*,
    error::{CudnnError, IntoResult},
    nan_propagation::NanPropagation,
    op_tensor::*,
    sys,
};
use std::{marker::PhantomData, mem::MaybeUninit};

/// The description of a Tensor Core operation.
///
///
/// As specified in the cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters),
/// admissible types for scaling parameters are `f32` and `f64` for `f32` and `f64` tensors
/// respectively.
pub struct OpTensorDescriptor<T: DataType, Op: OpTensorOp> {
    pub(crate) raw: sys::cudnnOpTensorDescriptor_t,
    comp_type: PhantomData<T>,
    op: Op,
}

impl<T, Op> OpTensorDescriptor<T, Op>
where
    T: DataType,
    Op: OpTensorOp,
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
    /// use cudnn::{Add, NanPropagation, OpTensorDescriptor};
    ///
    /// let op = Add;
    /// let nan_policy = NanPropagation::PropagateNaN;
    ///
    /// // We are stating that the computation must be done in f32.
    /// let desc = OpTensorDescriptor::<f32, _>::new(op, nan_policy)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(op: Op, nan_opt: NanPropagation) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreateOpTensorDescriptor(raw.as_mut_ptr()).into_result()?;
            let mut raw = raw.assume_init();

            sys::cudnnSetOpTensorDescriptor(raw, Op::into_raw(), T::into_raw(), nan_opt.into())
                .into_result()?;

            Ok(Self {
                raw,
                comp_type: PhantomData,
                op,
            })
        }
    }
}

impl<T: DataType, Op: OpTensorOp> Drop for OpTensorDescriptor<T, Op> {
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
pub trait SupportedOp<AType, BType, CType>
where
    Self: DataType,
    AType: DataType,
    BType: DataType,
    CType: DataType,
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
