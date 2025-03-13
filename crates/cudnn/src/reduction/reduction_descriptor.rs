use std::{marker::PhantomData, mem::MaybeUninit};

use crate::{
    sys, CudnnError, DataType, IndicesType, IntoResult, NanPropagation, ReduceIndices, ReduceOp,
};

/// Descriptor of a tensor reduction operation.
pub struct ReductionDescriptor<T>
where
    T: DataType,
{
    pub(crate) raw: sys::cudnnReduceTensorDescriptor_t,
    comp_type: PhantomData<T>,
}

impl<T> ReductionDescriptor<T>
where
    T: DataType,
{
    /// Creates a new tensor reduction descriptor.
    ///
    /// # Arguments
    ///
    /// * `op` - tensor reduction operation.
    /// * `non_opt` - NaN propagation policy.
    /// * `indices` - whether to compute indices or not.
    /// * `indices_type` - data type of the indices.
    ///
    /// **Do note** that requesting indices for operations other than min and max will
    /// result in an error.
    ///
    /// # Errors
    ///
    /// Returns errors if an unsupported combination of arguments is detected.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{IndicesType, NanPropagation, ReduceIndices, ReduceOp, ReductionDescriptor};
    ///
    /// let op = ReduceOp::Add;
    /// let nan_policy = NanPropagation::PropagateNaN;
    /// let indices = ReduceIndices::None;
    /// let indices_type = None;
    ///
    /// // We are stating that the computation must be done in f32.
    /// let desc = ReductionDescriptor::<f32>::new(op, nan_policy, indices, indices_type)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        op: ReduceOp,
        nan_opt: NanPropagation,
        indices: ReduceIndices,
        indices_type: impl Into<Option<IndicesType>>,
    ) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();
        let indices_type = indices_type.into().unwrap_or(IndicesType::U8);

        unsafe {
            sys::cudnnCreateReduceTensorDescriptor(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            sys::cudnnSetReduceTensorDescriptor(
                raw,
                op.into(),
                T::into_raw(),
                nan_opt.into(),
                indices.into(),
                indices_type.into(),
            )
            .into_result()?;

            Ok(Self {
                raw,
                comp_type: PhantomData,
            })
        }
    }
}

impl<T> Drop for ReductionDescriptor<T>
where
    T: DataType,
{
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyReduceTensorDescriptor(self.raw);
        }
    }
}
