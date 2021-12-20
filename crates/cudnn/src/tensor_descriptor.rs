use crate::{
    data_type::DataType,
    error::{CudnnError, IntoResult},
    sys,
    tensor_format::{SupportedType, TensorFormat},
};
use std::{
    marker::PhantomData,
    mem::{self, MaybeUninit},
};

/// A generic description of an n-dimensional dataset.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct TensorDescriptor<T, F, const D: usize>
where
    T: DataType,
    F: TensorFormat + SupportedType<T>,
{
    pub(crate) raw: sys::cudnnTensorDescriptor_t,
    data_type: PhantomData<T>,
    format: F,
}

impl<T, F, const D: usize> TensorDescriptor<T, F, D>
where
    T: DataType,
    F: TensorFormat + SupportedType<T>,
{
    /// Creates a generic tensor descriptor with the given memory format.
    ///
    /// # Arguments
    ///
    /// * shape - array containing the size of the tensor for every dimension. The size along
    /// unused dimensions should be set to 1.
    ///
    /// * format - tensor format.
    ///
    /// # Errors
    ///
    /// Returns an error if at least one of the elements of the array shape was negative or zero,
    /// the dimension was smaller than 3 or larger than `CUDNN_DIM_MAX`, or the total size of the
    /// tensor descriptor exceeds the maximum limit of 2 Giga-elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{TensorDescriptor, NCHW};
    ///
    /// let shape = [2, 25, 25];
    /// let format = NCHW;
    ///
    /// let desc = TensorDescriptor::<f32, _, 3>::new(shape, format)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(shape: [i32; D], format: F) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreateTensorDescriptor(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            sys::cudnnSetTensorNdDescriptorEx(
                raw,
                <F as TensorFormat>::into_raw(),
                <F as SupportedType<T>>::data_type(),
                shape.len() as i32,
                shape.as_ptr(),
            )
            .into_result()?;

            Ok(Self {
                raw,
                format,
                data_type: PhantomData,
            })
        }
    }
}

impl<T, F, const D: usize> Drop for TensorDescriptor<T, F, D>
where
    T: DataType,
    F: TensorFormat + SupportedType<T>,
{
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyTensorDescriptor(self.raw);
        }
    }
}
