use crate::{
    data_type::DataType,
    error::{CudnnError, IntoResult},
    sys,
    tensor::{SupportedType, TensorFormat},
};
use std::{
    marker::PhantomData,
    mem::{self, MaybeUninit},
};

/// A generic description of an n-dimensional filter dataset.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct FilterDescriptor<T, F, const D: usize>
where
    T: DataType,
    F: TensorFormat + SupportedType<T>,
{
    pub(crate) raw: sys::cudnnFilterDescriptor_t,
    data_type: PhantomData<T>,
    format: F,
}

impl<T, F, const D: usize> FilterDescriptor<T, F, D>
where
    T: DataType,
    F: TensorFormat + SupportedType<T>,
{
    /// Creates a generic filter descriptor with the given memory format.
    ///
    /// # Arguments
    ///
    /// * shape - array containing the size of the filter for every dimension.
    ///
    /// * format - tensor format. If set to [`NCHW`](TensorFormat::NCHW), then the layout of the
    /// filter is as follows: for D = 4, a 4D filter descriptor, the filter layout is in the form of
    /// KCRS, i.e. K represents the number of output feature maps, C is the number of input feature
    /// maps, R is the number of rows per filter, S is the number of columns per filter. For N = 3,
    /// a 3D filter descriptor, the number S (number of columns per filter) is omitted. For N = 5
    /// and greater, the layout of the higher dimensions immediately follows RS.
    ///
    /// # Errors
    ///
    /// Returns an error if at least one of the elements of the array shape was negative or zero,
    /// the dimension was smaller than 3 or larger than `CUDNN_DIM_MAX`, or the total size of the
    /// filter descriptor exceeds the maximum limit of 2 Giga-elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{FilterDescriptor, NCHW};
    ///
    /// let shape = [2, 2, 25, 25];
    /// let format = NCHW;
    ///
    /// let desc = FilterDescriptor::<f32, _, 4>::new(shape, format)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(shape: [i32; D], format: F) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreateFilterDescriptor(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            sys::cudnnSetFilterNdDescriptor(
                raw,
                <F as SupportedType<T>>::data_type(),
                <F as TensorFormat>::into_raw(),
                D as i32,
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

impl<T, F, const D: usize> Drop for FilterDescriptor<T, F, D>
where
    T: DataType,
    F: TensorFormat + SupportedType<T>,
{
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyFilterDescriptor(self.raw);
        }
    }
}
