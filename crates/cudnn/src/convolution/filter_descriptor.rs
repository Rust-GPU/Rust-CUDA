use crate::{sys, CudnnError, DataType, IntoResult, ScalarC, TensorFormat, VecType};
use std::{marker::PhantomData, mem::MaybeUninit};

/// A generic description of an n-dimensional filter dataset.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct FilterDescriptor<T>
where
    T: DataType,
{
    pub(crate) raw: sys::cudnnFilterDescriptor_t,
    data_type: PhantomData<T>,
}

impl<T> FilterDescriptor<T>
where
    T: DataType,
{
    /// Creates a generic filter descriptor with the given shape and memory format.
    ///
    /// # Arguments
    ///
    ///  * `shape` - slice containing the size of the filter for every dimension.
    ///  * `format` - tensor format. If set to [`Nchw`](ScalarC::Nchw), then the layout
    ///    of the filter is as follows: for D = 4, a 4D filter descriptor, the filter
    ///    layout is in the form of KCRS, i.e. K represents the number of output feature
    ///    maps, C is the number of input feature maps, R is the number of rows per
    ///    filter, S is the number of columns per filter. For N = 3, a 3D filter
    ///    descriptor, the number S (number of columns per filter) is omitted. For N = 5
    ///    and greater, the layout of the higher dimensions immediately follows RS.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetFilterNdDescriptor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if at least one of the elements of the array shape was negative
    /// or zero, the dimension was smaller than 3 or larger than `CUDNN_DIM_MAX`, or the
    /// total size of the filter descriptor exceeds the maximum limit of 2
    /// Giga-elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{FilterDescriptor, ScalarC};
    ///
    /// let shape = &[2, 2, 25, 25];
    /// let format = ScalarC::Nchw;
    ///
    /// let desc = FilterDescriptor::<f32>::new(shape, format)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(shape: &[i32], format: ScalarC) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();
        let ndims = shape.len();

        unsafe {
            sys::cudnnCreateFilterDescriptor(raw.as_mut_ptr()).into_result()?;

            let raw = raw.assume_init();

            sys::cudnnSetFilterNdDescriptor(
                raw,
                T::into_raw(),
                format.into(),
                ndims as i32,
                shape.as_ptr(),
            )
            .into_result()?;

            Ok(Self {
                raw,
                data_type: PhantomData,
            })
        }
    }

    /// Creates a generic filter descriptor with the given shape and vectorized memory format.
    ///
    /// # Arguments
    ///
    /// `shape` - slice containing the size of the filter for every dimension.
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
    /// use cudnn::{FilterDescriptor, Vec4};
    ///
    /// let shape = &[4, 32, 32, 32];
    ///
    /// let desc = FilterDescriptor::<i8>::new_vectorized::<Vec4>(shape)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_vectorized<V: VecType<T>>(shape: &[i32]) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        let ndims = shape.len();
        let format = TensorFormat::NchwVectC;

        unsafe {
            sys::cudnnCreateFilterDescriptor(raw.as_mut_ptr()).into_result()?;

            let raw = raw.assume_init();

            sys::cudnnSetFilterNdDescriptor(
                raw,
                V::into_raw(),
                format.into(),
                ndims as i32,
                shape.as_ptr(),
            )
            .into_result()?;

            Ok(Self {
                raw,
                data_type: PhantomData,
            })
        }
    }
}

impl<T> Drop for FilterDescriptor<T>
where
    T: DataType,
{
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyFilterDescriptor(self.raw);
        }
    }
}
