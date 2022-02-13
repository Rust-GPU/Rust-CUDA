use crate::{
    data_type::DataType,
    error::{CudnnError, IntoResult},
    sys, TensorFormat,
};
use std::{
    marker::PhantomData,
    mem::{self, MaybeUninit},
};

pub struct TensorDescriptorBuilder<'a, T>
where
    T: DataType,
{
    shape: &'a [i32],
    data_type: PhantomData<T>,
}

impl<'a, T> TensorDescriptorBuilder<'a, T>
where
    T: DataType,
{
    /// Creates a tensor descriptor builder with the given shape.
    ///
    /// # Arguments
    ///
    /// `shape` - slice containing the size of the tensor for every dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudnn::TensorDescriptorBuilder;
    ///
    /// let builder = TensorDescriptorBuilder::<f32>::new(&[5, 5, 10, 25]);
    /// ```
    pub fn new(shape: &'a [i32]) -> Self {
        Self {
            shape,
            data_type: PhantomData,
        }
    }

    /// Configures the strides for this tensor descriptor builder.
    ///
    /// # Arguments
    ///
    /// `strides` - strides for the tensor descriptor.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudnn::TensorDescriptorBuilder;
    ///
    /// let builder = TensorDescriptorBuilder::<f32>::new(&[5, 5, 10, 25]).strides(&[1250, 250, 25, 1]);
    /// ```
    pub fn strides(self, strides: &'a [i32]) -> TensorDescriptorBuilderStrides<'a, T> {
        TensorDescriptorBuilderStrides {
            builder: self,
            strides,
        }
    }

    /// Configures the format for this tensor descriptor builder.
    ///
    /// # Arguments
    ///
    /// `format` - format for the tensor descriptor.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudnn::{TensorFormat, TensorDescriptorBuilder};
    ///
    /// let builder = TensorDescriptorBuilder::<f32>::new(&[5, 5, 10, 25]).format(TensorFormat::Nchw);
    /// ```
    pub fn format(self, format: TensorFormat) -> TensorDescriptorBuilderFormat<'a, T> {
        TensorDescriptorBuilderFormat {
            builder: self,
            format,
        }
    }
}

pub struct TensorDescriptorBuilderStrides<'a, T>
where
    T: DataType,
{
    builder: TensorDescriptorBuilder<'a, T>,
    strides: &'a [i32],
}

impl<'a, T> TensorDescriptorBuilderStrides<'a, T>
where
    T: DataType,
{
    /// Creates a tensor descriptor from the provided configuration.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{TensorDescriptorBuilder, TensorFormat};
    ///
    /// let builder = TensorDescriptorBuilder::<f32>::new(&[5, 5, 10, 25])
    ///     .strides(&[1250, 250, 25, 1])
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if at least one of the elements of the shape slice was negative or zero,
    /// the dimension was smaller than 3 or larger than `CUDNN_DIM_MAX`, or the total size of the
    /// tensor descriptor exceeds the maximum limit of 2 Giga-elements.
    pub fn build(self) -> Result<TensorDescriptor<T>, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        let shape = self.builder.shape;
        let data_type = self.builder.data_type;
        let strides = self.strides;
        let ndims = shape.len();

        assert_eq!(
            ndims,
            strides.len(),
            "shape and strides length do not match."
        );

        unsafe {
            sys::cudnnCreateTensorDescriptor(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            sys::cudnnSetTensorNdDescriptor(
                raw,
                T::into_raw(),
                ndims as i32,
                shape.as_ptr(),
                strides.as_ptr(),
            )
            .into_result()?;

            Ok(TensorDescriptor { raw, data_type })
        }
    }
}

pub struct TensorDescriptorBuilderFormat<'a, T>
where
    T: DataType,
{
    builder: TensorDescriptorBuilder<'a, T>,
    format: TensorFormat,
}

impl<'a, T> TensorDescriptorBuilderFormat<'a, T>
where
    T: DataType,
{
    /// Creates a tensor descriptor from the provided configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{TensorDescriptorBuilder, TensorFormat};
    ///
    /// let builder = TensorDescriptorBuilder::<f32>::new(&[5, 5, 10, 25])
    ///     .format(TensorFormat::Nchw)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if at least one of the elements of the shape slice was negative or zero,
    /// the dimension was smaller than 3 or larger than `CUDNN_DIM_MAX`, or the total size of the
    /// tensor descriptor exceeds the maximum limit of 2 Giga-elements.
    pub fn build(self) -> Result<TensorDescriptor<T>, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        let shape = self.builder.shape;
        let data_type = self.builder.data_type;
        let format = self.format;
        let ndims = shape.len();

        unsafe {
            sys::cudnnCreateTensorDescriptor(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            sys::cudnnSetTensorNdDescriptorEx(
                raw,
                format.into(),
                T::into_raw(),
                ndims as i32,
                shape.as_ptr(),
            )
            .into_result()?;

            Ok(TensorDescriptor { raw, data_type })
        }
    }
}

/// A generic description of an n-dimensional dataset.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct TensorDescriptor<T>
where
    T: DataType,
{
    pub(crate) raw: sys::cudnnTensorDescriptor_t,
    data_type: PhantomData<T>,
}

impl<T> Drop for TensorDescriptor<T>
where
    T: DataType,
{
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyTensorDescriptor(self.raw);
        }
    }
}
