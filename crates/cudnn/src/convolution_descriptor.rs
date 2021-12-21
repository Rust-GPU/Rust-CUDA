use crate::{
    convolution_mode::ConvolutionMode,
    data_type::DataType,
    error::{CudnnError, IntoResult},
    math_type::MathType,
    sys,
    tensor_format::TensorFormat,
};

use std::{marker::PhantomData, mem::MaybeUninit};

/// A generic description of an n-dimensional convolution.
///
/// *Do note* that N can be either 2 or 3, respectively for a 2-d or a 3-d convolution.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct ConvolutionDescriptor<T: DataType, const N: usize> {
    pub(crate) raw: sys::cudnnConvolutionDescriptor_t,
    comp_type: PhantomData<T>,
}

impl<T: DataType, const N: usize> ConvolutionDescriptor<T, N> {
    /// Creates a new `ConvolutionDescriptor`.
    ///
    /// # Arguments
    ///
    /// * `padding` -  array of dimension N containing the zero-padding size for each dimension.
    /// For every dimension, the padding represents the number of extra zeros implicitly
    /// concatenated at the start and at the end of every element of that dimension.
    ///
    /// * `stride` -  array of dimension N containing the filter stride for each dimension.
    /// For every dimension, the filter stride represents the number of elements to slide to reach
    /// the next start of the filtering window of the next point.
    ///
    /// * `dilation` - array of dimension N containing the dilation factor for each dimension.
    ///
    /// * `groups` - number of groups to be used in the associated convolution.
    ///
    /// * `mode` - selects between [`Convolution`](ConvolutionMode::Convolution) and
    /// [`CrossCorrelation`](ConvolutionMode::CrossCorrelation).
    ///
    /// * `math_type` - indicates whether or not the use of tensor op is permitted in the library
    /// routines associated with a given convolution descriptor.
    ///
    /// # Examples
    ///
    /// This struct can be used both for 2-d and 3-d convolutions.
    ///
    /// ## 2-d convolution
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, ConvolutionDescriptor, MathType, ConvolutionMode};
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let groups = 1;
    /// let mode = ConvolutionMode::CrossCorrelation;
    /// let math_type = MathType::Default;
    ///
    /// let conv_desc = ConvolutionDescriptor::<f32, 2>::new(padding, stride, dilation, groups, mode, math_type)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## 3-d convolution
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, ConvolutionDescriptor, MathType, ConvolutionMode};
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0, 0];
    /// let stride = [1, 1, 1];
    /// let dilation = [1, 1, 1];
    /// let groups = 1;
    /// let mode = ConvolutionMode::CrossCorrelation;
    /// let math_type = MathType::Default;
    ///
    /// let conv_desc = ConvolutionDescriptor::<f32, 3>::new(padding, stride, dilation, groups, mode, math_type)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// This function returns an error if any element of stride and dilation is negative or 0, if
    /// any element of padding is negative or if N is greater than `CUDNN_DIM_MAX`.
    pub fn new(
        padding: [i32; N],
        stride: [i32; N],
        dilation: [i32; N],
        groups: u32,
        mode: ConvolutionMode,
        math_type: MathType,
    ) -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreateConvolutionDescriptor(raw.as_mut_ptr()).into_result()?;

            let mut conv_desc = Self {
                raw: raw.assume_init(),
                comp_type: PhantomData,
            };

            sys::cudnnSetConvolutionNdDescriptor(
                conv_desc.raw,
                N as i32,
                padding.as_ptr(),
                stride.as_ptr(),
                dilation.as_ptr(),
                mode.into(),
                T::into_raw(),
            )
            .into_result()?;

            conv_desc.set_math_type(math_type)?;

            sys::cudnnSetConvolutionGroupCount(conv_desc.raw, groups as i32).into_result()?;

            Ok(conv_desc)
        }
    }

    /// Sets the `MathType` for this convolution descriptor instance.
    pub fn set_math_type(&mut self, math_type: MathType) -> Result<(), CudnnError> {
        unsafe { sys::cudnnSetConvolutionMathType(self.raw, math_type.into()).into_result() }
    }
}

impl<T: DataType, const N: usize> Drop for ConvolutionDescriptor<T, N> {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyConvolutionDescriptor(self.raw);
        }
    }
}
