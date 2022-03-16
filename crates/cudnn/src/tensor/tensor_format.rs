use crate::sys;

/// Tensor formats in which each element of the tensor has scalar value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarC {
    /// This tensor format specifies that the data is laid out in the following order: batch size,
    /// feature maps, rows, columns. The strides are implicitly defined in such a way that the data
    /// are contiguous in memory with no padding between images, feature maps, rows, and columns;
    /// the columns are the inner dimension and the images are the outermost dimension.
    Nchw,
    /// This tensor format specifies that the data is laid out in the following order: batch size,
    /// rows, columns, feature maps. The strides are implicitly defined in such a way that the data
    /// are contiguous in memory with no padding between images, rows, columns, and feature maps; the
    /// feature maps are the inner dimension and the images are the outermost dimension.
    Nhwc,
}

impl From<ScalarC> for sys::cudnnTensorFormat_t {
    fn from(tensor_format: ScalarC) -> Self {
        match tensor_format {
            ScalarC::Nchw => sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            ScalarC::Nhwc => sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
        }
    }
}

/// Predefined layouts for tensors.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorFormat_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorFormat {
    /// Scalar valued formats.
    ///
    /// * `Nchw` format.
    ///
    /// * `Nhwc` format.
    ScalarC(ScalarC),
    /// Vector valued formats.
    ///
    /// This tensor format specifies that the data is laid out in the following order: batch size,
    /// feature maps, rows, columns. However, each element of the tensor is a vector of multiple
    /// feature maps.
    NchwVectC,
}

impl From<ScalarC> for TensorFormat {
    fn from(tensor_format: ScalarC) -> Self {
        Self::ScalarC(tensor_format)
    }
}

impl From<TensorFormat> for sys::cudnnTensorFormat_t {
    fn from(tensor_format: TensorFormat) -> Self {
        match tensor_format {
            TensorFormat::ScalarC(fmt) => fmt.into(),
            TensorFormat::NchwVectC => sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW_VECT_C,
        }
    }
}
