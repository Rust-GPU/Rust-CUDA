use crate::sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]

pub enum TensorFormat {
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
    /// This tensor format specifies that the data is laid out in the following order: batch size,
    /// feature maps, rows, columns. However, each element of the tensor is a vector of multiple
    /// feature maps.
    NchwVectC,
}

impl From<TensorFormat> for sys::cudnnTensorFormat_t {
    fn from(tensor_format: TensorFormat) -> Self {
        match tensor_format {
            TensorFormat::Nchw => sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            TensorFormat::Nhwc => sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
            TensorFormat::NchwVectC => sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW_VECT_C,
        }
    }
}