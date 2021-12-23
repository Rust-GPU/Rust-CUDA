use crate::{data_type, private, sys};

/// This tensor format specifies that the data is laid out in the following order: batch size,
/// feature maps, rows, columns. The strides are implicitly defined in such a way that the data
/// are contiguous in memory with no padding between images, feature maps, rows, and columns;
/// the columns are the inner dimension and the images are the outermost dimension.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NCHW;

/// This tensor format specifies that the data is laid out in the following order: batch size,
/// rows, columns, feature maps. The strides are implicitly defined in such a way that the data
/// are contiguous in memory with no padding between images, rows, columns, and feature maps; the
/// feature maps are the inner dimension and the images are the outermost dimension.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NHWC;

/// This tensor format specifies that the data is laid out in the following order: batch size,
/// feature maps, rows, columns. However, each element of the tensor is a vector of multiple
/// feature maps.
///
///  This format is only supported with tensor data types [`i8`] and [`u8`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NCHWVectC8x4;

/// This tensor format specifies that the data is laid out in the following order: batch size,
/// feature maps, rows, columns. However, each element of the tensor is a vector of multiple
/// feature maps.
///
///  This format is only supported with tensor data types [`i8`] and [`u8`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NCHWVectC8x32;

/// Specifies the raw type for the given tensor memory format.
pub trait TensorFormat: private::Sealed {
    fn into_raw() -> sys::cudnnTensorFormat_t;
}

macro_rules! impl_cudnn_tensor_format {
    ($safe_type:ident, $raw_type:ident) => {
        impl private::Sealed for $safe_type {}

        impl TensorFormat for $safe_type {
            fn into_raw() -> sys::cudnnTensorFormat_t {
                sys::cudnnTensorFormat_t::$raw_type
            }
        }
    };
}

impl_cudnn_tensor_format!(NCHW, CUDNN_TENSOR_NCHW);
impl_cudnn_tensor_format!(NHWC, CUDNN_TENSOR_NHWC);
impl_cudnn_tensor_format!(NCHWVectC8x4, CUDNN_TENSOR_NCHW_VECT_C);
impl_cudnn_tensor_format!(NCHWVectC8x32, CUDNN_TENSOR_NCHW_VECT_C);

/// Specifies the type supported by each tensor format.
pub trait SupportedType<T>
where
    Self: TensorFormat,
    T: data_type::DataType,
{
    fn data_type() -> sys::cudnnDataType_t;
}

macro_rules! impl_cudnn_supported_type {
    ($tensor_format:ident, $safe_type:ident, $raw_type:ident) => {
        impl SupportedType<$safe_type> for $tensor_format {
            fn data_type() -> sys::cudnnDataType_t {
                sys::cudnnDataType_t::$raw_type
            }
        }
    };
}

/// Data types supported by the NCHW  tensor format.
impl_cudnn_supported_type!(NCHW, f32, CUDNN_DATA_FLOAT);
impl_cudnn_supported_type!(NCHW, f64, CUDNN_DATA_DOUBLE);
impl_cudnn_supported_type!(NCHW, i8, CUDNN_DATA_INT8);
impl_cudnn_supported_type!(NCHW, u8, CUDNN_DATA_UINT8);
impl_cudnn_supported_type!(NCHW, i32, CUDNN_DATA_INT32);
impl_cudnn_supported_type!(NCHW, i64, CUDNN_DATA_INT64);

/// Data types supported by the NHWC tensor format.
impl_cudnn_supported_type!(NHWC, f32, CUDNN_DATA_FLOAT);
impl_cudnn_supported_type!(NHWC, f64, CUDNN_DATA_DOUBLE);
impl_cudnn_supported_type!(NHWC, i8, CUDNN_DATA_INT8);
impl_cudnn_supported_type!(NHWC, u8, CUDNN_DATA_UINT8);
impl_cudnn_supported_type!(NHWC, i32, CUDNN_DATA_INT32);
impl_cudnn_supported_type!(NHWC, i64, CUDNN_DATA_INT64);

/// Data types supported by the NCHWVectC8x4 tensor format.
impl_cudnn_supported_type!(NCHWVectC8x4, i8, CUDNN_DATA_INT8x4);
impl_cudnn_supported_type!(NCHWVectC8x4, u8, CUDNN_DATA_UINT8x4);

/// Data types supported by the NCHWVectC8x32 tensor format.
impl_cudnn_supported_type!(NCHWVectC8x32, i8, CUDNN_DATA_INT8x32);
