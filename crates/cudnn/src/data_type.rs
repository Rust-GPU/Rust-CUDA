use crate::{private, sys};

pub trait DataType: private::Sealed + cust::memory::DeviceCopy {
    /// Returns the corresponding raw cuDNN data type.
    fn into_raw() -> sys::cudnnDataType_t;
}

macro_rules! impl_cudnn_data_type {
    ($safe_type:ident, $raw_type:ident) => {
        impl private::Sealed for $safe_type {}

        impl DataType for $safe_type {
            fn into_raw() -> sys::cudnnDataType_t {
                sys::cudnnDataType_t::$raw_type
            }
        }
    };
}

impl_cudnn_data_type!(f32, CUDNN_DATA_FLOAT);
impl_cudnn_data_type!(f64, CUDNN_DATA_DOUBLE);
impl_cudnn_data_type!(i8, CUDNN_DATA_INT8);
impl_cudnn_data_type!(u8, CUDNN_DATA_UINT8);
impl_cudnn_data_type!(i32, CUDNN_DATA_INT32);
impl_cudnn_data_type!(i64, CUDNN_DATA_INT64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Vec4;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Vec32;

/// Vectorized data type. Vectorization size can be either 4 or 32 elements.
pub trait VecType<T>: private::Sealed
where
    T: DataType,
{
    /// Return the corresponding raw cuDNN data type.
    fn into_raw() -> sys::cudnnDataType_t;
}

impl private::Sealed for Vec4 {}

impl private::Sealed for Vec32 {}

macro_rules! impl_cudnn_vec_type {
    ($type:ident, $safe_type:ident, $raw_type:ident) => {
        impl VecType<$safe_type> for $type {
            fn into_raw() -> sys::cudnnDataType_t {
                sys::cudnnDataType_t::$raw_type
            }
        }
    };
}

impl_cudnn_vec_type!(Vec4, i8, CUDNN_DATA_INT8x4);
impl_cudnn_vec_type!(Vec32, i8, CUDNN_DATA_INT8x32);
impl_cudnn_vec_type!(Vec4, u8, CUDNN_DATA_UINT8x4);

/// Admissible data types for scaling parameters.
pub trait ScalingDataType<T>: DataType + private::Sealed
where
    T: DataType,
{
}

impl ScalingDataType<i8> for f32 {}
impl ScalingDataType<u8> for f32 {}
impl ScalingDataType<i32> for f32 {}
impl ScalingDataType<i64> for f32 {}
impl ScalingDataType<f32> for f32 {}

impl ScalingDataType<f64> for f64 {}
