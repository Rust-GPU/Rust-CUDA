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
