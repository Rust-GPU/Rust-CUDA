use crate::{private, DataType};

/// Supported data types configurations for convolution operations.
pub trait SupportedConv<X, W, Y>: private::Sealed + DataType
where
    X: DataType,
    W: DataType,
    Y: DataType,
{
}

impl SupportedConv<f32, f32, f32> for f32 {}
impl SupportedConv<f64, f64, f64> for f64 {}
impl SupportedConv<i8, i8, i8> for i32 {}
impl SupportedConv<i8, i8, f32> for i32 {}
impl SupportedConv<u8, i8, i8> for i32 {}
impl SupportedConv<u8, i8, f32> for i32 {}
impl SupportedConv<i32, i32, i32> for i32 {}
