mod pooling_descriptor;
mod pooling_mode;

pub use pooling_descriptor::*;
pub use pooling_mode::*;

use crate::{private, sys, CudnnContext, CudnnError, DataType, IntoResult, TensorDescriptor};
use cust::memory::GpuBuffer;

impl CudnnContext {
    /// This function computes the pooling of the input tensor and produces a smaller tensor in
    /// output.
    ///
    /// # Arguments
    ///
    /// * `pooling_desc` - descriptor of the pooling operation.
    ///
    /// * `alpha` - scaling factor for the result.
    ///
    /// * `x_desc` - descriptor for the input tensor.
    ///
    /// * `x` - data for the input tensor.
    ///
    /// * `beta` - scaling factor for the destination tensor.
    ///
    /// * `y_desc` - descriptor for the destination tensor.
    ///
    /// * `y` - data for the destination tensor.
    ///
    /// # Errors
    ///
    /// Returns errors if the batch size or channels dimensions of the two tensor differ or an
    /// invalid combination of arguments is detected.
    pub fn pooling_forward<T, U>(
        &self,
        pooling_desc: &PoolingDescriptor,
        alpha: T,
        x_desc: &TensorDescriptor<U>,
        x: &impl GpuBuffer<U>,
        beta: T,
        y_desc: &TensorDescriptor<U>,
        y: &mut impl GpuBuffer<U>,
    ) -> Result<(), CudnnError>
    where
        T: SupportedPoolFwd<U>,
        U: DataType,
    {
        let alpha_ptr = &alpha as *const T as *const _;
        let x_ptr = x.as_device_ptr().as_ptr() as *const _;

        let beta_ptr = &beta as *const T as *const _;
        let y_ptr = y.as_device_ptr().as_mut_ptr() as *mut _;

        unsafe {
            sys::cudnnPoolingForward(
                self.raw,
                pooling_desc.raw,
                alpha_ptr,
                x_desc.raw,
                x_ptr,
                beta_ptr,
                y_desc.raw,
                y_ptr,
            )
            .into_result()
        }
    }

    /// Computes the gradient of a pooling operation.
    ///
    /// # Arguments
    ///
    /// * `pooling_desc` - descriptor of the pooling operation.
    ///
    /// * `alpha` - scaling factor for the result.
    ///
    /// * `y_desc` - tensor descriptor for the output map.
    ///
    /// * `y` - data for the output map.
    ///
    /// * `dy_desc` - tensor descriptor for the differential of the output map.
    ///
    /// * `dy` - data foe the differential of the output map.
    ///
    /// * `x_desc` - tensor descriptor for the dropout input.
    ///
    /// * `x` - data for the dropout input.
    ///
    /// * `beta` - scaling factor for the destination tensor.
    ///
    /// * `dx_desc` - tensor descriptor for the input differential.
    ///
    /// * `dx` - data for the input differential.
    ///
    /// # Errors
    ///
    /// Returns errors if the dimensions or the strides of `y` and `dy` tensors differ or if the
    /// dimensions or the strides of `x` and `dx` tensors differ or if an unsupported combination
    /// of arguments is detected.
    pub fn pooling_backward<T, U>(
        &self,
        pooling_desc: &PoolingDescriptor,
        alpha: T,
        y_desc: &TensorDescriptor<U>,
        y: &impl GpuBuffer<U>,
        dy_desc: &TensorDescriptor<U>,
        dy: &impl GpuBuffer<U>,
        x_desc: &TensorDescriptor<U>,
        x: &impl GpuBuffer<U>,
        beta: T,
        dx_desc: &TensorDescriptor<U>,
        dx: &mut impl GpuBuffer<U>,
    ) -> Result<(), CudnnError>
    where
        T: SupportedPoolBwd<U>,
        U: DataType,
    {
        let alpha_ptr = &alpha as *const T as *const _;

        let y_ptr = y.as_device_ptr().as_ptr() as *const _;
        let dy_ptr = dy.as_device_ptr().as_ptr() as *const _;
        let x_ptr = x.as_device_ptr().as_ptr() as *const _;

        let beta_ptr = &beta as *const T as *const _;

        let dx_ptr = dx.as_device_ptr().as_mut_ptr() as *mut _;

        unsafe {
            sys::cudnnPoolingBackward(
                self.raw,
                pooling_desc.raw,
                alpha_ptr,
                y_desc.raw,
                y_ptr,
                dy_desc.raw,
                dy_ptr,
                x_desc.raw,
                x_ptr,
                beta_ptr,
                dx_desc.raw,
                dx_ptr,
            )
            .into_result()
        }
    }
}

/// Supported data type configurations for the pooling forward operation.
pub trait SupportedPoolFwd<T>: DataType + private::Sealed
where
    T: DataType,
{
}

impl SupportedPoolFwd<i8> for f32 {}
impl SupportedPoolFwd<u8> for f32 {}
impl SupportedPoolFwd<i32> for f32 {}
impl SupportedPoolFwd<i64> for f32 {}
impl SupportedPoolFwd<f32> for f32 {}
impl SupportedPoolFwd<f64> for f32 {}

impl SupportedPoolFwd<i8> for f64 {}
impl SupportedPoolFwd<u8> for f64 {}
impl SupportedPoolFwd<i32> for f64 {}
impl SupportedPoolFwd<i64> for f64 {}
impl SupportedPoolFwd<f32> for f64 {}
impl SupportedPoolFwd<f64> for f64 {}

/// Supported type configurations for the pooling backward operation.
pub trait SupportedPoolBwd<T>: DataType + private::Sealed
where
    T: DataType,
{
}

impl SupportedPoolBwd<f32> for f32 {}
impl SupportedPoolBwd<f64> for f64 {}
