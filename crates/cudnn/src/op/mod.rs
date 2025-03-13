mod op_tensor_descriptor;
mod op_tensor_op;

pub use op_tensor_descriptor::*;
pub use op_tensor_op::*;

use crate::{
    sys, CudnnContext, CudnnError, DataType, IntoResult, ScalingDataType, TensorDescriptor,
};
use cust::memory::GpuBuffer;

impl CudnnContext {
    /// This function computes a binary element-wise tensor core operation according to the
    /// following equation:
    ///
    /// C = OP( alpha * A , beta * B ) + gamma * C
    ///
    /// given the tensors A, B and C, and the scaling parameters alpha, beta and gamma.
    ///
    /// Each dimension of the input tensor A must match the corresponding dimension of the
    /// destination tensor C, and each dimension of the input tensor B must match the
    /// corresponding dimension of the destination tensor C or must be equal to 1.
    /// In the latter case, the same value from the input tensor B for those dimensions will be
    /// used to blend into the C tensor.
    ///
    /// # Arguments
    ///
    /// * `op_desc` - handle to a previously initialized binary op tensor descriptor.
    /// * `alpha` - scaling factor for the left operand.
    /// * `a_desc` - tensor descriptor for the left operand.
    /// * `a` - data for the left operand.
    /// * `beta` - scaling factor for the right operand.
    /// * `b_desc` - tensor descriptor for right operand.
    /// * `b` - data for the right operand.
    /// * `gamma` - scaling factor for the destination tensor.
    /// * `c_desc` - tensor descriptor for the destination tensor.
    /// * `c` - data for the destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{BinaryOp, BinaryOpTensorDescriptor, CudnnContext, NanPropagation, TensorDescriptor};
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let op = BinaryOp::Mul;
    /// let nan_policy = NanPropagation::PropagateNaN;
    ///
    /// let op_desc = BinaryOpTensorDescriptor::<f32>::new(op, nan_policy)?;
    ///
    /// let alpha = 1.0;
    /// let a_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 5], &[5, 5, 5, 1])?;
    /// let a = DeviceBuffer::<i8>::from_slice(&[2, 2, 2, 2, 2])?;
    ///
    /// let beta = 1.0;
    /// let b_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 5], &[5, 5, 5, 1])?;
    /// let b = DeviceBuffer::<i8>::from_slice(&[3, 3, 3, 3, 3])?;
    ///
    /// let gamma = 0.0;
    /// let c_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 5], &[5, 5, 5, 1])?;
    /// let mut c = DeviceBuffer::<i8>::from_slice(&[0, 0, 0, 0, 0])?;
    ///
    /// ctx.binary_tensor_op(&op_desc, alpha, &a_desc, &a, beta, &b_desc, &b, gamma, &c_desc, &mut c)?;
    ///
    /// let c_host = c.as_host_vec()?;
    ///
    /// assert!(c_host.iter().all(|x| *x == 6));
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn binary_tensor_op<CompT, T1, T2, T3>(
        &self,
        op_desc: &BinaryOpTensorDescriptor<CompT>,
        alpha: CompT,
        a_desc: &TensorDescriptor<T1>,
        a: &impl GpuBuffer<T1>,
        beta: CompT,
        b_desc: &TensorDescriptor<T2>,
        b: &impl GpuBuffer<T2>,
        gamma: CompT,
        c_desc: &TensorDescriptor<T3>,
        c: &mut impl GpuBuffer<T3>,
    ) -> Result<(), CudnnError>
    where
        CompT: SupportedOp<T1, T2, T3>,
        T1: DataType,
        T2: DataType,
        T3: DataType,
    {
        let a_data = a.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let b_data = b.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let c_data = c.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let alpha = &alpha as *const CompT as *const std::ffi::c_void;
        let beta = &beta as *const CompT as *const std::ffi::c_void;
        let gamma = &gamma as *const CompT as *const std::ffi::c_void;

        unsafe {
            sys::cudnnOpTensor(
                self.raw,
                op_desc.raw,
                alpha,
                a_desc.raw,
                a_data,
                beta,
                b_desc.raw,
                b_data,
                gamma,
                c_desc.raw,
                c_data,
            )
            .into_result()
        }
    }

    /// This function computes an unary element wise tensor core operation according to the
    /// following equation:
    ///
    /// C = OP ( alpha * A ) + gamma * C
    ///
    /// given the tensors A and C, and the scaling parameters alpha and gamma.
    ///
    /// Each dimension of the input tensor A must match the corresponding dimension of the
    /// destination tensor C
    ///
    /// # Arguments
    ///
    /// * `op_desc` - handle to a previously initialized unary op tensor descriptor.
    ///
    /// * `alpha` - scaling factor for the operand.
    ///
    /// * `a_desc` - tensor descriptor for the operand.
    ///
    /// * `a` - data for the operand.
    ///
    /// * `gamma` - scaling factor for the destination tensor.
    ///
    /// * `c_desc` - tensor descriptor for the destination tensor.
    ///
    /// * `c` - data for the destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, NanPropagation, TensorDescriptor, UnaryOp, UnaryOpTensorDescriptor};
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let op = UnaryOp::Sqrt;
    /// let nan_policy = NanPropagation::PropagateNaN;
    ///
    /// let op_desc = UnaryOpTensorDescriptor::<f32>::new(op, nan_policy)?;
    ///
    /// let alpha = 1.0;
    /// let a_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 5], &[5, 5, 5, 1])?;
    /// let a = DeviceBuffer::<i8>::from_slice(&[49, 49, 49, 49, 49])?;
    ///
    /// let gamma = 0.0;
    /// let c_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 5], &[5, 5, 5, 1])?;
    /// let mut c = DeviceBuffer::<i8>::from_slice(&[0, 0, 0, 0, 0])?;
    ///
    /// ctx.unary_tensor_op(&op_desc, alpha, &a_desc, &a, gamma, &c_desc, &mut c)?;
    ///
    /// let c_host = c.as_host_vec()?;
    ///
    /// assert!(c_host.iter().all(|x| *x == 7));
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn unary_tensor_op<CompT, T1, T2>(
        &self,
        op_desc: &UnaryOpTensorDescriptor<CompT>,
        alpha: CompT,
        a_desc: &TensorDescriptor<T1>,
        a: &impl GpuBuffer<T1>,
        gamma: CompT,
        c_desc: &TensorDescriptor<T2>,
        c: &mut impl GpuBuffer<T2>,
    ) -> Result<(), CudnnError>
    where
        CompT: SupportedOp<T1, T1, T2>,
        T1: DataType,
        T2: DataType,
    {
        let a_data = a.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let c_data = c.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let alpha = &alpha as *const CompT as *const std::ffi::c_void;
        let gamma = &gamma as *const CompT as *const std::ffi::c_void;

        unsafe {
            // The second tensor and the second scaling factors here are ignored.
            // We use the left operand twice to make cuDNN happy, as it won't accept a null pointer.
            sys::cudnnOpTensor(
                self.raw,
                op_desc.raw,
                alpha,
                a_desc.raw,
                a_data,
                alpha,
                a_desc.raw,
                a_data,
                gamma,
                c_desc.raw,
                c_data,
            )
            .into_result()
        }
    }

    /// This function adds two tensors in-place according to the following equation:
    ///
    /// C = alpha * A + gamma * C
    ///
    /// given the tensors A and C, and the scaling parameters alpha and gamma.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling factor for the operand.
    ///
    /// * `a_desc` - tensor descriptor for the operand.
    ///
    /// * `a` - data for the operand.
    ///
    /// * `gamma` - scaling factor for the destination tensor.
    ///
    /// * `c_desc` - tensor descriptor for the destination tensor.
    ///
    /// * `c` - data for the destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnAddTensor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns error if an unsupported configurations of arguments is detected.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, TensorDescriptor};
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let alpha = 1.0;
    /// let a_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 5], &[5, 5, 5, 1])?;
    /// let a = DeviceBuffer::<i8>::from_slice(&[4, 4, 4, 4, 4])?;
    ///
    /// let gamma = 1.0;
    /// let c_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 5], &[5, 5, 5, 1])?;
    /// let mut c = DeviceBuffer::<i8>::from_slice(&[6, 6, 6, 6, 6])?;
    ///
    /// ctx.add_assign(alpha, &a_desc, &a, gamma, &c_desc, &mut c)?;
    ///
    /// let c_host = c.as_host_vec()?;
    ///
    /// assert!(c_host.iter().all(|x| *x == 10));
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_assign<CompT, T1>(
        &self,
        alpha: CompT,
        a_desc: &TensorDescriptor<T1>,
        a: &impl GpuBuffer<T1>,
        gamma: CompT,
        c_desc: &TensorDescriptor<T1>,
        c: &mut impl GpuBuffer<T1>,
    ) -> Result<(), CudnnError>
    where
        CompT: ScalingDataType<T1>,
        T1: DataType,
    {
        let a_data = a.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let c_data = c.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let alpha = &alpha as *const CompT as *const std::ffi::c_void;
        let gamma = &gamma as *const CompT as *const std::ffi::c_void;

        unsafe {
            sys::cudnnAddTensor(
                self.raw, alpha, a_desc.raw, a_data, gamma, c_desc.raw, c_data,
            )
            .into_result()
        }
    }

    /// Sets all the elements of a tensor to a given value.
    ///
    /// # Arguments
    ///
    /// * `desc` - tensor descriptor.
    ///
    /// * `data` - data for the tensor.
    ///
    /// * `value` - value to set. Must be stored in host memory.
    ///
    /// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns error if an unsupported configurations of arguments is detected.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, ScalarC, TensorDescriptor};
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let value = 7.0;
    /// let desc = TensorDescriptor::<f32>::new_format(&[1, 1, 1, 5], ScalarC::Nchw)?;
    /// let mut data = DeviceBuffer::<f32>::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0])?;
    ///
    /// ctx.set(&desc, &mut data, value)?;
    ///
    /// let data_host = data.as_host_vec()?;
    ///
    /// assert!(data_host.iter().all(|x| (*x - value).abs() <= std::f32::EPSILON));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set<CompT, T>(
        &self,
        desc: &TensorDescriptor<T>,
        data: &mut impl GpuBuffer<T>,
        value: CompT,
    ) -> Result<(), CudnnError>
    where
        CompT: ScalingDataType<T>,
        T: DataType,
    {
        let data = data.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let value = &value as *const CompT as *const std::ffi::c_void;

        unsafe { sys::cudnnSetTensor(self.raw, desc.raw, data, value).into_result() }
    }

    /// This function scales all the element of a tensor by a given value.
    ///
    /// # Arguments
    ///
    ///   * `desc` - descriptor of the tensor to scale.
    ///   * `data` - data of the tensor.
    ///   * `value` - value in the host memory to a single value that all elements of
    ///     the tensor will be scaled with.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnScaleTensor)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns error if an unsupported configurations of arguments is detected.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, ScalarC, TensorDescriptor};
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let value = 7.0;
    /// let desc = TensorDescriptor::<i8>::new_format(&[1, 1, 1, 5], ScalarC::Nchw)?;
    /// let mut data = DeviceBuffer::<i8>::from_slice(&[2, 2, 2, 2, 2])?;
    ///
    /// ctx.scale(&desc, &mut data, value)?;
    ///
    /// let data_host = data.as_host_vec()?;
    ///
    /// assert!(data_host.iter().all(|x| *x == 14));
    /// # Ok(())
    /// # }
    /// ```
    pub fn scale<CompT, T>(
        &self,
        desc: &TensorDescriptor<T>,
        data: &mut impl GpuBuffer<T>,
        value: CompT,
    ) -> Result<(), CudnnError>
    where
        CompT: ScalingDataType<T>,
        T: DataType,
    {
        let data = data.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let value = &value as *const CompT as *const std::ffi::c_void;

        unsafe { sys::cudnnScaleTensor(self.raw, desc.raw, data, value).into_result() }
    }
}
