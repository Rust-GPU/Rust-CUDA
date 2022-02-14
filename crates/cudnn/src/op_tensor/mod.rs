mod op_tensor_descriptor;
mod op_tensor_op;

pub use op_tensor_descriptor::*;
pub use op_tensor_op::*;

use crate::{sys, CudnnContext, CudnnError, DataType, IntoResult, TensorDescriptor};
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
    /// * `op_desc` - handle to a previously initialized op tensor descriptor.
    ///
    /// * `alpha` - scaling factor for the left operand.
    ///
    /// * `a_desc` - tensor descriptor for the left operand.
    ///
    /// * `a` - data for the left operand.
    ///
    /// * `beta` - scaling factor for the right operand.
    ///
    /// * `b_desc` - tensor descriptor for right operand.
    ///
    /// * `b` - data for the right operand.
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
    pub fn binary_tensor_op<CompT, Op, T1, T2, T3>(
        &self,
        op_desc: &OpTensorDescriptor<CompT, Op>,
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
        CompT: DataType + SupportedOp<T1, T2, T3>,
        Op: OpTensorOp + BinaryOp,
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
    /// * `op_desc` - handle to a previously initialized op tensor descriptor.
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
    pub fn unary_tensor_op<CompT, Op, T1, T2>(
        &self,
        op_desc: &OpTensorDescriptor<CompT, Op>,
        alpha: CompT,
        a_desc: &TensorDescriptor<T1>,
        a: &impl GpuBuffer<T1>,
        gamma: CompT,
        c_desc: &TensorDescriptor<T2>,
        c: &mut impl GpuBuffer<T2>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T1, T2>,
        Op: OpTensorOp + UnaryOp,
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
    pub fn add_assign<CompT, T1, T2>(
        &self,
        alpha: CompT,
        a_desc: &TensorDescriptor<T1>,
        a: &impl GpuBuffer<T1>,
        gamma: CompT,
        c_desc: &TensorDescriptor<T2>,
        c: &mut impl GpuBuffer<T2>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T1, T2>,
        T1: DataType,
        T2: DataType,
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
}
