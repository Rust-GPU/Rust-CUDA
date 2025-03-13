mod indices_type;
mod reduce_indices;
mod reduce_op;
mod reduction_descriptor;

pub use indices_type::*;
pub use reduce_indices::*;
pub use reduce_op::*;
pub use reduction_descriptor::*;

use std::mem::MaybeUninit;

use cust::memory::GpuBuffer;

use crate::{
    sys, CudnnContext, CudnnError, DataType, IntoResult, ScalingDataType, TensorDescriptor,
};

impl CudnnContext {
    /// Returns the minimum size of the workspace to be passed to the reduction given
    /// the input and output tensors.
    ///
    /// # Arguments
    ///
    /// * `desc` - reduction descriptor.
    /// * `a_desc` - input tensor descriptor.
    /// * `c_desc` - output tensor descriptor.
    pub fn get_reduction_workspace_size<T, U, V>(
        &self,
        desc: &ReductionDescriptor<T>,
        a_desc: &TensorDescriptor<U>,
        c_desc: &TensorDescriptor<V>,
    ) -> Result<usize, CudnnError>
    where
        T: DataType,
        U: DataType,
        V: DataType,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetReductionWorkspaceSize(
                self.raw,
                desc.raw,
                a_desc.raw,
                c_desc.raw,
                size.as_mut_ptr(),
            )
            .into_result()?;

            Ok(size.assume_init())
        }
    }

    /// Returns the minimum size of the index space to be passed to the reduction given
    /// the input and output tensors.
    ///
    /// # Arguments
    ///
    /// * `desc` - reduction descriptor.
    /// * `a_desc` - input tensor descriptor.
    /// * `c_desc` - output tensor descriptor.
    pub fn get_reduction_indices_size<T, U, V>(
        &self,
        desc: &ReductionDescriptor<T>,
        a_desc: &TensorDescriptor<U>,
        c_desc: &TensorDescriptor<V>,
    ) -> Result<usize, CudnnError>
    where
        T: DataType,
        U: DataType,
        V: DataType,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetReductionIndicesSize(
                self.raw,
                desc.raw,
                a_desc.raw,
                c_desc.raw,
                size.as_mut_ptr(),
            )
            .into_result()?;

            Ok(size.assume_init())
        }
    }

    /// This function reduces tensor `a` by implementing the equation:
    ///
    /// C = alpha * reduce op ( A ) + gamma * C
    ///
    /// given tensors `a` and `c` and scaling factors `alpha` and `gamma`.
    /// Each dimension of the output tensor c must match the corresponding dimension of the
    /// input tensor a or must be equal to 1.
    ///
    /// The dimensions equal to 1 indicate the dimensions of a to be reduced.
    ///
    /// **Do note** that currently only the 32-bit indices type is supported and that the data types
    /// of the tensors A and C must match if of type double. In this case, alpha and gamma and are all
    /// assumed to be of type double.
    ///
    /// # Arguments
    ///
    /// * `desc` - tensor reduction descriptor.
    /// * `indices` - indices buffer in device memory.
    /// * `workspace` - workspace for the reduction operation.
    /// * `alpha` - scaling factor for the input tensor.
    /// * `a_desc` - tensor descriptor for the input tensor.
    /// * `a` - input tensor in device memory.
    /// * `gamma` -  scaling factor for the output tensor.
    /// * `c_desc` - tensor descriptor for the output tensor.
    /// * `c` - output tensor in device memory.
    ///
    /// # Errors
    ///
    /// Returns errors if an unsupported configuration of arguments is detected.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, NanPropagation, ReduceOp, ReduceIndices, ReductionDescriptor, TensorDescriptor};
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let op = ReduceOp::Add;
    /// let nan_policy = NanPropagation::PropagateNaN;
    /// let indices = ReduceIndices::None;
    /// let indices_type = None;
    ///
    /// let desc = ReductionDescriptor::<f32>::new(op, nan_policy, indices, indices_type)?;
    ///
    /// let alpha = 1.0;
    /// let a_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 5], &[5, 5, 5, 1])?;
    /// let a = DeviceBuffer::<i8>::from_slice(&[4, 4, 4, 4, 4])?;
    ///
    /// let gamma = 1.0;
    /// let c_desc = TensorDescriptor::<i8>::new_strides(&[1, 1, 1, 1], &[1, 1, 1, 1])?;
    /// let mut c = DeviceBuffer::<i8>::from_slice(&[0])?;
    ///
    /// let workspace_size = ctx.get_reduction_workspace_size(&desc, &a_desc, &c_desc)?;
    /// let mut workspace = unsafe { DeviceBuffer::uninitialized(workspace_size)? };
    ///
    /// let indices: Option<&mut DeviceBuffer<u8>> = None;
    ///
    /// ctx.reduce(&desc, indices, &mut workspace, alpha, &a_desc, &a, gamma, &c_desc, &mut c)?;
    ///
    /// let c_host = c.as_host_vec()?;
    ///
    /// assert!(c_host.iter().all(|x| *x == 20));
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn reduce<CompT, U, V>(
        &self,
        desc: &ReductionDescriptor<CompT>,
        indices: Option<&mut impl GpuBuffer<u8>>,
        workspace: &mut impl GpuBuffer<u8>,
        alpha: CompT,
        a_desc: &TensorDescriptor<U>,
        a: &impl GpuBuffer<U>,
        gamma: CompT,
        c_desc: &TensorDescriptor<V>,
        c: &mut impl GpuBuffer<V>,
    ) -> Result<(), CudnnError>
    where
        CompT: ScalingDataType<U>,
        U: DataType,
        V: DataType,
    {
        let (indices_ptr, indices_size) = {
            indices.map_or((std::ptr::null_mut(), 0), |indices| {
                (indices.as_device_ptr().as_mut_ptr() as _, indices.len())
            })
        };

        let workspace_ptr = workspace.as_device_ptr().as_mut_ptr() as _;
        let workspace_size = workspace.len();

        let a_data = a.as_device_ptr().as_ptr() as _;
        let c_data = c.as_device_ptr().as_mut_ptr() as _;

        let alpha = &alpha as *const CompT as _;
        let gamma = &gamma as *const CompT as _;

        unsafe {
            sys::cudnnReduceTensor(
                self.raw,
                desc.raw,
                indices_ptr,
                indices_size,
                workspace_ptr,
                workspace_size,
                alpha,
                a_desc.raw,
                a_data,
                gamma,
                c_desc.raw,
                c_data,
            )
            .into_result()?;
        }

        Ok(())
    }
}
