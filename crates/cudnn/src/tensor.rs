use crate::{
    data_type::DataType,
    error::CudnnError,
    tensor_descriptor::TensorDescriptor,
    tensor_format::{SupportedType, TensorFormat},
};
use cust::memory::{DeviceBuffer, DeviceCopy, GpuBox, GpuBuffer, UnifiedBuffer};

/// A cuDNN tensor generic over both unified and device memory.
pub struct Tensor<T, F, V, const D: usize>
where
    T: DataType,
    F: TensorFormat + SupportedType<T>,
    V: GpuBuffer<T>,
{
    data: V,
    descriptor: TensorDescriptor<T, F, D>,
}

impl<T, F, V, const D: usize> Tensor<T, F, V, D>
where
    T: DataType,
    F: TensorFormat + SupportedType<T>,
    V: GpuBuffer<T>,
{
    /// Returns a reference to the tensor's underlying data.
    pub fn data(&self) -> &V {
        &self.data
    }

    /// Returns a reference to the tensor's descriptor.
    pub fn descriptor(&self) -> &TensorDescriptor<T, F, D> {
        &self.descriptor
    }

    /// Creates a new cuDNN tensor.
    ///
    /// **Do note** that the minimum number of dimensions supported by cuDNN is equal to 3.
    ///
    /// # Arguments
    ///
    /// * `data` - underlying data.
    ///
    /// * `shape` - array that contain the size of the tensor for every dimension. The size along
    /// unused dimensions should be set to 1.
    ///
    /// * `format` - tensor format.
    ///
    /// **Do note** that you should always initialize a [`CudnnContext`](crate::CudnnContext) before
    /// allocating memory and creating tensors.
    ///
    /// # Examples
    ///
    /// The vector underlying data can be allocated either in device memory or in unified memory.
    ///
    /// ## Device Memory
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, Tensor, NCHW};
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let host_vec = vec![0.0; 2500];
    /// let data = DeviceBuffer::<f32>::from_slice(&host_vec)?;
    ///
    /// let shape =  [2, 2, 25, 25];
    /// let format = NCHW;
    ///
    /// let t = Tensor::new(data, shape, NCHW)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Unified Memory
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, Tensor, NCHW};
    /// use cust::memory::UnifiedBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let data = UnifiedBuffer::new(&0.0_f32, 2500)?;
    ///
    /// let shape =  [2, 2, 25, 25];
    /// let format = NCHW;
    ///
    /// let t = Tensor::new(data, shape, NCHW)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(data: V, shape: [i32; D], format: F) -> Result<Self, CudnnError> {
        let descriptor = TensorDescriptor::new(shape, format)?;

        assert_eq!(
            shape.iter().product::<i32>() as usize,
            data.len(),
            "error: number of elements does not match shape."
        );

        Ok(Tensor { data, descriptor })
    }
}
