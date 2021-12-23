mod filter_descriptor;
pub use filter_descriptor::*;

use crate::{
    data_type::DataType,
    error::CudnnError,
    tensor::{SupportedType, TensorFormat},
};
use cust::memory::{DeviceBuffer, DeviceCopy, GpuBox, GpuBuffer, UnifiedBuffer};

/// A cuDNN filter generic over both unified and device memory.
pub struct Filter<T, F, V, const D: usize>
where
    T: DataType,
    F: TensorFormat + SupportedType<T>,
    V: GpuBuffer<T>,
{
    data: V,
    descriptor: FilterDescriptor<T, F, D>,
}

impl<T, F, V, const D: usize> Filter<T, F, V, D>
where
    T: DataType,
    F: TensorFormat + SupportedType<T>,
    V: GpuBuffer<T>,
{
    /// Returns a reference to the filter's underlying data.
    pub fn data(&self) -> &V {
        &self.data
    }

    /// Returns a reference to the filter's descriptor.
    pub fn descriptor(&self) -> &FilterDescriptor<T, F, D> {
        &self.descriptor
    }

    /// Creates a new cuDNN filter.
    ///
    /// **Do note** that the minimum number of dimensions supported by cuDNN is equal to 3.
    ///
    /// # Arguments
    ///
    /// * `data` - underlying data.
    ///
    /// * `shape` - array that contain the size of the filter for every dimension.
    ///
    /// * `format` - tensor format. If set to [`NCHW`](TensorFormat::NCHW), then the layout of the
    /// filter is as follows: for D = 4, a 4D filter descriptor, the filter layout is in the form of
    /// KCRS, i.e. K represents the number of output feature maps, C is the number of input feature
    /// maps, R is the number of rows per filter, S is the number of columns per filter. For N = 3,
    /// a 3D filter descriptor, the number S (number of columns per filter) is omitted. For N = 5
    /// and greater, the layout of the higher dimensions immediately follows RS.
    ///
    /// **Do note** that you should always initialize a [`CudnnContext`](crate::CudnnContext) before
    /// allocating memory and creating filters.
    ///
    /// # Examples
    ///
    /// The underlying data can be allocated either in device memory or in unified memory.
    ///
    /// ## Device Memory
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext, Filter, NCHW};
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
    /// let t = Filter::new(data, shape, NCHW)?;
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
    /// use cudnn::{CudnnContext, Filter, NCHW};
    /// use cust::memory::UnifiedBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let data = UnifiedBuffer::new(&0.0_f32, 2500)?;
    ///
    /// let shape =  [2, 2, 25, 25];
    /// let format = NCHW;
    ///
    /// let t = Filter::new(data, shape, NCHW)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(data: V, shape: [i32; D], format: F) -> Result<Self, CudnnError> {
        let descriptor = FilterDescriptor::new(shape, format)?;

        assert_eq!(
            shape.iter().product::<i32>() as usize,
            data.len(),
            "number of elements does not match shape."
        );

        Ok(Self { data, descriptor })
    }
}
