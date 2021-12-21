use crate::{
    convolution_descriptor::ConvolutionDescriptor,
    convolution_fwd_algo::{BestHeuristic, ConvolutionFwdAlgo, SupportedConvFwd},
    data_type::*,
    error::{CudnnError, IntoResult},
    filter_descriptor::FilterDescriptor,
    nan_propagation::*,
    op_tensor_descriptor::*,
    sys,
    tensor::*,
    tensor_descriptor::TensorDescriptor,
    tensor_format::*,
};
use cust::memory::GpuBuffer;
use std::mem::{self, MaybeUninit};

/// cuDNN library context. It's the central structure required to interact with cuDNN.
/// It holds and manages internal memory allocations.
///
/// # Multi-thread Usage
///
/// While it is technically allowed to use the same context across threads, it is very suboptimal
/// and dangerous so we chose not to expose this functionality. Instead, you should create a context
/// for every thread as also recommended by the
/// [cuDNN docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreate).
///
/// # Multi-Device Usage
///
/// cuDNN contexts are tied to the current device through the current CUDA context, therefore,
/// for multi-device usage one cuDNN context must be created for every different device.
///
/// # Drop Overhead
///
/// cuDNN contexts hold the internal memory allocations required by the library, and will free those
/// resources on drop. They will also synchronize the entire device when dropping the context.
/// Therefore, you should minimize both the amount of contexts, and the amount of context drops.
/// You should generally allocate and drop context outside of performance critical code paths.
pub struct CudnnContext {
    raw: sys::cudnnHandle_t,
}

impl CudnnContext {
    /// Creates a new cuDNN context, allocating the required memory on both host and device.
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::CudnnContext;
    ///
    /// let ctx = CudnnContext::new()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreate(raw.as_mut_ptr()).into_result()?;
            let raw = raw.assume_init();

            Ok(Self { raw })
        }
    }

    /// Returns the version number of the underlying cuDNN library.
    pub fn version(&self) -> (u32, u32, u32) {
        unsafe {
            // cudnnGetVersion does not return a state as it never fails.
            let version = sys::cudnnGetVersion();
            (
                (version / 1000) as u32,
                ((version % 1000) / 100) as u32,
                (version % 100) as u32,
            )
        }
    }

    /// The the same version of a given cuDNN library can be compiled against different CUDA toolkit
    /// versions. This routine returns the CUDA toolkit version that the currently used cuDNN
    /// library has been compiled against.
    pub fn cuda_version(&self) -> (u32, u32, u32) {
        unsafe {
            // cudnnGetCudartVersion does not return a state as it never fails.
            let version = sys::cudnnGetCudartVersion();
            (
                (version / 1000) as u32,
                ((version % 1000) / 100) as u32,
                (version % 100) as u32,
            )
        }
    }

    /// This function implements the equation:
    ///
    /// C = alpha * A + beta * B + gamma * C
    ///
    /// given the tensors A, B and C, and the scaling parameters alpha, beta and gamma.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling factor for the left operand.
    ///
    /// * `a` - left operand.
    ///
    /// * `beta` - scaling factor for the right operand.
    ///
    /// * `b` - right operand.
    ///
    /// * `gamma` - scaling factor for the destination tensor.
    ///
    /// * `c` - destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    pub fn add<CompT, T1, F1, T2, F2, T3, F3, const D: usize>(
        &self,
        alpha: CompT,
        a: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        beta: CompT,
        b: &Tensor<T2, F2, impl GpuBuffer<T2>, D>,
        gamma: CompT,
        c: &mut Tensor<T3, F3, impl GpuBuffer<T3>, D>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T2, T3>,
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
    {
        let a_data = a.data().as_device_ptr().as_raw();
        let a_desc = a.descriptor();

        let b_data = b.data().as_device_ptr().as_raw();
        let b_desc = b.descriptor();

        let c_data = c.data().as_device_ptr().as_raw();
        let c_desc = c.descriptor();

        let add_op_desc =
            OpTensorDescriptor::<CompT>::new(OpTensorOp::Add, NanPropagation::PropagateNaN)?;

        unsafe {
            sys::cudnnOpTensor(
                self.raw,
                add_op_desc.raw,
                &alpha as *const CompT as *const std::ffi::c_void,
                a_desc.raw,
                a_data as *const std::ffi::c_void,
                &beta as *const CompT as *const std::ffi::c_void,
                b_desc.raw,
                b_data as *const std::ffi::c_void,
                &gamma as *const CompT as *const std::ffi::c_void,
                c_desc.raw,
                c_data as *mut std::ffi::c_void,
            )
            .into_result()
        }
    }

    /// This function implements the equation:
    ///
    /// C = alpha * A + gamma * C
    ///
    /// given the tensors A and C, and the scaling parameters alpha and gamma.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling factor for the operand.
    ///
    /// * `a` - operand.
    ///
    /// * `gamma` - scaling factor for the destination tensor.
    ///
    /// * `c` - destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    pub fn add_assign<CompT, T1, F1, T2, F2, const D: usize>(
        &self,
        alpha: CompT,
        a: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        gamma: CompT,
        c: &mut Tensor<T2, F2, impl GpuBuffer<T2>, D>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T1, T2>,
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
    {
        let a_data = a.data().as_device_ptr().as_raw();
        let a_desc = a.descriptor();

        let c_data = c.data().as_device_ptr().as_raw();
        let c_desc = c.descriptor();

        unsafe {
            sys::cudnnAddTensor(
                self.raw,
                &alpha as *const CompT as *const std::ffi::c_void,
                a_desc.raw,
                a_data as *const std::ffi::c_void,
                &gamma as *const CompT as *const std::ffi::c_void,
                c_desc.raw,
                c_data as *mut std::ffi::c_void,
            )
            .into_result()
        }
    }

    /// This function implements the equation:
    ///
    /// C = alpha * A * beta * B + gamma * C
    ///
    /// given the tensors A, B and C, and the scaling parameters alpha, beta and gamma.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling factor for the left operand.
    ///
    /// * `a` - left operand.
    ///
    /// * `beta` - scaling factor for the right operand.
    ///
    /// * `b` - right operand.
    ///
    /// * `gamma` - scaling factor for the destination tensor.
    ///
    /// * `c` - destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    pub fn mul<CompT, T1, F1, T2, F2, T3, F3, const D: usize>(
        &self,
        alpha: CompT,
        a: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        beta: CompT,
        b: &Tensor<T2, F2, impl GpuBuffer<T2>, D>,
        gamma: CompT,
        c: &mut Tensor<T3, F3, impl GpuBuffer<T3>, D>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T2, T3>,
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
    {
        let a_data = a.data().as_device_ptr().as_raw();
        let a_desc = a.descriptor();

        let b_data = b.data().as_device_ptr().as_raw();
        let b_desc = b.descriptor();

        let c_data = c.data().as_device_ptr().as_raw();
        let c_desc = c.descriptor();

        let mul_op_desc =
            OpTensorDescriptor::<CompT>::new(OpTensorOp::Mul, NanPropagation::PropagateNaN)?;

        unsafe {
            sys::cudnnOpTensor(
                self.raw,
                mul_op_desc.raw,
                &alpha as *const CompT as *const std::ffi::c_void,
                a_desc.raw,
                a_data as *const std::ffi::c_void,
                &beta as *const CompT as *const std::ffi::c_void,
                b_desc.raw,
                b_data as *const std::ffi::c_void,
                &gamma as *const CompT as *const std::ffi::c_void,
                c_desc.raw,
                c_data as *mut std::ffi::c_void,
            )
            .into_result()
        }
    }

    /// This function implements the equation:
    ///
    /// C = min (alpha * A, beta * B) + gamma * C
    ///
    /// given the tensors A, B and C, and the scaling parameters alpha, beta and gamma.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling factor for the left operand.
    ///
    /// * `a` - left operand.
    ///
    /// * `beta` - scaling factor for the right operand.
    ///
    /// * `b` - right operand.
    ///
    /// * `gamma` - scaling factor for the destination tensor.
    ///
    /// * `c` - destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    pub fn min<CompT, T1, F1, T2, F2, T3, F3, const D: usize>(
        &self,
        alpha: CompT,
        a: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        beta: CompT,
        b: &Tensor<T2, F2, impl GpuBuffer<T2>, D>,
        gamma: CompT,
        c: &mut Tensor<T3, F3, impl GpuBuffer<T3>, D>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T2, T3>,
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
    {
        let a_data = a.data().as_device_ptr().as_raw();
        let a_desc = a.descriptor();

        let b_data = b.data().as_device_ptr().as_raw();
        let b_desc = b.descriptor();

        let c_data = c.data().as_device_ptr().as_raw();
        let c_desc = c.descriptor();

        let min_op_desc =
            OpTensorDescriptor::<CompT>::new(OpTensorOp::Min, NanPropagation::PropagateNaN)?;

        unsafe {
            sys::cudnnOpTensor(
                self.raw,
                min_op_desc.raw,
                &alpha as *const CompT as *const std::ffi::c_void,
                a_desc.raw,
                a_data as *const std::ffi::c_void,
                &beta as *const CompT as *const std::ffi::c_void,
                b_desc.raw,
                b_data as *const std::ffi::c_void,
                &gamma as *const CompT as *const std::ffi::c_void,
                c_desc.raw,
                c_data as *mut std::ffi::c_void,
            )
            .into_result()
        }
    }

    /// This function implements the equation:
    ///
    /// C = max (alpha * A, beta * B) + gamma * C
    ///
    /// given the tensors A, B and C, and the scaling parameters alpha, beta and gamma.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling factor for the left operand.
    ///
    /// * `a` - left operand.
    ///
    /// * `beta` - scaling factor for the right operand.
    ///
    /// * `b` - right operand.
    ///
    /// * `gamma` - scaling factor for the destination tensor.
    ///
    /// * `c` - destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    pub fn max<CompT, T1, F1, T2, F2, T3, F3, const D: usize>(
        &self,
        alpha: CompT,
        a: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        beta: CompT,
        b: &Tensor<T2, F2, impl GpuBuffer<T2>, D>,
        gamma: CompT,
        c: &mut Tensor<T3, F3, impl GpuBuffer<T3>, D>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T2, T3>,
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
    {
        let a_data = a.data().as_device_ptr().as_raw();
        let a_desc = a.descriptor();

        let b_data = b.data().as_device_ptr().as_raw();
        let b_desc = b.descriptor();

        let c_data = c.data().as_device_ptr().as_raw();
        let c_desc = c.descriptor();

        let max_op_desc =
            OpTensorDescriptor::<CompT>::new(OpTensorOp::Max, NanPropagation::PropagateNaN)?;

        unsafe {
            sys::cudnnOpTensor(
                self.raw,
                max_op_desc.raw,
                &alpha as *const CompT as *const std::ffi::c_void,
                a_desc.raw,
                a_data as *const std::ffi::c_void,
                &beta as *const CompT as *const std::ffi::c_void,
                b_desc.raw,
                b_data as *const std::ffi::c_void,
                &gamma as *const CompT as *const std::ffi::c_void,
                c_desc.raw,
                c_data as *mut std::ffi::c_void,
            )
            .into_result()
        }
    }

    /// This function implements the equation:
    ///
    /// C = sqrt (alpha * A) + gamma * C
    ///
    /// given the tensors A and C, and the scaling parameters alpha and gamma.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling factor for the operand.
    ///
    /// * `a` - operand.
    ///
    /// * `gamma` - scaling factor for the destination tensor.
    ///
    /// * `c` - destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    pub fn sqrt<CompT, T1, F1, T2, F2, const D: usize>(
        &self,
        alpha: CompT,
        a: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        gamma: CompT,
        c: &mut Tensor<T2, F2, impl GpuBuffer<T2>, D>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T1, T2>,
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
    {
        let a_data = a.data().as_device_ptr().as_raw();
        let a_desc = a.descriptor();

        let c_data = c.data().as_device_ptr().as_raw();
        let c_desc = c.descriptor();

        let sqrt_op_desc =
            OpTensorDescriptor::<CompT>::new(OpTensorOp::Sqrt, NanPropagation::PropagateNaN)?;

        unsafe {
            // The second tensor and the second scaling factors here are ignored.
            // We use the left operand twice to make cuDNN happy, as it won't accept a null pointer.
            sys::cudnnOpTensor(
                self.raw,
                sqrt_op_desc.raw,
                &alpha as *const CompT as *const std::ffi::c_void,
                a_desc.raw,
                a_data as *const std::ffi::c_void,
                &alpha as *const CompT as *const std::ffi::c_void,
                a_desc.raw,
                a_data as *const std::ffi::c_void,
                &gamma as *const CompT as *const std::ffi::c_void,
                c_desc.raw,
                c_data as *mut std::ffi::c_void,
            )
            .into_result()
        }
    }

    /// This function implements the equation:
    ///
    /// C = NOT (alpha * A) + gamma * C
    ///
    /// given the tensors A and C, and the scaling parameters alpha and gamma.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling factor for the operand.
    ///
    /// * `a` - operand.
    ///
    /// * `gamma` - scaling factor for the destination tensor.
    ///
    /// * `c` - destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    pub fn not<CompT, T1, F1, T2, F2, const D: usize>(
        &self,
        alpha: CompT,
        a: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        gamma: CompT,
        c: &mut Tensor<T2, F2, impl GpuBuffer<T2>, D>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T1, T2>,
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
    {
        let a_data = a.data().as_device_ptr().as_raw();
        let a_desc = a.descriptor();

        let c_data = c.data().as_device_ptr().as_raw();
        let c_desc = c.descriptor();

        let not_op_desc =
            OpTensorDescriptor::<CompT>::new(OpTensorOp::Not, NanPropagation::PropagateNaN)?;

        unsafe {
            // The second tensor and the second scaling factors here are ignored.
            // We use the left operand twice to make cuDNN happy, as it won't accept a null pointer.
            sys::cudnnOpTensor(
                self.raw,
                not_op_desc.raw,
                &alpha as *const CompT as *const std::ffi::c_void,
                a_desc.raw,
                a_data as *const std::ffi::c_void,
                &alpha as *const CompT as *const std::ffi::c_void,
                a_desc.raw,
                a_data as *const std::ffi::c_void,
                &gamma as *const CompT as *const std::ffi::c_void,
                c_desc.raw,
                c_data as *mut std::ffi::c_void,
            )
            .into_result()
        }
    }

    /// This function serves as a heuristic for obtaining the best suited algorithm for
    /// `cudnnConvolutionForward()` for the given layer specifications.
    ///
    /// It will return the best algorithm according to an internal heuristic.
    ///
    /// # Arguments
    ///
    /// * `x_desc` - previously initialized tensor descriptor for the input map.
    ///
    /// * `w_desc` - previously initialized tensor descriptor for the filter map.
    ///
    /// * `y_desc` - previously initialized tensor descriptor for the output map.
    ///
    /// * `conv_desc` - previously initialized convolution descriptor.
    ///
    /// **Do note** that the best found algorithm `MathType` and the one supplied to the convolution
    /// descriptor's at its creation may differ, for this reason you should always manually set the
    /// math type of the convolution descriptor according to the one of the returned algorithm, as
    /// pictured in the following example.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///     ConvolutionDescriptor, ConvolutionMode, CudnnContext, FilterDescriptor, MathType,
    ///     TensorDescriptor, NCHW,
    /// };
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let groups = 1;
    /// let mode = ConvolutionMode::CrossCorrelation;
    /// let math_type = MathType::Default;
    ///
    /// // 2-dimensional convolution.
    /// let mut conv_desc = ConvolutionDescriptor::<f32, 2>::new(padding, stride, dilation, groups, mode, math_type)?;
    ///
    /// let input_desc = TensorDescriptor::<f32, _, 4>::new([3, 2, 5, 5,], NCHW)?;
    /// let filter_desc = FilterDescriptor::<f32, _, 4>::new([3, 2, 2, 2], NCHW)?;
    /// let output_desc = TensorDescriptor::<f32, _, 4>::new([3, 3, 4, 4], NCHW)?;
    ///
    /// let algo = ctx.get_convolution_forward_algorithm(&input_desc, &filter_desc, &output_desc, &conv_desc)?;
    ///
    /// conv_desc.set_math_type(algo.math_type())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_convolution_forward_algorithm<
        InType,
        InFmt,
        FilterType,
        FilterFmt,
        CompType,
        OutType,
        OutFmt,
        const D: usize,
        const N: usize,
    >(
        &self,
        x_desc: &TensorDescriptor<InType, InFmt, D>,
        w_desc: &FilterDescriptor<FilterType, FilterFmt, D>,
        y_desc: &TensorDescriptor<OutType, OutFmt, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
    ) -> Result<BestHeuristic, CudnnError>
    where
        InType: DataType,
        InFmt: TensorFormat + SupportedType<InType>,
        FilterType: DataType,
        FilterFmt: TensorFormat + SupportedType<FilterType>,
        CompType: DataType,
        OutType: DataType,
        OutFmt: TensorFormat + SupportedType<OutType>,
        BestHeuristic:
            SupportedConvFwd<InType, InFmt, FilterType, FilterFmt, CompType, OutType, OutFmt, D, N>,
    {
        let mut returned_algo_count = MaybeUninit::uninit();
        let mut perf_results = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetConvolutionForwardAlgorithm_v7(
                self.raw,
                x_desc.raw,
                w_desc.raw,
                conv_desc.raw,
                y_desc.raw,
                1,
                returned_algo_count.as_mut_ptr(),
                perf_results.as_mut_ptr(),
            )
            .into_result()?;

            let returned_algo_count = returned_algo_count.assume_init();

            match returned_algo_count {
                // This is general enough so that in the future it can be expanded to be more
                // complex.
                1 => {
                    let results: Vec<BestHeuristic> = {
                        let raw_results = std::slice::from_raw_parts(
                            perf_results.as_ptr(),
                            returned_algo_count as usize,
                        );

                        raw_results
                            .iter()
                            .copied()
                            .map(BestHeuristic::try_from)
                            .filter_map(Result::ok)
                            .collect()
                    };

                    let algo = results[0];

                    Ok(algo)
                }
                _ => return Err(CudnnError::BadParam),
            }
        }
    }

    /// This function returns the amount of GPU memory workspace the user needs to allocate to be
    /// able to call `cudnnConvolutionForward()` with the specified algorithm. The workspace
    /// allocated will then be passed to the routine `cudnnConvolutionForward()`.
    ///
    /// The specified algorithm can be the result of the call to
    /// [`get_convolution_forward_algorithm`](crate::CudnnContext::get_convolution_forward_algorithm)
    /// or can be chosen arbitrarily by the user. In the latter case workspace size can be directly
    /// obtained by calling [`workspace_size`](crate::BestHeuristic::workspace_size) on the returned
    /// algorithm.
    ///
    /// **Do note** that not every algorithm is available for every configuration of the input
    /// tensor and/or every configuration of the convolution descriptor.
    ///
    /// # Arguments
    ///
    /// * `x_desc` - previously initialized tensor descriptor for the input map.
    ///
    /// * `w_desc` - previously initialized tensor descriptor for the filter map.
    ///
    /// * `y_desc` - previously initialized tensor descriptor for the output map.
    ///
    /// * `conv_desc` - previously initialized convolution descriptor.
    ///
    /// * `algo` - chosen convolution algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///    ConvolutionDescriptor, ConvolutionMode, CudnnContext, FilterDescriptor,
    ///    ImplicitPrecompGemm, MathType, TensorDescriptor, NCHW,
    /// };
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let groups = 1;
    /// let mode = ConvolutionMode::CrossCorrelation;
    /// let math_type = MathType::Default;
    ///
    /// // 2-dimensional convolution.
    /// let mut conv_desc =
    ///     ConvolutionDescriptor::<f32, 2>::new(padding, stride, dilation, groups, mode, math_type)?;
    ///
    /// let input_desc = TensorDescriptor::<f32, _, 4>::new([3, 2, 5, 5], NCHW)?;
    /// let filter_desc = FilterDescriptor::<f32, _, 4>::new([3, 2, 2, 2], NCHW)?;
    /// let output_desc = TensorDescriptor::<f32, _, 4>::new([3, 3, 4, 4], NCHW)?;
    ///
    /// let algo = ImplicitPrecompGemm;
    ///
    /// let size = ctx.get_convolution_forward_workspace_size(
    ///     &input_desc,
    ///     &filter_desc,
    ///     &output_desc,
    ///     &conv_desc,
    ///     &algo,
    /// )?;
    ///
    /// let workspace: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(size)? };
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_convolution_forward_workspace_size<
        InType,
        InFmt,
        FilterType,
        FilterFmt,
        CompType,
        OutType,
        OutFmt,
        Algo,
        const D: usize,
        const N: usize,
    >(
        &self,
        x_desc: &TensorDescriptor<InType, InFmt, D>,
        w_desc: &FilterDescriptor<FilterType, FilterFmt, D>,
        y_desc: &TensorDescriptor<OutType, OutFmt, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
        algo: &Algo,
    ) -> Result<usize, CudnnError>
    where
        InType: DataType,
        InFmt: TensorFormat + SupportedType<InType>,
        FilterType: DataType,
        FilterFmt: TensorFormat + SupportedType<FilterType>,
        CompType: DataType,
        OutType: DataType,
        OutFmt: TensorFormat + SupportedType<OutType>,
        Algo: ConvolutionFwdAlgo
            + SupportedConvFwd<InType, InFmt, FilterType, FilterFmt, CompType, OutType, OutFmt, D, N>,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetConvolutionForwardWorkspaceSize(
                self.raw,
                x_desc.raw,
                w_desc.raw,
                conv_desc.raw,
                y_desc.raw,
                algo.into_raw(),
                size.as_mut_ptr(),
            )
            .into_result()?;

            Ok(size.assume_init())
        }
    }
}

impl Drop for CudnnContext {
    fn drop(&mut self) {
        unsafe {
            // This can be either a valid cuDNN handle or a null pointer.
            // Since it's getting dropped we shouldn't bother much.
            sys::cudnnDestroy(self.raw);
        }
    }
}
