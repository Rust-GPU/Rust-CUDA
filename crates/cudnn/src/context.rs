use crate::{
    convolution_algo::{
        BestHeuristic, ConvolutionBwdDataAlgo, ConvolutionBwdFilterAlgo, ConvolutionFwdAlgo,
        SupportedConvBwdData, SupportedConvBwdFilter, SupportedConvFwd,
    },
    convolution_descriptor::ConvolutionDescriptor,
    data_type::*,
    error::{CudnnError, IntoResult},
    filter::*,
    filter_descriptor::FilterDescriptor,
    nan_propagation::*,
    op_tensor_descriptor::*,
    sys,
    tensor::*,
    tensor_descriptor::TensorDescriptor,
    tensor_format::*,
};
use cust::memory::{GpuBox, GpuBuffer};
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
    /// `convolution_forward()` for the given layer specifications.
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
    /// math type of the convolution descriptor according to the one of the returned algorithm to
    /// get the best possible performance.
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
    /// conv_desc.set_math_type(algo.math_type())?; // Set math type.
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_convolution_forward_algorithm<
        T1,
        F1,
        T2,
        F2,
        CompType,
        T3,
        F3,
        const D: usize,
        const N: usize,
    >(
        &self,
        x_desc: &TensorDescriptor<T1, F1, D>,
        w_desc: &FilterDescriptor<T2, F2, D>,
        y_desc: &TensorDescriptor<T3, F3, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
    ) -> Result<BestHeuristic<sys::cudnnConvolutionFwdAlgo_t>, CudnnError>
    where
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        CompType: DataType,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
        BestHeuristic<sys::cudnnConvolutionFwdAlgo_t>:
            SupportedConvFwd<T1, F1, T2, F2, CompType, T3, F3, D, N>,
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
                    let results: Vec<BestHeuristic<sys::cudnnConvolutionFwdAlgo_t>> = {
                        let raw_results = std::slice::from_raw_parts(
                            perf_results.as_ptr(),
                            returned_algo_count as usize,
                        );

                        raw_results
                            .iter()
                            .copied()
                            .map(BestHeuristic::<sys::cudnnConvolutionFwdAlgo_t>::try_from)
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

    /// This function serves as a heuristic for obtaining the best suited algorithm for
    /// `convolution_backward_data()` for the given layer specifications.
    ///
    /// It will return the best algorithm according to an internal heuristic.
    ///
    /// # Arguments
    ///
    /// * `w_desc` - previously initialized filter descriptor.
    ///
    /// * `dy_desc` - previously initialized differential tensor descriptor for the output map.
    ///
    /// * `dx_desc` - previously initialized differential tensor descriptor for the input map.
    ///
    /// * `conv_desc` - previously initialized convolution descriptor.
    ///
    /// **Do note** that the best found algorithm `MathType` and the one supplied to the convolution
    /// descriptor's at its creation may differ, for this reason you should always manually set the
    /// math type of the convolution descriptor according to the one of the returned algorithm to
    /// get the best possible performance.
    pub fn get_convolution_backward_data_algorithm<
        T1,
        F1,
        T2,
        F2,
        CompType,
        T3,
        F3,
        const D: usize,
        const N: usize,
    >(
        &self,
        w_desc: &FilterDescriptor<T1, F1, D>,
        dy_desc: &TensorDescriptor<T2, F2, D>,
        dx_desc: &TensorDescriptor<T3, F3, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
    ) -> Result<BestHeuristic<sys::cudnnConvolutionBwdDataAlgo_t>, CudnnError>
    where
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        CompType: DataType,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
        BestHeuristic<sys::cudnnConvolutionBwdDataAlgo_t>:
            SupportedConvBwdData<T1, F1, T2, F2, CompType, T3, F3, D, N>,
    {
        let mut returned_algo_count = MaybeUninit::uninit();
        let mut perf_results = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetConvolutionBackwardDataAlgorithm_v7(
                self.raw,
                w_desc.raw,
                dy_desc.raw,
                conv_desc.raw,
                dx_desc.raw,
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
                    let results: Vec<BestHeuristic<sys::cudnnConvolutionBwdDataAlgo_t>> = {
                        let raw_results = std::slice::from_raw_parts(
                            perf_results.as_ptr(),
                            returned_algo_count as usize,
                        );

                        raw_results
                            .iter()
                            .copied()
                            .map(BestHeuristic::<sys::cudnnConvolutionBwdDataAlgo_t>::try_from)
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

    /// This function serves as a heuristic for obtaining the best suited algorithm for
    /// `convolution_backward_filter()` for the given layer specifications.
    ///
    /// It will return the best algorithm according to an internal heuristic.
    ///
    /// # Arguments
    ///
    /// * `x_desc` -  previously initialized tensor descriptor for the input map.
    ///
    /// * `dy_desc` - previously initialized differential tensor descriptor for the output map.
    ///
    /// * `dw_desc` - previously initialized differential tensor descriptor for the filter.
    ///
    /// * `conv_desc` - previously initialized convolution descriptor.
    ///
    /// **Do note** that the best found algorithm `MathType` and the one supplied to the convolution
    /// descriptor's at its creation may differ, for this reason you should always manually set the
    /// math type of the convolution descriptor according to the one of the returned algorithm to
    /// get the best possible performance.
    pub fn get_convolution_backward_filter_algorithm<
        T1,
        F1,
        T2,
        F2,
        CompType,
        T3,
        F3,
        const D: usize,
        const N: usize,
    >(
        &self,
        x_desc: &TensorDescriptor<T1, F1, D>,
        dy_desc: &TensorDescriptor<T2, F2, D>,
        dw_desc: &FilterDescriptor<T3, F3, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
    ) -> Result<BestHeuristic<sys::cudnnConvolutionBwdFilterAlgo_t>, CudnnError>
    where
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        CompType: DataType,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
        BestHeuristic<sys::cudnnConvolutionBwdFilterAlgo_t>:
            SupportedConvBwdFilter<T1, F1, T2, F2, CompType, T3, F3, D, N>,
    {
        let mut returned_algo_count = MaybeUninit::uninit();
        let mut perf_results = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                self.raw,
                x_desc.raw,
                dy_desc.raw,
                conv_desc.raw,
                dw_desc.raw,
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
                    let results: Vec<BestHeuristic<sys::cudnnConvolutionBwdFilterAlgo_t>> = {
                        let raw_results = std::slice::from_raw_parts(
                            perf_results.as_ptr(),
                            returned_algo_count as usize,
                        );

                        raw_results
                            .iter()
                            .copied()
                            .map(BestHeuristic::<sys::cudnnConvolutionBwdFilterAlgo_t>::try_from)
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
    /// able to call `convolution_forward()` with the specified algorithm. The workspace
    /// allocated will then be passed to the routine `convolution_forward()`.
    ///
    /// The specified algorithm can be the result of the call to
    /// [`get_convolution_forward_algorithm`](crate::CudnnContext::get_convolution_forward_algorithm)
    /// or can be chosen arbitrarily by the user. In the former case workspace size can be directly
    /// obtained by calling [`workspace_size`](crate::BestHeuristic::workspace_size) on the returned
    /// algorithm.
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
    /// **Do note** that not every algorithm is available for every configuration of the input
    /// tensor and/or every configuration of the convolution descriptor.
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
    /// let workspace = size.map(|size| unsafe { DeviceBuffer::<u8>::uninitialized(size).unwrap() });
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_convolution_forward_workspace_size<
        T1,
        F1,
        T2,
        F2,
        CompType,
        T3,
        F3,
        A,
        const D: usize,
        const N: usize,
    >(
        &self,
        x_desc: &TensorDescriptor<T1, F1, D>,
        w_desc: &FilterDescriptor<T2, F2, D>,
        y_desc: &TensorDescriptor<T3, F3, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
        algo: &A,
    ) -> Result<Option<usize>, CudnnError>
    where
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        CompType: DataType,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
        A: ConvolutionFwdAlgo + SupportedConvFwd<T1, F1, T2, F2, CompType, T3, F3, D, N>,
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

            Ok(match size.assume_init() {
                0 => None,
                size @ _ => Some(size),
            })
        }
    }

    /// This function returns the amount of GPU memory workspace the user needs to allocate to be
    /// able to call `convolution_backward_data()` with the specified algorithm. The workspace
    /// allocated will then be passed to the routine `convolution_backward_data()`.
    ///
    /// The specified algorithm can be the result of the call to
    /// [`get_convolution_backward_data_algorithm`](crate::CudnnContext::get_convolution_backward_data_algorithm)
    /// or can be chosen arbitrarily by the user. In the former case workspace size can be directly
    /// obtained by calling [`workspace_size`](crate::BestHeuristic::workspace_size) on the returned
    /// algorithm.
    ///
    /// # Arguments
    ///
    /// * `w_desc` - previously initialized filter descriptor.
    ///
    /// * `dy_desc` - previously initialized differential tensor descriptor for the output map.
    ///
    /// * `dx_desc` - previously initialized differential tensor descriptor for the input map.
    ///
    /// * `conv_desc` - previously initialized convolution descriptor.
    ///
    /// * `algo` - chosen convolution algorithm.
    ///
    ///  **Do note** that not every algorithm is available for every configuration of the input
    /// tensor and/or every configuration of the convolution descriptor.
    pub fn get_convolution_backward_data_workspace_size<
        T1,
        F1,
        T2,
        F2,
        CompType,
        T3,
        F3,
        A,
        const D: usize,
        const N: usize,
    >(
        &self,
        w_desc: &FilterDescriptor<T1, F1, D>,
        dy_desc: &TensorDescriptor<T2, F2, D>,
        dx_desc: &TensorDescriptor<T3, F3, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
        algo: &A,
    ) -> Result<Option<usize>, CudnnError>
    where
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        CompType: DataType,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
        A: ConvolutionBwdDataAlgo + SupportedConvFwd<T1, F1, T2, F2, CompType, T3, F3, D, N>,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetConvolutionBackwardDataWorkspaceSize(
                self.raw,
                w_desc.raw,
                dy_desc.raw,
                conv_desc.raw,
                dx_desc.raw,
                ConvolutionBwdDataAlgo::into_raw(algo),
                size.as_mut_ptr(),
            )
            .into_result()?;

            Ok(match size.assume_init() {
                0 => None,
                size @ _ => Some(size),
            })
        }
    }

    /// This function returns the amount of GPU memory workspace the user needs to allocate to be
    /// able to call `convolution_backward_filter()` with the specified algorithm. The workspace
    /// allocated will then be passed to the routine `convolution_backward_filter()`.
    ///
    /// The specified algorithm can be the result of the call to
    /// [`get_convolution_backward_filter_algorithm`](crate::CudnnContext::get_convolution_backward_filter_algorithm)
    /// or can be chosen arbitrarily by the user. In the former case workspace size can be directly
    /// obtained by calling [`workspace_size`](crate::BestHeuristic::workspace_size) on the returned
    /// algorithm.
    ///
    /// # Arguments
    ///
    /// * `x_desc` -  previously initialized tensor descriptor for the input map.
    ///
    /// * `dy_desc` - previously initialized differential tensor descriptor for the output map.
    ///
    /// * `dw_desc` - previously initialized differential tensor descriptor for the filter.
    ///
    /// * `conv_desc` - previously initialized convolution descriptor.
    ///
    /// * `algo` - chosen convolution algorithm.
    ///
    /// **Do note** that not every algorithm is available for every configuration of the input
    /// tensor and/or every configuration of the convolution descriptor.
    pub fn get_convolution_filter_workspace_size<
        T1,
        F1,
        T2,
        F2,
        CompType,
        T3,
        F3,
        A,
        const D: usize,
        const N: usize,
    >(
        &self,
        x_desc: &TensorDescriptor<T1, F1, D>,
        dy_desc: &TensorDescriptor<T2, F2, D>,
        dw_desc: &FilterDescriptor<T3, F3, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
        algo: &A,
    ) -> Result<Option<usize>, CudnnError>
    where
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        CompType: DataType,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
        A: ConvolutionBwdFilterAlgo
            + SupportedConvBwdFilter<T1, F1, T2, F2, CompType, T3, F3, D, N>,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetConvolutionBackwardFilterWorkspaceSize(
                self.raw,
                x_desc.raw,
                dy_desc.raw,
                conv_desc.raw,
                dw_desc.raw,
                ConvolutionBwdFilterAlgo::into_raw(algo),
                size.as_mut_ptr(),
            )
            .into_result()?;

            Ok(match size.assume_init() {
                0 => None,
                size @ _ => Some(size),
            })
        }
    }

    /// This function executes convolutions or cross-correlations over `x` using a filter specified
    /// with `w`, returning results in `y`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling parameter.
    ///
    /// * `x` - input map.
    ///
    /// * `w` - filter.
    ///
    /// * `conv_desc` - convolution descriptor.
    ///
    /// * `algo` - convolution algorithm that should be used to compute the result.
    ///
    /// * `workspace` -  a buffer to GPU memory to a workspace needed to be able to execute the
    /// specified algorithm. Must be left to `None` if the algorithm works in-place. The workspace
    /// dimension can be obtained with [`get_convolution_forward_workspace_size`].
    ///
    /// * `beta` - scaling parameter.
    ///
    /// * `y` - output map. It carries the result of the convolution.
    ///
    /// Scaling factors `alpha` and `beta` can be used to scale the input tensor and the output
    /// tensor respectively. They are used to blend the computation result with prior value in the
    /// output layer as follows: y = alpha * result + beta * y.
    ///
    /// **Do note** than not all possible configurations of layouts and data types for the operands
    /// are supported by cuDNN. Refer to the following link for the
    /// [complete list](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward).
    pub fn convolution_forward<
        T1,
        F1,
        T2,
        F2,
        CompType,
        T3,
        F3,
        A,
        W,
        const D: usize,
        const N: usize,
    >(
        &self,
        alpha: CompType,
        x: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        w: &Filter<T2, F2, impl GpuBuffer<T2>, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
        algo: &A,
        workspace: &Option<W>,
        beta: CompType,
        y: &mut Tensor<T3, F3, impl GpuBuffer<T3>, D>,
    ) -> Result<(), CudnnError>
    where
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        CompType: DataType,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
        A: ConvolutionFwdAlgo + SupportedConvFwd<T1, F1, T2, F2, CompType, T3, F3, D, N>,
        W: GpuBuffer<u8>,
    {
        let x_data = x.data().as_device_ptr().as_raw();
        let x_desc = x.descriptor();

        let w_data = w.data().as_device_ptr().as_raw();
        let w_desc = w.descriptor();

        let y_data = y.data().as_device_ptr().as_raw_mut();
        let y_desc = y.descriptor();

        // If the workspace size is 0 then the algorithm can work in-place and cuDNN expects a null
        // pointer.
        let (workspace_ptr, workspace_size): (*mut u8, usize) = {
            workspace
                .as_ref()
                .map_or((std::ptr::null_mut(), 0), |workspace| {
                    (workspace.as_device_ptr().as_raw_mut(), workspace.len())
                })
        };

        unsafe {
            sys::cudnnConvolutionForward(
                self.raw,
                &alpha as *const CompType as *const std::ffi::c_void,
                x_desc.raw,
                x_data as *const std::ffi::c_void,
                w_desc.raw,
                w_data as *const std::ffi::c_void,
                conv_desc.raw,
                algo.into_raw(),
                workspace_ptr as *mut std::ffi::c_void,
                workspace_size,
                &beta as *const CompType as *const std::ffi::c_void,
                y_desc.raw,
                y_data as *mut std::ffi::c_void,
            )
            .into_result()
        }
    }

    /// This function computes the convolution data gradient of the tensor `dy`, where `y` is the
    /// output of the forward convolution in `convolution_forward`.
    ///
    /// It uses the specified algo, and returns the results in the output tensor `dx`. Scaling
    /// factors `alpha` and `beta` can be used to scale the computed result or accumulate with the
    /// current `dx`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling parameter.
    ///
    /// * `w` - filter.
    ///
    /// * `dy` - output map gradient.
    ///
    /// * `conv_desc` - previously initialized convolution description. The one defined for the
    /// forward pass in suitable to be used here provided that it refers to the same layer.
    ///
    /// * `algo` - convolution algorithm that should be used to compute the result.
    ///
    /// * `workspace` -  a buffer to GPU memory to a workspace needed to be able to execute the
    /// specified algorithm. Must be left to `None` if the algorithm works in-place. The workspace
    /// dimension can be obtained with [`get_convolution_forward_workspace_size`].
    ///
    /// * `beta` - scaling parameter.
    ///
    /// * `dx` - input map gradient.
    pub fn convolution_backward_data<
        T1,
        F1,
        T2,
        F2,
        CompType,
        T3,
        F3,
        A,
        W,
        const D: usize,
        const N: usize,
    >(
        &self,
        alpha: CompType,
        w: &Filter<T1, F1, impl GpuBuffer<T1>, D>,
        dy: &Tensor<T2, F2, impl GpuBuffer<T2>, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
        algo: &A,
        workspace: &Option<W>,
        beta: CompType,
        dx: &mut Tensor<T3, F3, impl GpuBuffer<T3>, D>,
    ) -> Result<(), CudnnError>
    where
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        CompType: DataType,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
        A: ConvolutionBwdDataAlgo + SupportedConvBwdData<T1, F1, T2, F2, CompType, T3, F3, D, N>,
        W: GpuBuffer<u8>,
    {
        let w_data = w.data().as_device_ptr().as_raw();
        let w_desc = w.descriptor();

        let dy_data = dy.data().as_device_ptr().as_raw();
        let dy_desc = dy.descriptor();

        let dx_data = dx.data().as_device_ptr().as_raw_mut();
        let dx_desc = dx.descriptor();

        let (workspace_ptr, workspace_size): (*mut u8, usize) = {
            workspace
                .as_ref()
                .map_or((std::ptr::null_mut(), 0), |workspace| {
                    (workspace.as_device_ptr().as_raw_mut(), workspace.len())
                })
        };

        unsafe {
            sys::cudnnConvolutionBackwardData(
                self.raw,
                &alpha as *const CompType as *const std::ffi::c_void,
                w_desc.raw,
                w_data as *const std::ffi::c_void,
                dy_desc.raw,
                dy_data as *const std::ffi::c_void,
                conv_desc.raw,
                algo.into_raw(),
                workspace_ptr as *mut std::ffi::c_void,
                workspace_size,
                &beta as *const CompType as *const std::ffi::c_void,
                dx_desc.raw,
                dx_data as *mut std::ffi::c_void,
            )
            .into_result()
        }
    }

    /// This function computes the convolution weight (filter) gradient of the tensor `dy`, where
    /// `y` is the output of the forward convolution in `convolution_forward()`.
    ///
    /// It uses the specified `algo`, and returns the results in the output tensor `dw`. Scaling
    /// factors `alpha` and `beta` can be used to scale the computed result or accumulate with the
    /// current `dw`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling parameter.
    ///
    /// * `x` - input map.
    ///
    /// * `dy` - output map gradient.
    ///
    /// * `conv_desc` - previously initialized convolution description. The one defined for the
    /// forward pass in suitable to be used here provided that it refers to the same layer.
    ///
    /// * `algo` - convolution algorithm that should be used to compute the result.
    ///
    /// * `workspace` -  a buffer to GPU memory to a workspace needed to be able to execute the
    /// specified algorithm. Must be left to `None` if the algorithm works in-place. The workspace
    /// dimension can be obtained with [`get_convolution_forward_workspace_size`].
    ///
    /// * `beta` - scaling parameter.
    ///
    /// * `dw` - filter gradient.
    pub fn convolution_backward_filter<
        T1,
        F1,
        T2,
        F2,
        CompType,
        T3,
        F3,
        A,
        W,
        const D: usize,
        const N: usize,
    >(
        &self,
        alpha: CompType,
        x: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        dy: &Tensor<T2, F2, impl GpuBuffer<T2>, D>,
        conv_desc: &ConvolutionDescriptor<CompType, N>,
        algo: &A,
        workspace: &Option<W>,
        beta: CompType,
        dw: &mut Filter<T3, F3, impl GpuBuffer<T3>, D>,
    ) -> Result<(), CudnnError>
    where
        T1: DataType,
        F1: TensorFormat + SupportedType<T1>,
        T2: DataType,
        F2: TensorFormat + SupportedType<T2>,
        CompType: DataType,
        T3: DataType,
        F3: TensorFormat + SupportedType<T3>,
        A: ConvolutionBwdFilterAlgo
            + SupportedConvBwdFilter<T1, F1, T2, F2, CompType, T3, F3, D, N>,
        W: GpuBuffer<u8>,
    {
        let x_data = x.data().as_device_ptr().as_raw();
        let x_desc = x.descriptor();

        let dy_data = dy.data().as_device_ptr().as_raw();
        let dy_desc = dy.descriptor();

        let dw_data = dw.data().as_device_ptr().as_raw_mut();
        let dw_desc = dw.descriptor();

        let (workspace_ptr, workspace_size): (*mut u8, usize) = {
            workspace
                .as_ref()
                .map_or((std::ptr::null_mut(), 0), |workspace| {
                    (workspace.as_device_ptr().as_raw_mut(), workspace.len())
                })
        };

        unsafe {
            sys::cudnnConvolutionBackwardFilter(
                self.raw,
                &alpha as *const CompType as *const std::ffi::c_void,
                x_desc.raw,
                x_data as *const std::ffi::c_void,
                dy_desc.raw,
                dy_data as *const std::ffi::c_void,
                conv_desc.raw,
                algo.into_raw(),
                workspace_ptr as *mut std::ffi::c_void,
                workspace_size,
                &beta as *const CompType as *const std::ffi::c_void,
                dw_desc.raw,
                dw_data as *mut std::ffi::c_void,
            )
            .into_result()
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
