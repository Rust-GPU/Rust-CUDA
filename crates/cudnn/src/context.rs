use crate::{
    convolution::{
        BestHeuristic, ConvolutionBwdDataAlgo, ConvolutionBwdFilterAlgo, ConvolutionDescriptor,
        ConvolutionFwdAlgo, Filter, FilterDescriptor, SupportedConvBwdData, SupportedConvBwdFilter,
        SupportedConvFwd,
    },
    data_type::*,
    dropout_descriptor::*,
    error::{CudnnError, IntoResult},
    nan_propagation::*,
    op_tensor::*,
    rnn::{RnnDataDescriptor, RnnDataLayout, RnnDataType, RnnDescriptor, RnnMode, SupportedPrec},
    sys,
    tensor::*,
    ForwardMode,
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
/// You should generally create and drop context outside of performance critical code paths.
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

    /// This function sets the user's CUDA stream in the cuDNN handle.
    ///
    /// The new stream will be used to launch cuDNN GPU kernels or to synchronize to this stream
    /// when cuDNN kernels are launched in the internal streams.
    ///
    /// If the cuDNN library stream is not set, all kernels use the default (NULL) stream.
    /// Setting the user stream in the cuDNN handle guarantees the issue-order execution of cuDNN
    /// calls and other GPU kernels launched in the same stream.
    ///
    /// # Arguments
    ///
    /// `stream` - the CUDA stream to be written to the cuDNN handle.
    ///
    /// # Errors
    ///
    /// Returns error if the supplied stream in invalid or a mismatch if found between the user
    /// stream and the cuDNN handle context.
    pub fn set_stream(&mut self, stream: &cust::stream::Stream) -> Result<(), CudnnError> {
        unsafe {
            sys::cudnnSetStream(self.raw, stream.as_inner() as sys::cudaStream_t).into_result()
        }
    }

    /// This function computes a binary element-wise tensor core operations according to the
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
    pub fn binary_tensor_op<CompT, Op, T1, F1, T2, F2, T3, F3, const D: usize>(
        &self,
        op_desc: &OpTensorDescriptor<CompT, Op>,
        alpha: CompT,
        a: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        beta: CompT,
        b: &Tensor<T2, F2, impl GpuBuffer<T2>, D>,
        gamma: CompT,
        c: &mut Tensor<T3, F3, impl GpuBuffer<T3>, D>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T2, T3>,
        Op: OpTensorOp + BinaryOp,
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

        unsafe {
            sys::cudnnOpTensor(
                self.raw,
                op_desc.raw,
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
    /// * `a` - operand.
    ///
    /// * `gamma` - scaling factor for the destination tensor.
    ///
    /// * `c` - destination tensor. This tensor is written after being read.
    ///
    /// **Do note** that the scaling factors must be stored in host memory. All tensor formats up
    /// to dimension five (5) are supported. This routine does not support tensor formats beyond
    /// these dimensions.
    pub fn unary_tensor_op<CompT, Op, T1, F1, T2, F2, const D: usize>(
        &self,
        op_desc: &OpTensorDescriptor<CompT, Op>,
        alpha: CompT,
        a: &Tensor<T1, F1, impl GpuBuffer<T1>, D>,
        gamma: CompT,
        c: &mut Tensor<T2, F2, impl GpuBuffer<T2>, D>,
    ) -> Result<(), CudnnError>
    where
        CompT: DataType + SupportedOp<T1, T1, T2>,
        Op: OpTensorOp + UnaryOp,
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
            // The second tensor and the second scaling factors here are ignored.
            // We use the left operand twice to make cuDNN happy, as it won't accept a null pointer.
            sys::cudnnOpTensor(
                self.raw,
                op_desc.raw,
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

    /// This function is used to query the amount of space required to store the states of the
    /// random number generators
    pub fn get_dropout_states_size(&self) -> Result<usize, CudnnError> {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnDropoutGetStatesSize(self.raw, size.as_mut_ptr()).into_result()?;

            Ok(size.assume_init())
        }
    }

    /// This function is used to query the amount of reserve needed to run dropout with the input
    /// dimensions given by `x_desc`.
    ///
    /// The same reserve space is expected to be passed to `dropout_forward()` and
    /// `dropout_backward()`, and its contents is expected to remain unchanged between
    /// `dropout_forward()` and `dropout_backward` calls.
    ///
    /// # Arguments
    ///
    /// `x_desc` - a previously initialized tensor descriptor, describing input to a dropout
    /// operation.
    pub fn get_dropout_reserved_space_size<T, F, const D: usize>(
        &self,
        x_desc: &TensorDescriptor<T, F, D>,
    ) -> Result<usize, CudnnError>
    where
        T: DataType,
        F: TensorFormat + SupportedType<T>,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnDropoutGetStatesSize(self.raw, size.as_mut_ptr()).into_result()?;

            Ok(size.assume_init())
        }
    }

    /// This function performs forward dropout operation over `x` returning results in `y`.
    ///
    /// The approximate dropout fraction of x values will be replaced by a 0, and the rest will be
    /// scaled by 1 / (1 - dropout), i.e. the value configured in `dropout_desc`.
    ///
    /// This function should not be running concurrently with another `dropout_forward()` function
    /// using the same states, as defined in the `DropoutDescriptor`.
    ///
    /// # Arguments
    ///
    /// * `dropout_descriptor` - previously created dropout descriptor.
    ///
    /// * `x` - input tensor.
    ///
    /// * `y` - destination tensor.
    ///
    /// * `reserve_space` - user-allocated GPU memory used by this function. It is expected that the
    /// contents of reserveSpace does not change between `dropout_forward()` and
    /// `dropout_backward()` calls.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of elements in `x` and `y` differs and if `reserve_space` is
    /// less than the value returned by `get_dropout_reserve_space_size`.
    pub fn dropout_forward<T, F1, F2, const D: usize>(
        &self,
        dropout_desc: &DropoutDescriptor<impl GpuBuffer<u8>>,
        x: &Tensor<T, F1, impl GpuBuffer<T>, D>,
        y: &mut Tensor<T, F2, impl GpuBuffer<T>, D>,
        reserve_space: &mut impl GpuBuffer<u8>,
    ) -> Result<(), CudnnError>
    where
        T: DataType,
        F1: TensorFormat + SupportedType<T>,
        F2: TensorFormat + SupportedType<T>,
    {
        let x_desc = x.descriptor();
        let x_data = x.data().as_device_ptr().as_raw();

        let y_desc = y.descriptor();
        let y_data = y.data().as_device_ptr().as_raw_mut();

        let reserve_space_ptr = reserve_space.as_device_ptr().as_raw_mut();

        unsafe {
            sys::cudnnDropoutForward(
                self.raw,
                dropout_desc.raw,
                x_desc.raw,
                x_data as *const std::ffi::c_void,
                y_desc.raw,
                y_data as *mut std::ffi::c_void,
                reserve_space_ptr as *mut std::ffi::c_void,
                reserve_space.len(),
            )
            .into_result()
        }
    }

    /// This function performs backward dropout operation over `dy` returning results in `dx`.
    ///
    /// If during forward dropout operation value from `x` was propagated to `y` then
    /// during backward operation value from `dy` will be propagated to `dx`, otherwise, `dx`
    /// value will be set to 0.
    ///
    /// # Arguments
    ///
    /// * `dropout_descriptor` - previously created dropout descriptor.
    ///
    /// * `dy` - input tensor.
    ///
    /// * `dx` - destination tensor.
    ///
    /// * `reserve_space` - user-allocated GPU memory used by this function. It is expected that the
    /// contents of reserveSpace does not change between `dropout_forward()` and
    /// `dropout_backward()` calls.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of elements in `dx` and `dy` differs and if `reserve_space`
    /// is less than the value returned by `get_dropout_reserve_space_size`.
    pub fn dropout_backward<T, F1, F2, const D: usize>(
        &self,
        dropout_desc: &DropoutDescriptor<impl GpuBuffer<u8>>,
        dy: &Tensor<T, F1, impl GpuBuffer<T>, D>,
        dx: &mut Tensor<T, F2, impl GpuBuffer<T>, D>,
        reserve_space: &mut impl GpuBuffer<u8>,
    ) -> Result<(), CudnnError>
    where
        T: DataType,
        F1: TensorFormat + SupportedType<T>,
        F2: TensorFormat + SupportedType<T>,
    {
        let dy_desc = dy.descriptor();
        let dy_data = dy.data().as_device_ptr().as_raw();

        let dx_desc = dx.descriptor();
        let dx_data = dx.data().as_device_ptr().as_raw_mut();

        let reserve_space_ptr = reserve_space.as_device_ptr().as_raw_mut();

        unsafe {
            sys::cudnnDropoutBackward(
                self.raw,
                dropout_desc.raw,
                dy_desc.raw,
                dy_data as *const std::ffi::c_void,
                dx_desc.raw,
                dx_data as *mut std::ffi::c_void,
                reserve_space_ptr as *mut std::ffi::c_void,
                reserve_space.len(),
            )
            .into_result()
        }
    }

    /// Creates and initializes a generic dropout descriptor.
    ///
    /// # Arguments
    ///
    /// * `dropout` - probability with which the value from input is set to zero during the dropout
    /// layer.
    ///
    /// * `states` - user-allocated GPU memory that will hold random number generator states.
    ///
    /// * `seed` - seed used to initialize random number generator states.
    ///
    /// **Do note** that the exact amount of memory can be obtained with `get_dropout_states_size`.
    ///
    /// # Errors
    ///
    /// Return errors if `states` size is less than that returned by `get_dropout_states_size`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{CudnnContext};
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let size = ctx.get_dropout_states_size()?;
    /// let states = unsafe { DeviceBuffer::uninitialized(size)? };
    ///
    /// let dropout = 0.5;
    /// let seed = 123;
    ///
    /// let dropout_desc = ctx.create_dropout_descriptor(dropout, states, seed)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_dropout_descriptor<T: GpuBuffer<u8>>(
        &self,
        dropout: f32,
        states: T,
        seed: u64,
    ) -> Result<DropoutDescriptor<T>, CudnnError> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            sys::cudnnCreateDropoutDescriptor(raw.as_mut_ptr()).into_result()?;

            let mut raw = raw.assume_init();

            sys::cudnnSetDropoutDescriptor(
                raw,
                self.raw,
                dropout,
                states.as_device_ptr().as_raw_mut() as *mut std::ffi::c_void,
                states.len(),
                seed,
            )
            .into_result()?;

            Ok(DropoutDescriptor::new(raw, states))
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
    pub fn get_convolution_backward_filter_workspace_size<
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
        workspace: Option<&mut W>,
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
    ///
    /// **Do note** than not all possible configurations of layouts and data types for the operands
    /// are supported by cuDNN. Refer to the following link for the
    /// [complete list](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardData).
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
        workspace: Option<&mut W>,
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
    ///
    /// **Do note** than not all possible configurations of layouts and data types for the operands
    /// are supported by cuDNN. Refer to the following link for the
    /// [complete list](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardFilter).
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
        workspace: Option<&mut W>,
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

    /// Computes the work and reserve space buffer sizes based on the RNN network geometry stored
    /// in `rnn_desc`, designated usage (inference or training) defined by the `mode` argument, and
    /// the current RNN data dimensions are retrieved from `x_desc`.
    ///
    /// When RNN data dimensions change, this function must be called again because the RNN
    /// temporary buffer sizes are not monotonic.
    ///
    /// # Arguments
    ///
    /// * `rnn_desc` - a RNN descriptor.
    ///
    /// * `mode` - specifies whether the temporary buffers are used in inference or training mode.
    /// The reserve-space buffer is not used during inference. Therefore, the returned size of the
    /// reserve space buffer will be `None` when the `mode` argument is `ForwardMode::Inference`.
    ///
    /// * `x_desc` - a RNN data descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error is an incompatible or unsupported combination of input arguments was
    /// detected.
    pub fn get_rnn_temp_space_sizes<T1, T2, L>(
        &self,
        rnn_desc: &RnnDescriptor<T1, T2>,
        mode: ForwardMode,
        x_desc: &RnnDataDescriptor<T1, L>,
    ) -> Result<(usize, Option<usize>), CudnnError>
    where
        T1: DataType + RnnDataType,
        T2: DataType + SupportedPrec<T1>,
        L: RnnDataLayout,
    {
        let mut workspace_size = MaybeUninit::uninit();
        let mut reserve_space_size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetRNNTempSpaceSizes(
                self.raw,
                rnn_desc.raw,
                mode.into(),
                x_desc.raw,
                workspace_size.as_mut_ptr(),
                reserve_space_size.as_mut_ptr(),
            )
            .into_result()?;

            Ok((
                workspace_size.assume_init(),
                match reserve_space_size.assume_init() {
                    0 => None,
                    size @ _ => Some(size),
                },
            ))
        }
    }

    /// This function returns the required size of the weight space buffer in bytes.
    /// The weight space buffer holds all RNN weight matrices and bias vectors.
    ///
    /// # Arguments
    ///
    /// `rnn_desc` - an RNN descriptor.
    pub fn get_rnn_weight_space_size<T1, T2>(
        &self,
        rnn_desc: &RnnDescriptor<T1, T2>,
    ) -> Result<usize, CudnnError>
    where
        T1: DataType + RnnDataType,
        T2: DataType + SupportedPrec<T1>,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetRNNWeightSpaceSize(self.raw, rnn_desc.raw, size.as_mut_ptr())
                .into_result()?;

            Ok(size.assume_init())
        }
    }

    pub fn rnn_forward<T1, T2, L>(
        &self,
        rnn_desc: &RnnDescriptor<T1, T2>,
        forward_mode: ForwardMode,
        device_seq_lengths: &impl GpuBuffer<i32>,
        x_desc: &RnnDataDescriptor<T1, L>,
        x: &impl GpuBuffer<T1>,
        y_desc: &RnnDataDescriptor<T1, L>,
        y: &impl GpuBuffer<T1>,
        h_desc: &TensorDescriptor<T1, NCHW, 3>,
        hx: &impl GpuBuffer<T1>,
        hy: &mut impl GpuBuffer<T1>,
        c_desc: Option<&TensorDescriptor<T1, NCHW, 3>>,
        cx: Option<&impl GpuBuffer<T1>>,
        cy: Option<&mut impl GpuBuffer<T1>>,
        work_space: &mut impl GpuBuffer<u8>,
        weight_space: &mut impl GpuBuffer<u8>,
        reserve_space: &mut impl GpuBuffer<u8>,
    ) -> Result<(), CudnnError>
    where
        T1: DataType + RnnDataType,
        T2: DataType + SupportedPrec<T1>,
        L: RnnDataLayout,
        NCHW: SupportedType<T1>,
    {
        todo!()
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
