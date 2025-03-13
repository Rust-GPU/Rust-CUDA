mod convolution_algo;
mod convolution_config;
mod convolution_descriptor;
mod convolution_mode;
mod filter_descriptor;

pub use convolution_algo::*;
pub use convolution_config::*;
pub use convolution_descriptor::*;
pub use convolution_mode::*;
pub use filter_descriptor::*;

use crate::{
    sys, ActivationDescriptor, CudnnContext, CudnnError, DataType, IntoResult, TensorDescriptor,
};
use cust::memory::GpuBuffer;
use std::mem::MaybeUninit;

impl CudnnContext {
    /// This function serves as a heuristic for obtaining the best suited algorithm for
    /// `convolution_forward()` for the given layer specifications.
    ///
    /// It will return the best algorithm according to an internal heuristic.
    ///
    /// # Arguments
    ///
    ///   * `x_desc` - previously initialized tensor descriptor for the input map.
    ///   * `w_desc` - previously initialized tensor descriptor for the filter map.
    ///   * `y_desc` - previously initialized tensor descriptor for the output map.
    ///   * `conv_desc` - previously initialized convolution descriptor.
    ///
    /// **Do note** that the best found algorithm `MathType` and the one supplied to the
    /// convolution descriptor's at its creation may differ, for this reason you should
    /// always manually set the math type of the convolution descriptor according to the
    /// one of the returned algorithm to get the best possible performance.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionForwardAlgorithm_v7)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid combination of arguments is passed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///     ConvDescriptor, ConvMode, CudnnContext, FilterDescriptor, ScalarC,
    ///     TensorDescriptor
    /// };
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let mode = ConvMode::CrossCorrelation;
    ///
    /// let mut conv_desc = ConvDescriptor::<f32>::new(padding, stride, dilation, mode)?;
    ///
    /// let x_desc = TensorDescriptor::<f32>::new_format(&[3, 2, 5, 5,], ScalarC::Nchw)?;
    /// let w_desc = FilterDescriptor::<f32>::new(&[3, 2, 2, 2], ScalarC::Nchw)?;
    /// let y_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    ///
    /// let res = ctx.get_convolution_forward_algorithm(&x_desc, &w_desc, &y_desc, &conv_desc)?;
    ///
    /// let algo = res.algo();
    /// let math_type = res.math_type();
    ///
    /// conv_desc.set_math_type(math_type)?; // Sets math type.
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_convolution_forward_algorithm<T1, T2, CompT, T3>(
        &self,
        x_desc: &TensorDescriptor<T1>,
        w_desc: &FilterDescriptor<T2>,
        y_desc: &TensorDescriptor<T3>,
        conv_desc: &ConvDescriptor<CompT>,
    ) -> Result<BestHeuristic<ConvFwdAlgo>, CudnnError>
    where
        T1: DataType,
        T2: DataType,
        CompT: SupportedConv<T1, T2, T3>,
        T3: DataType,
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
                    let results: Vec<BestHeuristic<ConvFwdAlgo>> = {
                        let raw_results = std::slice::from_raw_parts(
                            perf_results.as_ptr(),
                            returned_algo_count as usize,
                        );

                        raw_results
                            .iter()
                            .copied()
                            .map(BestHeuristic::<ConvFwdAlgo>::try_from)
                            .filter_map(Result::ok)
                            .collect()
                    };

                    let algo = results[0];
                    Ok(algo)
                }
                _ => Err(CudnnError::BadParam),
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
    ///   * `w_desc` - previously initialized filter descriptor.
    ///   * `dy_desc` - previously initialized differential tensor descriptor for the
    ///     output map.
    ///   * `dx_desc` - previously initialized differential tensor descriptor for the
    ///     input map.
    ///   * `conv_desc` - previously initialized convolution descriptor.
    ///
    /// **Do note** that the best found algorithm `MathType` must be set manually on the
    /// convolution descriptor.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionBackwardDataAlgorithm_v7)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid combination of arguments is passed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///     ConvDescriptor, ConvMode, CudnnContext, FilterDescriptor, ScalarC,
    ///     TensorDescriptor
    /// };
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let mode = ConvMode::CrossCorrelation;
    ///
    /// let mut conv_desc = ConvDescriptor::<f32>::new(padding, stride, dilation, mode)?;
    ///
    /// let w_desc = FilterDescriptor::<f32>::new(&[3, 2, 2, 2], ScalarC::Nchw)?;
    /// let dy_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    /// let dx_desc = TensorDescriptor::<f32>::new_format(&[3, 2, 5, 5,], ScalarC::Nchw)?;
    ///
    /// let res = ctx.get_convolution_backward_data_algorithm(&w_desc, &dy_desc, &dx_desc, &conv_desc)?;
    ///
    /// let algo = res.algo();
    /// let math_type = res.math_type();
    ///
    /// conv_desc.set_math_type(math_type)?; // Sets math type.
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_convolution_backward_data_algorithm<T1, T2, CompT, T3>(
        &self,
        w_desc: &FilterDescriptor<T1>,
        dy_desc: &TensorDescriptor<T2>,
        dx_desc: &TensorDescriptor<T3>,
        conv_desc: &ConvDescriptor<CompT>,
    ) -> Result<BestHeuristic<ConvBwdDataAlgo>, CudnnError>
    where
        T1: DataType,
        T2: DataType,
        CompT: SupportedConv<T1, T2, T3>,
        T3: DataType,
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
                    let results: Vec<BestHeuristic<ConvBwdDataAlgo>> = {
                        let raw_results = std::slice::from_raw_parts(
                            perf_results.as_ptr(),
                            returned_algo_count as usize,
                        );

                        raw_results
                            .iter()
                            .copied()
                            .map(BestHeuristic::<ConvBwdDataAlgo>::try_from)
                            .filter_map(Result::ok)
                            .collect()
                    };

                    let algo = results[0];
                    Ok(algo)
                }
                _ => Err(CudnnError::BadParam),
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
    ///   * `x_desc` -  previously initialized tensor descriptor for the input map.
    ///   * `dy_desc` - previously initialized differential tensor descriptor for the
    ///     output map.
    ///   * `dw_desc` - previously initialized differential tensor descriptor for the
    ///     filter.
    ///   * `conv_desc` - previously initialized convolution descriptor.
    ///
    /// **Do note** that the best found algorithm `MathType` must be set manually on the
    /// convolution descriptor.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionBackwardFilterAlgorithm_v7)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid combination of arguments is passed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///     ConvDescriptor, ConvMode, CudnnContext, FilterDescriptor, ScalarC,
    ///     TensorDescriptor
    /// };
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let mode = ConvMode::CrossCorrelation;
    ///
    /// let mut conv_desc = ConvDescriptor::<f32>::new(padding, stride, dilation, mode)?;
    ///
    /// let x_desc = TensorDescriptor::<f32>::new_format(&[3, 2, 5, 5,], ScalarC::Nchw)?;
    /// let dy_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    /// let dw_desc = FilterDescriptor::<f32>::new(&[3, 2, 2, 2], ScalarC::Nchw)?;
    ///
    /// let res = ctx.get_convolution_backward_filter_algorithm(&x_desc, &dy_desc, &dw_desc, &conv_desc)?;
    ///
    /// let algo = res.algo();
    /// let math_type = res.math_type();
    ///
    /// conv_desc.set_math_type(math_type)?; // Sets math type.
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_convolution_backward_filter_algorithm<T1, T2, CompT, T3>(
        &self,
        x_desc: &TensorDescriptor<T1>,
        dy_desc: &TensorDescriptor<T2>,
        dw_desc: &FilterDescriptor<T3>,
        conv_desc: &ConvDescriptor<CompT>,
    ) -> Result<BestHeuristic<ConvBwdFilterAlgo>, CudnnError>
    where
        T1: DataType,
        T2: DataType,
        CompT: SupportedConv<T1, T2, T3>,
        T3: DataType,
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
                    let results: Vec<BestHeuristic<ConvBwdFilterAlgo>> = {
                        let raw_results = std::slice::from_raw_parts(
                            perf_results.as_ptr(),
                            returned_algo_count as usize,
                        );

                        raw_results
                            .iter()
                            .copied()
                            .map(BestHeuristic::<ConvBwdFilterAlgo>::try_from)
                            .filter_map(Result::ok)
                            .collect()
                    };

                    let algo = results[0];
                    Ok(algo)
                }
                _ => Err(CudnnError::BadParam),
            }
        }
    }

    /// This function returns the amount of GPU memory workspace the user needs to
    /// allocate to be able to call `convolution_forward()` with the specified
    /// algorithm. The workspace allocated will then be passed to the routine
    /// `convolution_forward()`.
    ///
    /// The specified algorithm can be the result of the call to
    /// [`get_convolution_forward_algorithm`](crate::CudnnContext::get_convolution_forward_algorithm)
    /// or can be chosen arbitrarily by the user. In the former case workspace size can
    /// be directly obtained by calling
    /// [`workspace_size()`](crate::BestHeuristic::workspace_size) on the returned
    /// algorithm.
    ///
    /// # Arguments
    ///
    /// * `x_desc` - previously initialized tensor descriptor for the input map.
    /// * `w_desc` - previously initialized tensor descriptor for the filter map.
    /// * `y_desc` - previously initialized tensor descriptor for the output map.
    /// * `conv_desc` - previously initialized convolution descriptor.
    /// * `algo` - chosen convolution algorithm.
    ///
    /// **Do note** that not every algorithm is available for every configuration of the
    /// input tensor and/or every configuration of the convolution descriptor.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionForwardWorkspaceSize)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid combination of arguments is passed or the
    /// combination of the tensor descriptors, filter descriptor and convolution
    /// descriptor is not supported for the specified algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///    ConvDescriptor, ConvFwdAlgo, ConvMode, CudnnContext, FilterDescriptor,
    ///    MathType, ScalarC, TensorDescriptor
    /// };
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let mode = ConvMode::CrossCorrelation;
    ///
    /// // 2-dimensional convolution.
    /// let mut conv_desc =
    ///     ConvDescriptor::<f32>::new(padding, stride, dilation, mode)?;
    ///
    /// let x_desc = TensorDescriptor::<f32>::new_format(&[3, 2, 5, 5,], ScalarC::Nchw)?;
    /// let w_desc = FilterDescriptor::<f32>::new(&[3, 2, 2, 2], ScalarC::Nchw)?;
    /// let y_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    ///
    /// let algo = ConvFwdAlgo::ImplicitPrecompGemm;
    ///
    /// let size = ctx.get_convolution_forward_workspace_size(
    ///     &x_desc,
    ///     &w_desc,
    ///     &y_desc,
    ///     &conv_desc,
    ///     algo,
    /// )?;
    ///
    /// let workspace = size.map(|size| unsafe { DeviceBuffer::<u8>::uninitialized(size).unwrap() });
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_convolution_forward_workspace_size<T1, T2, CompT, T3>(
        &self,
        x_desc: &TensorDescriptor<T1>,
        w_desc: &FilterDescriptor<T2>,
        y_desc: &TensorDescriptor<T3>,
        conv_desc: &ConvDescriptor<CompT>,
        algo: ConvFwdAlgo,
    ) -> Result<Option<usize>, CudnnError>
    where
        T1: DataType,
        T2: DataType,
        CompT: SupportedConv<T1, T2, T3>,
        T3: DataType,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetConvolutionForwardWorkspaceSize(
                self.raw,
                x_desc.raw,
                w_desc.raw,
                conv_desc.raw,
                y_desc.raw,
                algo.into(),
                size.as_mut_ptr(),
            )
            .into_result()?;

            Ok(match size.assume_init() {
                0 => None,
                size => Some(size),
            })
        }
    }

    /// This function returns the amount of GPU memory workspace the user needs to
    /// allocate to be able to call `convolution_backward_data()` with the specified
    /// algorithm. The workspace allocated will then be passed to the routine
    /// `convolution_backward_data()`.
    ///
    /// The specified algorithm can be the result of the call to
    /// [`get_convolution_backward_data_algorithm`](crate::CudnnContext::get_convolution_backward_data_algorithm)
    /// or can be chosen arbitrarily by the user. In the former case workspace size can
    /// be directly obtained by calling
    /// [`workspace_size`](crate::BestHeuristic::workspace_size) on the returned
    /// algorithm.
    ///
    /// # Arguments
    ///
    /// * `w_desc` - previously initialized filter descriptor.
    /// * `dy_desc` - previously initialized differential tensor descriptor for the
    ///   output map.
    /// * `dx_desc` - previously initialized differential tensor descriptor for the
    ///   input map.
    /// * `conv_desc` - previously initialized convolution descriptor.
    /// * `algo` - chosen convolution algorithm.
    ///
    ///  **Do note** that not every algorithm is available for every configuration of
    /// the input tensor and/or every configuration of the convolution descriptor.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionBackwardDataWorkspaceSize)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid combination of arguments is passed or the
    /// combination of the tensor descriptors, filter descriptor and convolution
    /// descriptor is not supported for the specified algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///    ConvDescriptor, ConvBwdDataAlgo, ConvMode, CudnnContext, FilterDescriptor,
    ///    MathType, ScalarC, TensorDescriptor
    /// };
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let mode = ConvMode::CrossCorrelation;
    ///
    /// // 2-dimensional convolution.
    /// let mut conv_desc =
    ///     ConvDescriptor::<f32>::new(padding, stride, dilation, mode)?;
    ///
    /// let w_desc = FilterDescriptor::<f32>::new(&[3, 2, 2, 2], ScalarC::Nchw)?;
    /// let dy_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    /// let dx_desc = TensorDescriptor::<f32>::new_format(&[3, 2, 5, 5,], ScalarC::Nchw)?;
    ///
    /// let algo = ConvBwdDataAlgo::Algo0;
    ///
    /// let size = ctx.get_convolution_backward_data_workspace_size(
    ///     &w_desc,
    ///     &dy_desc,
    ///     &dx_desc,
    ///     &conv_desc,
    ///     algo,
    /// )?;
    ///
    /// let workspace = size.map(|size| unsafe { DeviceBuffer::<u8>::uninitialized(size).unwrap() });
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_convolution_backward_data_workspace_size<T1, T2, CompT, T3>(
        &self,
        w_desc: &FilterDescriptor<T1>,
        dy_desc: &TensorDescriptor<T2>,
        dx_desc: &TensorDescriptor<T3>,
        conv_desc: &ConvDescriptor<CompT>,
        algo: ConvBwdDataAlgo,
    ) -> Result<Option<usize>, CudnnError>
    where
        T1: DataType,
        T2: DataType,
        CompT: SupportedConv<T1, T2, T3>,
        T3: DataType,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetConvolutionBackwardDataWorkspaceSize(
                self.raw,
                w_desc.raw,
                dy_desc.raw,
                conv_desc.raw,
                dx_desc.raw,
                algo.into(),
                size.as_mut_ptr(),
            )
            .into_result()?;

            Ok(match size.assume_init() {
                0 => None,
                size => Some(size),
            })
        }
    }

    /// This function returns the amount of GPU memory workspace the user needs to
    /// allocate to be able to call `convolution_backward_filter()` with the specified
    /// algorithm. The workspace allocated will then be passed to the routine
    /// `convolution_backward_filter()`.
    ///
    /// The specified algorithm can be the result of the call to
    /// [`get_convolution_backward_filter_algorithm`](crate::CudnnContext::get_convolution_backward_filter_algorithm)
    /// or can be chosen arbitrarily by the user. In the former case workspace size can
    /// be directly obtained by calling
    /// [`workspace_size`](crate::BestHeuristic::workspace_size) on the returned
    /// algorithm.
    ///
    /// # Arguments
    ///
    /// * `x_desc` -  previously initialized tensor descriptor for the input map.
    /// * `dy_desc` - previously initialized differential tensor descriptor for the
    ///   output map.
    /// * `dw_desc` - previously initialized differential tensor descriptor for the
    ///   filter.
    /// * `conv_desc` - previously initialized convolution descriptor.
    /// * `algo` - chosen convolution algorithm.
    ///
    /// **Do note** that not every algorithm is available for every configuration of the
    /// input tensor and/or every configuration of the convolution descriptor.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionBackwardFilterWorkspaceSize)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid combination of arguments is passed or the
    /// combination of the tensor descriptors, filter descriptor and convolution
    /// descriptor is not supported for the specified algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///    ConvDescriptor, ConvBwdFilterAlgo, ConvMode, CudnnContext, FilterDescriptor,
    ///    MathType, ScalarC, TensorDescriptor
    /// };
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let mode = ConvMode::CrossCorrelation;
    ///
    /// // 2-dimensional convolution.
    /// let mut conv_desc =
    ///     ConvDescriptor::<f32>::new(padding, stride, dilation, mode)?;
    ///
    /// let x_desc = TensorDescriptor::<f32>::new_format(&[3, 2, 5, 5,], ScalarC::Nchw)?;
    /// let dy_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    /// let dw_desc = FilterDescriptor::<f32>::new(&[3, 2, 2, 2], ScalarC::Nchw)?;
    ///
    /// let algo = ConvBwdFilterAlgo::Algo0;
    ///
    /// let size = ctx.get_convolution_backward_filter_workspace_size(
    ///     &x_desc,
    ///     &dy_desc,
    ///     &dw_desc,
    ///     &conv_desc,
    ///     algo,
    /// )?;
    ///
    /// let workspace = size.map(|size| unsafe { DeviceBuffer::<u8>::uninitialized(size).unwrap() });
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_convolution_backward_filter_workspace_size<T1, T2, CompT, T3>(
        &self,
        x_desc: &TensorDescriptor<T1>,
        dy_desc: &TensorDescriptor<T2>,
        dw_desc: &FilterDescriptor<T3>,
        conv_desc: &ConvDescriptor<CompT>,
        algo: ConvBwdFilterAlgo,
    ) -> Result<Option<usize>, CudnnError>
    where
        T1: DataType,
        T2: DataType,
        CompT: SupportedConv<T1, T2, T3>,
        T3: DataType,
    {
        let mut size = MaybeUninit::uninit();

        unsafe {
            sys::cudnnGetConvolutionBackwardFilterWorkspaceSize(
                self.raw,
                x_desc.raw,
                dy_desc.raw,
                conv_desc.raw,
                dw_desc.raw,
                algo.into(),
                size.as_mut_ptr(),
            )
            .into_result()?;

            Ok(match size.assume_init() {
                0 => None,
                size => Some(size),
            })
        }
    }

    /// This function executes convolutions or cross-correlations over `x` using a
    /// filter specified with `w`, returning results in `y`.
    ///
    /// # Arguments
    ///
    ///   * `alpha` - scaling parameter.
    ///   * `x_desc` - input map descriptor.
    ///   * `x` - input map data.
    ///   * `w_desc` - filter descriptor.
    ///   * `w` - filter data.
    ///   * `conv_desc` - convolution descriptor.
    ///   * `algo` - convolution algorithm that should be used to compute the result.
    ///   * `work_space` -  a buffer to GPU memory to a workspace needed to be able to
    ///     execute the specified algorithm. Must be left to `None` if the algorithm
    ///     works in-place. The workspace dimension can be obtained with
    ///     `get_convolution_forward_workspace_size`.
    ///   * `beta` - scaling parameter.
    ///   * `y_desc` - output map descriptor.
    ///   * `y` - output map data. It carries the result of the convolution. Scaling
    ///     factors `alpha` and `beta` can be used to scale the input tensor and the
    ///     output tensor respectively. They are used to blend the computation result
    ///     with prior value in the output layer as follows: y = alpha * result + beta *
    ///     y
    ///
    /// **Do note** than not all possible configurations of layouts and data types for
    /// the operands are supported by cuDNN. Refer to the following link for the
    /// [complete
    /// list](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward)
    /// and for in-depth explanation of the API behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid or unsupported combination of argument is passed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///     ConvDescriptor, ConvFwdAlgo, ConvMode, CudnnContext, FilterDescriptor,
    ///     ScalarC, TensorDescriptor
    /// };
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let mode = ConvMode::CrossCorrelation;
    ///
    /// let conv_desc = ConvDescriptor::<f32>::new(padding, stride, dilation, mode)?;
    ///
    /// # let data = vec![1.0_f32; 150];
    /// # let x = DeviceBuffer::from_slice(&data)?;
    /// # let w = DeviceBuffer::from_slice(&data[..24])?;
    /// # let mut y = DeviceBuffer::from_slice(&data[..144])?;
    /// let x_desc = TensorDescriptor::<f32>::new_format(&[3, 2, 5, 5,], ScalarC::Nchw)?;
    /// let w_desc = FilterDescriptor::<f32>::new(&[3, 2, 2, 2], ScalarC::Nchw)?;
    /// let y_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    ///
    /// let algo = ConvFwdAlgo::ImplicitPrecompGemm;
    ///
    /// let size = ctx.get_convolution_forward_workspace_size(
    ///     &x_desc,
    ///     &w_desc,
    ///     &y_desc,
    ///     &conv_desc,
    ///     algo,
    /// )?;
    ///
    /// let mut workspace = size.map(|size| unsafe { DeviceBuffer::<u8>::uninitialized(size).unwrap() });
    ///
    /// let alpha = 1.;
    /// let beta = 0.;
    ///
    /// ctx.convolution_forward(
    ///     alpha,
    ///     &x_desc,
    ///     &x,
    ///     &w_desc,
    ///     &w,
    ///     &conv_desc,
    ///     algo,
    ///     workspace.as_mut(),
    ///     beta,
    ///     &y_desc,
    ///     &mut y
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn convolution_forward<T1, T2, CompT, T3, W>(
        &self,
        alpha: CompT,
        x_desc: &TensorDescriptor<T1>,
        x: &impl GpuBuffer<T1>,
        w_desc: &FilterDescriptor<T2>,
        w: &impl GpuBuffer<T2>,
        conv_desc: &ConvDescriptor<CompT>,
        algo: ConvFwdAlgo,
        work_space: Option<&mut W>,
        beta: CompT,
        y_desc: &TensorDescriptor<T3>,
        y: &mut impl GpuBuffer<T3>,
    ) -> Result<(), CudnnError>
    where
        T1: DataType,
        T2: DataType,
        CompT: SupportedConv<T1, T2, T3>,
        T3: DataType,
        W: GpuBuffer<u8>,
    {
        let x_data = x.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let w_data = w.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let y_data = y.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let alpha = &alpha as *const CompT as *const std::ffi::c_void;
        let beta = &beta as *const CompT as *const std::ffi::c_void;

        // If the size is 0 then the algorithm can work in-place and cuDNN expects a null
        // pointer.
        let (work_space_ptr, work_space_size) = {
            work_space.map_or((std::ptr::null_mut(), 0), |work_space| {
                (
                    work_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void,
                    work_space.len(),
                )
            })
        };

        unsafe {
            sys::cudnnConvolutionForward(
                self.raw,
                alpha,
                x_desc.raw,
                x_data,
                w_desc.raw,
                w_data,
                conv_desc.raw,
                algo.into(),
                work_space_ptr,
                work_space_size,
                beta,
                y_desc.raw,
                y_data,
            )
            .into_result()
        }
    }

    /// This function applies a bias and then an activation to the convolutions or
    /// cross-correlation output:
    ///
    /// y = act ( alpha * conv(x) + beta * z + bias )
    ///
    /// Results are returned in y.
    ///
    /// # Arguments
    ///
    ///   * `alpha` - scaling parameter.
    ///   * `x_desc` - input map descriptor.
    ///   * `x` - input map data.
    ///   * `w_desc` - filter descriptor.
    ///   * `w` - filter data.
    ///   * `conv_desc` - convolution descriptor.
    ///   * `algo` - convolution algorithm that should be used to compute the result.
    ///   * `work_space` -  a buffer to GPU memory to a workspace needed to be able to
    ///     execute the specified algorithm. Must be left to `None` if the algorithm
    ///     works in-place. The workspace dimension can be obtained with
    ///     `get_convolution_forward_workspace_size`.
    ///   * `beta` - scaling parameter.
    ///   * `z_desc` - descriptor for the z tensor.
    ///   * `z` - data for the z tensor.
    ///   * `bias_desc` - descriptor for the bias tensor.
    ///   * `bias` - data for the bias tensor.
    ///   * `activation_desc` - neuron activation function descriptor.
    ///   * `y_desc` - output map descriptor.
    ///   * `y` - data for the output map.
    ///
    /// **Do note** that `y_desc` and `z_desc` should match.
    ///
    /// cuDNN
    /// [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward)
    /// may offer additional information about the APi behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid or unsupported combination of argument is passed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{
    ///     ActivationDescriptor, ActivationMode, ConvDescriptor, ConvFwdAlgo, ConvMode,
    ///     CudnnContext, FilterDescriptor, NanPropagation, ScalarC, TensorDescriptor
    /// };
    /// use cust::memory::DeviceBuffer;
    ///
    /// let ctx = CudnnContext::new()?;
    ///
    /// let padding = [0, 0];
    /// let stride = [1, 1];
    /// let dilation = [1, 1];
    /// let mode = ConvMode::CrossCorrelation;
    ///
    /// let conv_desc = ConvDescriptor::<f32>::new(padding, stride, dilation, mode)?;
    ///
    /// # let data = vec![1.0_f32; 150];
    /// # let x = DeviceBuffer::from_slice(&data)?;
    /// # let w = DeviceBuffer::from_slice(&data[..24])?;
    /// # let z = DeviceBuffer::from_slice(&data[..144])?;
    /// # let bias = DeviceBuffer::from_slice(&data[..3])?;
    /// # let mut y = DeviceBuffer::from_slice(&data[..144])?;
    /// let x_desc = TensorDescriptor::<f32>::new_format(&[3, 2, 5, 5,], ScalarC::Nchw)?;
    /// let w_desc = FilterDescriptor::<f32>::new(&[3, 2, 2, 2], ScalarC::Nchw)?;
    /// let y_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    ///
    /// let algo = ConvFwdAlgo::ImplicitPrecompGemm;
    ///
    /// let size = ctx.get_convolution_forward_workspace_size(
    ///     &x_desc,
    ///     &w_desc,
    ///     &y_desc,
    ///     &conv_desc,
    ///     algo,
    /// )?;
    ///
    /// let mut workspace = size.map(|size| unsafe { DeviceBuffer::<u8>::uninitialized(size).unwrap() });
    ///
    /// let alpha = 1.;
    /// let beta = 0.;
    ///
    /// let z_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    /// let bias_desc = TensorDescriptor::<f32>::new_format(&[1, 3, 1, 1], ScalarC::Nchw)?;
    ///
    /// let mode = ActivationMode::Relu;
    /// let nan_opt = NanPropagation::NotPropagateNaN;
    /// let coefficient = None;
    ///
    /// let activation_desc = ActivationDescriptor::new(mode, nan_opt, coefficient)?;
    ///
    /// ctx.convolution_bias_act_forward(
    ///     alpha,
    ///     &x_desc,
    ///     &x,
    ///     &w_desc,
    ///     &w,
    ///     &conv_desc,
    ///     algo,
    ///     workspace.as_mut(),
    ///     beta,
    ///     &z_desc,
    ///     &z,
    ///     &bias_desc,
    ///     &bias,
    ///     &activation_desc,
    ///     &y_desc,
    ///     &mut y
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn convolution_bias_act_forward<T1, T2, CompT, T3, W>(
        &self,
        alpha: CompT,
        x_desc: &TensorDescriptor<T1>,
        x: &impl GpuBuffer<T1>,
        w_desc: &FilterDescriptor<T2>,
        w: &impl GpuBuffer<T2>,
        conv_desc: &ConvDescriptor<CompT>,
        algo: ConvFwdAlgo,
        work_space: Option<&mut W>,
        beta: CompT,
        z_desc: &TensorDescriptor<T3>,
        z: &impl GpuBuffer<T3>,
        bias_desc: &TensorDescriptor<CompT>,
        bias: &impl GpuBuffer<CompT>,
        activation_desc: &ActivationDescriptor,
        y_desc: &TensorDescriptor<T3>,
        y: &mut impl GpuBuffer<T3>,
    ) -> Result<(), CudnnError>
    where
        T1: DataType,
        T2: DataType,
        CompT: SupportedConv<T1, T2, T3>,
        T3: DataType,
        W: GpuBuffer<u8>,
    {
        let x_data = x.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let w_data = w.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let z_data = z.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let bias_data = bias.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let y_data = y.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let alpha = &alpha as *const CompT as *const std::ffi::c_void;
        let beta = &beta as *const CompT as *const std::ffi::c_void;

        let (work_space_ptr, work_space_size) = {
            work_space.map_or((std::ptr::null_mut(), 0), |work_space| {
                (
                    work_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void,
                    work_space.len(),
                )
            })
        };

        unsafe {
            sys::cudnnConvolutionBiasActivationForward(
                self.raw,
                alpha,
                x_desc.raw,
                x_data,
                w_desc.raw,
                w_data,
                conv_desc.raw,
                algo.into(),
                work_space_ptr,
                work_space_size,
                beta,
                z_desc.raw,
                z_data,
                bias_desc.raw,
                bias_data,
                activation_desc.raw,
                y_desc.raw,
                y_data,
            )
            .into_result()
        }
    }

    /// This function computes the convolution data gradient of the tensor `dy`, where
    /// `y` is the output of the forward convolution in `convolution_forward`.
    ///
    /// It uses the specified algo, and returns the results in the output tensor `dx`.
    /// Scaling factors `alpha` and `beta` can be used to scale the computed result or
    /// accumulate with the current `dx`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - scaling parameter.
    /// * `w_desc` - filter descriptor.
    /// * `w` - filter data.
    /// * `dy_desc` - output map gradient descriptor.
    /// * `dy` - output map gradient data.
    /// * `conv_desc` - previously initialized convolution description. The one defined
    ///   for the forward pass in suitable to be used here provided that it refers to
    ///   the same layer.
    /// * `algo` - convolution algorithm that should be used to compute the result.
    /// * `work_space` -  a buffer to GPU memory to a workspace needed to be able to
    ///   execute the specified algorithm. Must be left to `None` if the algorithm works
    ///   in-place. The workspace dimension can be obtained with
    ///   [`get_convolution_backward_data_workspace_size()`](crate::CudnnContext::get_convolution_backward_data_workspace_size).
    /// * `beta` - scaling parameter.
    /// * `dx_desc` - input map gradient descriptor.
    /// * `dx` - input map gradient data.
    ///
    /// **Do note** than not all possible configurations of layouts and data types for
    /// the operands are supported by cuDNN. Refer to the following link for the
    /// [complete
    /// list](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardData)
    /// and for an in-depth explanation of the API behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid or unsupported combination of argument is passed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{ConvBwdDataAlgo, FilterDescriptor, ScalarC, TensorDescriptor};
    /// use cust::memory::DeviceBuffer;
    ///
    /// # use cudnn::{CudnnContext, ConvDescriptor, ConvMode};
    /// # let ctx = CudnnContext::new()?;
    /// # let padding = [0, 0];
    /// # let stride = [1, 1];
    /// # let dilation = [1, 1];
    /// # let mode = ConvMode::CrossCorrelation;
    /// # let conv_desc = ConvDescriptor::<f32>::new(padding, stride, dilation, mode)?;
    /// # let data = vec![1.0_f32; 150];
    /// # let mut dx = DeviceBuffer::from_slice(&data)?;
    /// # let w = DeviceBuffer::from_slice(&data[..24])?;
    /// # let dy = DeviceBuffer::from_slice(&data[..144])?;
    /// let dx_desc = TensorDescriptor::<f32>::new_format(&[3, 2, 5, 5,], ScalarC::Nchw)?;
    /// let w_desc = FilterDescriptor::<f32>::new(&[3, 2, 2, 2], ScalarC::Nchw)?;
    /// let dy_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    ///
    /// let algo = ConvBwdDataAlgo::Algo0;
    ///
    /// let size = ctx.get_convolution_backward_data_workspace_size(
    ///     &w_desc,
    ///     &dy_desc,
    ///     &dx_desc,
    ///     &conv_desc,
    ///     algo,
    /// )?;
    ///
    /// let mut workspace = size.map(|size| unsafe { DeviceBuffer::<u8>::uninitialized(size).unwrap() });
    ///
    /// let alpha = 1.;
    /// let beta = 0.;
    ///
    /// ctx.convolution_backward_data(
    ///     alpha,
    ///     &w_desc,
    ///     &w,
    ///     &dy_desc,
    ///     &dy,
    ///     &conv_desc,
    ///     algo,
    ///     workspace.as_mut(),
    ///     beta,
    ///     &dx_desc,
    ///     &mut dx,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn convolution_backward_data<T1, T2, CompT, T3, W>(
        &self,
        alpha: CompT,
        w_desc: &FilterDescriptor<T1>,
        w: &impl GpuBuffer<T1>,
        dy_desc: &TensorDescriptor<T2>,
        dy: &impl GpuBuffer<T2>,
        conv_desc: &ConvDescriptor<CompT>,
        algo: ConvBwdDataAlgo,
        work_space: Option<&mut W>,
        beta: CompT,
        dx_desc: &TensorDescriptor<T3>,
        dx: &mut impl GpuBuffer<T3>,
    ) -> Result<(), CudnnError>
    where
        T1: DataType,
        T2: DataType,
        CompT: SupportedConv<T1, T2, T3>,
        T3: DataType,
        W: GpuBuffer<u8>,
    {
        let w_data = w.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let dy_data = dy.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let dx_data = dx.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let alpha = &alpha as *const CompT as *const std::ffi::c_void;
        let beta = &beta as *const CompT as *const std::ffi::c_void;

        let (work_space_ptr, work_space_size) = {
            work_space.map_or((std::ptr::null_mut(), 0), |work_space| {
                (
                    work_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void,
                    work_space.len(),
                )
            })
        };

        unsafe {
            sys::cudnnConvolutionBackwardData(
                self.raw,
                alpha,
                w_desc.raw,
                w_data,
                dy_desc.raw,
                dy_data,
                conv_desc.raw,
                algo.into(),
                work_space_ptr,
                work_space_size,
                beta,
                dx_desc.raw,
                dx_data,
            )
            .into_result()
        }
    }

    /// This function computes the convolution reserve (filter) gradient of the tensor
    /// `dy`, where `y` is the output of the forward convolution in
    /// `convolution_forward()`.
    ///
    /// It uses the specified `algo`, and returns the results in the output tensor `dw`.
    /// Scaling factors `alpha` and `beta` can be used to scale the computed result or
    /// accumulate with the current `dw`.
    ///
    /// # Arguments
    ///
    ///   * `alpha` - scaling parameter.
    ///   * `x_desc` - input ma descriptor.
    ///   * `x` - input map data.
    ///   * `dy_desc` - output map gradient descriptor.
    ///   * `y` - output map gradient data.
    ///   * `conv_desc` - previously initialized convolution description. The one
    ///     defined for the forward pass in suitable to be used here provided that it
    ///     refers to the same layer.
    ///   * `algo` - convolution algorithm that should be used to compute the result.
    ///   * `work_space` -  a buffer to GPU memory to a workspace needed to be able to
    ///     execute the specified algorithm. Must be left to `None` if the algorithm
    ///     works in-place. The workspace dimension can be obtained with
    ///     [`get_convolution_backward_data_workspace_size()`](crate::CudnnContext::get_convolution_backward_data_workspace_size).
    ///   * `beta` - scaling parameter.
    ///   * `dw_desc` - filter gradient descriptor.
    ///   * `dw` - filter gradient data.
    ///
    /// **Do note** than not all possible configurations of layouts and data types for
    /// the operands are supported by cuDNN. Refer to the following link for the
    /// [complete
    /// list](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardFilter)
    /// and for an in-depth explanation of the API behavior.
    ///
    /// # Errors
    ///
    /// Returns errors if an invalid or unsupported combination of argument is passed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use cudnn::{ConvBwdFilterAlgo, FilterDescriptor, ScalarC, TensorDescriptor};
    /// use cust::memory::DeviceBuffer;
    ///
    /// # use cudnn::{CudnnContext, ConvDescriptor, ConvMode};
    /// # let ctx = CudnnContext::new()?;
    /// # let padding = [0, 0];
    /// # let stride = [1, 1];
    /// # let dilation = [1, 1];
    /// # let mode = ConvMode::CrossCorrelation;
    /// # let conv_desc = ConvDescriptor::<f32>::new(padding, stride, dilation, mode)?;
    /// # let data = vec![1.0_f32; 150];
    /// # let x = DeviceBuffer::from_slice(&data)?;
    /// # let mut dw = DeviceBuffer::from_slice(&data[..24])?;
    /// # let dy = DeviceBuffer::from_slice(&data[..144])?;
    /// let x_desc = TensorDescriptor::<f32>::new_format(&[3, 2, 5, 5,], ScalarC::Nchw)?;
    /// let dw_desc = FilterDescriptor::<f32>::new(&[3, 2, 2, 2], ScalarC::Nchw)?;
    /// let dy_desc = TensorDescriptor::<f32>::new_format(&[3, 3, 4, 4], ScalarC::Nchw)?;
    ///
    /// let algo = ConvBwdFilterAlgo::Algo0;
    ///
    /// let size = ctx.get_convolution_backward_filter_workspace_size(
    ///     &x_desc,
    ///     &dy_desc,
    ///     &dw_desc,
    ///     &conv_desc,
    ///     algo,
    /// )?;
    ///
    /// let mut workspace = size.map(|size| unsafe { DeviceBuffer::<u8>::uninitialized(size).unwrap() });
    ///
    /// let alpha = 1.;
    /// let beta = 0.;
    ///
    /// ctx.convolution_backward_filter(
    ///     alpha,
    ///     &x_desc,
    ///     &x,
    ///     &dy_desc,
    ///     &dy,
    ///     &conv_desc,
    ///     algo,
    ///     workspace.as_mut(),
    ///     beta,
    ///     &dw_desc,
    ///     &mut dw,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn convolution_backward_filter<T1, T2, CompT, T3, W>(
        &self,
        alpha: CompT,
        x_desc: &TensorDescriptor<T1>,
        x: &impl GpuBuffer<T1>,
        dy_desc: &TensorDescriptor<T2>,
        y: &impl GpuBuffer<T2>,
        conv_desc: &ConvDescriptor<CompT>,
        algo: ConvBwdFilterAlgo,
        work_space: Option<&mut W>,
        beta: CompT,
        dw_desc: &FilterDescriptor<T3>,
        dw: &mut impl GpuBuffer<T3>,
    ) -> Result<(), CudnnError>
    where
        T1: DataType,
        T2: DataType,
        CompT: SupportedConv<T1, T2, T3>,
        T3: DataType,
        W: GpuBuffer<u8>,
    {
        let x_data = x.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let dy_data = y.as_device_ptr().as_ptr() as *const std::ffi::c_void;
        let dw_data = dw.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void;

        let alpha = &alpha as *const CompT as *const std::ffi::c_void;
        let beta = &beta as *const CompT as *const std::ffi::c_void;

        let (work_space_ptr, work_space_size) = {
            work_space.map_or((std::ptr::null_mut(), 0), |work_space| {
                (
                    work_space.as_device_ptr().as_mut_ptr() as *mut std::ffi::c_void,
                    work_space.len(),
                )
            })
        };

        unsafe {
            sys::cudnnConvolutionBackwardFilter(
                self.raw,
                alpha,
                x_desc.raw,
                x_data,
                dy_desc.raw,
                dy_data,
                conv_desc.raw,
                algo.into(),
                work_space_ptr,
                work_space_size,
                beta,
                dw_desc.raw,
                dw_data,
            )
            .into_result()
        }
    }
}
