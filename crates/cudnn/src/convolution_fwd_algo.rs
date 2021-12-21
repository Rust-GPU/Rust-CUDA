use crate::{
    data_type::DataType,
    determinism::Determinism,
    error::{CudnnError, IntoResult},
    math_type::MathType,
    private, sys,
    tensor_format::{NCHWVectC8x32, NCHWVectC8x4, TensorFormat, NCHW, NHWC},
};

// Convolution Forward Algorithms (as listed in the docs at:
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionFwdAlgo_t)

/// This algorithm expresses the convolution as a matrix product without actually explicitly
/// forming the matrix that holds the input tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplicitGemm;

/// This algorithm expresses convolution as a matrix product without actually explicitly forming
/// the matrix that holds the input tensor data, but still needs some memory workspace to
/// pre-compute some indices in order to facilitate the implicit construction of the matrix that
/// holds the input tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplicitPrecompGemm;

/// This algorithm expresses the convolution as an explicit matrix product. A significant memory
/// workspace is needed to store the matrix that holds the input tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Gemm;

/// This algorithm expresses the convolution as a direct convolution (for example, without
/// implicitly or explicitly doing a matrix multiplication).
///
/// **Do note** that this is currently not implemented in cuDNN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Direct;

/// This algorithm uses the Fast-Fourier Transform approach to compute the convolution. A
/// significant memory workspace is needed to store intermediate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft;

/// This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
/// A significant memory workspace is needed to store intermediate results, but less than
/// [`Fft`], for large size images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FftTiling;

/// This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
/// sized workspace is needed to store intermediate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Winograd;

/// This algorithm uses the Winograd Transform approach to compute the convolution. A significant
/// workspace may be needed to store intermediate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WinogradNonFused;

/// The best suited algorithm according to the layer specifications obtained through a heuristic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BestHeuristic {
    raw: sys::cudnnConvolutionFwdAlgo_t,
    time: f32,
    workspace_size: usize,
    determinism: Determinism,
    math_type: MathType,
}

impl TryFrom<sys::cudnnConvolutionFwdAlgoPerf_t> for BestHeuristic {
    type Error = CudnnError;

    fn try_from(raw: sys::cudnnConvolutionFwdAlgoPerf_t) -> Result<Self, Self::Error> {
        let sys::cudnnConvolutionFwdAlgoPerf_t {
            algo,
            status,
            time,
            memory,
            determinism,
            mathType,
            ..
        } = raw;
        status.into_result().map(|_| Self {
            raw: algo,
            time,
            workspace_size: memory,
            determinism: Determinism::from(determinism),
            math_type: mathType.into(),
        })
    }
}

// ConvolutionFwdAlgo implementations.

pub trait ConvolutionFwdAlgo: private::Sealed {
    fn into_raw(&self) -> sys::cudnnConvolutionFwdAlgo_t;
}

macro_rules! impl_convolution_fwd_algo {
    ($safe_type:ident, $raw_type:ident) => {
        impl private::Sealed for $safe_type {}

        impl ConvolutionFwdAlgo for $safe_type {
            fn into_raw(&self) -> sys::cudnnConvolutionFwdAlgo_t {
                sys::cudnnConvolutionFwdAlgo_t::$raw_type
            }
        }
    };
}

impl_convolution_fwd_algo!(ImplicitGemm, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
#[rustfmt::skip]
impl_convolution_fwd_algo!(ImplicitPrecompGemm,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
impl_convolution_fwd_algo!(Gemm, CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
impl_convolution_fwd_algo!(Direct, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT);
impl_convolution_fwd_algo!(Fft, CUDNN_CONVOLUTION_FWD_ALGO_FFT);
impl_convolution_fwd_algo!(FftTiling, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING);
impl_convolution_fwd_algo!(Winograd, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);
impl_convolution_fwd_algo!(WinogradNonFused, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);

impl private::Sealed for BestHeuristic {}

impl ConvolutionFwdAlgo for BestHeuristic {
    fn into_raw(&self) -> sys::cudnnConvolutionFwdAlgo_t {
        self.raw
    }
}

impl BestHeuristic {
    /// Returns the math type associated to the optimal algorithm.
    pub fn math_type(&self) -> MathType {
        self.math_type
    }

    /// Returns the workspace size associated to the optimal algorithm.
    pub fn workspace_size(&self) -> usize {
        self.workspace_size
    }

    /// Returns the determinism of the optimal algorithm.
    pub fn determinism(&self) -> Determinism {
        self.determinism
    }
}

pub trait SupportedConvFwd<
    InType,
    InFmt,
    FilterType,
    FilterFmt,
    CompType,
    OutType,
    OutFmt,
    const D: usize,
    const N: usize,
> where
    Self: ConvolutionFwdAlgo,
    InType: DataType,
    InFmt: TensorFormat,
    FilterType: DataType,
    FilterFmt: TensorFormat,
    CompType: DataType,
    OutType: DataType,
    OutFmt: TensorFormat,
{
}

macro_rules! impl_supported_conv_fwd {
    ($conv_fwd_algo:ty, $in_type:ty, $in_fmt:ty, $filter_type:ty, $filter_fmt:ty, $comp_type:ty, $out_type:ty, $out_fmt:ty, $dim_operands:expr, $dim_conv:expr) => {
        impl
            SupportedConvFwd<
                $in_type,
                $in_fmt,
                $filter_type,
                $filter_fmt,
                $comp_type,
                $out_type,
                $out_fmt,
                $dim_operands,
                $dim_conv,
            > for $conv_fwd_algo
        {
        }
    };
}

// Admissible configuration for the forward convolution (as specified in
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward)

#[rustfmt::skip]
mod supported_conv_fwd_impls {
    use super::*;
    /// ImplicitGemm supported configurations for 2-d convolutions and filter format equal to NCHW.
    impl_supported_conv_fwd!(ImplicitGemm, f32, NCHW, f32, NCHW, f32, f32, NCHW, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, f32, NHWC, f32, NCHW, f32, f32, NCHW, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, f32, NCHW, f32, NCHW, f32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, f32, NHWC, f32, NCHW, f32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, f64, NCHW, f64, NCHW, f64, f64, NCHW, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, f64, NHWC, f64, NCHW, f64, f64, NCHW, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, f64, NCHW, f64, NCHW, f64, f64, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, f64, NHWC, f64, NCHW, f64, f64, NHWC, 4, 2);

    /// ImplicitPrecompGemm supported configurations for 2-d convolutions and filter format equal to
    /// NCHW.
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f32, NCHW, f32, NCHW, f32, f32, NCHW, 4, 2);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f32, NHWC, f32, NCHW, f32, f32, NCHW, 4, 2);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f32, NCHW, f32, NCHW, f32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f32, NHWC, f32, NCHW, f32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f64, NCHW, f64, NCHW, f64, f64, NCHW, 4, 2);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f64, NHWC, f64, NCHW, f64, f64, NCHW, 4, 2);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f64, NCHW, f64, NCHW, f64, f64, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f64, NHWC, f64, NCHW, f64, f64, NHWC, 4, 2);

    /// Gemm supported configurations for 2-d convolutions and filter format equal to NCHW.
    impl_supported_conv_fwd!(Gemm, f32, NCHW, f32, NCHW, f32, f32, NCHW, 4, 2);
    impl_supported_conv_fwd!(Gemm, f32, NHWC, f32, NCHW, f32, f32, NCHW, 4, 2);
    impl_supported_conv_fwd!(Gemm, f32, NCHW, f32, NCHW, f32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(Gemm, f32, NHWC, f32, NCHW, f32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(Gemm, f64, NCHW, f64, NCHW, f64, f64, NCHW, 4, 2);
    impl_supported_conv_fwd!(Gemm, f64, NHWC, f64, NCHW, f64, f64, NCHW, 4, 2);
    impl_supported_conv_fwd!(Gemm, f64, NCHW, f64, NCHW, f64, f64, NHWC, 4, 2);
    impl_supported_conv_fwd!(Gemm, f64, NHWC, f64, NCHW, f64, f64, NHWC, 4, 2);

    /// Fft supported configurations for 2-d convolutions and filter format equal to NCHW.
    impl_supported_conv_fwd!(Fft, f32, NCHW, f32, NCHW, f32, f32, NCHW, 4, 2);

    /// Fft supported configurations for 2-d convolutions and filter format equal to NCHW.
    impl_supported_conv_fwd!(FftTiling, f32, NCHW, f32, NCHW, f32, f32, NCHW, 4, 2);
    impl_supported_conv_fwd!(FftTiling, f64, NCHW, f64, NCHW, f64, f64, NCHW, 4, 2);

    /// 2-d convolutions supported with NCHWVectC memory format.
    impl_supported_conv_fwd!(ImplicitGemm, i8, NCHWVectC8x4, i8, NCHWVectC8x4, i8, i8, NCHWVectC8x4, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, u8, NCHWVectC8x4, u8, NCHWVectC8x4, u8, u8, NCHWVectC8x4, 4, 2);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, i8, NCHWVectC8x32, i8, NCHWVectC8x32, i8, i8, NCHWVectC8x32, 4, 2);

    /// 2-d convolutions supported with NHWC memory format.
    impl_supported_conv_fwd!(ImplicitGemm, i8, NHWC, i8, NHWC, i32, i8, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, i8, NHWC, i8, NHWC, i32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, u8, NHWC, u8, NHWC, i32, u8, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, i8, NHWC, u8, NHWC, i32, u8, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, u8, NHWC, u8, NHWC, i32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitGemm, i8, NHWC, u8, NHWC, i32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f32, NHWC, f32, NHWC, f32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f64, NHWC, f64, NHWC, f64, f64, NHWC, 4, 2);

    /// ImplicitGemm supported configurations for 3-d convolutions and filter format equal to NCHW.
    impl_supported_conv_fwd!(ImplicitGemm, f32, NCHW, f32, NCHW, f32, f32, NCHW, 5, 3);
    impl_supported_conv_fwd!(ImplicitGemm, f32, NHWC, f32, NCHW, f32, f32, NCHW, 5, 3);
    impl_supported_conv_fwd!(ImplicitGemm, f32, NCHW, f32, NCHW, f32, f32, NHWC, 5, 3);
    impl_supported_conv_fwd!(ImplicitGemm, f32, NHWC, f32, NCHW, f32, f32, NHWC, 5, 3);
    impl_supported_conv_fwd!(ImplicitGemm, f64, NCHW, f64, NCHW, f64, f64, NCHW, 5, 3);
    impl_supported_conv_fwd!(ImplicitGemm, f64, NHWC, f64, NCHW, f64, f64, NCHW, 5, 3);
    impl_supported_conv_fwd!(ImplicitGemm, f64, NCHW, f64, NCHW, f64, f64, NHWC, 5, 3);
    impl_supported_conv_fwd!(ImplicitGemm, f64, NHWC, f64, NCHW, f64, f64, NHWC, 5, 3);

    /// ImplicitPrecompGemm supported configurations for 3-d convolutions and filter format equal to
    /// NCHW.
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f32, NCHW, f32, NCHW, f32, f32, NCHW, 5, 3);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f32, NHWC, f32, NCHW, f32, f32, NCHW, 5, 3);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f32, NCHW, f32, NCHW, f32, f32, NHWC, 5, 3);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f32, NHWC, f32, NCHW, f32, f32, NHWC, 5, 3);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f64, NCHW, f64, NCHW, f64, f64, NCHW, 5, 3);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f64, NHWC, f64, NCHW, f64, f64, NCHW, 5, 3);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f64, NCHW, f64, NCHW, f64, f64, NHWC, 5, 3);
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f64, NHWC, f64, NCHW, f64, f64, NHWC, 5, 3);

    /// Fft supported configurations for 3-d convolutions and filter format equal to NCHW.
    impl_supported_conv_fwd!(FftTiling, f32, NCHW, f32, NCHW, f32, f32, NCHW, 5, 3);
    impl_supported_conv_fwd!(FftTiling, f64, NCHW, f64, NCHW, f64, f64, NCHW, 5, 3);

    /// Supported configuration for 3-d convolution with filter format equal to NHWC.
    impl_supported_conv_fwd!(ImplicitPrecompGemm, f32, NHWC, f32, NHWC, f32, f32, NHWC, 5, 3);

    /// BestHeuristic supported configurations. Its the set union of all those above.
    impl_supported_conv_fwd!(BestHeuristic, f32, NCHW, f32, NCHW, f32, f32, NCHW, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, f32, NHWC, f32, NCHW, f32, f32, NCHW, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, f32, NCHW, f32, NCHW, f32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, f32, NHWC, f32, NCHW, f32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, f64, NCHW, f64, NCHW, f64, f64, NCHW, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, f64, NHWC, f64, NCHW, f64, f64, NCHW, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, f64, NCHW, f64, NCHW, f64, f64, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, f64, NHWC, f64, NCHW, f64, f64, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, i8, NCHWVectC8x4, i8, NCHWVectC8x4, i8, i8, NCHWVectC8x4, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, u8, NCHWVectC8x4, u8, NCHWVectC8x4, u8, u8, NCHWVectC8x4, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, i8, NCHWVectC8x32, i8, NCHWVectC8x32, i8, i8, NCHWVectC8x32, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, i8, NHWC, i8, NHWC, i32, i8, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, i8, NHWC, i8, NHWC, i32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, u8, NHWC, u8, NHWC, i32, u8, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, i8, NHWC, u8, NHWC, i32, u8, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, u8, NHWC, u8, NHWC, i32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, i8, NHWC, u8, NHWC, i32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, f32, NHWC, f32, NHWC, f32, f32, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, f64, NHWC, f64, NHWC, f64, f64, NHWC, 4, 2);
    impl_supported_conv_fwd!(BestHeuristic, f32, NCHW, f32, NCHW, f32, f32, NCHW, 5, 3);
    impl_supported_conv_fwd!(BestHeuristic, f32, NHWC, f32, NCHW, f32, f32, NCHW, 5, 3);
    impl_supported_conv_fwd!(BestHeuristic, f32, NCHW, f32, NCHW, f32, f32, NHWC, 5, 3);
    impl_supported_conv_fwd!(BestHeuristic, f32, NHWC, f32, NCHW, f32, f32, NHWC, 5, 3);
    impl_supported_conv_fwd!(BestHeuristic, f64, NCHW, f64, NCHW, f64, f64, NCHW, 5, 3);
    impl_supported_conv_fwd!(BestHeuristic, f64, NHWC, f64, NCHW, f64, f64, NCHW, 5, 3);
    impl_supported_conv_fwd!(BestHeuristic, f64, NCHW, f64, NCHW, f64, f64, NHWC, 5, 3);
    impl_supported_conv_fwd!(BestHeuristic, f64, NHWC, f64, NCHW, f64, f64, NHWC, 5, 3);
    impl_supported_conv_fwd!(BestHeuristic, f32, NHWC, f32, NHWC, f32, f32, NHWC, 5, 3);
}
