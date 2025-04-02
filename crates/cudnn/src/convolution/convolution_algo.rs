use crate::{CudnnError, Determinism, IntoResult, MathType};

/// The best suited algorithm according to the layer specifications obtained through a heuristic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BestHeuristic<T>
where
    T: Copy,
{
    algo: T,
    time: f32,
    workspace_size: usize,
    determinism: Determinism,
    math_type: MathType,
}

impl<T> BestHeuristic<T>
where
    T: Copy,
{
    /// Return the contained algorithm.
    pub fn algo(&self) -> T {
        self.algo
    }

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

/// Convolution forward algorithms as listed in the [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionFwdAlgo_t).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvFwdAlgo {
    /// This algorithm expresses the convolution as a matrix product without actually explicitly
    /// forming the matrix that holds the input tensor data.
    ImplicitGemm,
    /// This algorithm expresses convolution as a matrix product without actually explicitly forming
    /// the matrix that holds the input tensor data, but still needs some memory workspace to
    /// pre-compute some indices in order to facilitate the implicit construction of the matrix that
    /// holds the input tensor data.
    ImplicitPrecompGemm,
    /// This algorithm expresses the convolution as an explicit matrix product. A significant memory
    /// workspace is needed to store the matrix that holds the input tensor data.
    Gemm,
    /// This algorithm expresses the convolution as a direct convolution (for example, without
    /// implicitly or explicitly doing a matrix multiplication).
    ///
    /// **Do note** that this is currently not implemented in cuDNN.
    Direct,
    /// This algorithm uses the Fast-Fourier Transform approach to compute the convolution. A
    /// significant memory workspace is needed to store intermediate results.
    Fft,
    /// This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
    /// A significant memory workspace is needed to store intermediate results, but less than
    /// [`ConvFwdAlgo::Fft`], for large size images.
    FftTiling,
    /// This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
    /// sized workspace is needed to store intermediate results.
    Winograd,
    /// This algorithm uses the Winograd Transform approach to compute the convolution. A
    /// significant workspace may be needed to store intermediate results.
    WinogradNonFused,
}

impl From<ConvFwdAlgo> for cudnn_sys::cudnnConvolutionFwdAlgo_t {
    fn from(algo: ConvFwdAlgo) -> Self {
        match algo {
            ConvFwdAlgo::ImplicitGemm => Self::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            ConvFwdAlgo::ImplicitPrecompGemm => {
                Self::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            }
            ConvFwdAlgo::Gemm => Self::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            ConvFwdAlgo::Direct => Self::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            ConvFwdAlgo::Fft => Self::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            ConvFwdAlgo::FftTiling => Self::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
            ConvFwdAlgo::Winograd => Self::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            ConvFwdAlgo::WinogradNonFused => Self::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        }
    }
}

impl From<cudnn_sys::cudnnConvolutionFwdAlgo_t> for ConvFwdAlgo {
    fn from(algo: cudnn_sys::cudnnConvolutionFwdAlgo_t) -> Self {
        use cudnn_sys::cudnnConvolutionFwdAlgo_t::*;
        match algo {
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => ConvFwdAlgo::ImplicitGemm,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => ConvFwdAlgo::ImplicitPrecompGemm,
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM => ConvFwdAlgo::Gemm,
            CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => ConvFwdAlgo::Direct,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT => ConvFwdAlgo::Fft,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => ConvFwdAlgo::FftTiling,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => ConvFwdAlgo::Winograd,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => ConvFwdAlgo::WinogradNonFused,
            CUDNN_CONVOLUTION_FWD_ALGO_COUNT => unreachable!(),
        }
    }
}

/// BestHeuristic for the forward convolution algorithm.
impl TryFrom<cudnn_sys::cudnnConvolutionFwdAlgoPerf_t> for BestHeuristic<ConvFwdAlgo> {
    type Error = CudnnError;

    fn try_from(raw: cudnn_sys::cudnnConvolutionFwdAlgoPerf_t) -> Result<Self, Self::Error> {
        let cudnn_sys::cudnnConvolutionFwdAlgoPerf_t {
            algo,
            status,
            time,
            memory,
            determinism,
            mathType,
            ..
        } = raw;
        status.into_result().map(|_| Self {
            algo: algo.into(),
            time,
            workspace_size: memory,
            determinism: Determinism::from(determinism),
            math_type: mathType.into(),
        })
    }
}

/// Convolution backward data algorithms as listed in the [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBwdDataAlgo_t).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvBwdDataAlgo {
    /// This algorithm expresses the convolution as a sum of matrix products without actually explicitly
    /// forming the matrix that holds the input tensor data. The sum is done using the atomic add
    /// operation, thus the results are non-deterministic.
    Algo0,
    /// This algorithm expresses the convolution as a matrix product without actually explicitly forming
    /// the matrix that holds the input tensor data. The results are deterministic.
    Algo1,
    /// This algorithm uses the Fast-Fourier Transform approach to compute the convolution. A
    /// significant memory workspace is needed to store intermediate results.
    Fft,
    /// This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
    /// A significant memory workspace is needed to store intermediate results, but less than
    /// [`ConvBwdDataAlgo::Fft`], for large size images.
    FftTiling,
    /// This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
    /// sized workspace is needed to store intermediate results.
    Winograd,
    /// This algorithm uses the Winograd Transform approach to compute the convolution. A
    /// significant workspace may be needed to store intermediate results.
    WinogradNonFused,
}

impl From<ConvBwdDataAlgo> for cudnn_sys::cudnnConvolutionBwdDataAlgo_t {
    fn from(algo: ConvBwdDataAlgo) -> Self {
        match algo {
            ConvBwdDataAlgo::Algo0 => Self::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            ConvBwdDataAlgo::Algo1 => Self::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
            ConvBwdDataAlgo::Fft => Self::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
            ConvBwdDataAlgo::FftTiling => Self::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
            ConvBwdDataAlgo::Winograd => Self::CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
            ConvBwdDataAlgo::WinogradNonFused => {
                Self::CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
            }
        }
    }
}

impl From<cudnn_sys::cudnnConvolutionBwdDataAlgo_t> for ConvBwdDataAlgo {
    fn from(algo: cudnn_sys::cudnnConvolutionBwdDataAlgo_t) -> Self {
        use cudnn_sys::cudnnConvolutionBwdDataAlgo_t::*;
        match algo {
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 => Self::Algo0,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 => Self::Algo1,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT => Self::Fft,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING => Self::FftTiling,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD => Self::Winograd,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED => Self::WinogradNonFused,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT => unreachable!(),
        }
    }
}

/// BestHeuristic for the backward data convolution algorithm.
impl TryFrom<cudnn_sys::cudnnConvolutionBwdDataAlgoPerf_t> for BestHeuristic<ConvBwdDataAlgo> {
    type Error = CudnnError;

    fn try_from(raw: cudnn_sys::cudnnConvolutionBwdDataAlgoPerf_t) -> Result<Self, Self::Error> {
        let cudnn_sys::cudnnConvolutionBwdDataAlgoPerf_t {
            algo,
            status,
            time,
            memory,
            determinism,
            mathType,
            ..
        } = raw;
        status.into_result().map(|_| Self {
            algo: algo.into(),
            time,
            workspace_size: memory,
            determinism: Determinism::from(determinism),
            math_type: mathType.into(),
        })
    }
}

/// Convolution backward filter algorithms as listed in the [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBwdFilterAlgo_t).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvBwdFilterAlgo {
    /// This algorithm expresses the convolution as a sum of matrix products without actually explicitly
    /// forming the matrix that holds the input tensor data. The sum is done using the atomic add
    /// operation, thus the results are non-deterministic.
    Algo0,
    /// This algorithm expresses the convolution as a matrix product without actually explicitly forming
    /// the matrix that holds the input tensor data. The results are deterministic.
    Algo1,
    /// This algorithm is similar to `Algo0` but uses some small workspace to pre-compute some
    /// indices. The results are also non-deterministic.
    Algo3,
    /// This algorithm uses the Fast-Fourier Transform approach to compute the convolution. A
    /// significant memory workspace is needed to store intermediate results.
    Fft,
    /// This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
    /// A significant memory workspace is needed to store intermediate results, but less than
    /// [`ConvBwdFilterAlgo::Fft`], for large size images.
    FftTiling,
    /// This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
    /// sized workspace is needed to store intermediate results.
    Winograd,
    /// This algorithm uses the Winograd Transform approach to compute the convolution. A
    /// significant workspace may be needed to store intermediate results.
    WinogradNonFused,
}

impl From<ConvBwdFilterAlgo> for cudnn_sys::cudnnConvolutionBwdFilterAlgo_t {
    fn from(algo: ConvBwdFilterAlgo) -> Self {
        match algo {
            ConvBwdFilterAlgo::Algo0 => {
                cudnn_sys::cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
            }
            ConvBwdFilterAlgo::Algo1 => {
                cudnn_sys::cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
            }
            ConvBwdFilterAlgo::Algo3 => {
                cudnn_sys::cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3
            }
            ConvBwdFilterAlgo::Fft => {
                cudnn_sys::cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT
            }
            ConvBwdFilterAlgo::FftTiling => {
                cudnn_sys::cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING
            }
            ConvBwdFilterAlgo::Winograd => {
                cudnn_sys::cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD
            }
            ConvBwdFilterAlgo::WinogradNonFused => {
                cudnn_sys::cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED
            }
        }
    }
}

impl From<cudnn_sys::cudnnConvolutionBwdFilterAlgo_t> for ConvBwdFilterAlgo {
    fn from(algo: cudnn_sys::cudnnConvolutionBwdFilterAlgo_t) -> Self {
        use cudnn_sys::cudnnConvolutionBwdFilterAlgo_t::*;
        match algo {
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 => Self::Algo0,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 => Self::Algo1,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 => Self::Algo3,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT => Self::Fft,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING => Self::FftTiling,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD => Self::Winograd,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED => Self::WinogradNonFused,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT => unreachable!(),
        }
    }
}

/// BestHeuristic for the backward filter convolution algorithm.
impl TryFrom<cudnn_sys::cudnnConvolutionBwdFilterAlgoPerf_t> for BestHeuristic<ConvBwdFilterAlgo> {
    type Error = CudnnError;

    fn try_from(raw: cudnn_sys::cudnnConvolutionBwdFilterAlgoPerf_t) -> Result<Self, Self::Error> {
        let cudnn_sys::cudnnConvolutionBwdFilterAlgoPerf_t {
            algo,
            status,
            time,
            memory,
            determinism,
            mathType,
            ..
        } = raw;
        status.into_result().map(|_| Self {
            algo: algo.into(),
            time,
            workspace_size: memory,
            determinism: Determinism::from(determinism),
            math_type: mathType.into(),
        })
    }
}
