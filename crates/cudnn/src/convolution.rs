use crate::{private, sys};

/// Enum used to configure a convolution descriptor. The filter used for the convolution can be
/// applied in two different ways, corresponding mathematically to a convolution or to a
/// cross-correlation. A cross-correlation is equivalent to a convolution with its filter
/// rotated by 180 degrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvolutionMode {
    /// Convolution operation.
    Convolution,
    /// Cross Correlation operation.
    CrossCorrelation,
}

impl ConvolutionMode {
    /// Returns the raw cuDNN type associated to the given variant.
    pub fn into_raw(&self) -> sys::cudnnConvolutionMode_t {
        match self {
            Self::Convolution => sys::cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
            Self::CrossCorrelation => sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        }
    }
}

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
pub struct AlgoGemm;

/// This algorithm expresses the convolution as a direct convolution (for example, without
/// implicitly or explicitly doing a matrix multiplication).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlgoDirect;

/// This algorithm uses the Fast-Fourier Transform approach to compute the convolution. A
/// significant memory workspace is needed to store intermediate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlgoFft;

/// This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
/// A significant memory workspace is needed to store intermediate results, but less than
/// [`AlgoFft`], for large size images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlgoFftTiling;

/// This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
/// sized workspace is needed to store intermediate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlgoWinograd;

/// This algorithm uses the Winograd Transform approach to compute the convolution. A significant
/// workspace may be needed to store intermediate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgoWinogradNonFused;

pub trait ConvolutionFwdAlgo: private::Sealed {
    fn into_raw() -> sys::cudnnConvolutionFwdAlgo_t;
}

macro_rules! impl_convolution_fwd_algo {
    ($safe_type:ident, $raw_type:ident) => {
        impl ConvolutionFwdAlgo for $safe_type {
            fn into_raw() -> sys::cudnnConvolutionFwdAlgo_t {
                sys::cudnnConvolutionFwdAlgo_t::$raw_type
            }
        }
    };
}

pub trait SupportedConvFwd {}
