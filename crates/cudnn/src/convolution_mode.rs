use crate::sys;

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

impl From<sys::cudnnConvolutionMode_t> for ConvolutionMode {
    fn from(raw: sys::cudnnConvolutionMode_t) -> Self {
        match raw {
            sys::cudnnConvolutionMode_t::CUDNN_CONVOLUTION => Self::Convolution,
            sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION => Self::CrossCorrelation,
        }
    }
}

impl From<ConvolutionMode> for sys::cudnnConvolutionMode_t {
    /// Returns the raw cuDNN type associated to the given variant.
    fn from(convolution_mode: ConvolutionMode) -> sys::cudnnConvolutionMode_t {
        match convolution_mode {
            ConvolutionMode::Convolution => sys::cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
            ConvolutionMode::CrossCorrelation => {
                sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION
            }
        }
    }
}
