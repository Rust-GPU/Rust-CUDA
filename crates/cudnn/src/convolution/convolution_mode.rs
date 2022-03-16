use crate::sys;

/// Enum used to configure a convolution descriptor.
///
/// The filter used for the convolution can be applied in two different ways, corresponding
/// mathematically to a convolution or to a cross-correlation.
///
/// A cross-correlation is equivalent to a convolution with its filter rotated by 180 degrees.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionMode_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvMode {
    /// Convolution operation.
    Convolution,
    /// Cross Correlation operation.
    CrossCorrelation,
}

impl From<ConvMode> for sys::cudnnConvolutionMode_t {
    fn from(convolution_mode: ConvMode) -> sys::cudnnConvolutionMode_t {
        match convolution_mode {
            ConvMode::Convolution => sys::cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
            ConvMode::CrossCorrelation => sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        }
    }
}
