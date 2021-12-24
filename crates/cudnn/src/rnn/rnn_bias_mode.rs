use crate::sys;

/// Specifies the number of bias vectors for a recurrent neural network function.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnBiasMode {
    /// Applies a recurrent neural network cell formula that does not use biases.
    NoBias,
    /// Applies a recurrent neural network cell formula that uses one input bias vector in the
    /// input GEMM.
    SingleInpBias,
    /// Applies a recurrent neural network cell formula that uses uses two bias vectors.
    DoubleBias,
    /// Applies a recurrent neural network cell formula that uses one recurrent bias vector in the
    /// recurrent GEMM.
    SingleRecurrentBias,
}

impl From<sys::cudnnRNNBiasMode_t> for RnnBiasMode {
    fn from(raw: sys::cudnnRNNBiasMode_t) -> Self {
        match raw {
            sys::cudnnRNNBiasMode_t::CUDNN_RNN_NO_BIAS => Self::NoBias,
            sys::cudnnRNNBiasMode_t::CUDNN_RNN_SINGLE_INP_BIAS => Self::SingleInpBias,
            sys::cudnnRNNBiasMode_t::CUDNN_RNN_DOUBLE_BIAS => Self::DoubleBias,
            sys::cudnnRNNBiasMode_t::CUDNN_RNN_SINGLE_REC_BIAS => Self::SingleRecurrentBias,
        }
    }
}

impl From<RnnBiasMode> for sys::cudnnRNNBiasMode_t {
    fn from(mode: RnnBiasMode) -> Self {
        match mode {
            RnnBiasMode::NoBias => sys::cudnnRNNBiasMode_t::CUDNN_RNN_NO_BIAS,
            RnnBiasMode::SingleInpBias => sys::cudnnRNNBiasMode_t::CUDNN_RNN_SINGLE_INP_BIAS,
            RnnBiasMode::DoubleBias => sys::cudnnRNNBiasMode_t::CUDNN_RNN_DOUBLE_BIAS,
            RnnBiasMode::SingleRecurrentBias => sys::cudnnRNNBiasMode_t::CUDNN_RNN_SINGLE_REC_BIAS,
        }
    }
}
