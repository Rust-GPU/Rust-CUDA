/// Specifies the number of bias vectors for a recurrent neural network function.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNBiasMode_t)
/// may offer additional information about the APi behavior.
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

impl From<RnnBiasMode> for cudnn_sys::cudnnRNNBiasMode_t {
    fn from(mode: RnnBiasMode) -> Self {
        use cudnn_sys::cudnnRNNBiasMode_t::*;
        match mode {
            RnnBiasMode::NoBias => CUDNN_RNN_NO_BIAS,
            RnnBiasMode::SingleInpBias => CUDNN_RNN_SINGLE_INP_BIAS,
            RnnBiasMode::DoubleBias => CUDNN_RNN_DOUBLE_BIAS,
            RnnBiasMode::SingleRecurrentBias => CUDNN_RNN_SINGLE_REC_BIAS,
        }
    }
}
