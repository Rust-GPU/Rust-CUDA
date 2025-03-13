use crate::sys;

/// Specifies the behavior of the first layer in a recurrent neural network.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNInputMode_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnInputMode {
    /// A biased matrix multiplication is performed at the input of the first recurrent layer.
    LinearInput,
    /// No operation is performed at the input of the first recurrent layer. If `SkipInput` is used
    /// the leading dimension of the input tensor must be equal to the hidden state size of the
    /// network.
    SkipInput,
}

impl From<RnnInputMode> for sys::cudnnRNNInputMode_t {
    fn from(mode: RnnInputMode) -> Self {
        match mode {
            RnnInputMode::LinearInput => sys::cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
            RnnInputMode::SkipInput => sys::cudnnRNNInputMode_t::CUDNN_SKIP_INPUT,
        }
    }
}
