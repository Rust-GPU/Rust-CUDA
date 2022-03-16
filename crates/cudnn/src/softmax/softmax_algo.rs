use crate::sys;

/// Specifies the implementation of the softmax function.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxAlgorithm_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoftmaxAlgo {
    /// This implementation applies the straightforward softmax operation.
    Fast,
    /// This implementation scales each point of the softmax input domain by its maximum value
    /// to avoid potential floating point overflows in the softmax evaluation.
    Accurate,
    /// This entry performs the log softmax operation, avoiding overflows by scaling each point in
    /// the input domain as in the accurate version.
    Log,
}

impl From<SoftmaxAlgo> for sys::cudnnSoftmaxAlgorithm_t {
    fn from(algo: SoftmaxAlgo) -> Self {
        match algo {
            SoftmaxAlgo::Fast => Self::CUDNN_SOFTMAX_FAST,
            SoftmaxAlgo::Accurate => Self::CUDNN_SOFTMAX_ACCURATE,
            SoftmaxAlgo::Log => Self::CUDNN_SOFTMAX_ACCURATE,
        }
    }
}
