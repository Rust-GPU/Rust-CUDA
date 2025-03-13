use crate::sys;

/// Specifies the recurrence pattern for a recurrent neural network.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDirectionMode_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnDirectionMode {
    /// The network iterates recurrently from the first input to the last.
    Unidirectional,
    /// Each layer of the network iterates recurrently from the first input to the last and
    /// separately from the last input to the first. The outputs of the two are concatenated at
    /// each iteration giving the output of the layer.
    Bidirectional,
}

impl From<RnnDirectionMode> for sys::cudnnDirectionMode_t {
    fn from(mode: RnnDirectionMode) -> Self {
        match mode {
            RnnDirectionMode::Unidirectional => Self::CUDNN_UNIDIRECTIONAL,
            RnnDirectionMode::Bidirectional => Self::CUDNN_BIDIRECTIONAL,
        }
    }
}
