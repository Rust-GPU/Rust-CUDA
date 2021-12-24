use crate::sys;

/// Specifies the recurrence pattern for a recurrent neural network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnDirectionMode {
    /// The network iterates recurrently from the first input to the last.
    Unidirectional,
    /// Each layer of the network iterates recurrently from the first input to the last and
    /// separately from the last input to the first. The outputs of the two are concatenated at
    /// each iteration giving the output of the layer.
    Bidirectional,
}

impl From<sys::cudnnDirectionMode_t> for RnnDirectionMode {
    fn from(raw: sys::cudnnDirectionMode_t) -> Self {
        match raw {
            sys::cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL => Self::Unidirectional,
            sys::cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL => Self::Bidirectional,
        }
    }
}

impl From<RnnDirectionMode> for sys::cudnnDirectionMode_t {
    fn from(mode: RnnDirectionMode) -> Self {
        match mode {
            RnnDirectionMode::Unidirectional => sys::cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL,
            RnnDirectionMode::Bidirectional => sys::cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL,
        }
    }
}
