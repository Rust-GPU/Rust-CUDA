use crate::sys;

/// Specifies the type of recurrent neural network used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnMode {
    /// A single-gate recurrent neural network with a ReLU activation function.
    RnnReLu,
    /// A single-gate recurrent neural network with a tanh activation function.
    RnnTanh,
    /// A four-gate Long Short-Term Memory (LSTM) network with no peephole connections.
    Lstm,
    /// A three-gate network consisting of Gated Recurrent Units.
    Gru,
}

impl From<sys::cudnnRNNMode_t> for RnnMode {
    fn from(raw: sys::cudnnRNNMode_t) -> Self {
        match raw {
            sys::cudnnRNNMode_t::CUDNN_RNN_RELU => Self::RnnReLu,
            sys::cudnnRNNMode_t::CUDNN_RNN_TANH => Self::RnnTanh,
            sys::cudnnRNNMode_t::CUDNN_LSTM => Self::Lstm,
            sys::cudnnRNNMode_t::CUDNN_GRU => Self::Gru,
        }
    }
}

impl From<RnnMode> for sys::cudnnRNNMode_t {
    fn from(mode: RnnMode) -> Self {
        match mode {
            RnnMode::RnnReLu => sys::cudnnRNNMode_t::CUDNN_RNN_RELU,
            RnnMode::RnnTanh => sys::cudnnRNNMode_t::CUDNN_RNN_TANH,
            RnnMode::Lstm => sys::cudnnRNNMode_t::CUDNN_LSTM,
            RnnMode::Gru => sys::cudnnRNNMode_t::CUDNN_GRU,
        }
    }
}
