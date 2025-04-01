/// Specifies the type of recurrent neural network used.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNMode_t)
/// may offer additional information about the APi behavior.
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

impl From<RnnMode> for cudnn_sys::cudnnRNNMode_t {
    fn from(mode: RnnMode) -> Self {
        use cudnn_sys::cudnnRNNMode_t::*;
        match mode {
            RnnMode::RnnReLu => CUDNN_RNN_RELU,
            RnnMode::RnnTanh => CUDNN_RNN_TANH,
            RnnMode::Lstm => CUDNN_LSTM,
            RnnMode::Gru => CUDNN_GRU,
        }
    }
}
