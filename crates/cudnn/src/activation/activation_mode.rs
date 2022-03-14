use crate::sys;

/// Specifies a neuron activation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationMode {
    /// Selects the sigmoid function.
    Sigmoid,
    /// Selects the rectified linear function.
    Relu,
    /// Selects the hyperbolic tangent function.
    Tanh,
    /// Selects the clipped rectified linear function.
    ClippedRelu,
    /// Selects the exponential linear function.
    Elu,
    /// Selects the swish function.
    Swish,
}

impl From<ActivationMode> for sys::cudnnActivationMode_t {
    fn from(mode: ActivationMode) -> Self {
        match mode {
            ActivationMode::Sigmoid => Self::CUDNN_ACTIVATION_SIGMOID,
            ActivationMode::Relu => Self::CUDNN_ACTIVATION_RELU,
            ActivationMode::Tanh => Self::CUDNN_ACTIVATION_TANH,
            ActivationMode::ClippedRelu => Self::CUDNN_ACTIVATION_CLIPPED_RELU,
            ActivationMode::Elu => Self::CUDNN_ACTIVATION_ELU,
            ActivationMode::Swish => Self::CUDNN_ACTIVATION_SWISH,
        }
    }
}
