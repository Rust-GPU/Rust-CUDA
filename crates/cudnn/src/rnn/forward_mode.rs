use crate::sys;

/// Specifies inference or training mode in RNN API.
///
/// This parameter allows the cuDNN library to tune more precisely the size of the workspace buffer
/// that could be different in inference and training regimes.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnForwardMode_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForwardMode {
    /// Selects the inference mode.
    Inference,
    /// Selects the training mode.
    Training,
}

impl From<ForwardMode> for sys::cudnnForwardMode_t {
    fn from(mode: ForwardMode) -> Self {
        match mode {
            ForwardMode::Training => sys::cudnnForwardMode_t::CUDNN_FWD_MODE_TRAINING,
            ForwardMode::Inference => sys::cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
        }
    }
}
