use crate::sys;

/// Selects the LSTM cell clipping mode.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNClipMode_t)
/// may offer additional information about the APi behavior.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnClipMode {
    /// Disables LSTM cell clipping.
    ClipNone,
    /// Enables LSTM cell clipping.
    ClipMinMax,
}

impl From<RnnClipMode> for sys::cudnnRNNClipMode_t {
    fn from(mode: RnnClipMode) -> Self {
        match mode {
            RnnClipMode::ClipNone => sys::cudnnRNNClipMode_t::CUDNN_RNN_CLIP_NONE,
            RnnClipMode::ClipMinMax => sys::cudnnRNNClipMode_t::CUDNN_RNN_CLIP_MINMAX,
        }
    }
}
