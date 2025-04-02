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

impl From<RnnClipMode> for cudnn_sys::cudnnRNNClipMode_t {
    fn from(mode: RnnClipMode) -> Self {
        use cudnn_sys::cudnnRNNClipMode_t::*;
        match mode {
            RnnClipMode::ClipNone => CUDNN_RNN_CLIP_NONE,
            RnnClipMode::ClipMinMax => CUDNN_RNN_CLIP_MINMAX,
        }
    }
}
