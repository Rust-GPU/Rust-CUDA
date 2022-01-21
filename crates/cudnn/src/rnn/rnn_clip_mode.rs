use crate::sys;

/// Selects the LSTM cell clipping mode.
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
