use crate::sys;

/// Selects how buffers holding gradients of the loss function, computed with respect to trainable
/// parameters, are updated.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnWgradMode_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WGradMode {
    /// A weight gradient component, corresponding to a new batch of inputs, overwrites previously
    /// stored weight gradients in the output buffer.
    Set,
    /// A weight gradient component corresponding to a new batch of inputs is added to previously
    /// evaluated weight gradients. Before using this mode, the buffer holding weight gradients
    /// should be initialized to zero. Alternatively, the first API call outputting to an
    /// uninitialized buffer should use the `WGradMode::Set` variant.
    Add,
}

impl From<WGradMode> for sys::cudnnWgradMode_t {
    fn from(mode: WGradMode) -> Self {
        match mode {
            WGradMode::Set => Self::CUDNN_WGRAD_MODE_SET,
            WGradMode::Add => Self::CUDNN_WGRAD_MODE_ADD,
        }
    }
}
