use crate::sys;

/// Specifies a group of weights or biases for the multi-head attention layer.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMultiHeadAttnWeightKind_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttnWeight {
    /// Selects the input projection weights for queries.
    QWeights,
    /// Selects the input projection weights for keys.
    KWeights,
    /// Selects the input projection weights for values.
    VWeights,
    /// Selects the output projection weights.
    OWeights,
    /// Selects the input projection biases for queries.
    QBiases,
    /// Selects the input projection biases for keys.
    KBiases,
    /// Selects the input projection biases for values.
    VBiases,
    /// Selects the output projection biases.
    OBiases,
}

impl From<AttnWeight> for sys::cudnnMultiHeadAttnWeightKind_t {
    fn from(kind: AttnWeight) -> Self {
        match kind {
            AttnWeight::QWeights => Self::CUDNN_MH_ATTN_Q_WEIGHTS,
            AttnWeight::KWeights => Self::CUDNN_MH_ATTN_K_WEIGHTS,
            AttnWeight::VWeights => Self::CUDNN_MH_ATTN_V_WEIGHTS,
            AttnWeight::OWeights => Self::CUDNN_MH_ATTN_O_WEIGHTS,
            AttnWeight::QBiases => Self::CUDNN_MH_ATTN_Q_BIASES,
            AttnWeight::KBiases => Self::CUDNN_MH_ATTN_K_BIASES,
            AttnWeight::VBiases => Self::CUDNN_MH_ATTN_V_BIASES,
            AttnWeight::OBiases => Self::CUDNN_MH_ATTN_O_BIASES,
        }
    }
}
