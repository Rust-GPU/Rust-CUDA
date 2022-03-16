use crate::sys;

/// Indicates whether a given cuDNN routine should propagate Nan numbers.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnNanPropagation_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NanPropagation {
    /// NaN numbers are not propagated.
    NotPropagateNaN,
    /// Nan numbers are propagated.
    PropagateNaN,
}

impl From<NanPropagation> for sys::cudnnNanPropagation_t {
    fn from(nan_propagation: NanPropagation) -> sys::cudnnNanPropagation_t {
        match nan_propagation {
            NanPropagation::NotPropagateNaN => sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            NanPropagation::PropagateNaN => sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        }
    }
}
