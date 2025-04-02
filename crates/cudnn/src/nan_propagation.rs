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

impl From<NanPropagation> for cudnn_sys::cudnnNanPropagation_t {
    fn from(nan_propagation: NanPropagation) -> cudnn_sys::cudnnNanPropagation_t {
        use cudnn_sys::cudnnNanPropagation_t::*;
        match nan_propagation {
            NanPropagation::NotPropagateNaN => CUDNN_NOT_PROPAGATE_NAN,
            NanPropagation::PropagateNaN => CUDNN_PROPAGATE_NAN,
        }
    }
}
