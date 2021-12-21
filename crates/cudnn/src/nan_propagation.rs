use crate::sys;

/// Indicates whether a given cuDNN routine should propagate Nan numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NanPropagation {
    /// NaN numbers are not propagated.
    NotPropagateNaN,
    /// Nan numbers are propagated.
    PropagateNaN,
}

impl From<NanPropagation> for sys::cudnnNanPropagation_t {
    /// Returns the corresponding raw cuDNN variant of the enum.
    fn from(nan_propagation: NanPropagation) -> sys::cudnnNanPropagation_t {
        match nan_propagation {
            NanPropagation::NotPropagateNaN => sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            NanPropagation::PropagateNaN => sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        }
    }
}

impl From<sys::cudnnNanPropagation_t> for NanPropagation {
    /// Returns the corresponding raw cuDNN variant of the enum.
    fn from(raw: sys::cudnnNanPropagation_t) -> Self {
        match raw {
            sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN => NanPropagation::NotPropagateNaN,
            sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN => NanPropagation::PropagateNaN,
        }
    }
}
