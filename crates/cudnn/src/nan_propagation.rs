use crate::sys;

/// Indicates whether a given cuDNN routine should propagate Nan numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NanPropagation {
    /// NaN numbers are not propagated.
    NotPropagateNaN,
    /// Nan numbers are propagated.
    PropagateNaN,
}

impl NanPropagation {
    /// Returns the corresponding raw cuDNN variant of the enum.
    pub fn into_raw(self) -> sys::cudnnNanPropagation_t {
        match self {
            NanPropagation::NotPropagateNaN => sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            NanPropagation::PropagateNaN => sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        }
    }
}
