use crate::sys;

/// Enum stating whether or not the computed results are deterministic (reproducible).
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDeterminism_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Determinism {
    /// Results are guaranteed to be reproducible.
    Deterministic,
    /// Results are not guaranteed to be reproducible.
    NonDeterministic,
}

impl From<sys::cudnnDeterminism_t> for Determinism {
    fn from(raw: sys::cudnnDeterminism_t) -> Self {
        match raw {
            sys::cudnnDeterminism_t::CUDNN_DETERMINISTIC => Self::Deterministic,
            sys::cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC => Self::NonDeterministic,
        }
    }
}

impl From<Determinism> for sys::cudnnDeterminism_t {
    fn from(determinism: Determinism) -> Self {
        match determinism {
            Determinism::Deterministic => sys::cudnnDeterminism_t::CUDNN_DETERMINISTIC,
            Determinism::NonDeterministic => sys::cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        }
    }
}
