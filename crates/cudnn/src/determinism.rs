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

impl From<cudnn_sys::cudnnDeterminism_t> for Determinism {
    fn from(raw: cudnn_sys::cudnnDeterminism_t) -> Self {
        use cudnn_sys::cudnnDeterminism_t::*;
        match raw {
            CUDNN_DETERMINISTIC => Self::Deterministic,
            CUDNN_NON_DETERMINISTIC => Self::NonDeterministic,
        }
    }
}

impl From<Determinism> for cudnn_sys::cudnnDeterminism_t {
    fn from(determinism: Determinism) -> Self {
        use cudnn_sys::cudnnDeterminism_t::*;
        match determinism {
            Determinism::Deterministic => CUDNN_DETERMINISTIC,
            Determinism::NonDeterministic => CUDNN_NON_DETERMINISTIC,
        }
    }
}
