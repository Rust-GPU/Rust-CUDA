use crate::{
    backend::{Descriptor, EngineCfgBuilder, Graph},
    sys, CudnnContext, CudnnError, IntoResult,
};

pub enum HeuristicMode {
    A,
    B,
}

impl From<HeuristicMode> for sys::cudnnBackendHeurMode_t {
    fn from(mode: HeuristicMode) -> Self {
        match mode {
            HeuristicMode::A => sys::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_A,
            HeuristicMode::B => sys::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_B,
        }
    }
}
