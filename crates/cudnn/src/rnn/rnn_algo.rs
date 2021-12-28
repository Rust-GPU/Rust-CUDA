use crate::sys;

/// A recurrent neural network algorithm.
///
/// **Do note** that double precision is only supported by `RnnAlgo::AlgoStandard`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnAlgo {
    AlgoStandard,
    AlgoPersistStatic,
    AlgoPersistDynamic,
}

impl From<sys::cudnnRNNAlgo_t> for RnnAlgo {
    fn from(raw: sys::cudnnRNNAlgo_t) -> Self {
        match raw {
            sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD => Self::AlgoStandard,
            sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_STATIC => Self::AlgoPersistStatic,
            sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_DYNAMIC => Self::AlgoPersistDynamic,
            // This whole enumeration is not documented in the cuDNN docs, the 3 fields above
            // are just briefly mentioned and the others never appear. I therefore assume they are
            // of no use.
            _ => unreachable!(),
        }
    }
}

impl From<RnnAlgo> for sys::cudnnRNNAlgo_t {
    fn from(algo: RnnAlgo) -> Self {
        match algo {
            RnnAlgo::AlgoStandard => sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD,
            RnnAlgo::AlgoPersistStatic => sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_STATIC,
            RnnAlgo::AlgoPersistDynamic => sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
        }
    }
}
