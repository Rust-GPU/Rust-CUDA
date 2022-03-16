use crate::sys;

/// A recurrent neural network algorithm.
///
/// **Do note** that double precision is only supported by `RnnAlgo::Standard`.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNAlgo_t)
/// may offer additional information about the APi behavior.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnAlgo {
    Standard,
    PersistStatic,
    PersistDynamic,
    PersistStaticSmallH,
}

impl From<RnnAlgo> for sys::cudnnRNNAlgo_t {
    fn from(algo: RnnAlgo) -> Self {
        match algo {
            RnnAlgo::Standard => sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD,
            RnnAlgo::PersistStatic => sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_STATIC,
            RnnAlgo::PersistDynamic => sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
            RnnAlgo::PersistStaticSmallH => sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_STATIC,
        }
    }
}
