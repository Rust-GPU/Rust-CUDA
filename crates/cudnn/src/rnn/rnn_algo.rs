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

impl From<RnnAlgo> for cudnn_sys::cudnnRNNAlgo_t {
    fn from(algo: RnnAlgo) -> Self {
        use cudnn_sys::cudnnRNNAlgo_t::*;
        match algo {
            RnnAlgo::Standard => CUDNN_RNN_ALGO_STANDARD,
            RnnAlgo::PersistStatic => CUDNN_RNN_ALGO_PERSIST_STATIC,
            RnnAlgo::PersistDynamic => CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
            RnnAlgo::PersistStaticSmallH => CUDNN_RNN_ALGO_PERSIST_STATIC,
        }
    }
}
