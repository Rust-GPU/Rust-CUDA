use crate::sys;

/// The data layout for input and output of a recurrent neural network.
///
/// cuDNN [docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNDataLayout_t)
/// may offer additional information about the APi behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnDataLayout {
    /// Data layout is padded, with outer stride from one time-step to the next.
    SeqMajorUnpacked,
    /// The sequence length is sorted and packed as in the basic RNN API.
    SeqMajorPacked,
    /// Data layout is padded, with outer stride from one batch to the next.
    BatchMajorUnpacked,
}

impl From<RnnDataLayout> for sys::cudnnRNNDataLayout_t {
    fn from(rnn_data_layout: RnnDataLayout) -> Self {
        match rnn_data_layout {
            RnnDataLayout::SeqMajorUnpacked => {
                sys::cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED
            }
            RnnDataLayout::SeqMajorPacked => {
                sys::cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED
            }
            RnnDataLayout::BatchMajorUnpacked => {
                sys::cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
            }
        }
    }
}
