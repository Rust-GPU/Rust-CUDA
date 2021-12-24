use crate::sys;

/// Data layout is padded, with outer stride from one time-step to the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqMajorUnpacked;

/// The sequence length is sorted and packed as in the basic RNN API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqMajorPacked;

/// Data layout is padded, with outer stride from one batch to the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchMajorUnpacked;

/// The data layout for input and output of a recurrent neural network.
pub trait RnnDataLayout {
    fn into_raw() -> sys::cudnnRNNDataLayout_t;
}

macro_rules! impl_rnn_data_layout {
    ($layout:ident, $raw:ident) => {
        impl RnnDataLayout for $layout {
            fn into_raw() -> sys::cudnnRNNDataLayout_t {
                sys::cudnnRNNDataLayout_t::$raw
            }
        }
    };
}

impl_rnn_data_layout!(SeqMajorPacked, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED);
impl_rnn_data_layout!(SeqMajorUnpacked, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED);
impl_rnn_data_layout!(
    BatchMajorUnpacked,
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
);
