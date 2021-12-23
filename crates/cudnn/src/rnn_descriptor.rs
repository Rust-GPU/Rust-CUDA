use crate::sys;

/// A description of an recurrent neural network operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RnnDescriptor {
    raw: sys::cudnnRNNDescriptor_t,
}
