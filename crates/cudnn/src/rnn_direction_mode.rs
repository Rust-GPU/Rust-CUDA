use crate::sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnDirectionMdoe {
    Unidirectional,
    Bidirectional,
}
