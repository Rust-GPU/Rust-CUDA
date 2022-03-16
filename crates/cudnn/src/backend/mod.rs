#![allow(warnings)]

mod conv_bwd_data;
mod conv_bwd_filter;
mod conv_cfg;
mod conv_fwd;
mod descriptor;
mod engine;
mod engine_cfg;
mod engine_heuristic;
mod execution_plan;
mod graph;
mod matmul;
mod matmul_cfg;
mod operation;
mod pointwise;
mod pointwise_cfg;
mod pointwise_mode;
mod reduction;
mod reduction_cfg;
mod reduction_mode;
mod tensor;

pub use conv_bwd_data::*;
pub use conv_bwd_filter::*;
pub use conv_cfg::*;
pub use conv_fwd::*;
pub use descriptor::*;
pub use engine::*;
pub use engine_cfg::*;
pub use engine_heuristic::*;
pub use execution_plan::*;
pub use graph::*;
pub use matmul::*;
pub use matmul_cfg::*;
pub use operation::*;
pub use pointwise::*;
pub use pointwise_cfg::*;
pub use pointwise_mode::*;
pub use reduction::*;
pub use reduction_cfg::*;
pub use reduction_mode::*;
pub use tensor::*;

pub trait FloatDataType: crate::DataType {
    fn wrap(self) -> Real;
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Real {
    Float(f32),
    Double(f64),
}

impl FloatDataType for f32 {
    fn wrap(self) -> Real {
        Real::Float(self)
    }
}

impl FloatDataType for f64 {
    fn wrap(self) -> Real {
        Real::Double(self)
    }
}
