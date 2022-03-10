use crate::{
    backend::{ConvCfg, Descriptor, MatMulCfg, PointwiseCfg, Real, ReductionCfg, Tensor},
    sys,
};

#[non_exhaustive]
#[derive(Clone, PartialEq, Debug)]
pub enum Operation {
    Pointwise {
        raw: Descriptor,
        cfg: PointwiseCfg,
        x: Tensor,
        y: Tensor,
        b: Option<Tensor>,
        alpha: Option<Real>,
        beta: Option<Real>,
    },
    ConvFwd {
        raw: Descriptor,
        cfg: ConvCfg,
        alpha: Real,
        beta: Real,
        w: Tensor,
        x: Tensor,
        y: Tensor,
    },
    ConvBwdData {
        raw: Descriptor,
        cfg: ConvCfg,
        alpha: Real,
        beta: Real,
        w: Tensor,
        dx: Tensor,
        dy: Tensor,
    },
    ConvBwdFilter {
        raw: Descriptor,
        cfg: ConvCfg,
        alpha: Real,
        beta: Real,
        dw: Tensor,
        x: Tensor,
        dy: Tensor,
    },
    MatMul {
        raw: Descriptor,
        cfg: MatMulCfg,
        a: Tensor,
        b: Tensor,
        c: Tensor,
    },
    Reduction {
        raw: Descriptor,
        cfg: ReductionCfg,
        x: Tensor,
        y: Tensor,
    },
}
