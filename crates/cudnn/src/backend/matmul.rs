use cust::memory::bytemuck::Contiguous;

use crate::{
    backend::{Descriptor, MatMulCfg, Operation, Tensor},
    sys, CudnnError, DataType, IntoResult,
};

#[derive(Clone, Default, PartialEq, Eq, Hash, Debug)]
pub struct MatMulBuilder {
    cfg: Option<MatMulCfg>,
    a: Option<Tensor>,
    b: Option<Tensor>,
    c: Option<Tensor>,
}

impl MatMulBuilder {
    pub fn set_cfg(mut self, cfg: MatMulCfg) -> Self {
        self.cfg = Some(cfg);
        self
    }

    pub fn set_a(mut self, a: Tensor) -> Self {
        self.a = Some(a);
        self
    }

    pub fn set_b(mut self, b: Tensor) -> Self {
        self.b = Some(b);
        self
    }

    pub fn set_c(mut self, c: Tensor) -> Self {
        self.c = Some(c);
        self
    }

    pub fn build(self) -> Result<Operation, CudnnError> {
        let a = self.a.expect("a matrix is required.");
        let b = self.b.expect("b matrix is required");
        let c = self.c.expect("c matrix is required");
        let cfg = self.cfg.expect("matmul configuration is required.");

        unsafe {
            let mut raw = Descriptor::new(
                sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
            )?;

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_ADESC,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &a.raw.inner(),
            )?;

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_BDESC,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &b.raw.inner(),
            )?;

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_CDESC,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &c.raw.inner(),
            )?;

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_DESC,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &cfg.raw.inner(),
            )?;

            raw.finalize()?;

            Ok(Operation::MatMul { raw, cfg, a, b, c })
        }
    }
}
