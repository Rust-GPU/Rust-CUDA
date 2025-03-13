use crate::{
    backend::{ConvCfg, Descriptor, FloatDataType, Operation, Real, Tensor},
    sys, CudnnError, DataType, IntoResult,
};

pub struct ConvBwdDataBuilder {
    cfg: Option<ConvCfg>,
    alpha: Option<Real>,
    beta: Option<Real>,
    w: Option<Tensor>,
    dx: Option<Tensor>,
    dy: Option<Tensor>,
}

impl ConvBwdDataBuilder {
    pub fn set_cfg(mut self, cfg: ConvCfg) -> Self {
        self.cfg = Some(cfg);
        self
    }

    pub fn set_alpha<T>(mut self, alpha: T) -> Self
    where
        T: FloatDataType,
    {
        self.alpha = Some(alpha.wrap());
        self
    }

    pub fn set_beta<T>(mut self, beta: T) -> Self
    where
        T: FloatDataType,
    {
        self.beta = Some(beta.wrap());
        self
    }

    pub fn set_w(mut self, w: Tensor) -> Self {
        self.w = Some(w);
        self
    }

    pub fn set_dx(mut self, dx: Tensor) -> Self {
        self.dx = Some(dx);
        self
    }

    pub fn set_dy(mut self, dy: Tensor) -> Self {
        self.dy = Some(dy);
        self
    }

    pub fn build(self) -> Result<Operation, CudnnError> {
        let cfg = self.cfg.expect("convolution configuration is required.");

        let w = self.w.expect("w tensor is required");
        let dx = self.dx.expect("dx tensor is required.");
        let dy = self.dy.expect("dy tensor is required.");

        let alpha = self.alpha.unwrap_or(Real::Float(1.0));
        let beta = self.beta.unwrap_or(Real::Float(0.0));

        unsafe {
            let mut raw = Descriptor::new(
                sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
            )?;

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &cfg.raw.inner(),
            )
            ?;

            match self.alpha {
                Some(Real::Float(ref alpha)) => raw.set_attribute(
                    sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                    sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT,
                    1,
                    alpha,
                )?,
                Some(Real::Double(ref alpha)) => raw.set_attribute(
                    sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                    sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    alpha,
                )?,
                None => (),
            }

            match self.beta {
                Some(Real::Float(ref beta)) => raw.set_attribute(
                    sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                    sys::cudnnBackendAttributeType_t::CUDNN_TYPE_FLOAT,
                    1,
                    beta,
                )?,
                Some(Real::Double(ref beta)) => raw.set_attribute(
                    sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                    sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DOUBLE,
                    1,
                    beta,
                )?,
                None => (),
            }

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &w.raw.inner(),
            )?;

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &dx.raw.inner(),
            )?;

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &dy.raw.inner(),
            )?;

            raw.finalize()?;

            Ok(Operation::ConvBwdData {
                raw,
                cfg,
                alpha,
                beta,
                w,
                dx,
                dy,
            })
        }
    }
}
