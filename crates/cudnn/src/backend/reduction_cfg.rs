use crate::{
    backend::{Descriptor, ReductionMode},
    sys, CudnnError, DataType, IntoResult,
};

#[derive(Clone, Default, PartialEq, Eq, Hash, Debug)]
pub struct ReductionCfgBuilder {
    math_precision: Option<sys::cudnnDataType_t>,
    mode: Option<ReductionMode>,
}

impl ReductionCfgBuilder {
    pub fn set_math_precision<T>(mut self) -> Self
    where
        T: DataType,
    {
        self.math_precision = Some(T::into_raw());
        self
    }

    pub fn set_mode(mut self, mode: ReductionMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn build(self) -> Result<ReductionCfg, CudnnError> {
        let math_precision = self
            .math_precision
            .unwrap_or(sys::cudnnDataType_t::CUDNN_DATA_FLOAT);

        let mode: sys::cudnnReduceTensorOp_t =
            self.mode.expect("reduction mode is required.").into();

        unsafe {
            let mut raw = Descriptor::new(
                sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_REDUCTION_DESCRIPTOR,
            )?;

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_COMP_TYPE,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
                1,
                &math_precision,
            )?;

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_REDUCTION_OPERATOR,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                1,
                &mode,
            )?;

            raw.finalize()?;

            Ok(ReductionCfg { raw })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReductionCfg {
    pub(crate) raw: Descriptor,
}
