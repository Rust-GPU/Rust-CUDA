use crate::{backend::Descriptor, sys, CudnnError, DataType, IntoResult};

#[derive(Clone, Default, PartialEq, Eq, Hash, Debug)]
pub struct MatMulCfgBuilder {
    compt_type: Option<sys::cudnnDataType_t>,
}

impl MatMulCfgBuilder {
    pub fn set_comp_type<T>(mut self) -> Self
    where
        T: DataType,
    {
        self.compt_type = Some(T::into_raw());
        self
    }

    pub fn build(self) -> Result<MatMulCfg, CudnnError> {
        let compt_type = self.compt_type.expect("computation type is rquired");

        unsafe {
            let mut raw = Descriptor::new(
                sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_MATMUL_DESCRIPTOR,
            )?;

            raw.set_attribute(
                sys::cudnnBackendAttributeName_t::CUDNN_ATTR_MATMUL_COMP_TYPE,
                sys::cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
                1,
                &compt_type,
            )?;

            raw.finalize()?;

            Ok(MatMulCfg { raw })
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MatMulCfg {
    pub(crate) raw: Descriptor,
}
